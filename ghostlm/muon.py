"""Muon optimizer for the hidden matrices, with AdamW for everything else.

Muon (Keller Jordan, 2024) replaces each 2D weight's momentum update with its
nearest semi-orthogonal matrix, computed by a few Newton-Schulz iterations.
Updates are rescaled by 0.2 * sqrt(max(rows, cols)) so their RMS matches
AdamW's (Moonshot AI, "Muon is Scalable for LLM Training", 2025); that lets
Muon share AdamW's learning rate, weight decay and LR schedule unchanged.

``normuon=True`` adds NorMuon's per-neuron normalization (Li et al., 2025):
orthogonalized updates have very uneven row norms, so each output row is
divided by an EMA of its mean squared update before the RMS rescale.
``cautious_wd=True`` only decays coordinates whose update has the same sign
as the weight (cautious weight decay).
"""

import torch


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately orthogonalize ``g`` (2D) with a quintic Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    dtype = torch.bfloat16 if g.is_cuda else torch.float32
    x = g.to(dtype)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * A @ A
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """One optimizer over two kinds of param groups.

    Groups with ``use_muon=True`` must hold only 2D tensors and get Muon
    updates; all other groups get AdamW. Every group keeps an ``lr`` key, so
    a trainer that rewrites ``group["lr"]`` each step drives both halves.
    """

    def __init__(self, param_groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, momentum=0.95, ns_steps=5, normuon=False,
                 cautious_wd=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, ns_steps=ns_steps, use_muon=False,
                        normuon=normuon, cautious_wd=cautious_wd)
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            if group["use_muon"]:
                bad = [tuple(p.shape) for p in group["params"] if p.ndim != 2]
                if bad:
                    raise ValueError(f"Muon groups take 2D params only, got shapes {bad}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["use_muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(p.grad)
                    update = p.grad.add(buf, alpha=group["momentum"])  # Nesterov
                    update = zeropower_via_newtonschulz5(update, group["ns_steps"])
                    if group["normuon"]:
                        if "row_sq" not in state:
                            state["row_sq"] = torch.zeros(p.size(0), 1, device=p.device, dtype=p.dtype)
                        state["row_sq"].lerp_(update.square().mean(dim=1, keepdim=True),
                                              1 - group["betas"][1])
                        update = update / (state["row_sq"].sqrt() + group["eps"])
                        update.mul_(0.2 * p.numel() ** 0.5 / (update.norm() + group["eps"]))
                    else:
                        update.mul_(0.2 * max(p.size(0), p.size(1)) ** 0.5)
                    self._decay(p, update, lr * wd, group["cautious_wd"])
                    p.add_(update, alpha=-lr)
                else:
                    beta1, beta2 = group["betas"]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.lerp_(p.grad, 1 - beta1)
                    v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    m_hat = m / (1 - beta1 ** state["step"])
                    v_hat = v / (1 - beta2 ** state["step"])
                    update = m_hat / v_hat.sqrt().add_(group["eps"])
                    self._decay(p, update, lr * wd, group["cautious_wd"])
                    p.add_(update, alpha=-lr)
        return loss

    @staticmethod
    def _decay(p: torch.Tensor, update: torch.Tensor, amount: float, cautious: bool) -> None:
        if amount == 0:
            return
        if cautious:
            p.sub_(p * (update * p > 0) * amount)
        else:
            p.mul_(1 - amount)
