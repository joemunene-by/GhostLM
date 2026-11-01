"""The JAX port must agree with the PyTorch model and the MLX trainer's maths on a tiny shape."""

import json

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = jax.numpy

from ghostlm.config import GhostLMConfig  # noqa: E402
from ghostlm.model import GhostLM  # noqa: E402
from ghostlm.tpu import optim as tpu_optim  # noqa: E402
from ghostlm.tpu.data import CurriculumSampler, val_batches  # noqa: E402
from ghostlm.tpu.model import GhostLMFlax, cross_entropy, from_torch_state, init_params, param_count, to_torch_state  # noqa: E402

VOCAB, EOS = 101, 100


def tiny_config(**overrides) -> GhostLMConfig:
    c = GhostLMConfig.from_preset("ghost-small-v0.5")
    c.n_layers, c.d_model, c.n_heads, c.n_kv_heads, c.d_ff = 2, 64, 4, 2, 128
    c.use_qk_norm = c.attn_gate = c.value_residual = c.intra_doc_mask = True
    c.vocab_size, c.eos_token_id, c.context_length, c.dropout = VOCAB, EOS, 16, 0.0
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def tokens(rng, batch=2, length=16):
    idx = rng.integers(0, VOCAB - 1, size=(batch, length))
    idx[0, 5] = EOS
    idx[1, 9] = EOS
    return idx


def test_logits_match_pytorch():
    cfg = tiny_config()
    torch.manual_seed(0)
    torch_model = GhostLM(cfg).eval()
    params = from_torch_state(init_params(cfg), torch_model.state_dict())
    assert param_count(params) == sum(p.numel() for p in torch_model.parameters())
    idx = tokens(np.random.default_rng(0))
    with torch.no_grad():
        expected = torch_model(torch.from_numpy(idx))
        expected = expected[0] if isinstance(expected, tuple) else expected
    got = GhostLMFlax(cfg).apply({"params": params}, jnp.asarray(idx))
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=2e-4, rtol=1e-3)


def test_torch_state_round_trip():
    cfg = tiny_config()
    params = init_params(cfg, seed=3)
    back = from_torch_state(init_params(cfg, seed=4), to_torch_state(params))
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(back)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_loss_ignores_padding_and_matches_torch():
    cfg = tiny_config()
    torch.manual_seed(1)
    torch_model = GhostLM(cfg).eval()
    params = from_torch_state(init_params(cfg), torch_model.state_dict())
    idx = tokens(np.random.default_rng(1))
    targets = np.roll(idx, -1, axis=1)
    targets[:, -1] = -1
    logits = GhostLMFlax(cfg).apply({"params": params}, jnp.asarray(idx))
    got = float(cross_entropy(logits, jnp.asarray(targets)))
    with torch.no_grad():
        ref = torch_model(torch.from_numpy(idx))
        ref = ref[0] if isinstance(ref, tuple) else ref
        expected = torch.nn.functional.cross_entropy(
            ref.reshape(-1, VOCAB).float(), torch.from_numpy(targets).reshape(-1), ignore_index=-1)
    assert abs(got - float(expected)) < 2e-4


@pytest.mark.parametrize("optimizer", ["adamw", "normuon"])
def test_optimizer_lowers_loss(optimizer):
    cfg = tiny_config(optimizer=optimizer, learning_rate=3e-3, warmup_steps=1, max_steps=50, cautious_wd=True)
    params = init_params(cfg, seed=0)
    opt = tpu_optim.Optimizer(cfg, params)
    state = opt.init(params)
    model = GhostLMFlax(cfg)
    idx = tokens(np.random.default_rng(2), batch=4)
    targets = np.roll(idx, -1, axis=1)
    targets[:, -1] = -1
    x, y = jnp.asarray(idx), jnp.asarray(targets)
    loss_fn = jax.jit(jax.value_and_grad(lambda p: cross_entropy(model.apply({"params": p}, x), y)))
    update = jax.jit(opt.update)
    first, _ = loss_fn(params)
    for step in range(12):
        loss, grads = loss_fn(params)
        grads, _ = tpu_optim.clip_by_global_norm(grads, cfg.grad_clip)
        params, state = update(params, grads, state, jnp.float32(tpu_optim.lr_at(step, cfg)))
    last, _ = loss_fn(params)
    assert float(last) < float(first) - 0.05
    assert int(state[("step",)]) == 12


def test_lr_schedule_matches_pytorch_trainer():
    from types import SimpleNamespace
    from ghostlm.trainer import GhostTrainer
    for schedule in ("cosine", "wsd"):
        cfg = tiny_config(lr_schedule=schedule, warmup_steps=10, max_steps=100, learning_rate=1e-3)
        for step in (0, 5, 9, 10, 50, 79, 80, 99):
            stand_in = SimpleNamespace(step=step, config=cfg)
            stand_in.wsd_decay_start = lambda: GhostTrainer.wsd_decay_start(stand_in)
            reference = GhostTrainer.get_lr(stand_in)
            assert tpu_optim.lr_at(step, cfg) == pytest.approx(reference)


def test_sampler_matches_torch_dataset(tmp_path):
    from ghostlm.curriculum import DEFAULT_GENERALIST_CURRICULUM
    from ghostlm.dataset import MultiDomainBinDataset
    rng = np.random.default_rng(0)
    bins = {}
    for domain in ("general_web", "code"):
        arr = rng.integers(0, VOCAB, size=5000).astype(np.uint16)
        path = tmp_path / f"train.{domain}.bin"
        arr.tofile(path)
        (tmp_path / f"train.{domain}.meta.json").write_text(json.dumps({"dtype": "uint16"}))
        bins[domain] = str(path)
    cfg = tiny_config(context_length=16, seed=42)
    sampler = CurriculumSampler(bins, 16, DEFAULT_GENERALIST_CURRICULUM, seed=42)
    torch_ds = MultiDomainBinDataset(bins, cfg, DEFAULT_GENERALIST_CURRICULUM, progress_fn=lambda: 0.0, seed=42)
    it = iter(torch_ds)
    for n in range(6):
        tx, ty = next(it)
        x, y = sampler.sample(n, 0.0)
        np.testing.assert_array_equal(x, tx.numpy())
        np.testing.assert_array_equal(y, ty.numpy())
    xb, yb = sampler.batch(0, 3, 0.0)
    assert xb.shape == (3, 16) and yb.shape == (3, 16)


def test_val_batches_pad_last_target(tmp_path):
    arr = np.arange(33, dtype=np.uint16)
    path = tmp_path / "val.bin"
    arr.tofile(path)
    batches = list(val_batches(path, context=16, batch_size=2, num_batches=5))
    assert len(batches) == 1
    x, y = batches[0]
    assert x.shape == (2, 16)
    assert y[0, 0] == 1 and y[1, -1] == 32
