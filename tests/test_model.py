"""GhostLM unit tests — validates model architecture, tokenizer, and config."""

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


def test_config_defaults():
    """Test that default configuration values are set correctly."""
    config = GhostLMConfig()
    assert config.vocab_size == 50257
    assert config.n_layers == 6
    assert config.d_model == 512
    assert config.n_heads == 8


def test_config_presets():
    """Test that preset configurations return correct hyperparameters."""
    tiny = GhostLMConfig.from_preset("ghost-tiny")
    assert tiny.n_layers == 2
    assert tiny.d_model == 256

    small = GhostLMConfig.from_preset("ghost-small")
    assert small.n_layers == 6
    assert small.d_model == 512

    medium = GhostLMConfig.from_preset("ghost-medium")
    assert medium.n_layers == 12
    assert medium.d_model == 768


def test_config_model_size():
    """Test that model_size() returns a human-readable string."""
    config = GhostLMConfig.from_preset("ghost-small")
    size_str = config.model_size()
    assert isinstance(size_str, str)
    assert size_str.endswith("M") or size_str.endswith("B")


def test_tokenizer_encode_decode():
    """Test that encoding and decoding preserves text content."""
    tokenizer = GhostTokenizer()
    text = "CVE-2023-1234 is a critical buffer overflow"
    ids = tokenizer.encode(text)

    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)

    decoded = tokenizer.decode(ids)
    assert text.lower() in decoded.lower()


def test_tokenizer_special_tokens():
    """Test that BOS and EOS tokens are correctly added."""
    tokenizer = GhostTokenizer()
    ids = tokenizer.encode("test text", add_bos=True, add_eos=True)

    assert ids[0] == tokenizer._special_tokens[GhostTokenizer.BOS]
    assert ids[-1] == tokenizer._special_tokens[GhostTokenizer.EOS]


def test_tokenizer_chunk_text():
    """Test that long text is split into overlapping chunks correctly."""
    tokenizer = GhostTokenizer()
    long_text = "security vulnerability " * 200
    chunks = tokenizer.chunk_text(long_text, chunk_size=100, overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_model_forward_pass():
    """Test that model forward pass produces correct output shapes."""
    config = GhostLMConfig.from_preset("ghost-tiny")
    config.vocab_size = 50261
    config.context_length = 64

    model = GhostLM(config)
    x = torch.randint(0, 50261, (2, 64))

    logits, loss = model(x)

    assert logits.shape == (2, 64, 50261)
    assert loss is None


def test_model_forward_with_loss():
    """Test that model computes loss when targets are provided."""
    config = GhostLMConfig.from_preset("ghost-tiny")
    config.vocab_size = 50261
    config.context_length = 64

    model = GhostLM(config)
    x = torch.randint(0, 50261, (2, 64))

    logits, loss = model(x, targets=x)

    assert loss is not None
    assert loss.item() > 0


def test_model_generate():
    """Test autoregressive generation produces expected output length."""
    config = GhostLMConfig.from_preset("ghost-tiny")
    config.vocab_size = 50261
    config.context_length = 64

    model = GhostLM(config)
    x = torch.randint(0, 50261, (1, 10))

    generated = model.generate(x, max_new_tokens=20)

    assert generated.shape == (1, 30)


def test_model_num_params():
    """Test that parameter count is within expected range for ghost-tiny."""
    config = GhostLMConfig.from_preset("ghost-tiny")
    config.vocab_size = 50261
    config.context_length = 64

    model = GhostLM(config)
    n_params = model.num_params()

    assert n_params > 0
    assert n_params < 50_000_000


def test_model_with_rope():
    """Test that model works with RoPE enabled."""
    config = GhostLMConfig.from_preset("ghost-tiny")
    config.vocab_size = 50261
    config.context_length = 64
    config.use_rope = True

    model = GhostLM(config)
    x = torch.randint(0, 50261, (2, 32))

    logits, loss = model(x)
    assert logits.shape == (2, 32, 50261)

    # RoPE model should not have pos_embedding
    assert not hasattr(model, "pos_embedding")


def test_model_with_rope_and_loss():
    """Test that RoPE model computes loss correctly."""
    config = GhostLMConfig.from_preset("ghost-tiny")
    config.vocab_size = 50261
    config.context_length = 64
    config.use_rope = True

    model = GhostLM(config)
    x = torch.randint(0, 50261, (2, 32))

    logits, loss = model(x, targets=x)
    assert loss is not None
    assert loss.item() > 0


def test_model_with_flash_attention():
    """Test that model works with Flash Attention enabled."""
    config = GhostLMConfig.from_preset("ghost-tiny")
    config.vocab_size = 50261
    config.context_length = 64
    config.use_flash_attention = True

    model = GhostLM(config)
    x = torch.randint(0, 50261, (2, 32))

    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 32, 50261)
    assert loss is not None
    assert loss.item() > 0


def test_model_with_rope_and_flash_attention():
    """Test that RoPE and Flash Attention work together."""
    config = GhostLMConfig.from_preset("ghost-tiny")
    config.vocab_size = 50261
    config.context_length = 64
    config.use_rope = True
    config.use_flash_attention = True

    model = GhostLM(config)
    x = torch.randint(0, 50261, (2, 32))

    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 32, 50261)
    assert loss is not None


def test_rope_and_learned_pos_output_differ():
    """Test that RoPE and learned positional embeddings produce different outputs."""
    torch.manual_seed(42)
    config_rope = GhostLMConfig.from_preset("ghost-tiny")
    config_rope.vocab_size = 50261
    config_rope.context_length = 64
    config_rope.use_rope = True

    torch.manual_seed(42)
    config_learned = GhostLMConfig.from_preset("ghost-tiny")
    config_learned.vocab_size = 50261
    config_learned.context_length = 64
    config_learned.use_rope = False

    model_rope = GhostLM(config_rope)
    model_learned = GhostLM(config_learned)

    x = torch.randint(0, 50261, (1, 16))

    logits_rope, _ = model_rope(x)
    logits_learned, _ = model_learned(x)

    # Outputs should differ since position encoding method is different
    assert not torch.allclose(logits_rope, logits_learned)


def test_model_generate_with_rope():
    """Test autoregressive generation works with RoPE."""
    config = GhostLMConfig.from_preset("ghost-tiny")
    config.vocab_size = 50261
    config.context_length = 64
    config.use_rope = True

    model = GhostLM(config)
    x = torch.randint(0, 50261, (1, 10))

    generated = model.generate(x, max_new_tokens=20)
    assert generated.shape == (1, 30)


def test_gqa_shapes_and_compact_cache():
    """GQA projects K/V at n_kv_heads and caches them compactly."""
    config = GhostLMConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128,
        vocab_size=200, context_length=32, dropout=0.0, use_rope=True,
    )
    model = GhostLM(config)
    head_dim = 64 // 4

    # QKV projection: 4 query heads + 2 K heads + 2 V heads.
    assert model.blocks[0].attn.c_qkv.out_features == (4 + 2 + 2) * head_dim

    x = torch.randint(0, 200, (2, 8))
    logits, _, kv = model(x, use_cache=True)
    assert logits.shape == (2, 8, 200)
    # Cached K/V stay at n_kv_heads — half the memory of MHA here.
    assert kv[0][0].shape == (2, 2, 8, head_dim)

    # Optimizer setup must categorize every parameter.
    model.configure_optimizers(config)


def test_gqa_defaults_to_mha():
    """n_kv_heads=None must reproduce the historical MHA checkpoint shapes."""
    config = GhostLMConfig(
        n_layers=1, d_model=64, n_heads=4, d_ff=128,
        vocab_size=200, context_length=32,
    )
    model = GhostLM(config)
    assert model.blocks[0].attn.c_qkv.out_features == 3 * 64
    assert model.blocks[0].attn.n_rep == 1


def test_qk_norm_modules_and_forward():
    """QK-norm adds per-head RMSNorms and keeps the forward pass finite."""
    config = GhostLMConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=128,
        vocab_size=200, context_length=32, dropout=0.0,
        use_rope=True, use_rmsnorm=True, use_qk_norm=True,
    )
    model = GhostLM(config)
    attn = model.blocks[0].attn
    assert attn.q_norm.weight.shape == (16,)
    assert attn.k_norm.weight.shape == (16,)

    x = torch.randint(0, 200, (2, 8))
    logits, loss = model(x, targets=x)
    assert torch.isfinite(loss)

    # The norm weights must land in the no-decay group.
    model.configure_optimizers(config)


def test_rope_base_flows_into_rotary_embedding():
    """config.rope_base must change the RoPE frequency table."""
    base_cfg = GhostLMConfig(
        n_layers=1, d_model=64, n_heads=4, d_ff=128,
        vocab_size=200, context_length=32, use_rope=True,
    )
    long_cfg = GhostLMConfig(
        n_layers=1, d_model=64, n_heads=4, d_ff=128,
        vocab_size=200, context_length=32, use_rope=True,
        rope_base=1_000_000.0,
    )
    rope_default = GhostLM(base_cfg).blocks[0].attn.rope
    rope_long = GhostLM(long_cfg).blocks[0].attn.rope
    assert not torch.allclose(rope_default.inv_freq, rope_long.inv_freq)
    # Higher base means slower-rotating high dims (smaller frequencies).
    assert rope_long.inv_freq[-1] < rope_default.inv_freq[-1]
