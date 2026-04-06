"""GhostLM unit tests — validates model architecture, tokenizer, and config."""

import pytest
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
