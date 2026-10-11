import math

import pytest
import torch

from tenslam.model import (ModelConfig, RMSNorm, TenslamLM, apply_rope, build_rope_cache)
from tenslam.sanity import check_backward, check_forward, check_overfit

SMALL = ModelConfig(vocab_size=64, n_layers=2, d_model=32, n_heads=4, d_ff=64, max_seq_len=32)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return TenslamLM(SMALL)


def batch(b=2, t=16, v=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, v, (b, t + 1), generator=g)
    return data[:, :-1], data[:, 1:]


def test_full_config_is_about_30m_parameters():
    m = TenslamLM(ModelConfig.from_toml("configs/model.toml"))
    assert m.cfg.n_layers == 8 and m.cfg.d_model == 512 and m.cfg.n_heads == 8
    assert m.cfg.max_seq_len == 512 and m.cfg.vocab_size == 8192
    assert 29_000_000 <= m.num_params() <= 31_000_000


def test_parameter_count_matches_formula():
    c = SMALL
    per_block = 4 * c.d_model**2 + 3 * c.d_model * c.d_ff + 2 * c.d_model
    expected = c.vocab_size * c.d_model + c.n_layers * per_block + c.d_model  # tied head
    assert TenslamLM(c).num_params() == expected


def test_embeddings_are_tied(model):
    assert model.lm_head.weight is model.tok_emb.weight


def test_forward_shape_and_initial_loss_near_ln_vocab(model):
    x, y = batch()
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 64)
    assert abs(loss.item() - math.log(64)) < 0.3


def test_forward_without_targets_returns_no_loss(model):
    assert model(batch()[0])[1] is None


def test_causality_future_tokens_cannot_change_past_logits(model):
    model.eval()
    x, _ = batch()
    x2 = x.clone()
    x2[:, 10:] = (x2[:, 10:] + 1) % 64  # change only positions >= 10
    with torch.no_grad():
        a, b = model(x)[0], model(x2)[0]
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-5)


def test_batch_items_are_independent(model):
    model.eval()
    x, _ = batch()
    with torch.no_grad():
        both, alone = model(x)[0], model(x[:1])[0]
    assert torch.allclose(both[:1], alone, atol=1e-5)


def test_ignore_index_excludes_positions_from_loss(model):
    x, y = batch()
    y2 = y.clone(); y2[:, 8:] = -100
    _, full = model(x, y); _, part = model(x, y2)
    _, first_half = model(x[:, :8], y[:, :8])
    assert not torch.isclose(full, part)
    assert torch.isclose(part, first_half, atol=1e-5)  # causal: first 8 positions unaffected


def test_all_parameters_get_gradients_and_update(model):
    r = check_backward(model, *batch(), lr=1e-2)
    assert r["ok"], r


def test_init_is_deterministic_per_seed():
    torch.manual_seed(5); a = TenslamLM(SMALL)
    torch.manual_seed(5); b = TenslamLM(SMALL)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_residual_projections_use_scaled_init():
    m = TenslamLM(ModelConfig(**{**SMALL.to_dict(), "d_model": 128, "n_heads": 4, "d_ff": 256}))
    std_plain = m.blocks[0].attn.qkv.weight.std().item()
    std_proj = m.blocks[0].attn.proj.weight.std().item()
    assert std_proj < std_plain * 0.7  # 0.02/sqrt(2L) vs 0.02


def test_rmsnorm_matches_formula():
    x = torch.randn(3, 5, 16)
    n = RMSNorm(16, eps=1e-5)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(n(x), expected, atol=1e-6)


def test_rope_preserves_norm_and_position_zero_is_identity():
    cos, sin = build_rope_cache(16, 8, 10000.0)
    x = torch.randn(1, 2, 16, 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    assert torch.allclose(x[:, :, 0], y[:, :, 0], atol=1e-6)


def test_rope_scores_depend_only_on_relative_position():
    cos, sin = build_rope_cache(32, 8, 10000.0)
    q, k = torch.randn(8), torch.randn(8)

    def score(m, n):
        qm = apply_rope(q.view(1, 1, 1, 8).expand(1, 1, 32, 8), cos, sin)[0, 0, m]
        kn = apply_rope(k.view(1, 1, 1, 8).expand(1, 1, 32, 8), cos, sin)[0, 0, n]
        return (qm @ kn).item()

    assert score(5, 2) == pytest.approx(score(20, 17), abs=1e-4)  # same offset 3
    assert score(5, 2) != pytest.approx(score(5, 3), abs=1e-4)


def test_attention_matches_manual_masked_softmax(model):
    attn = model.blocks[0].attn
    x = torch.randn(2, 12, 32)
    cos, sin = model.rope_cos, model.rope_sin
    got = attn(x, cos, sin)
    b, t, c = x.shape
    q, k, v = attn.qkv(x).view(b, t, 3, 4, 8).permute(2, 0, 3, 1, 4)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = q @ k.transpose(-1, -2) / math.sqrt(8)
    scores = scores.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
    want = attn.proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, c))
    assert torch.allclose(got, want, atol=1e-5)


def test_sequence_longer_than_context_is_rejected(model):
    with pytest.raises(AssertionError):
        model(torch.zeros(1, 33, dtype=torch.long))


def test_bad_head_configuration_is_rejected():
    with pytest.raises(AssertionError):
        ModelConfig(d_model=30, n_heads=4)


def test_check_forward_reports_ok(model):
    r = check_forward(model, *batch())
    assert r["ok"] and r["logits_shape"] == (2, 16, 64)


def test_tiny_model_can_overfit_a_single_batch():
    torch.manual_seed(0)
    m = TenslamLM(SMALL)
    r = check_overfit(m, *batch(b=2, t=16), steps=150, lr=3e-3, log_every=1000)
    assert r["final_loss"] < 0.1 * r["first_loss"], r
