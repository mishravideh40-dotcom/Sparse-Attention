import torch

from sparse_attention.sliding_window import sliding_window_attention, sliding_window_attention_fast


def _check_matches_reference(batch, seq_len, head_dim, window_size, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)

    ref_out, _ = sliding_window_attention(q, k, v, window_size=window_size)
    fast_out, fast_weights = sliding_window_attention_fast(q, k, v, window_size=window_size)

    assert fast_out.shape == ref_out.shape
    assert torch.allclose(fast_out, ref_out, atol=1e-5), (
        f"batch={batch} seq_len={seq_len} head_dim={head_dim} window_size={window_size}"
    )
    return fast_weights


def test_matches_reference_typical_case():
    _check_matches_reference(batch=2, seq_len=20, head_dim=8, window_size=5)


def test_matches_reference_window_size_one():
    # each query can only see itself
    weights = _check_matches_reference(batch=1, seq_len=10, head_dim=4, window_size=1)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_matches_reference_window_covers_whole_sequence():
    # window_size >= seq_len -> every query sees its entire causal history,
    # same case the reference tests against plain dense causal attention
    _check_matches_reference(batch=1, seq_len=12, head_dim=6, window_size=12)
    _check_matches_reference(batch=1, seq_len=12, head_dim=6, window_size=100)


def test_matches_reference_seq_len_one():
    _check_matches_reference(batch=1, seq_len=1, head_dim=4, window_size=3)


def test_weights_sum_to_one_and_no_nans():
    torch.manual_seed(1)
    batch, seq_len, head_dim, window_size = 3, 15, 8, 4
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)

    _, weights = sliding_window_attention_fast(q, k, v, window_size=window_size)
    assert not torch.isnan(weights).any()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq_len), atol=1e-6)


def test_fast_weights_shape_is_windowed_not_full():
    batch, seq_len, head_dim, window_size = 1, 20, 4, 6
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)
    _, weights = sliding_window_attention_fast(q, k, v, window_size=window_size)
    assert weights.shape == (batch, seq_len, window_size)
