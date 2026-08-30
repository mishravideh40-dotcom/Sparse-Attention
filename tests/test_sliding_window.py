import torch

from sparse_attention.dense_attention import causal_mask, dense_attention
from sparse_attention.sliding_window import sliding_window_attention, sliding_window_mask


def test_mask_shape_and_dtype():
    mask = sliding_window_mask(10, 3)
    assert mask.shape == (10, 10)
    assert mask.dtype == torch.bool


def test_mask_never_sees_future():
    mask = sliding_window_mask(10, 4)
    causal = causal_mask(10)
    # sliding window must never allow anything the causal mask forbids
    assert torch.all(~mask | causal)


def test_mask_window_size_respected():
    window_size = 3
    mask = sliding_window_mask(10, window_size)
    for i in range(10):
        allowed = mask[i].nonzero().flatten().tolist()
        assert allowed == list(range(max(0, i - window_size + 1), i + 1))


def test_reduces_to_dense_causal_when_window_covers_all():
    torch.manual_seed(0)
    seq_len, head_dim = 12, 6
    q = torch.randn(1, seq_len, head_dim)
    k = torch.randn(1, seq_len, head_dim)
    v = torch.randn(1, seq_len, head_dim)

    dense_out, _ = dense_attention(q, k, v, mask=causal_mask(seq_len))
    sparse_out, _ = sliding_window_attention(q, k, v, window_size=seq_len)

    assert torch.allclose(dense_out, sparse_out, atol=1e-6)


def test_weights_are_zero_outside_window():
    window_size = 3
    torch.manual_seed(0)
    seq_len, head_dim = 8, 4
    q = torch.randn(1, seq_len, head_dim)
    k = torch.randn(1, seq_len, head_dim)
    v = torch.randn(1, seq_len, head_dim)

    _, weights = sliding_window_attention(q, k, v, window_size=window_size)
    mask = sliding_window_mask(seq_len, window_size)
    outside = ~mask
    assert torch.allclose(weights[0][outside], torch.zeros_like(weights[0][outside]), atol=1e-8)


def test_weights_sum_to_one():
    torch.manual_seed(1)
    seq_len, head_dim = 8, 4
    q = torch.randn(1, seq_len, head_dim)
    k = torch.randn(1, seq_len, head_dim)
    v = torch.randn(1, seq_len, head_dim)
    _, weights = sliding_window_attention(q, k, v, window_size=3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, seq_len), atol=1e-6)
