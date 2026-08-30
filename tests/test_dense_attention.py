import torch
import torch.nn.functional as F

from sparse_attention.dense_attention import causal_mask, dense_attention, manual_softmax


def test_manual_softmax_matches_torch():
    x = torch.randn(4, 10, 10)
    ours = manual_softmax(x, dim=-1)
    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(ours, ref, atol=1e-6)
    # every row of a softmax must sum to 1
    assert torch.allclose(ours.sum(dim=-1), torch.ones(4, 10), atol=1e-6)


def test_dense_attention_matches_reference_no_mask():
    torch.manual_seed(0)
    batch, seq_len, head_dim = 2, 16, 8
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)

    ours, _ = dense_attention(q, k, v, mask=None)
    ref = F.scaled_dot_product_attention(q, k, v)

    assert torch.allclose(ours, ref, atol=1e-5)


def test_dense_attention_matches_reference_causal():
    torch.manual_seed(0)
    batch, seq_len, head_dim = 2, 16, 8
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)
    mask = causal_mask(seq_len)

    ours, _ = dense_attention(q, k, v, mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.allclose(ours, ref, atol=1e-5)


def test_causal_mask_shape_and_values():
    mask = causal_mask(5)
    assert mask.shape == (5, 5)
    assert mask.dtype == torch.bool
    # position 0 can only see itself
    assert mask[0].tolist() == [True, False, False, False, False]
    # position 4 (last) can see everyone
    assert mask[4].tolist() == [True, True, True, True, True]


def test_attention_weights_sum_to_one():
    torch.manual_seed(1)
    q = torch.randn(1, 8, 4)
    k = torch.randn(1, 8, 4)
    v = torch.randn(1, 8, 4)
    mask = causal_mask(8)

    _, weights = dense_attention(q, k, v, mask=mask)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
