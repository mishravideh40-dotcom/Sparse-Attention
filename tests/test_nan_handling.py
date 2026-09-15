import torch

from sparse_attention.dense_attention import dense_attention, manual_softmax


def test_softmax_fully_masked_row_is_zero_not_nan():
    x = torch.tensor([[1.0, 2.0, 3.0], [float("-inf"), float("-inf"), float("-inf")]])
    weights = manual_softmax(x, dim=-1)
    assert not torch.isnan(weights).any()
    assert torch.allclose(weights[1], torch.zeros(3))
    # untouched row still behaves like a normal softmax
    assert torch.allclose(weights[0].sum(), torch.tensor(1.0))


def test_softmax_partially_masked_rows_unaffected():
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    x[2, 3:] = float("-inf")  # one row partially masked, still has valid entries
    weights = manual_softmax(x, dim=-1)
    assert not torch.isnan(weights).any()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_dense_attention_fully_masked_query_gives_zero_output():
    torch.manual_seed(0)
    seq_len, head_dim = 6, 4
    q = torch.randn(1, seq_len, head_dim)
    k = torch.randn(1, seq_len, head_dim)
    v = torch.randn(1, seq_len, head_dim)

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    mask[0, :] = False  # query 0 is allowed to attend to nothing

    out, weights = dense_attention(q, k, v, mask=mask)

    assert not torch.isnan(out).any()
    assert not torch.isnan(weights).any()
    assert torch.allclose(weights[0, 0], torch.zeros(seq_len))
    assert torch.allclose(out[0, 0], torch.zeros(head_dim))

    # every other (fully-visible) row is a completely normal softmax
    assert torch.allclose(weights[0, 1:].sum(dim=-1), torch.ones(seq_len - 1), atol=1e-6)


def test_dense_attention_mixed_masked_and_unmasked_queries():
    torch.manual_seed(1)
    seq_len, head_dim = 5, 3
    q = torch.randn(1, seq_len, head_dim)
    k = torch.randn(1, seq_len, head_dim)
    v = torch.randn(1, seq_len, head_dim)

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    mask[1, :] = False
    mask[3, :] = False

    out, weights = dense_attention(q, k, v, mask=mask)

    assert not torch.isnan(out).any()
    for masked_row in (1, 3):
        assert torch.allclose(out[0, masked_row], torch.zeros(head_dim))
    for ok_row in (0, 2, 4):
        assert torch.allclose(weights[0, ok_row].sum(), torch.tensor(1.0), atol=1e-6)
