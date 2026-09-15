import torch

from sparse_attention.bigbird import bigbird_attention_fast
from sparse_attention.dense_attention import causal_mask, dense_attention


def test_global_rows_match_dense_causal_exactly():
    # Global query rows attend to their entire causal history by
    # construction -- for those rows the fast kernel should be bit-for-bit
    # the same as plain causal dense attention, since it computes them the
    # exact same way (no randomness or gathering involved for these rows).
    torch.manual_seed(0)
    batch, seq_len, head_dim = 2, 30, 8
    num_global = 5
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)

    out, info = bigbird_attention_fast(q, k, v, window_size=4, num_global=num_global, num_random=3)
    dense_out, _ = dense_attention(q[:, :num_global, :], k, v, mask=causal_mask(seq_len)[:num_global])

    assert info["num_global_rows"] == num_global
    assert torch.allclose(out[:, :num_global, :], dense_out, atol=1e-6)


def test_regular_rows_reduce_to_dense_causal_when_window_covers_all():
    # num_global=0, num_random=0, window_size >= seq_len: every row is a
    # "regular" row whose window alone already covers its entire causal
    # history, so the whole output must equal plain causal dense attention
    # exactly -- no randomness is involved in this configuration at all.
    torch.manual_seed(1)
    batch, seq_len, head_dim = 1, 16, 6
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)

    out, info = bigbird_attention_fast(q, k, v, window_size=seq_len, num_global=0, num_random=0)
    dense_out, _ = dense_attention(q, k, v, mask=causal_mask(seq_len))

    assert info["num_global_rows"] == 0
    assert torch.allclose(out, dense_out, atol=1e-5)


def test_no_nan_and_weights_sum_to_one():
    torch.manual_seed(2)
    batch, seq_len, head_dim = 2, 64, 8
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)

    out, info = bigbird_attention_fast(
        q, k, v, window_size=8, num_global=4, num_random=6, generator=torch.Generator().manual_seed(0)
    )
    assert not torch.isnan(out).any()

    assert torch.allclose(
        info["global_weights"].sum(dim=-1), torch.ones(batch, info["num_global_rows"]), atol=1e-6
    )
    num_regular = seq_len - info["num_global_rows"]
    assert torch.allclose(info["regular_weights"].sum(dim=-1), torch.ones(batch, num_regular), atol=1e-6)


def test_regular_rows_never_attend_to_future():
    torch.manual_seed(3)
    batch, seq_len, head_dim = 1, 50, 4
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)

    _, info = bigbird_attention_fast(
        q, k, v, window_size=6, num_global=3, num_random=5, generator=torch.Generator().manual_seed(0)
    )
    reg_indices, reg_valid, num_global_rows = info["regular_indices"], info["regular_valid"], info["num_global_rows"]
    row_positions = torch.arange(num_global_rows, seq_len).unsqueeze(1)  # absolute row index per regular row
    # every *valid* gathered index must be <= that row's own absolute position
    violations = reg_valid & (reg_indices > row_positions)
    assert not violations.any()


def test_output_shape():
    batch, seq_len, head_dim = 3, 40, 8
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)
    out, _ = bigbird_attention_fast(q, k, v, window_size=5, num_global=2, num_random=4)
    assert out.shape == (batch, seq_len, head_dim)


def test_zero_global_zero_random_is_pure_sliding_window():
    # with num_global=0 and num_random=0, every row is regular and its
    # candidate set is exactly the sliding window -- compare directly
    # against the existing sliding_window_attention reference.
    from sparse_attention.sliding_window import sliding_window_attention

    torch.manual_seed(4)
    batch, seq_len, head_dim, window_size = 2, 25, 6, 5
    q = torch.randn(batch, seq_len, head_dim)
    k = torch.randn(batch, seq_len, head_dim)
    v = torch.randn(batch, seq_len, head_dim)

    out, _ = bigbird_attention_fast(q, k, v, window_size=window_size, num_global=0, num_random=0)
    ref_out, _ = sliding_window_attention(q, k, v, window_size=window_size)
    assert torch.allclose(out, ref_out, atol=1e-5)
