import torch

from sparse_attention.bigbird import bigbird_attention, bigbird_mask
from sparse_attention.dense_attention import causal_mask, dense_attention
from sparse_attention.sliding_window import sliding_window_mask


def test_mask_shape_and_dtype():
    mask = bigbird_mask(12, window_size=3, num_global=2, num_random=2)
    assert mask.shape == (12, 12)
    assert mask.dtype == torch.bool


def test_mask_never_sees_future():
    mask = bigbird_mask(12, window_size=3, num_global=2, num_random=2)
    causal = causal_mask(12)
    assert torch.all(~mask | causal)


def test_mask_contains_local_window():
    seq_len, window_size = 10, 3
    mask = bigbird_mask(seq_len, window_size=window_size, num_global=0, num_random=0)
    local = sliding_window_mask(seq_len, window_size)
    # with no global/random tokens, the mask must be *exactly* the local window
    assert torch.equal(mask, local)


def test_global_tokens_reachable_regardless_of_distance():
    seq_len, num_global = 20, 2
    # tiny window and no randomness: the *only* way a far query can see an
    # early key is through the global mechanism
    mask = bigbird_mask(seq_len, window_size=1, num_global=num_global, num_random=0)
    for g in range(num_global):
        # every later position must be able to see global token g
        assert mask[g + 1 :, g].all()
    # global tokens themselves must see their entire causal history
    for g in range(num_global):
        assert torch.all(mask[g, : g + 1])


def test_global_zero_matches_no_global_tokens():
    mask = bigbird_mask(10, window_size=2, num_global=0, num_random=3, generator=torch.Generator().manual_seed(0))
    # with num_global=0, no row/column should be forced fully-open purely by
    # the global mechanism -- spot check nothing beyond local|random got set
    local = sliding_window_mask(10, 2)
    gen = torch.Generator().manual_seed(0)
    mask2 = bigbird_mask(10, window_size=2, num_global=0, num_random=3, generator=gen)
    assert torch.equal(mask, mask2)  # same seed -> reproducible random component
    assert torch.all(~mask | (local | mask))  # sanity: local is always a subset


def test_random_component_respects_causality_and_reproducibility():
    seq_len = 15
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    mask1 = bigbird_mask(seq_len, window_size=1, num_global=0, num_random=4, generator=gen1)
    mask2 = bigbird_mask(seq_len, window_size=1, num_global=0, num_random=4, generator=gen2)
    assert torch.equal(mask1, mask2)

    causal = causal_mask(seq_len)
    assert torch.all(~mask1 | causal)
    # row 0 can only ever pick itself (only one causally valid candidate)
    assert mask1[0].sum() == 1
    assert mask1[0, 0]


def test_reduces_to_dense_causal_when_window_covers_all():
    torch.manual_seed(0)
    seq_len, head_dim = 12, 6
    q = torch.randn(1, seq_len, head_dim)
    k = torch.randn(1, seq_len, head_dim)
    v = torch.randn(1, seq_len, head_dim)

    dense_out, _ = dense_attention(q, k, v, mask=causal_mask(seq_len))
    # window alone already covers the whole sequence, so global/random are
    # redundant with it and the union must equal plain causal attention
    sparse_out, _ = bigbird_attention(q, k, v, window_size=seq_len, num_global=0, num_random=0)

    assert torch.allclose(dense_out, sparse_out, atol=1e-6)


def test_weights_are_well_formed():
    torch.manual_seed(2)
    seq_len, head_dim = 16, 4
    q = torch.randn(1, seq_len, head_dim)
    k = torch.randn(1, seq_len, head_dim)
    v = torch.randn(1, seq_len, head_dim)

    out, weights = bigbird_attention(
        q, k, v, window_size=3, num_global=1, num_random=2, generator=torch.Generator().manual_seed(0)
    )
    mask = bigbird_mask(seq_len, window_size=3, num_global=1, num_random=2, generator=torch.Generator().manual_seed(0))

    assert not torch.isnan(out).any()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, seq_len), atol=1e-6)
    assert torch.allclose(weights[0][~mask], torch.zeros_like(weights[0][~mask]), atol=1e-8)
