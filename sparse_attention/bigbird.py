import math

import torch

from .dense_attention import causal_mask, dense_attention, manual_softmax
from .sliding_window import sliding_window_mask


def bigbird_mask(
    seq_len: int,
    window_size: int,
    num_global: int,
    num_random: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Causal BigBird-style mask: union of local (sliding-window), global,
    and random attention, always intersected with causality.

    - local: same as sliding_window_mask -- position i sees the previous
      window_size - 1 positions plus itself.
    - global: the first num_global positions are "global tokens" -- they see
      every earlier position, and every later position sees them. This is
      what lets a designated token stay reachable regardless of distance,
      unlike pure sliding-window (see WRITEUP.md).
    - random: for each query i, num_random keys are sampled uniformly from
      i's causally-valid history (positions 0..i). This is BigBird's third
      component -- it gives the sparse graph a small chance of connecting
      any two positions, rather than only ever the fixed local/global set.

    All three are unioned then AND-ed with the causal mask, so nothing here
    can see the future no matter how local/global/random combine.
    """
    causal = causal_mask(seq_len)
    local = sliding_window_mask(seq_len, window_size)

    global_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    if num_global > 0:
        global_ids = torch.arange(min(num_global, seq_len))
        global_mask[global_ids, :] = True  # global tokens attend to everyone...
        global_mask[:, global_ids] = True  # ...and everyone attends to them
        # (causality is enforced by the final `& causal` below)

    random_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    if num_random > 0:
        for i in range(seq_len):
            num_candidates = i + 1  # positions 0..i are causally valid keys
            k = min(num_random, num_candidates)
            chosen = torch.randperm(num_candidates, generator=generator)[:k]
            random_mask[i, chosen] = True

    combined = local | global_mask | random_mask
    return combined & causal


def bigbird_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    num_global: int,
    num_random: int,
    generator: torch.Generator | None = None,
):
    """Reference BigBird-style attention: dense attention restricted to the
    local+global+random mask. Like sliding_window_attention, this is a
    correctness baseline (full QK^T is still computed), not the
    compute-efficient gather/block-sparse form."""
    seq_len = q.shape[-2]
    mask = bigbird_mask(seq_len, window_size, num_global, num_random, generator=generator)
    return dense_attention(q, k, v, mask=mask)


def _bigbird_regular_row_indices(
    seq_len: int,
    window_size: int,
    num_global: int,
    num_random: int,
    generator: torch.Generator | None = None,
    oversample: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed-width, per-row candidate key list for every query row,
    fully vectorized (no Python loop over seq_len) -- this is what makes
    the fast kernel below actually fast to *set up*, not just to run.

    Only meaningful for "regular" rows (see bigbird_attention_fast for why
    global rows are handled separately); it's still computed for every row
    here since it's cheap and the caller slices out the rows it needs.

    Returns (indices, valid), both (seq_len, window_size + num_global +
    num_random). indices[i, j] is a key position (only meaningful where
    valid[i, j] is True); padding slots point at position 0 but are masked
    out via valid, exactly like sliding_window_attention_fast's padding.

    Each slot category explicitly excludes positions already claimed by an
    earlier category (window, then global, then random) so no key is ever
    double-counted in the softmax the way it would be if the same key
    showed up as two separate "True" slots. The random category is filled
    by oversampling `num_random * oversample` uniform candidates per row
    and keeping the first `num_random` that survive the window/global
    exclusion filter -- a rejection-sampling approximation, not the exact
    same algorithm bigbird_mask's random component uses (which enumerates
    every causally-valid non-excluded position, an O(seq_len) operation per
    row that's exactly what a fast kernel needs to avoid). At the tail end
    of a long sequence (i.e. most rows, where seq_len is much larger than
    the oversample pool) this essentially never runs dry; near the very
    start it can occasionally return fewer than num_random random slots
    (marked invalid, same graceful degradation bigbird_mask uses when
    num_candidates < num_random).
    """
    K = window_size + num_global + num_random
    positions = torch.arange(seq_len)

    # --- window slots: identical construction to sliding_window_mask, but
    # as explicit gather indices instead of a boolean grid.
    window_offsets = torch.arange(window_size) - (window_size - 1)
    window_idx = positions.unsqueeze(1) + window_offsets.unsqueeze(0)  # (seq_len, window_size)
    window_valid = window_idx >= 0
    window_idx = window_idx.clamp(min=0)

    # --- global slots: the first num_global positions, excluded if a row's
    # window already covers them (a global id g is in row i's window iff
    # 0 <= i - g < window_size).
    if num_global > 0:
        global_ids = torch.arange(num_global)
        global_idx = global_ids.unsqueeze(0).expand(seq_len, num_global)
        causally_valid = global_ids.unsqueeze(0) <= positions.unsqueeze(1)
        diff = positions.unsqueeze(1) - global_ids.unsqueeze(0)
        already_in_window = (diff >= 0) & (diff < window_size)
        global_valid = causally_valid & ~already_in_window
    else:
        global_idx = torch.zeros(seq_len, 0, dtype=torch.long)
        global_valid = torch.zeros(seq_len, 0, dtype=torch.bool)

    # --- random slots: oversample, then reject candidates that collide
    # with window or global, then take the first num_random survivors.
    if num_random > 0:
        pool = max(num_random * oversample, num_random)
        raw = torch.randint(0, seq_len, (seq_len, pool), generator=generator)
        candidate = raw % (positions.unsqueeze(1) + 1)  # fold into [0, i]

        diff_w = positions.unsqueeze(1) - candidate
        in_window = (diff_w >= 0) & (diff_w < window_size)
        in_global = candidate < num_global
        excluded = in_window | in_global

        # Stable sort puts non-excluded candidates first without disturbing
        # their (already random) relative order.
        order = torch.argsort(excluded.float(), dim=-1, stable=True)
        random_idx = candidate.gather(1, order[:, :num_random])
        random_valid = ~excluded.gather(1, order[:, :num_random])
    else:
        random_idx = torch.zeros(seq_len, 0, dtype=torch.long)
        random_valid = torch.zeros(seq_len, 0, dtype=torch.bool)

    indices = torch.cat([window_idx, global_idx, random_idx], dim=1)
    valid = torch.cat([window_valid, global_valid, random_valid], dim=1)
    assert indices.shape == (seq_len, K)
    return indices, valid


def bigbird_attention_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    num_global: int,
    num_random: int,
    generator: torch.Generator | None = None,
):
    """Compute-efficient BigBird-style attention. Unlike sliding-window,
    BigBird's per-query key set isn't a fixed contiguous offset, so it
    can't be gathered with a single `unfold` -- this kernel instead splits
    queries into two groups with very different shapes and handles each
    the cheap way:

    - "Global" query rows (the first num_global positions) attend to their
      *entire* causal history by definition (that's what makes them
      global) -- there's no sparsity to exploit for these rows, so they're
      just computed with plain causal dense_attention, restricted to only
      those num_global query rows (cost O(num_global * seq_len), not
      O(seq_len^2), since there are few of them).
    - "Regular" rows (everything else) really are sparse -- at most
      window_size + num_global + num_random keys each, a small constant --
      and are computed via a padded gather, the same shape of trick as
      sliding_window_attention_fast, using indices from
      _bigbird_regular_row_indices.

    Requires q, k, v shaped exactly (batch, seq_len, head_dim), same
    constraint as sliding_window_attention_fast and for the same reason.

    Returns:
        out: (batch, seq_len, head_dim)
        info: dict with the intermediate global/regular weights and the
            regular rows' gather indices/validity, for testing/inspection
            -- there's no single (batch, seq_len, seq_len)-shaped weights
            tensor the way dense_attention returns, because the two row
            groups don't share a common attended-set width.
    """
    batch, seq_len, head_dim = q.shape
    num_global_eff = min(num_global, seq_len)

    indices, valid = _bigbird_regular_row_indices(
        seq_len, window_size, num_global_eff, num_random, generator=generator
    )

    if num_global_eff > 0:
        global_mask = causal_mask(seq_len)[:num_global_eff]
        global_out, global_weights = dense_attention(q[:, :num_global_eff, :], k, v, mask=global_mask)
    else:
        global_out = q.new_zeros(batch, 0, head_dim)
        global_weights = None

    reg_indices = indices[num_global_eff:]
    reg_valid = valid[num_global_eff:]
    num_regular = seq_len - num_global_eff

    if num_regular > 0:
        k_gathered = k[:, reg_indices, :]  # (batch, num_regular, K, head_dim)
        v_gathered = v[:, reg_indices, :]
        q_regular = q[:, num_global_eff:, :]

        scores = torch.einsum("brd,brkd->brk", q_regular, k_gathered) / math.sqrt(head_dim)
        scores = scores.masked_fill(~reg_valid.unsqueeze(0), float("-inf"))
        reg_weights = manual_softmax(scores, dim=-1)
        reg_out = torch.einsum("brk,brkd->brd", reg_weights, v_gathered)
    else:
        reg_out = q.new_zeros(batch, 0, head_dim)
        reg_weights = None

    out = torch.cat([global_out, reg_out], dim=1)
    info = {
        "num_global_rows": num_global_eff,
        "global_weights": global_weights,
        "regular_weights": reg_weights,
        "regular_indices": reg_indices,
        "regular_valid": reg_valid,
    }
    return out, info
