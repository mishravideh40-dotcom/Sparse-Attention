import torch

from .dense_attention import causal_mask, dense_attention
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
