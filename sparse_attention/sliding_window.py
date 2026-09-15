import math

import torch

from .dense_attention import dense_attention, manual_softmax


def sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """True = allowed. Causal: position i attends only to itself and the
    previous (window_size - 1) positions -- never the future, never
    anything older than the window."""
    positions = torch.arange(seq_len)
    i = positions.unsqueeze(1)  # (seq_len, 1) query position
    j = positions.unsqueeze(0)  # (1, seq_len) key position
    causal = j <= i
    within_window = (i - j) < window_size
    return causal & within_window


def sliding_window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int):
    """Reference sliding-window attention: dense attention restricted to a
    local causal window via masking. Same math as dense_attention, just a
    stricter mask -- this is the correctness baseline; a compute-efficient
    (gather-based) version comes later for the benchmark."""
    seq_len = q.shape[-2]
    mask = sliding_window_mask(seq_len, window_size)
    return dense_attention(q, k, v, mask=mask)


def sliding_window_attention_fast(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int):
    """Compute-efficient sliding-window attention: only ever computes
    q_i . k_j for the window_size keys actually inside query i's window,
    instead of computing the full (seq_len, seq_len) score matrix and
    masking most of it away, as sliding_window_attention (the reference
    above) does. This is the O(seq_len * window_size) kernel that section 1
    of WRITEUP.md describes as "coming later" -- it now exists, is checked
    against the reference for correctness, and is what makes the benchmark
    in WRITEUP.md section 4 measure a *real* speedup/memory reduction
    instead of only a theoretical one.

    q, k, v: exactly (batch, seq_len, head_dim) -- unlike the reference
    version (which broadcasts over arbitrary leading dims via a mask),
    this kernel pads/unfolds along a specific batch+seq_len layout, so the
    batch dimension must be explicit.

    Returns:
        out: (batch, seq_len, head_dim)
        weights: (batch, seq_len, window_size) -- note this is *not* the
            same shape as the reference's (batch, seq_len, seq_len); slot j
            here is key position (i - window_size + 1 + j), not literal
            key index j.
    """
    batch, seq_len, head_dim = q.shape

    # Pad K, V at the *start* with window_size - 1 zero vectors, so every
    # query's window [i - window_size + 1, i] reads as a contiguous slice
    # of the padded tensor, even for queries near the start of the sequence.
    pad = window_size - 1
    k_padded = torch.nn.functional.pad(k, (0, 0, pad, 0))
    v_padded = torch.nn.functional.pad(v, (0, 0, pad, 0))

    # unfold(dim=1, size=window_size, step=1): for each query position,
    # gather the window_size keys/values ending there. Shape comes out as
    # (batch, seq_len, head_dim, window_size); permute to put window_size
    # next to seq_len so it reads as "seq_len windows of window_size keys".
    k_windows = k_padded.unfold(1, window_size, 1).permute(0, 1, 3, 2)
    v_windows = v_padded.unfold(1, window_size, 1).permute(0, 1, 3, 2)

    # Window slot j (0-indexed) at query i is original key position
    # i - window_size + 1 + j. Slots pointing at a position < 0 are the
    # zero-padding, not a real key -- a real key can legitimately score 0,
    # so padding must be excluded by position, not by score value.
    positions = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    slot = torch.arange(window_size).unsqueeze(0)  # (1, window_size)
    valid = (positions - window_size + 1 + slot) >= 0  # (seq_len, window_size)

    scores = torch.einsum("bsd,bswd->bsw", q, k_windows) / math.sqrt(head_dim)
    scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    weights = manual_softmax(scores, dim=-1)  # (batch, seq_len, window_size)
    out = torch.einsum("bsw,bswd->bsd", weights, v_windows)
    return out, weights
