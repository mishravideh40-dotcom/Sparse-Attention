import torch

from .dense_attention import dense_attention


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
