import math

import torch


def manual_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax implemented by hand: exp(x_i) / sum(exp(x_j))."""
    # Subtract the max before exponentiating -- exp() of a big number overflows
    # to inf, but shifting every value down by the row max keeps everything
    # <= 0 before exp(), so the result is identical while staying finite.
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def causal_mask(seq_len: int) -> torch.Tensor:
    """True = allowed. Position i may only attend to positions j <= i."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    """Manual dense attention: matmul -> mask -> softmax -> matmul.

    q, k, v: (..., seq_len, head_dim)
    mask: (seq_len, seq_len) bool, True = allowed position. Broadcasts over
        any leading batch/head dims.

    Returns:
        out: (..., seq_len, head_dim)
        weights: (..., seq_len, seq_len) attention probabilities, for
            inspection/plotting/reuse by sparse variants.
    """
    head_dim = q.shape[-1]

    # Step 1: similarity between every query and every key.
    scores = q @ k.transpose(-2, -1)  # (..., seq_len, seq_len)
    scores = scores / math.sqrt(head_dim)  # keeps dot-product scale in check

    # Step 2: forbidden pairs get -inf so softmax turns them into 0 weight.
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Step 3: scores -> probabilities that sum to 1 along the key dimension.
    weights = manual_softmax(scores, dim=-1)

    # Step 4: blend the values according to those probabilities.
    out = weights @ v  # (..., seq_len, head_dim)
    return out, weights
