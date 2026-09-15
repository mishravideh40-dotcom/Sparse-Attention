"""Tiny 2-layer char-level GPT used for the dense-vs-sparse quality eval.

Deliberately minimal: single-head attention per block (multi-head adds
parameters but no new insight for "does this sparsity pattern hurt
modeling quality"), reusing this repo's own dense_attention/
sliding_window/bigbird functions as the attention mechanism rather than
`nn.MultiheadAttention` -- the point of this eval is to exercise the
from-scratch implementations end-to-end, not to build a competitive model.
"""
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from .bigbird import bigbird_mask
from .dense_attention import causal_mask, dense_attention
from .sliding_window import sliding_window_mask

AttentionKind = str  # "dense" | "sliding_window" | "bigbird"


def make_mask_fn(
    kind: AttentionKind,
    window_size: int = 16,
    num_global: int = 4,
    num_random: int = 8,
    seed: int = 0,
) -> Callable[[int], torch.Tensor]:
    """Returns a function seq_len -> mask, so a Block doesn't need to know
    which sparsity pattern it's using -- only that block_size determines
    the mask shape. BigBird gets a fresh, seeded generator per call so the
    random component is reproducible across runs/models with the same
    seed, rather than reusing a generator whose internal state would drift
    between train and eval calls."""

    def mask_fn(seq_len: int) -> torch.Tensor:
        if kind == "dense":
            return causal_mask(seq_len)
        if kind == "sliding_window":
            return sliding_window_mask(seq_len, window_size)
        if kind == "bigbird":
            generator = torch.Generator().manual_seed(seed)
            return bigbird_mask(seq_len, window_size, num_global, num_random, generator=generator)
        raise ValueError(f"unknown attention kind: {kind}")

    return mask_fn


class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, mask_fn: Callable[[int], torch.Tensor]):
        super().__init__()
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.mask_fn = mask_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        mask = self.mask_fn(seq_len)
        q, k, v = self.query(x), self.key(x), self.value(x)
        out, _ = dense_attention(q, k, v, mask=mask)  # "dense_attention" = the shared matmul/mask/softmax core
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, n_embd: int, mask_fn: Callable[[int], torch.Tensor]):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, mask_fn)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class CharGPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int = 64
    n_layer: int = 2
    attention_kind: AttentionKind = "dense"
    window_size: int = 16
    num_global: int = 4
    num_random: int = 8
    seed: int = 0


class CharGPT(nn.Module):
    def __init__(self, config: CharGPTConfig):
        super().__init__()
        self.config = config
        mask_fn = make_mask_fn(
            config.attention_kind,
            window_size=config.window_size,
            num_global=config.num_global,
            num_random=config.num_random,
            seed=config.seed,
        )
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config.n_embd, mask_fn) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        seq_len = idx.shape[-1]
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
