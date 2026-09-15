"""Wall-clock + memory benchmark: dense vs. sliding-window vs. BigBird-style
attention across sequence length.

Important honesty note (see WRITEUP.md section 5): every variant here is a
*correctness reference* built on top of `dense_attention` -- it always
materializes a full (seq_len, seq_len) score/weight matrix and only differs
in which entries the mask zeroes out. So the wall-clock and *actual* memory
numbers below are not expected to show a sparsity speedup; a real speedup
requires a gather/block-sparse kernel that never allocates the full matrix
in the first place. What we *can* honestly measure and plot is (a) that
these reference implementations cost the same or slightly more than dense
(masking is extra work, not less), and (b) the "theoretical" memory a
proper sparse kernel would use, computed from how many entries the mask
actually keeps -- which is the number that motivates writing that kernel.
"""
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sparse_attention.bigbird import bigbird_mask
from sparse_attention.dense_attention import causal_mask, dense_attention
from sparse_attention.sliding_window import sliding_window_mask

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
HEAD_DIM = 64
BATCH = 1
N_REPS = 5
BYTES_PER_ELEM = 4  # float32
OUT_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"


def build_variants(seq_len: int, generator: torch.Generator):
    return {
        "dense": causal_mask(seq_len),
        "sliding_window": sliding_window_mask(seq_len, window_size=64),
        "bigbird": bigbird_mask(
            seq_len, window_size=64, num_global=32, num_random=32, generator=generator
        ),
    }


def time_forward(q, k, v, mask, n_reps=N_REPS):
    # one untimed warmup call so the first-call allocator overhead doesn't
    # leak into the measurement
    dense_attention(q, k, v, mask=mask)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        dense_attention(q, k, v, mask=mask)
        times.append(time.perf_counter() - t0)
    return min(times)


def run():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []

    for seq_len in SEQ_LENS:
        torch.manual_seed(0)
        q = torch.randn(BATCH, seq_len, HEAD_DIM)
        k = torch.randn(BATCH, seq_len, HEAD_DIM)
        v = torch.randn(BATCH, seq_len, HEAD_DIM)
        generator = torch.Generator().manual_seed(0)

        masks = build_variants(seq_len, generator)
        materialized_bytes = BATCH * seq_len * seq_len * BYTES_PER_ELEM

        for name, mask in masks.items():
            elapsed = time_forward(q, k, v, mask)
            nnz = mask.sum().item()
            sparse_equiv_bytes = BATCH * nnz * BYTES_PER_ELEM
            rows.append(
                {
                    "variant": name,
                    "seq_len": seq_len,
                    "time_s": elapsed,
                    "materialized_bytes": materialized_bytes,
                    "sparse_equivalent_bytes": sparse_equiv_bytes,
                    "mask_density": nnz / (seq_len * seq_len),
                }
            )
            print(
                f"{name:15s} seq_len={seq_len:5d}  time={elapsed*1000:8.2f} ms  "
                f"density={nnz / (seq_len * seq_len):.4f}"
            )

    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_time(rows)
    plot_memory(rows)
    print(f"\nSaved results.csv, time.png, memory.png to {OUT_DIR}")


def plot_time(rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("dense", "sliding_window", "bigbird"):
        xs = [r["seq_len"] for r in rows if r["variant"] == name]
        ys = [r["time_s"] * 1000 for r in rows if r["variant"] == name]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("wall-clock time (ms, min of 5 reps)")
    ax.set_title("Forward pass time: dense vs. sparse reference implementations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time.png", dpi=150)
    plt.close(fig)


def plot_memory(rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = sorted(set(r["seq_len"] for r in rows))
    dense_mb = [
        next(r for r in rows if r["variant"] == "dense" and r["seq_len"] == s)["materialized_bytes"] / 1e6
        for s in xs
    ]
    ax.plot(xs, dense_mb, marker="o", linestyle="--", color="black", label="materialized (all variants, as-built)")

    for name in ("sliding_window", "bigbird"):
        ys = [
            next(r for r in rows if r["variant"] == name and r["seq_len"] == s)["sparse_equivalent_bytes"] / 1e6
            for s in xs
        ]
        ax.plot(xs, ys, marker="o", label=f"{name} (theoretical, gather-based)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("attention-matrix memory (MB)")
    ax.set_title("Materialized (as-built) vs. theoretical sparse-kernel memory")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "memory.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
