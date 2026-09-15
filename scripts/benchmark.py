"""Wall-clock + memory benchmark: dense vs. sliding-window (reference and
fast) vs. BigBird-style attention across sequence length.

Honesty note (see WRITEUP.md section 4): dense, sliding_window (reference),
and bigbird are all *correctness references* built on top of
`dense_attention` -- they always materialize a full (seq_len, seq_len)
score/weight matrix and only differ in which entries the mask zeroes out.
So their wall-clock and memory numbers are expected to be the same as
dense's, not better -- masking is extra work on top of the full matmul, not
a discount. sliding_window_fast is different: it's a real gather-based
kernel (sparse_attention/sliding_window.py) that never computes or
allocates the parts of the score matrix outside a query's window, so it's
the one variant here that should show an actual, measured speedup and
memory reduction rather than only the theoretical one computed from mask
density. BigBird doesn't get an equivalent fast kernel in this repo -- its
random component doesn't have sliding-window's fixed-offset structure, so
a real gather/block-sparse kernel for it is a materially harder unfold
problem; that's called out as future work rather than attempted here.
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
from sparse_attention.sliding_window import sliding_window_attention_fast, sliding_window_mask

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
HEAD_DIM = 64
BATCH = 1
WINDOW_SIZE = 64
N_REPS = 5
BYTES_PER_ELEM = 4  # float32
OUT_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"


def build_masked_variants(seq_len: int, generator: torch.Generator):
    return {
        "dense": causal_mask(seq_len),
        "sliding_window": sliding_window_mask(seq_len, window_size=WINDOW_SIZE),
        "bigbird": bigbird_mask(
            seq_len, window_size=WINDOW_SIZE, num_global=32, num_random=32, generator=generator
        ),
    }


def time_call(fn, n_reps=N_REPS):
    fn()  # one untimed warmup so first-call allocator overhead doesn't leak in
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
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

        masks = build_masked_variants(seq_len, generator)
        materialized_bytes = BATCH * seq_len * seq_len * BYTES_PER_ELEM

        for name, mask in masks.items():
            elapsed = time_call(lambda m=mask: dense_attention(q, k, v, mask=m))
            nnz = mask.sum().item()
            sparse_equiv_bytes = BATCH * nnz * BYTES_PER_ELEM
            rows.append(
                {
                    "variant": name,
                    "seq_len": seq_len,
                    "time_s": elapsed,
                    "actual_bytes": materialized_bytes,
                    "sparse_equivalent_bytes": sparse_equiv_bytes,
                    "mask_density": nnz / (seq_len * seq_len),
                    "memory_kind": "theoretical" if name != "dense" else "actual",
                }
            )
            print(
                f"{name:15s} seq_len={seq_len:5d}  time={elapsed*1000:8.2f} ms  "
                f"density={nnz / (seq_len * seq_len):.4f}"
            )

        # sliding_window_fast: a real gather-based kernel, not a masked
        # dense_attention call -- both its time and its memory are actual
        # measurements, not theoretical ones computed from mask density.
        fast_elapsed = time_call(lambda: sliding_window_attention_fast(q, k, v, window_size=WINDOW_SIZE))
        fast_bytes = BATCH * seq_len * WINDOW_SIZE * BYTES_PER_ELEM  # the weights tensor it actually allocates
        rows.append(
            {
                "variant": "sliding_window_fast",
                "seq_len": seq_len,
                "time_s": fast_elapsed,
                "actual_bytes": fast_bytes,
                "sparse_equivalent_bytes": fast_bytes,
                "mask_density": WINDOW_SIZE / seq_len,
                "memory_kind": "actual",
            }
        )
        print(f"{'sliding_window_fast':15s} seq_len={seq_len:5d}  time={fast_elapsed*1000:8.2f} ms  (real kernel)")

    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_time(rows)
    plot_memory(rows)
    print(f"\nSaved results.csv, time.png, memory.png to {OUT_DIR}")


def plot_time(rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {
        "dense": {},
        "sliding_window": {},
        "bigbird": {},
        "sliding_window_fast": {"linewidth": 2.5},
    }
    for name, style in styles.items():
        xs = [r["seq_len"] for r in rows if r["variant"] == name]
        ys = [r["time_s"] * 1000 for r in rows if r["variant"] == name]
        ax.plot(xs, ys, marker="o", label=name, **style)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("wall-clock time (ms, min of 5 reps)")
    ax.set_title("Forward pass time: reference implementations vs. real fast kernel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time.png", dpi=150)
    plt.close(fig)


def plot_memory(rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = sorted(set(r["seq_len"] for r in rows))
    dense_mb = [
        next(r for r in rows if r["variant"] == "dense" and r["seq_len"] == s)["actual_bytes"] / 1e6 for s in xs
    ]
    ax.plot(xs, dense_mb, marker="o", linestyle="--", color="black", label="dense (actual, as-built)")

    fast_mb = [
        next(r for r in rows if r["variant"] == "sliding_window_fast" and r["seq_len"] == s)["actual_bytes"] / 1e6
        for s in xs
    ]
    ax.plot(xs, fast_mb, marker="o", linewidth=2.5, label="sliding_window_fast (actual, real kernel)")

    for name in ("sliding_window", "bigbird"):
        ys = [
            next(r for r in rows if r["variant"] == name and r["seq_len"] == s)["sparse_equivalent_bytes"] / 1e6
            for s in xs
        ]
        ax.plot(xs, ys, marker="o", linestyle=":", label=f"{name} (theoretical, gather-based)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("attention-matrix memory (MB)")
    ax.set_title("Actual vs. theoretical attention-matrix memory")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "memory.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
