"""Checklist item 6: train a tiny 2-layer char-level GPT on TinyShakespeare
with dense, sliding-window, and BigBird attention, and compare loss.

Run `python scripts/download_data.py` first if data/tinyshakespeare.txt
doesn't exist yet.
"""
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sparse_attention.char_gpt import CharGPT, CharGPTConfig

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tinyshakespeare.txt"
OUT_DIR = ROOT / "quality_eval_results"

BLOCK_SIZE = 128
BATCH_SIZE = 32
N_EMBD = 64
N_LAYER = 2
MAX_ITERS = 500
EVAL_EVERY = 25
EVAL_ITERS = 20
LR = 3e-3
SEED = 1337

# Sparsity params chosen to be meaningfully sparse relative to BLOCK_SIZE=128
# (see WRITEUP.md section 3 for why: too wide a window/too many global+random
# tokens just degenerates back to dense and the comparison is uninformative).
WINDOW_SIZE = 16
NUM_GLOBAL = 4
NUM_RANDOM = 8


def load_data():
    text = DATA_PATH.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, generator):
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(data, block_size, batch_size, generator)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one(kind: str, train_data, val_data, vocab_size, data_generator):
    torch.manual_seed(SEED)  # same init for every variant -> fair comparison
    config = CharGPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        attention_kind=kind,
        window_size=WINDOW_SIZE,
        num_global=NUM_GLOBAL,
        num_random=NUM_RANDOM,
        seed=SEED,
    )
    model = CharGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = []
    t0 = time.perf_counter()
    for step in range(MAX_ITERS + 1):
        if step % EVAL_EVERY == 0 or step == MAX_ITERS:
            train_loss = estimate_loss(model, train_data, BLOCK_SIZE, BATCH_SIZE, data_generator)
            val_loss = estimate_loss(model, val_data, BLOCK_SIZE, BATCH_SIZE, data_generator)
            elapsed = time.perf_counter() - t0
            history.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})
            print(f"[{kind:15s}] step {step:4d}  train {train_loss:.4f}  val {val_loss:.4f}  ({elapsed:.1f}s)")

        if step == MAX_ITERS:
            break
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, data_generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return history


def run():
    OUT_DIR.mkdir(exist_ok=True)
    train_data, val_data, vocab_size = load_data()
    print(f"vocab_size={vocab_size}  train_chars={len(train_data)}  val_chars={len(val_data)}")

    all_history = {}
    for kind in ("dense", "sliding_window", "bigbird"):
        data_generator = torch.Generator().manual_seed(SEED)  # identical batch order per variant
        all_history[kind] = train_one(kind, train_data, val_data, vocab_size, data_generator)

    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "step", "train_loss", "val_loss"])
        for kind, hist in all_history.items():
            for row in hist:
                writer.writerow([kind, row["step"], row["train_loss"], row["val_loss"]])

    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, hist in all_history.items():
        steps = [r["step"] for r in hist]
        val_losses = [r["val_loss"] for r in hist]
        ax.plot(steps, val_losses, marker="o", markersize=3, label=kind)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss (cross-entropy, nats)")
    ax.set_title("Char-GPT quality: dense vs. sparse attention on TinyShakespeare")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss.png", dpi=150)
    plt.close(fig)

    print("\nFinal validation loss:")
    for kind, hist in all_history.items():
        print(f"  {kind:15s} {hist[-1]['val_loss']:.4f}")
    print(f"\nSaved results.csv and loss.png to {OUT_DIR}")


if __name__ == "__main__":
    run()
