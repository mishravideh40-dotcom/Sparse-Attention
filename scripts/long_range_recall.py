"""Synthetic long-range recall task -- validates the theoretical claim in
WRITEUP.md section 3 (sliding-window attention structurally loses
distant-but-relevant tokens) with a real experiment, instead of only
asserting it, and shows BigBird's global-token mechanism actually
recovering what sliding-window can't.

Task ("needle in a haystack", one digit): each training example is

    KEY  <253 random filler digits>  ?  KEY

The digit at position 0 (KEY, 0-9) must be reproduced immediately after
the '?' marker at the very end -- 254 positions later. Every filler digit
in between is drawn i.i.d. uniform, so there's nothing to learn there, and
nothing at the query position correlates with anything except position 0's
KEY. A model can only beat random guessing (10 choices -> log(10) nats,
10% accuracy) at the query position if it can actually attend back to
position 0 -- however far that is.

window_size is set far smaller than the gap, so plain sliding-window
attention is *structurally* incapable of ever seeing position 0 from the
query position, no matter how long it trains: this isn't an empirical
trend, it's provable from the mask itself. BigBird's num_global includes
position 0 as a global token, which is guaranteed reachable from anywhere
regardless of distance -- the exact mechanism WRITEUP.md section 3
describes.
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sparse_attention.char_gpt import CharGPT, CharGPTConfig

OUT_DIR = Path(__file__).resolve().parent.parent / "long_range_recall_results"

NUM_DIGITS = 10  # vocab: digits 0-9 plus '?' query marker
QUERY_TOKEN = NUM_DIGITS  # token id 10 == '?'
VOCAB_SIZE = NUM_DIGITS + 1

SEQ_LEN = 256  # KEY(1) + filler(253) + '?'(1) + KEY(1)
GAP = SEQ_LEN - 3
BATCH_SIZE = 64
N_EMBD = 64
N_LAYER = 2
MAX_ITERS = 2000
EVAL_EVERY = 100
EVAL_ITERS = 15
LR = 3e-3
SEED = 2027

WINDOW_SIZE = 8  # << GAP (253): sliding-window cannot structurally bridge this
NUM_GLOBAL = 4  # positions 0-3 are global; position 0 (the KEY) is always included
NUM_RANDOM = 8

RANDOM_BASELINE_LOSS = math.log(NUM_DIGITS)  # best possible without recall
RANDOM_BASELINE_ACC = 1 / NUM_DIGITS


def make_batch(batch_size: int, generator: torch.Generator):
    key = torch.randint(0, NUM_DIGITS, (batch_size, 1), generator=generator)
    filler = torch.randint(0, NUM_DIGITS, (batch_size, GAP), generator=generator)
    query = torch.full((batch_size, 1), QUERY_TOKEN, dtype=torch.long)
    seq = torch.cat([key, filler, query, key], dim=1)  # (batch, SEQ_LEN)
    x = seq[:, :-1]
    y = seq[:, 1:]
    return x, y


@torch.no_grad()
def eval_recall(model, generator):
    model.eval()
    losses, correct, total = [], 0, 0
    query_pos = SEQ_LEN - 2  # index of '?' within x/y (0-indexed)
    for _ in range(EVAL_ITERS):
        x, y = make_batch(BATCH_SIZE, generator)
        logits, _ = model(x)
        query_logits = logits[:, query_pos, :]
        target = y[:, query_pos]
        loss = torch.nn.functional.cross_entropy(query_logits, target)
        losses.append(loss.item())
        correct += (query_logits.argmax(dim=-1) == target).sum().item()
        total += target.numel()
    model.train()
    return sum(losses) / len(losses), correct / total


def train_one(kind: str, data_generator: torch.Generator):
    torch.manual_seed(SEED)  # same init for every variant -> fair comparison
    config = CharGPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=SEQ_LEN - 1,
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
    for step in range(MAX_ITERS + 1):
        if step % EVAL_EVERY == 0 or step == MAX_ITERS:
            recall_loss, recall_acc = eval_recall(model, data_generator)
            history.append({"step": step, "recall_loss": recall_loss, "recall_acc": recall_acc})
            print(f"[{kind:15s}] step {step:4d}  recall_loss {recall_loss:.4f}  recall_acc {recall_acc:.3f}")
        if step == MAX_ITERS:
            break
        x, y = make_batch(BATCH_SIZE, data_generator)
        _, loss = model(x, y)  # full-sequence LM loss (mostly irreducible filler noise, but
        optimizer.zero_grad(set_to_none=True)  # its gradient at the query position is exactly
        loss.backward()  # the recall signal being measured above)
        optimizer.step()

    return history


def run():
    OUT_DIR.mkdir(exist_ok=True)
    print(
        f"SEQ_LEN={SEQ_LEN}  gap={GAP}  window_size={WINDOW_SIZE}  num_global={NUM_GLOBAL}  "
        f"random-guess ceiling: loss={RANDOM_BASELINE_LOSS:.4f} acc={RANDOM_BASELINE_ACC:.3f}\n"
    )

    all_history = {}
    for kind in ("dense", "sliding_window", "bigbird"):
        data_generator = torch.Generator().manual_seed(SEED)  # identical example order per variant
        all_history[kind] = train_one(kind, data_generator)

    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "step", "recall_loss", "recall_acc"])
        for kind, hist in all_history.items():
            for row in hist:
                writer.writerow([kind, row["step"], row["recall_loss"], row["recall_acc"]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for kind, hist in all_history.items():
        steps = [r["step"] for r in hist]
        axes[0].plot(steps, [r["recall_loss"] for r in hist], marker="o", markersize=3, label=kind)
        axes[1].plot(steps, [r["recall_acc"] for r in hist], marker="o", markersize=3, label=kind)
    axes[0].axhline(RANDOM_BASELINE_LOSS, linestyle="--", color="gray", label="random-guess ceiling")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("loss at query position (nats)")
    axes[0].set_title("Recall loss")
    axes[0].legend(fontsize=8)
    axes[1].axhline(RANDOM_BASELINE_ACC, linestyle="--", color="gray", label="random-guess floor")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("top-1 accuracy at query position")
    axes[1].set_title("Recall accuracy")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Long-range recall: gap={GAP} tokens, window_size={WINDOW_SIZE}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recall.png", dpi=150)
    plt.close(fig)

    print("\nFinal recall performance:")
    for kind, hist in all_history.items():
        print(f"  {kind:15s} loss={hist[-1]['recall_loss']:.4f}  acc={hist[-1]['recall_acc']:.3f}")
    print(f"\nSaved results.csv and recall.png to {OUT_DIR}")


if __name__ == "__main__":
    run()
