# Sparse Attention from Scratch

Manual (from-scratch) implementation of dense attention and two sparse
attention patterns (sliding-window, BigBird-style local+global+random),
built for the Postman AI/ML Recruitment Task (Task 1).

## Setup

```
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Structure

- `sparse_attention/dense_attention.py` — manual dense attention (matmul +
  mask + softmax), no `F.scaled_dot_product_attention`. Reference for
  everything else; also where NaN-safe softmax lives (fully-masked queries
  produce zero output instead of NaN).
- `sparse_attention/sliding_window.py` — sliding-window sparse attention,
  implemented as dense attention restricted by a windowed causal mask.
- `sparse_attention/bigbird.py` — BigBird-style sparse attention: union of
  local (sliding-window) + global (fixed reachable-from-anywhere tokens) +
  random (reproducible, causality-respecting) masks.
- `tests/` — correctness harness: dense vs. PyTorch's own kernel, sliding
  window and BigBird vs. dense at full coverage, mask/weight invariants,
  and NaN-handling edge cases (23 tests, `pytest -q`).
- `examples/walkthrough.py` — small worked numeric example (4 tokens)
  showing dense vs. sliding-window attention weights and outputs side by
  side; referenced directly in `WRITEUP.md`.
- `scripts/benchmark.py` — wall-clock + memory benchmark across
  seq_len 512→8192 for all three attention variants; writes
  `benchmark_results/{results.csv,time.png,memory.png}`.

Run the tests:

```
pytest -q
```

Run the worked example:

```
python -m examples.walkthrough
```

Run the benchmark:

```
python scripts/benchmark.py
```

## Status

Implemented and tested: manual dense attention, sliding-window sparse
attention, BigBird-style local+global+random sparse attention, NaN-safe
softmax, and the wall-clock/memory benchmark — with a correctness harness
covering items 1-4 (checklist items 1, 2, 3, 4, and 5).

Not implemented (time-boxed submission): the char-GPT quality eval on
TinyShakespeare comparing dense vs. sparse loss (item 6). See `WRITEUP.md`
section 5.

See `WRITEUP.md` for the full explanation of dense attention, both sparse
patterns, the NaN fix, why sliding-window attention loses information that
BigBird's global tokens recover, and the benchmark findings (including why
wall-clock time doesn't yet improve, and why memory would with a proper
sparse kernel) — with real numbers from `examples/walkthrough.py` and
`benchmark_results/`.
