# Sparse Attention from Scratch

Manual (from-scratch) implementation of dense attention and two sparse attention
patterns, built for the Postman AI/ML Recruitment Task (Task 1).

## Setup

```
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Structure

- `sparse_attention/dense_attention.py` — manual dense attention (matmul +
  mask + softmax), no `F.scaled_dot_product_attention`. Reference for
  everything else.
- `sparse_attention/sliding_window.py` — sliding-window sparse attention,
  implemented as dense attention restricted by a windowed causal mask.
- `tests/` — correctness harness: dense vs. PyTorch's own kernel, sliding
  window vs. dense at full window width, and mask/weight invariants
  (11 tests, `pytest -q`).
- `examples/walkthrough.py` — small worked numeric example (4 tokens)
  showing dense vs. sliding-window attention weights and outputs side by
  side; referenced directly in `WRITEUP.md`.

Run the tests:

```
pytest -q
```

Run the worked example:

```
python -m examples.walkthrough
```

## Status

Implemented and tested: manual dense attention, sliding-window sparse
attention, correctness harness for both.

Not implemented (time-boxed submission): BigBird-style local+global+random
sparse attention, NaN handling for fully-masked queries, the benchmark
script, and the char-GPT quality eval. See `WRITEUP.md` section 4 for what's
missing and why, including the reasoning for the NaN fix even though it
isn't implemented.

See `WRITEUP.md` for the full explanation of dense attention, the
sliding-window implementation, and why sliding-window attention loses
information relative to dense attention (with real numbers from
`examples/walkthrough.py`).
