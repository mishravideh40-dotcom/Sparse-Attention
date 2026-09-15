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
  both `sliding_window_attention` (masked correctness reference) and
  `sliding_window_attention_fast` (real gather-based kernel that never
  computes the masked-out entries — ~97× faster, ~128× less memory than
  dense at seq_len=8192, measured, not estimated).
- `sparse_attention/bigbird.py` — BigBird-style sparse attention: union of
  local (sliding-window) + global (fixed reachable-from-anywhere tokens) +
  random (reproducible, causality-respecting) masks, both `bigbird_attention`
  (masked correctness reference) and `bigbird_attention_fast` (real kernel —
  global query rows via plain causal attention, regular rows via a
  vectorized padded gather; ~2.7× faster than the reference at seq_len=8192,
  but with a real, measured *memory regression* from its fancy-indexing
  gather-copy — see WRITEUP.md section 4 for the honest numbers).
- `tests/` — correctness harness: dense vs. PyTorch's own kernel, sliding
  window and BigBird vs. dense at full coverage, both fast kernels vs. their
  masked references (exactly, where randomness isn't involved), mask/weight
  invariants, and NaN-handling edge cases (35 tests, `pytest -q`).
- `examples/walkthrough.py` — small worked numeric example (4 tokens)
  showing dense vs. sliding-window attention weights and outputs side by
  side; referenced directly in `WRITEUP.md`.
- `scripts/benchmark.py` — wall-clock + memory benchmark across
  seq_len 512→8192 for all five variants (three reference implementations
  plus both real fast kernels); writes
  `benchmark_results/{results.csv,time.png,memory.png}`.
- `sparse_attention/char_gpt.py` + `scripts/quality_eval.py` — tiny 2-layer
  char-level GPT trained on TinyShakespeare with each attention variant as
  a drop-in swap; writes `quality_eval_results/{results.csv,loss.png}`.
- `scripts/download_data.py` — fetches TinyShakespeare into
  `data/tinyshakespeare.txt` (gitignored; run this first for the quality
  eval).

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

Run the quality eval (downloads TinyShakespeare first, ~1MB):

```
python scripts/download_data.py
python scripts/quality_eval.py
```

## Status

All six Task 1 checklist items implemented: manual dense attention,
sliding-window sparse attention, BigBird-style local+global+random sparse
attention, NaN-safe softmax, a wall-clock/memory benchmark (seq_len
512→8192), and a char-GPT quality eval on TinyShakespeare — with a 35-test
correctness harness for items 1-4 and saved, inspectable output
(CSV + plots) for items 5-6. Beyond the checklist minimum: both sparse
patterns got a real fast kernel, not just a masked reference —
`sliding_window_attention_fast` (~97× faster, ~128× less memory than dense
at seq_len=8192) and `bigbird_attention_fast` (~2.7× faster at seq_len=8192,
but ~2× *more* memory than dense — a real, honestly-reported tradeoff from
its gather-copy strategy, not a clean win).

See `WRITEUP.md` for the full explanation: dense attention, both sparse
patterns and their fast kernels, the NaN fix, why sliding-window attention
loses information that BigBird's global tokens recover, the full benchmark
findings for all five variants (including why the BigBird kernel is faster
but *not* smaller, and why that's the honest result rather than a
disappointment to hide), and the quality-eval result — sparse attention
actually *matched or beat* dense at this small scale, and why that's a
real, explainable finding rather than a contradiction of section 3's
example.
