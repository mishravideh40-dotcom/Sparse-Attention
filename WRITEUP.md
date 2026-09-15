# Sparse Attention from Scratch — Writeup

Postman AI/ML Recruitment Task, Task 1. This writeup covers what's implemented
(manual dense attention, sliding-window and BigBird-style sparse attention,
NaN-safe softmax, and the correctness harness for all of it), why
sliding-window attention loses information relative to dense attention and
how BigBird's global/random components address that gap, and what's out of
scope given the time available for this submission.

## 1. What's implemented

**Dense attention** (`sparse_attention/dense_attention.py`) is the textbook
mechanism built from primitives, not `F.scaled_dot_product_attention`:

```
scores  = (Q @ K^T) / sqrt(head_dim)
scores  = scores.masked_fill(~mask, -inf)   # forbidden pairs -> 0 weight
weights = manual_softmax(scores)             # exp(x - max) / sum, for stability
out     = weights @ V
```

`manual_softmax` subtracts the row max before exponentiating. This is not
optional: without it, a raw score of e.g. 50 overflows `exp()` to `inf` and
the whole row becomes NaN. Shifting by the max keeps every exponent `<= 0`
while leaving the ratio — and therefore the softmax output — unchanged.

**Sliding-window attention** (`sparse_attention/sliding_window.py`) reuses
`dense_attention` with a stricter mask: position `i` may attend only to
itself and the previous `window_size - 1` positions. It is intentionally
implemented as "dense attention + a smaller mask," not a gather-based
kernel — the goal at this stage is a *correctness reference*, i.e., the
ground truth that a compute-efficient version would later be checked
against. The mask being causal *and* windowed is enforced directly in
`sliding_window_mask`, not bolted on afterward.

**BigBird-style attention** (`sparse_attention/bigbird.py`) unions three
mask components, then intersects with causality:

- **local** — identical to the sliding-window mask above.
- **global** — the first `num_global` positions are global tokens: they can
  attend to their entire causal history, and every later position can
  attend to them regardless of distance. This is the mechanism that fixes
  the exact failure mode described in section 3 below.
- **random** — for each query `i`, `num_random` keys are sampled uniformly
  (via a `torch.Generator` for reproducibility) from `i`'s causally-valid
  history (`0..i`). This gives the sparse graph a nonzero chance of
  connecting any two positions instead of only ever the fixed local/global
  set, which is the property BigBird's authors lean on for their
  expressiveness guarantees.

`bigbird_mask(seq_len, window_size, num_global, num_random, generator=None)`
combines these as `(local | global | random) & causal`; `num_global=0,
num_random=0` degenerates to exactly the sliding-window mask, which is
tested directly (`test_mask_contains_local_window`).

**NaN handling** (`sparse_attention/dense_attention.py`, `manual_softmax`):
a fully-masked query row (every score `-inf`) previously computed
`-inf - (-inf) = nan` when subtracting the row max for numerical stability,
corrupting the whole row and propagating through `weights @ v`. The fix
detects rows where the max is `-inf` and swaps their max to `0` before
subtracting (so the row stays all `-inf`, `exp(-inf) = 0`), then guards the
softmax denominator against `0/0` on those same rows. The net effect: a
fully-masked query now produces an all-zero weight row and all-zero output,
deterministically, instead of NaN. This matters in practice for BigBird-like
patterns where a padding/masked position could otherwise end up with zero
causally-valid keys.

## 2. Correctness harness

23 tests across `tests/test_dense_attention.py`, `tests/test_sliding_window.py`,
`tests/test_bigbird.py`, and `tests/test_nan_handling.py` (`pytest -q` →
`23 passed`):

- **Dense vs. PyTorch reference**: `dense_attention` is checked against
  `F.scaled_dot_product_attention`, both unmasked and with a causal mask,
  to `atol=1e-5`. This is the anchor for everything else — if dense attention
  didn't match the reference kernel exactly, no comparison built on top of it
  would mean anything.
- **Softmax correctness**: `manual_softmax` matches `torch.softmax`, and
  every row sums to 1.
- **Sliding window reduces to dense**: when `window_size >= seq_len`, sliding
  window attention must produce *bit-identical* output to dense causal
  attention (`atol=1e-6`) — a window that covers the whole sequence is just
  dense attention with extra bookkeeping. This single test would catch most
  indexing bugs in the windowing logic.
- **Mask properties**: shape/dtype, "never attends to the future" (the
  sliding-window mask is a strict subset of the causal mask), and "the
  allowed set for position `i` is exactly `[i - window_size + 1, i]` clipped
  to `0`" are each checked directly, independent of any attention math.
- **Weights are well-formed**: zero outside the window, sum to 1 inside it.
- **BigBird mask properties**: never sees the future; degenerates to exactly
  the sliding-window mask when global/random are both zero; a global token
  is reachable from *every* later position even with `window_size=1` (i.e.
  the only path to it is the global mechanism, not local overlap); the
  random component is reproducible given the same `torch.Generator` seed and
  never picks a causally-invalid key (row 0 can only ever pick itself, since
  it has exactly one valid candidate).
- **NaN handling**: a fully-masked softmax row is all-zero with no NaNs; a
  *partially* masked row (some but not all entries `-inf`) is completely
  unaffected and still sums to 1; `dense_attention` with a query that has
  zero allowed keys produces zero output for that query and leaves every
  other (normally-masked) row's softmax untouched, including a mix of
  masked and unmasked queries in the same batch.

Each test isolates one property rather than eyeballing end-to-end output,
so a failure points at *what* broke (masking, softmax, windowing) rather
than just "attention is wrong somewhere."

## 3. Why sliding-window attention loses information

`examples/walkthrough.py` runs a 4-token, head_dim=2 example through both
mechanisms so the difference is visible in actual numbers rather than
asserted abstractly. `Q = K = [[1,0], [0,1], [1,1], [-1,1]]`,
`V = [[1,10], [2,20], [3,30], [4,40]]` (token index encoded in `V` so it's
obvious which tokens contribute to which output).

For the last query (token 3, vector `[-1, 1]`), **dense** causal attention
produces:

```
weights = [0.065, 0.266, 0.131, 0.539]
output  = [3.144, 31.440]
```

Token 1 gets weight **0.266** — the second-largest weight in the row, ahead
of token 2's 0.131. That's because `q3 · k1 = 1`, a genuinely high
similarity score, not noise.

With `window_size=2`, query 3 can only see tokens 2 and 3 (window =
`[i-1, i]`). Tokens 0 and 1 are *structurally* excluded — not down-weighted,
removed from the softmax entirely:

```
weights = [0.000, 0.000, 0.196, 0.804]
output  = [3.804, 38.044]
```

Token 1's 0.266 of attention mass doesn't get redistributed proportionally;
it collapses entirely onto tokens 2 and 3, which is why the output shifts
from `[3.144, 31.44]` to `[3.804, 38.04]` — a real, non-trivial change, not
rounding error.

**The general failure mode**: sliding-window attention assumes that
"relevant" and "recent" are the same thing. That's a fine assumption for
local syntax (adjacent tokens in code, nearby words in a sentence) but
false whenever a distant token is semantically important regardless of
position — a variable's definition far above its use, a document's title
influencing every later section, an instruction given once near the start
of a conversation. Dense attention would give that token real weight (as
token 1 got 0.266 above); sliding-window attention gives it exactly zero,
permanently, no matter how relevant it is, because the mask decides
relevance by distance alone before the query/key similarity is ever
computed.

**BigBird's global tokens fix exactly this**, and it's checkable on the same
4-token example, not just asserted: marking tokens 0 and 1 as global
(`bigbird_mask(4, window_size=2, num_global=2, num_random=0)`) happens to
make the union `local | global` cover every causally-valid pair in this tiny
example, so the mask becomes identical to plain causal attention — and
`bigbird_attention` reproduces the dense output for query 3 *exactly*:
`weights = [0.065, 0.266, 0.131, 0.539]`, `output = [3.144, 31.440]`,
matching dense to the last digit. Compare that to the sliding-window-only
result two paragraphs up (`output = [3.804, 38.044]`): the only change is
which mask bits are turned on, and token 1's dropped 0.266 of attention mass
comes right back. That's the mechanism, not a coincidence of this example —
in a real model, whichever tokens you designate global (in practice often
fixed positions like `[BOS]`/`[CLS]`, or task-specific anchor tokens) stay
reachable from arbitrarily far away, while ordinary tokens still only get
the cheap local window plus a few random long-range connections.

The **random** component in BigBird exists for a different reason than
global tokens do: it gives the attention graph a non-zero probability of
connecting any two positions at all, which (per the BigBird paper's
theoretical argument) helps the sparse pattern retain expressive/
theoretical properties of full attention (e.g. Turing-completeness) that a
purely local+global pattern doesn't guarantee on its own. It's a
different kind of guarantee than global tokens give — global tokens
guarantee *specific* positions stay reachable; random edges guarantee the
graph as a whole doesn't have blind spots baked into its structure.

## 4. Benchmark: wall-clock and memory, seq_len 512 → 8192

`scripts/benchmark.py` times a forward pass (min of 5 reps, one untimed
warmup) and computes attention-matrix memory for dense, sliding-window
(`window_size=64`), and BigBird (`window_size=64, num_global=32,
num_random=32`) at `seq_len ∈ {512, 1024, 2048, 4096, 8192}`, `head_dim=64`.
Raw numbers in `benchmark_results/results.csv`, plots in
`benchmark_results/time.png` and `benchmark_results/memory.png`.

**Wall-clock time is essentially identical across all three variants at
every sequence length** (e.g. at `seq_len=8192`: dense 395ms, sliding-window
392ms, BigBird 394ms) — the three lines in `time.png` overlap almost
exactly, all growing quadratically with `seq_len`. This is not a bug, and
it's the most important honest result of this benchmark: **every variant in
this repo is built on `dense_attention`**, so every one of them computes the
full `Q @ K^T` and materializes a full `(seq_len, seq_len)` weight matrix —
the mask only decides which entries of that already-computed matrix get
zeroed before the second matmul. Masking is a small constant-factor
overhead on top of dense attention's cost, not a discount, because the
expensive part (the full matmul) still happens either way. A sparsity
pattern only saves time or memory once it's paired with a kernel that
*skips* computing the masked-out entries instead of computing and then
discarding them — e.g. a gather-based implementation that only ever forms
`Q @ K_window^T` for the keys actually in a query's window/global/random
set. That kernel is exactly what "reference implementation" in section 1
was set up to eventually be checked against, and is the natural next step
after this submission.

**Memory tells the story sparsity is actually for.** `mask_density` (the
fraction of the `seq_len × seq_len` grid each pattern keeps) drops sharply
with `seq_len`: sliding-window goes from 11.7% density at `seq_len=512` to
0.78% at `seq_len=8192`; BigBird from 20.1% to 1.53%. `memory.png` plots the
*as-built* materialized memory (identical for all three variants, per the
paragraph above, black dashed line) against the *theoretical* memory a
gather-based kernel would need if it only ever stored the `mask_density`
fraction of entries (colored lines). At `seq_len=8192` that's roughly a
**128× reduction** for sliding-window and a **65× reduction** for BigBird
versus the as-built dense matrix (2.09 MB and 4.11 MB respectively vs.
268.44 MB) — the entire point of sparse attention, visible even though this
repo's current implementations don't yet realize it.

## 5. Quality eval: char-GPT on TinyShakespeare

`sparse_attention/char_gpt.py` is a minimal 2-layer, single-head-per-layer
char-level GPT (embed dim 64, `nn.LayerNorm` + residual + 4x MLP per block)
that plugs `dense_attention` directly in as its attention mechanism —
single-head, not multi-head, because the question here is "does this
sparsity pattern hurt modeling quality," which doesn't need extra heads to
answer. `scripts/quality_eval.py` trains three otherwise-identical models
(same init seed, same batch order, same 500 AdamW steps, block_size=128) —
one per attention kind — on TinyShakespeare (1.1MB, downloaded via
`scripts/download_data.py`, 90/10 train/val char split), and plots
validation loss. Sparse settings: `window_size=16` (so local context is
~12.5% of the 128-token block — meaningfully sparse, not window≈block_size,
which would just degenerate back to dense per section 1), `num_global=4`,
`num_random=8` for BigBird.

Final validation loss (500 steps): **dense 2.0810, sliding-window 2.0307,
BigBird 2.0441** — both sparse variants outperformed dense, with
sliding-window lowest. `quality_eval_results/loss.png` shows sliding-window
pulling ahead of dense by roughly step 75 and staying ahead for the rest of
training; BigBird tracks close behind it.

This is a genuinely interesting result, not the one section 3 might predict
in isolation, and it's worth being honest about rather than cherry-picking
an explanation: at this *scale* (2 layers, embed dim 64, 500 steps, a
1MB dataset), a restricted attention pattern is acting as a form of
inductive bias / implicit regularization rather than an information
bottleneck. Character-level Shakespeare is dominated by short-range
structure — spelling, common bigrams/trigrams, word boundaries, line
rhythm — so a 16-token window already contains most of what predicts the
next character most of the time, and the model doesn't have to spend its
very limited capacity (single head, 64-dim) learning to *ignore* the 87.5%
of the window sliding-window already excludes by construction. Dense
attention has to learn that suppression itself, which is extra work for a
tiny model in a 500-step budget. This doesn't contradict section 3's
example — full attention still strictly dominates in expressive power (it
*can* recover anything sliding-window can, plus long-range dependencies
sliding-window structurally cannot) — it just shows that "more expressive"
and "easier to optimize well at small scale, in few steps" are different
axes. The gap would be expected to narrow or reverse with more capacity,
more steps, or a task with real long-range dependencies (e.g. needing to
recall a character name introduced many lines earlier) — which is exactly
the kind of task section 3's worked example is designed to isolate.

## 6. Scope of this submission

Implemented and tested: manual dense attention, sliding-window sparse
attention, BigBird-style local+global+random sparse attention, NaN-safe
softmax, the wall-clock/memory benchmark, and the char-GPT quality eval —
covering every item in the Task 1 checklist (1 through 6), each backed by
either a passing test suite (items 1-4) or a runnable script with saved,
inspectable output (items 5-6).

The depth-over-completeness note in the task brief is why sections 1-3
here go into the actual mechanism and worked numeric examples rather than
listing checklist items; the implemented patterns are correct
(verified against PyTorch's own kernel, and against each other) and the
writeup demonstrates why
the sparsity pattern matters, not just that it runs.
