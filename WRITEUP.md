# Sparse Attention from Scratch — Writeup

Postman AI/ML Recruitment Task, Task 1. This writeup covers what's implemented
(manual dense attention, sliding-window sparse attention, and the correctness
harness comparing both against reference), why sliding-window attention loses
information relative to dense attention, and what's out of scope given the
time available for this submission.

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

## 2. Correctness harness

11 tests across `tests/test_dense_attention.py` and
`tests/test_sliding_window.py` (`pytest -q` → `11 passed`):

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

This is precisely the gap BigBird-style attention (local + global +
random) is designed to close: a small set of **global tokens** are made
visible to every query regardless of distance, so structurally important
content (like token 1 in this example, if it happened to be a global
token) survives the sparsification instead of being silently dropped. The
**random** component exists for a different reason — it gives the
attention graph a non-zero probability of connecting any two positions,
which (per the BigBird paper's theoretical argument) helps the sparse
pattern retain the expressive/theoretical properties of full attention
(e.g., Turing-completeness) that a purely local+global pattern doesn't
guarantee on its own.

## 4. Scope of this submission

Implemented and tested: manual dense attention, sliding-window sparse
attention, and a correctness harness for both (checklist items 1 and 2).

Not implemented in this submission, due to time constraints: BigBird-style
local+global+random block-sparse attention (item 3), NaN handling for
fully-masked queries (item 4), the wall-clock/memory benchmark across
seq_len 512→8192 (item 5), and the char-GPT quality eval on TinyShakespeare
(item 6).

For item 4 specifically, the fix is straightforward given the existing
code and worth stating even though it isn't implemented: a fully-masked
query row has every score set to `-inf` before softmax, so
`x - x.max(dim=-1)` becomes `-inf - (-inf)`, which is `nan`, and that NaN
propagates through the whole row and then through `weights @ V`. The
correct behavior is to detect rows where every entry is masked (`~mask`
is all-`True` for that row) and force their output to zero (or to the
row's own value via a residual, depending on the surrounding model) rather
than let a fully-excluded query silently corrupt the batch with NaNs.

The depth-over-completeness note in the task brief is why sections 1-3
here go into the actual mechanism and a worked numeric example rather than
listing checklist items; the two implemented patterns are correct
(verified against PyTorch's own kernel) and the writeup demonstrates why
the sparsity pattern matters, not just that it runs.
