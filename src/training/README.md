# Training

The single-cell runner, the pilot grid, and the curve fit for $L(N, D) = E + A/N^\alpha + B/D^\beta$ that chooses what size the shipped drafter should be.

| file | role | status |
|---|---|---|
| `train.py` | one cell: optimiser, schedule, loop, baselines | done |
| `evaluate.py` | exact full-split eval + per-pick breakdown | done |
| `checkpoint.py` | save/restore params + metadata | done |
| `run.py` | CLI entry point for a single cell | done |
| `grid.py` | the sweep | not started |
| `scaling_fit.py` | the curve fit | not started |

## The fit is a sizing procedure

$L(N, D)$ is not fit here to make a claim about scaling exponents. It is fit so that, given the compute budget available for the final run, the compute-optimal $(N^*, D^*)$ tells us how wide the drafter should be and how much data it should see — and so the arm with the better curve at that budget is the one that ships. See `docs/PROJECT_PLAN.md` §6.

It follows Chinchilla's robust procedure: Huber loss on log-residuals rather than least squares on raw loss, since raw least squares over-weights the small-$N$, high-loss corner, with bootstrapped confidence intervals rather than bare point estimates.

## Running one cell

    python -m src.training.run --processed-dir data/processed/FIN.PremierDraft \
        --width 64 --epochs 1 --out-dir runs/attn_d64

`--data-fraction` is the grid's D axis. It subsamples **drafts**, not rows: a fraction drawn over rows would leave most drafts present with a few picks missing, which is a different and easier distribution than seeing fewer complete drafts.

**Use `--epochs`, not `--steps`, for grid cells.** At fixed steps a small `--data-fraction` silently means many passes over a small set, and the fitted $\beta$ then measures data repetition rather than data scale. `run.py` warns past two passes.

## Throughput is the current blocker

The pilot runs managed **561 examples/second on CPU** — `jax.default_backend()` returns `cpu` and there is one device. That is 2.3 hours per epoch at $d=64$, roughly 37 hours per epoch at $d=256$, and makes the grid infeasible. Everything downstream of the fit is gated on moving to a GPU, which for a model this size is an afternoon of rented time.

## Checkpoints

One directory per run holding `params.msgpack` and `metadata.json`, written whenever validation loss improves. flax msgpack rather than orbax — a cell is 1MB to 64MB, so there is nothing to shard.

**`flax.serialization.from_bytes` does not validate against its template.** Handed a d=64 template and d=32 bytes it returns the d=32 arrays and raises nothing (verified against flax 0.12.9). Comparing the restored parameter count to the metadata does not catch it either, since both come from the same file and agree with each other. `restore` therefore checks structure, shapes and dtypes explicitly. Without it, a checkpoint written before an architecture change would load silently into the wrong shapes, evaluate without complaint, and report a plausible but wrong loss for a grid cell — the sort of error a curve fit absorbs rather than reveals.

## Resuming an interrupted run

A Colab runtime caps at 12h and is reclaimed after 90 minutes idle, so a long
cell will be interrupted. `--resume` continues one:

    python -m src.training.run --processed-dir data/processed/FIN.PremierDraft \
        --width 64 --steps 3000 --out-dir runs/attn_d64 \
        --max-seconds 3600 --resume

`--max-seconds` stops at the next `--eval-every` boundary once that much wall
clock has passed and exits **75** instead of 0, leaving a resumable state
behind. Re-invoking the identical command continues it. Anything else non-zero
is a real failure, so a caller driving a remote session can tell "there is more
to do" from "this broke" without parsing output.

This is deliberately *not* the same artefact as the best-val checkpoint above.
That one holds the parameters any later analysis wants; it carries no optimiser
moments, no step counter and no position in the shuffled batch stream.
Restarting from it would reset Adam to zero partway along a decayed learning
rate and replay data the run had already seen. So `resume.msgpack` and
`resume.json` sit alongside it holding current params, optimiser state,
best-so-far, the history, and the stream position -- written atomically at every
evaluation boundary, and deleted once the run completes so a finished cell
cannot be mistaken for a resumable one.

**The batch stream position is restored exactly, not approximately.**
`BatchStream` saves `(reshuffles, cursor)` rather than the permuted order
itself -- the order is a few million int64s, and it is anyway a pure function of
the seed and the number of reshuffles so far, so restoring replays that many
permutations in milliseconds. `tests/test_checkpoint.py` asserts that a run
chopped into four segments lands on bit-for-bit identical parameters to an
uninterrupted one, with an identical loss curve. A resume that silently
replayed or skipped data would still converge, just to a different place, and
would put a discontinuity at every interruption that the scaling fit would read
as structure.

**A resume across a config change is refused, not adapted.** The saved state
carries a fingerprint of the arm, model config, train config and training-row
count; a mismatch raises. Continuing a d=64 run into a d=128 tree would produce
a continuous loss curve and a meaningless result -- the same failure mode the
explicit shape checking above exists to prevent, and harder to spot.

## Why the per-pick breakdown is a first-class output

The aggregate loss averages fourteen quite different problems. Measured on the FIN val split: **7.1% of rows have a one-card pack and loss identically zero**, and 21.4% have a pack of three or fewer.

Picks that are exactly zero are harmless to the exponents — they scale $A$, $B$ and $E$ but leave $\alpha$ and $\beta$ alone. The risk is picks 11–12: easy but not trivial, so they saturate at small $N$ while the hard picks keep improving, and a subset that stops responding to $N$ while the rest continues bends the aggregate curve in a way that reads as an exponent. `summarise_by_pick` reports all-picks and picks-0-8 side by side.

**Headline numbers are the picks-0-8 slice**, for the same reason: it is where a decision actually exists.
