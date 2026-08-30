# Training and analysis

The grid runner (sweeping model size x dataset fraction x architecture x seed), the curve-fitting code for $L(N, D) = E + A/N^\alpha + B/D^\beta$, and the matched-state disagreement measurement used to validate the fitted $E$ against a real human Bayes-error floor.

| file | role | status |
|---|---|---|
| `train.py` | one cell: optimiser, schedule, loop, baselines | done |
| `evaluate.py` | exact full-split eval + per-pick breakdown | done |
| `checkpoint.py` | save/restore params + metadata | done |
| `run.py` | CLI entry point for a single cell | done |
| `grid.py` | the sweep | not started |
| `scaling_fit.py` | the curve fit | not started |
| `bayes_floor.py` | the floor measurement | blocked on PROJECT_PLAN 6a |

## Running one cell

    python -m src.training.run --processed-dir data/processed/FIN.PremierDraft \
        --width 64 --steps 3000 --out-dir runs/attn_d64

`--data-fraction` is the grid's D axis. It subsamples **drafts**, not rows: a
fraction drawn over rows would leave most drafts present with a few picks
missing, which is a different and easier distribution than seeing fewer
complete drafts.

## Checkpoints

One directory per run holding `params.msgpack` and `metadata.json`, written
whenever validation loss improves. flax msgpack rather than orbax -- a cell is
1MB to 64MB, so there is nothing to shard.

**`flax.serialization.from_bytes` does not validate against its template.**
Handed a d=64 template and d=32 bytes it returns the d=32 arrays and raises
nothing (verified against flax 0.12.9). Comparing the restored parameter count
to the metadata does not catch it either, since both come from the same file
and agree with each other. `restore` therefore checks structure, shapes and
dtypes explicitly. Without it, a checkpoint written before an architecture
change would load silently into the wrong shapes, evaluate without complaint,
and report a plausible but wrong loss for a grid cell -- the sort of error a
scaling fit absorbs rather than reveals.

## Why the per-pick breakdown is a first-class output

The aggregate loss averages fourteen quite different problems. Measured on the
FIN val split: **7.1% of rows have a one-card pack and loss identically zero**,
and 21.4% have a pack of three or fewer.

Picks that are exactly zero are harmless to the exponents -- they scale $A$,
$B$ and $E$ but leave $\alpha$ and $\beta$ alone. The risk is picks 11-12:
easy but not trivial, so they saturate at small $N$ while the hard picks keep
improving, and a subset that stops responding to $N$ while the rest continues
bends the aggregate curve in a way that reads as an exponent. `summarise_by_pick`
reports all-picks and picks-0-8 side by side so the fit can be checked both ways
before the grid commits to one.
