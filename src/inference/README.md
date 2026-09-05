# src/inference

The section 8 deliverable: a trained checkpoint that takes a pack and a
pool and returns ranked picks.

| module | what it is |
| --- | --- |
| `probe.py` | `PickProbe` -- restore a checkpoint, pad ids, get a distribution over pack slots. Shared with `src/analysis/synergy.py`, which probes a model rather than using one. |
| `drafter.py` | `Drafter` -- the same thing in card *names*, with a ranked list out and the guards that names make possible. The CLI lives here. |
| `metrics.py` | top-1, top-3 and calibration over a split, reported all-picks and picks-0-8. |

## Why names and not ids

Every id in range is a legal card, so a caller who got their ids from the
wrong `vocab.json` -- a different set, a re-ingest that reordered it, a
guess -- gets a confident ranking of a pack they did not ask about, with
no shape error anywhere to catch it. `Drafter` takes names and refuses the
ones it does not recognise, reporting all of them at once.

Two further guards follow from the same reasoning. A pack longer than the
corpus geometry is rejected rather than truncated to fit. And the pack and
pick numbers are *derived* from the pool size, via the pool-as-prefix
identity in `src/data/dataset.py`, rather than supplied: they are not
independent of it, and a state where they disagree is one no draft
produces and the model will answer anyway.

## Why the metrics are reported twice

Top-k is trivially 1.0 on a pack of k cards or fewer, and a one-card pack
is a perfectly calibrated 100%-confidence bin for any model whatsoever. On
FIN those are 7.1% and 21.4% of rows for k=1 and k=3. So every figure is
reported over all picks and over picks 0-8, and the headline is picks 0-8
-- the same slice, and the same reasoning, as `summarise_by_pick` in
`src/training/evaluate.py`.

## Usage

```bash
# rank a pack you type
python -m src.inference.drafter \
    --checkpoint runs/bdh_d64_s92000 \
    --processed-dir data/processed/FIN.PremierDraft \
    --pack "Vivi Ornitier" "Tifa, Martial Artist" --pool "Cloud, Midgar Mercenary"

# rank real validation packs, and report the metrics
python -m src.inference.drafter \
    --checkpoint runs/bdh_d64_s92000 \
    --processed-dir data/processed/FIN.PremierDraft \
    --examples 3 --evaluate --json-out picks.json
```
