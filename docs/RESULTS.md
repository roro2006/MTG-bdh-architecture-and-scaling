# Results

Everything here is a *pilot*: one width, one seed, one learning rate shared
across both arms, 0.33 of an epoch. It exists to show the harness produces
sane numbers before the grid spends real compute, which is stage 3 of
`PROJECT_PLAN.md` §9's build order. Nothing below is a scaling result and
none of it licenses a claim about either architecture.

## The first end-to-end runs

Both arms, `--width 64 --steps 3000 --batch-size 512 --learning-rate 3e-4
--seed 0`, on FIN.PremierDraft. CPU-only JAX, both arms unfused, run
concurrently on the same machine.

| | attention | BDH |
|---|---|---|
| parameters | 260,289 | 258,177 |
| arm parameters | 100,416 | 98,304 |
| best val loss | 0.9402 (step 2,750) | **0.9191** (step 3,000) |
| val loss, all picks | 0.9323 | **0.9080** |
| val loss, picks 0–8 | 1.1473 | **1.1150** |
| val accuracy, picks 0–8 | 0.5755 | **0.5890** |
| wall clock | 2,745s | 3,363s |
| epochs | 0.326 | 0.326 |

Baselines on the same split: uniform 1.7994, pick-rate prior 1.5662
(accuracy 0.4526).

**The models learn something real.** Both arms sit ~0.63 nats below the
pick-rate prior. That baseline already knows how good every card is in the
abstract, so beating it by that margin is evidence of pool-conditional
behaviour rather than card quality memorised twice.

**Iso-parameter sizing holds.** 258,177 against 260,289 is a 0.8% gap, and
the arms themselves are within 2.2%. `neuron_multiplier=4` does what
`ARCHITECTURE.md` says it does.

**Neither arm is converged and BDH least of all.** Attention peaked at step
2,750 and was flat-to-worse by 3,000; BDH's best was its last step and was
still falling. Whatever margin BDH holds here is a lower bound on the
margin at convergence, which is a reason to distrust the comparison rather
than to like it.

## What this does not show

The temptation is to read "BDH wins by 0.03 nats on real decisions" as a
finding. It is not one, for three reasons that all have to be closed before
the grid runs.

**The learning rate is untuned and shared.** 3e-4 was the default, applied
to both arms. `lr_sweep.py` exists because an unswept rate lets tuning
degradation get absorbed into the fitted exponent, and the identical
argument applies across the architecture axis: if 3e-4 happens to suit BDH
better, this table reports that and nothing else. Sweeping per arm as well
as per width is a prerequisite, not a refinement.

**One seed.** No error bar, so a 0.03 nat gap has no scale to be judged
against.

**BDH is ahead per parameter and behind per second.** 3,363s against
2,745s, a 1.23× ratio. That is the whole model, not the arm, measured on
CPU with two runs contending for one machine — indicative only, and not a
substitute for the FLOP accounting. But it is the shape
`ARCHITECTURE.md`'s fairness note predicts, and it is why both the
iso-parameter and iso-FLOP fits get reported rather than whichever one
reads better.

## Forced picks dilute the headline loss

The by-pick breakdown matters more than the aggregate. Loss falls
monotonically as the pack empties, and the last pick is not a decision at
all:

| pick | pack size | attention | BDH | uniform |
|-----:|----------:|----------:|----:|--------:|
| 0 | 14 | 1.1500 | 1.1654 | 2.6391 |
| 1 | 13 | 1.2479 | 1.1957 | 2.5649 |
| 4 | 10 | 1.1782 | 1.1366 | 2.3026 |
| 8 | 6 | 1.0033 | 0.9800 | 1.7918 |
| 11 | 3 | 0.6371 | 0.6269 | 1.0986 |
| 12 | 2 | 0.3640 | 0.3588 | 0.6931 |
| 13 | 1 | 0.0000 | 0.0000 | 0.0000 |

42,072 rows — 7.1% of the val split — are single-card packs contributing
identically zero, and picks 11–13 are 21% of the split at a third of the
average loss.

This is the same distortion `PROJECT_PLAN.md` §6a found in the floor
measurement, showing up now in the loss itself, and it has a direct
consequence for stage 5: **the fit should be run on the picks 0–8 number,
not the all-picks number.** The fitted $E$ is meant to be compared against a
human-disagreement floor that §6a has already concluded must be measured
where a real decision exists. Fitting $E$ over a population that is 7%
zeros and comparing it to a floor measured over picks 0–8 would compare two
different quantities and the mismatch would look like a finding.

## Measured activation density

Taken with `src/training/density.py` on the trained checkpoint, over the
val split. The initialised column is the same measurement on a 20-step
model, kept because it is the number an acceptance check would have
reported if taken at the wrong time.

| | initialised | trained |
|---|---|---|
| query | 0.5024 | 0.3949 |
| gate | 0.2446 | 0.1779 |
| score | 0.2483 | 0.1566 |

**Training genuinely sparsifies the arm.** Every fraction falls well below
its initialisation value, and the initialisation values are exactly what
`bdh_arm.py` predicts from a symmetric encoder (~0.5, and ~0.25 for a
product of two such). This is the sparsity half of `PROJECT_PLAN.md` §3a's
acceptance gate, and it passes — on a trained model, which is the only
place the question means anything.

**The FLOP consequence remains small, as predicted.** At width 64 the
measured density puts the perfectly-sparse bound at 5,365,063 FLOPs against
a dense 6,615,040 — **18.9% skippable**, out of a ceiling of 22.8% that
even zero density could not beat. The three encodes are 84% of the ideal
total and are paid at any density.

So `ARCHITECTURE.md`'s conclusion survives its own measurement: BDH starts
53% more expensive than the arm it is compared against, sparsity recovers
about a fifth of that, and a sparsity-based efficiency claim does not
survive at iso-parameter sizing. The interesting question stays quality per
parameter and per dense FLOP.

## Reproducing

```
python -m src.training.run --processed-dir data/processed/FIN.PremierDraft \
    --out-dir runs/attn_d64_s3000 --arm attention \
    --width 64 --steps 3000 --eval-every 250 --seed 0

python -m src.training.run --processed-dir data/processed/FIN.PremierDraft \
    --out-dir runs/bdh_d64_s3000 --arm bdh \
    --width 64 --steps 3000 --eval-every 250 --seed 0

python -m src.training.density --checkpoint runs/bdh_d64_s3000 \
    --processed-dir data/processed/FIN.PremierDraft
```

`runs/` is gitignored, so the numbers above are the record; each run's
`metrics.json` carries the full learning curve, the by-pick table, the
parameter breakdown and the exact config that produced it.
