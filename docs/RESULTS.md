# Results

Two converged runs, one width, one seed each. That is enough to answer some
questions and not others, and the split matters:

- **Answered.** Whether the harness produces a real drafter (yes), whether
  it uses its pool (yes, heavily), whether it learned card interaction
  rather than colour-matching (yes, about half the pool effect is genuinely
  pairwise), and what the binding constraint on quality is at this width
  (capacity, not steps).
- **Not answered.** Whether BDH or cross-attention is the better arm.
  Nothing below separates them, and one seed per arm at one width could not
  have. `PROJECT_PLAN.md` §6's grid is what settles it.

**A boundary that makes older numbers unreadable.** The feature table was
rebuilt from 65 columns to 119 (15 global keywords + 73 mechanics). Loss is
not comparable across that change. Every number in this file
postdates it except those in the [superseded section](#superseded-the-cpu-pilot),
which is kept only so its retraction is on the record rather than implied by
deletion.

## The converged comparison

Both arms, `--width 64 --steps 92000 --batch-size 512 --learning-rate 3e-4
--seed 0`, on FIN.PremierDraft at commit `a53e5a27`. 92,000 steps at batch
512 is 47.1M examples, ten passes over the 4,711,938 training rows. Two T4
sessions, two segments each, both `completed`.

| | BDH | attention |
|---|---|---|
| parameters | 261,633 | 263,745 |
| val loss, all picks | 0.8214 | 0.8212 |
| val accuracy, all picks | 0.6841 | 0.6834 |
| val loss, picks 0–8 | 1.0037 | 1.0033 |
| val accuracy, picks 0–8 | 0.6248 | 0.6239 |
| best val loss | 0.8222 | 0.8211 |
| best step | 88,250 | 88,250 |
| training wall clock | 2,695s | 2,390s |
| throughput | 17,479 ex/s | 19,709 ex/s |

Nothing in the quality rows is bolded on purpose. Marking a winner per row
would assert an ordering across differences of 0.0002 that the next
paragraph spends its length denying, and a reader skims the bold.

Baselines on the same val split, identical for both. They are reported per
slice because the aggregate ones cannot be compared against a picks-0-8
model number — the mismatch §7 warns about when it makes picks 0–8 the
headline slice:

| | uniform | pick-rate prior | prior accuracy |
|---|---|---|---|
| all picks (589,008 rows) | 1.7994 | 1.5662 | 0.4526 |
| picks 0–8 (378,648 rows) | 2.2671 | **1.9474** | **0.3630** |

The prior is much weaker on the decision slice than in aggregate, which is
what you would expect: forced and near-forced picks are where knowing a
card's overall pick rate goes furthest.

**The arms tie, and the tie is the result.** 0.0002 on all picks and 0.0004
on the headline slice, at 0.8% different parameter counts. Both arms peaked
at the same step. A gap that small is not a small win — it is two runs
landing in the same place, well inside what a seed change moves, and it
should not be reported as an ordering in either direction.

**The models are real drafters.** On the headline slice, 1.0037 against a
1.9474 pick-rate prior is **0.94 nats** below a baseline that already knows
how good every card is in the abstract, and accuracy is 0.6248 against
0.3630 — 26 points. That margin has to come from conditioning on the pack
and pool, and the
[pool probes](#the-model-uses-its-pool-and-not-only-for-colour) confirm
directly that it does.

**Attention is the cheaper arm here.** 19,709 ex/s against 17,479, some 13%,
which is the opposite of what the fused-kernel work is meant to address and
worth remembering before reading a per-step cost off this table.
`ARCHITECTURE.md` predicts this shape at iso-parameter sizing.

## Ten epochs is past the point of return

Best val loss reached by each epoch boundary:

| epoch | BDH | attention | BDH gain |
|------:|----:|----------:|---------:|
| 1 | 0.8657 | 0.8699 | — |
| 2 | 0.8509 | 0.8509 | 0.0148 |
| 3 | 0.8407 | 0.8421 | 0.0102 |
| 4 | 0.8345 | 0.8367 | 0.0062 |
| 5 | 0.8301 | 0.8309 | 0.0044 |
| 6 | 0.8273 | 0.8275 | 0.0028 |
| 7 | 0.8241 | 0.8257 | 0.0032 |
| 8 | 0.8236 | 0.8226 | 0.0005 |
| 9 | 0.8227 | 0.8222 | 0.0009 |
| 10 | 0.8222 | 0.8211 | 0.0005 |

**The constraint is capacity, not steps.** The last 20,000 steps move inside
a 0.0040 band for BDH and 0.0043 for attention. Both arms peak at step
88,250 of 92,000 and both finish slightly above their own best. Train loss
ends 0.04 below val. The remaining passes are buying memorisation, and more
of them would buy more of it.

**Three epochs is the grid's budget, and it costs a known amount.** Epoch 3
is 0.0185 nats short of epoch 10 for a third of the compute. It costs that
almost identically in both arms — 0.0014 apart at epoch 3 against 0.0002 at
epoch 10 — which is what matters for a comparison, though it does tilt very
slightly toward BDH. This is a budget decision and not a convergence claim:
every grid cell will be truncated, larger cells sit further from their own
converged loss than smaller ones, and that biases the fitted $\alpha$ mildly
optimistic and inflates $E$. Holding the pass count *constant* across cells
is what keeps the truncation out of $\beta$. See `grid.py`'s
`DEFAULT_EPOCHS`.

**This is why the width ladder had to move.** A plateau at 261k parameters
is a statement about 261k parameters. `LADDER` now starts at the measured
d=64 anchor and climbs 128/256/512, spanning 61× in $N$.

## The model uses its pool, and not only for colour

`PROJECT_PLAN.md` §7 asks for this explicitly, and it is the part top-1
agreement cannot answer. Run with `src/analysis/synergy.py` against both
converged checkpoints, 15,988 val rows with a non-empty pool.

### Pool ablation

Re-score real picks with the pool replaced by a decoy of the same size.
`permuted` borrows another drafter's real pool — coherent, right colours,
right curve, wrong drafter — and is the harder control. `random` samples
card ids uniformly and destroys pool coherence too.

| | BDH | attention |
|---|---|---|
| loss, real pool | 0.8323 | 0.8346 |
| loss, permuted decoy | 2.5670 | 2.5512 |
| **pool is worth** | **+1.7347 nats** | **+1.7166 nats** |
| loss, random decoy | 1.9362 | 1.8675 |
| pool is worth (random) | +1.1039 | +1.0329 |
| accuracy, real → permuted | 0.6790 → 0.3734 | 0.6754 → 0.3726 |
| top-1 pick changes | 59.2% of rows | 59.0% of rows |

*(The real-pool loss here is above the 0.8214 headline because empty-pool
rows are excluded: their two scorings are identical by construction and
would dilute every number toward zero.)*

**The pool is worth more than the entire uniform baseline.** +1.73 nats
against a uniform loss of 1.7994. Handed the wrong pool the model does not
degrade toward guessing — it goes *past* uniform to 2.57, because it
confidently commits to the colours the decoy implies. That is the signature
of a model reading the pool hard, not ignoring it. Accuracy falls 30 points
and the top-1 pick changes on 59% of rows.

The permuted decoy costing *more* than the random one (1.73 vs 1.10) is the
right way round and worth stating: a coherent-but-wrong pool is a more
convincing lie than noise, so it misleads the model further.

### One pack, three pools

Hold a pack fixed and vary only the pool. Spearman correlation is against
the empty-pool ranking.

| | BDH | attention |
|---|---|---|
| rank correlation, drafter's own pool vs. empty | +0.31 | +0.26 |
| rank correlation, another drafter's pool vs. empty | +0.77 | +0.66 |
| largest log-probability swing on one candidate | 5.03 nats | 3.99 nats |

Both arms switch their top pick away from the empty-pool answer as soon as
a real pool is present. A colour-matcher would barely reorder.

### Pairwise synergy, with the colour control

Lift is `log p(candidate | pool of 6 copies of anchor) − log p(candidate |
empty pool)`, over a 12 × 40 matrix of cards drawn from the 362 that
actually appear in a pack. Colourless cards are excluded from the colour
split rather than assigned to a side.

| | BDH | attention |
|---|---|---|
| mean lift, shares a colour | +1.599 | +1.429 |
| mean lift, shares no colour | +0.010 | −0.156 |
| **colour gap** | **1.589** | **1.585** |
| within-colour spread | 1.765 | 1.632 |
| **interaction spread** | **1.456** | **1.328** |
| **interaction variance share** | **0.483** | **0.517** |

**Colour is a large effect and it is not the whole effect.** Sharing a
colour with the pool is worth ~1.59 nats in both arms, and a candidate
sharing none is worth roughly zero. Read alone that is colour-matching.

The number that goes past it is the last row. A lift matrix is inflated by
two one-card properties that have nothing to do with pairing: a candidate
that gains from *any* pool (a row effect) and an anchor that shouts its
colour at *every* candidate (a column effect). `interaction_residual`
subtracts both, plus the grand mean, and what remains is the part that
depends on which candidate met which anchor. **About half the variance
survives** — 48% for BDH, 52% for attention. A purely additive model, which
is what a colour-matcher plus a card-quality prior would produce, scores
zero here.

That the residual spread is as large in the colour-*disjoint* half (1.24
BDH / 1.01 attention) as in the colour-sharing half (1.13 / 1.10) says the
same thing from the other direction: the pairwise structure is not confined
to pairs that colour already explains.

The ranked pairs make it legible. The strongest *raw* lifts are dominated by
one-card effects — `Eden, Seat of the Sanctum ← pool of Plains` is a land
responding to a colour beacon. After removing main effects the list changes
character, and both arms independently surface the same pairs:

| interaction | BDH | attention |
|---|---|---|
| `Ice Magic ← Counterspell` | +3.20 | +2.79 |
| `Ice Magic ← Swallowed by Leviathan` | +3.74 | +3.95 |
| `Ice Magic ← Shantotto, Tactician Magician` | +4.16 | +3.88 |
| `The Earth Crystal ← Nyxbloom Ancient` | +3.25 | +3.36 |

Two independently trained models, different architectures, agreeing on which
cards belong together — and the agreement tracks the mechanic columns rather
than colour. Reading the attributes out of `card_features.npz` rather than
off the card faces:

- `Ice Magic` is the top interaction candidate for *both* arms, and it is
  mono-blue with `instant_speed` and `bounce`. Every one of its strongest
  anchors carries `instant_speed` too, and two of them
  (`Counterspell`, `Syncopate`) carry `counterspell`. `Scorpion Sentinel`,
  attention's top anchor, is the exception that proves the point: mono-blue
  with no mechanic flags at all, and it ranks lower for BDH.
- `The Earth Crystal` is mono-green at mana value 4 with `activated_mana_cost`
  and `cost_reduction`; `Nyxbloom Ancient` is mono-green at mana value 7. A
  cost-reducer paired with the most expensive thing to reduce.

Colour alone cannot produce these rankings — the anchors are all the *same*
colour as the candidate, so the colour term is constant across them and the
ordering has to come from somewhere else. This is the mechanical-feature
columns doing the job §3b argued for, and it is the strongest evidence in
this file that the rebuild from 65 to 119 columns bought something.

**The honest caveat.** A pool of six copies of one common is a state no
drafter ever holds. It is deliberately exaggerated so the anchor's
contribution clears the noise floor, at the cost of asking the model about a
position outside its training distribution. Read the sign, the ordering and
the variance share; do not read the magnitudes as nats a real draft would
produce. External validation against CubeCobra co-occurrence, which §7 names
as the follow-up, is what would close that gap.

## Forced picks dilute the headline loss

The by-pick breakdown, on the converged BDH and attention runs. 42,072 rows
per pick.

| pick | pack size | BDH | attention | uniform |
|-----:|----------:|----:|----------:|--------:|
| 0 | 14 | 1.0077 | 1.0066 | 2.6391 |
| 1 | 13 | 1.0907 | 1.0928 | 2.5649 |
| 2 | 12 | 1.0778 | 1.0776 | 2.4849 |
| 3 | 11 | 1.0633 | 1.0604 | 2.3979 |
| 4 | 10 | 1.0213 | 1.0227 | 2.3026 |
| 5 | 9 | 1.0039 | 1.0025 | 2.1972 |
| 6 | 8 | 0.9633 | 0.9628 | 2.0794 |
| 7 | 7 | 0.9167 | 0.9151 | 1.9459 |
| 8 | 6 | 0.8882 | 0.8888 | 1.7918 |
| 9 | 5 | 0.8212 | 0.8209 | 1.6094 |
| 10 | 4 | 0.7284 | 0.7279 | 1.3863 |
| 11 | 3 | 0.5837 | 0.5839 | 1.0986 |
| 12 | 2 | 0.3340 | 0.3340 | 0.6931 |
| 13 | 1 | 0.0000 | 0.0000 | 0.0000 |

42,072 rows — 7.1% of the val split — are single-card packs contributing
identically zero, and picks 11–13 are 21% of the split at well under half
the average loss. **The fit runs on the picks 0–8 number**, because the
fitted $E$ is compared against a human-disagreement floor that §7 concluded
must be measured where a real decision exists.

**Pick 0 is the exception to the monotone trend**, and it is informative
rather than noise. Loss falls monotonically from pick 1 to pick 13 as the
pack empties, but pick 0 (1.0077) sits *below* picks 1–3. Pick 0 is the only
row with an empty pool, so it is pure card quality with nothing to condition
on and nothing to get wrong — the hardest picks are the early ones where a
pool exists but is still short. The converged model makes this visible; the
pilot did not.

## Measured activation density

Taken with `src/training/density.py` on the val split. Both columns are
119-column models; the difference between them is training length.

| | 3,000 steps | converged (92,000) |
|---|---|---|
| query | 0.3949 | 0.3851 |
| gate | 0.1779 | 0.1626 |
| score | 0.1566 | 0.1478 |
| skippable | 18.9% | **19.2%** |

**Training genuinely sparsifies the arm, and keeps doing it.** Every
fraction is well below its initialisation value (~0.5 for query, ~0.25 for
the products, exactly what `bdh_arm.py` predicts from a symmetric encoder),
and every fraction falls further between 3,000 steps and convergence. This
is the sparsity half of §3a's acceptance gate, passed on a converged model,
which is the only place the question means anything.

**The FLOP consequence remains small, as predicted.** At width 64 the
measured density puts the perfectly-sparse bound at 5,345,824 FLOPs against
a dense 6,615,040 — 19.2% skippable, out of a ceiling of 22.8% that even
zero density could not beat. The three encodes are 84% of the ideal total
and are paid at any density.

So `ARCHITECTURE.md`'s conclusion survives its own measurement on a
converged model: BDH starts 53% more expensive than the arm it is compared
against, sparsity recovers about a fifth of that, and a sparsity-based
efficiency claim does not survive at iso-parameter sizing. Note also that
unstructured sparsity wins memory traffic rather than FLOPs — a GPU
multiplies by zero as fast as by anything else — so the 19.2% is an upper
bound on a saving that needs block structure to be collected at all.

## Superseded: the CPU pilot

**Retracted.** An earlier version of this file reported BDH ahead of
attention — 0.9080 against 0.9323 on all picks, 1.1150 against 1.1473 on
picks 0–8 — from a pair of CPU runs at `--steps 3000`, which is 0.33 of an
epoch, on the 65-column feature table. That table is gone and those runs
were nowhere near converged.

**The converged runs contradict it.** At ten epochs on 119 columns the arms
tie to four decimal places. What the pilot measured was which arm descends
faster early, not where either one lands: BDH's best was its last step and
still falling, while attention had already peaked and gone flat. A gap read
off two curves at different points on their descent is a statement about
convergence rate, and the file presented it as a statement about quality.

Nothing in the pilot should be cited. The three caveats it listed — untuned
shared learning rate, one seed, and cost measured on a contended CPU — all
still stand, and the first two apply to the converged runs as well.

## Reproducing

The converged runs, via the Colab driver (see `scripts/README.md`):

```
scripts/colab_run.sh --gpu T4 --set FIN --arm bdh       --width 64 --steps 92000
scripts/colab_run.sh --gpu T4 --set FIN --arm attention --width 64 --steps 92000
```

A grid cell is sized in epochs instead, which the driver now takes directly:

```
scripts/colab_run.sh --gpu A100 --set FIN --arm bdh --width 256 \
    --epochs 3 --data-fraction 0.25
```

The probes, locally against a retrieved checkpoint:

```
python -m src.analysis.synergy --checkpoint runs/bdh_d64_s92000 \
    --processed-dir data/processed/FIN.PremierDraft \
    --rows 16384 --anchors 40 --candidates 12 --json-out runs/bdh_d64_s92000/synergy.json

python -m src.training.density --checkpoint runs/bdh_d64_s92000 \
    --processed-dir data/processed/FIN.PremierDraft
```

**Use `--epochs` rather than `--steps` for any grid cell.** §6 requires the
$D$ axis to be data scale; at fixed steps a small `--data-fraction` silently
means many passes and $\beta$ becomes a repetition exponent. `run.py` warns
past two passes and `grid.py` converts epochs to steps per cell.

`runs/` is gitignored, so the numbers above are the record. Each run's
`metrics.json` carries the full learning curve, the by-pick table, the
parameter breakdown and the exact config that produced it; `synergy.json`
and `density.json` carry the probe outputs.
