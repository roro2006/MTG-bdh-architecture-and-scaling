# Project Plan

## 0. What's being tested, and what's deliberately out of scope

The central claim under test: the scaling form

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

is fit independently for the attention arm and the BDH arm described in `ARCHITECTURE.md`, on identical data and a shared input representation, to see whether $\alpha$ and $\beta$ hold the same values outside the open-vocabulary language modeling regime both architectures were originally validated on. A second, harder question rides along with it: whether the fitted $E$ converges toward an independently measured human disagreement rate, rather than being accepted as a free-fit constant the way it usually is.

This first version of the project is scoped to one set, one event type (`PremierDraft`), and the pick-prediction task alone. Multi-set transfer scaling, MoE routing scaling, and the circuit-level interpretability work that the composite-embedding design in `ARCHITECTURE.md` sets up for are all real follow-ons, but they depend on this study's trained checkpoints and shouldn't be running concurrently with it — trying to do all of it at once is the fastest way to end up with none of it done well.

## 1. Task formalization

At pick $t$: the current pack, the pool accumulated so far, and pack/pick number as scalar features, are the inputs; the label is the card the human actually took (`pick` in 17lands' `draft_data_public`). Loss is computed only over the cards physically present in the pack — never the full vocabulary — which is what makes this a genuinely closed, combinatorially bounded task rather than an open-vocabulary one wearing a small vocabulary's clothes.

Splits are drawn on `draft_id`, not on individual rows. All ~45 picks belonging to one draft land in the same split. This is the same discipline as the addition-transformer exercise's train/test partition — getting it wrong here is subtler, because a leak wouldn't look like memorization of a single example, it would look like the model quietly learning something about a specific draft's trajectory that it has no business knowing at pick 3.

## 2. Data

Single-set `draft_data_public.<SET>.PremierDraft.csv.gz` from 17lands, vocabulary built from its `pack_card_*` columns. See `DATA.md` for what's been verified about this source directly, including actual file sizes pulled from the bucket rather than taken from documentation.

Before the train/val/test split is drawn, a separate subset is carved out: states where the same pack-and-pool combination recurs across different drafts and different players. This subset is never trained on — it exists solely to measure how often two people facing the same decision disagree, which is the input to the Bayes-floor check in §6. (What "the same state" means is now an open question rather than a settled one — see §6a.)

Exclusion is at *draft* granularity, not row granularity. Dropping only the matched row would leave that row's label sitting inside the pool of the very next pick of the same draft, which is in training — the answer would leak through the pool even though the row itself was withheld. `split_by_draft` supports row-level exclusion as an escape hatch if draft-level turns out to cost too much of the corpus, and using it would have to be declared.

## 3. Architecture

Full design is in `ARCHITECTURE.md`. The two things worth restating here because they drive the grid design directly:

**3a — the BDH port is the long pole.** There's no existing JAX implementation to build from; the reference implementation is a bare PyTorch script. Before any grid compute goes anywhere near it, it needs to pass its own acceptance test: train stably on a small toy task, show no NaNs, and actually exhibit the sparse/positive activation pattern the architecture is supposed to produce. Given this is a few-months-old, single-paper architecture, the honest expectation is that this stage takes real debugging time and shouldn't be assumed to just work on the first pass.

**3b — parameter counts get derived by hand, not just read off `count_params()`.** Same discipline as the original addition-transformer exercise's exact term-by-term derivation. This matters more here than it did there, because BDH's sparsity means an iso-parameter comparison and an iso-FLOP comparison are genuinely different experiments — both get run and reported (§5).

## 4. Grid design

A pilot grid comes first and exists purely to catch pipeline bugs before spending real compute: a handful of model sizes crossed with a couple of data fractions, one seed, both architectures. If loss curves don't look sane here, nothing downstream is worth running yet.

The full grid: five or six model sizes, log-spaced from roughly half a million to fifty million parameters (comfortably inside both the addition-transformer's demonstrated free-TPU regime and BDH's own tested 10M–1B range), crossed with four or five log-spaced fractions of the training set, two seeds per cell, both architectures. That's on the order of 80 runs. If wall-clock time turns out to be the binding constraint, seeds get cut before grid coverage does — averaging out noise matters less than actually having enough points to fit an exponent against.

## 5. Fitting

Both curves are fit using the same robust procedure the Chinchilla paper used — a Huber loss on log-residuals rather than naive least-squares on raw loss, since raw least-squares over-weights the small-$N$, high-loss corner of the grid. Exponents are reported with bootstrapped confidence intervals, not bare point estimates. Both the iso-parameter and iso-FLOP fits are reported side by side, for the reason given in `ARCHITECTURE.md`'s fairness note.

## 6. The Bayes-error floor

For every recurring pack/pool state in the held-out matched-state subset, compute the empirical spread of what different humans actually picked, and turn that into a single floor number — the cross-entropy of that empirical distribution against itself, which is the loss no model can beat if the underlying human choice is genuinely stochastic at that state.

The interesting result isn't just "the fitted $E$ matches the floor." A mismatch is worth reporting too, and worth being direct about in the writeup rather than glossing over — if $E$ comes in below the measured floor, that's a sign the model (or the split) is leaking something it shouldn't have access to, not a sign the architecture beat human unpredictability.

### 6a. Measured: exact state matching does not work, and the obvious relaxation is a trap

This was checked on the **full FIN corpus — all 5,889,954 picks across 140,237 drafts** — as soon as the ingest pipeline could produce it, rather than being left as a stage-6 discovery. Two findings, both load-bearing:

**Exact `(pack, pool, pack_number, pick_number)` matching yields zero recurring states.** Not "few" — zero, across the entire corpus. The reason is structural rather than a sample-size artifact: the pool is a near-unique fingerprint almost immediately. By pack 0 pick 2, when the pool holds just *two* cards, 380 of 386 pools in a 386-draft sample are already distinct; from pool size 12 onward every pool is unique. Since pool size is fully determined by `pack_number` and `pick_number`, an exact match needs a pool collision, and pools stop colliding almost at once. Going from a 16k-pick sample to the full 5.9M-pick corpus — a 363x increase — moved this number from zero to zero. It is a statement about how fast the state space opens up, not about how many samples were drawn.

**Dropping the pool from the state key — the natural first relaxation — concentrates the recurrence on decisions that are not decisions.** Matching on `(pack, pack_number, pick_number)` alone does find recurrence: 1,186,676 rows, 20.15% of the corpus, in 114,099 groups. But the distribution across the draft is fatal to the measurement:

| pick | pack size | rows in a recurring state | share of that pick |
|-----:|----------:|--------------------------:|-------------------:|
| 0–3  | 14–11     | 0                         | 0.0%               |
| 4–8  | 10–6      | 1,377                     | <0.3%              |
| 9    | 5         | 4,575                     | 1.1%               |
| 10   | 4         | 50,972                    | 12.1%              |
| 11   | 3         | 292,861                   | 69.6%              |
| 12   | 2         | 416,188                   | 98.9%              |
| 13   | 1         | 420,703                   | 100.0%             |

95.2% of all recurrence sits at picks 11–13, where the pack holds three cards or fewer; at pick 13 the pack holds exactly one card and the loss is identically zero. Packs collide there for the trivial reason that a 1-card pack has only 363 possible values — those 420,703 rows fall into just 277 distinct groups. Meanwhile the entire region where a real decision exists, picks 0 through 8, contributes **1,377 rows: 0.12% of the recurrence, and 0.02% of those picks**. Nothing at all before pick 4.

A floor measured over the relaxed subset would therefore be measured almost entirely over forced non-choices and would come back near zero — an apparently clean number that means nothing, and one that would make any fitted $E$ look badly miscalibrated for reasons having nothing to do with the models. Restricting the same relaxation to picks 0–8 to avoid that gives 1,377 rows across ~679 groups, which is far too thin to fit a stable floor and is itself concentrated at picks 7–8.

So the fallback named in §8 is not a contingency any more; it is the primary path, and it needs to be a relaxation that stays in the part of the draft where a real decision exists (roughly picks 0–8, where the pack still holds six or more cards). The relaxation has to be *chosen and justified*, not defaulted into. The candidates worth weighing:

- **Coarsen the pool, keep the pack exact.** Condition on a summary of the pool — its color distribution, curve, creature/spell split — rather than its exact contents. Two drafters with the same pack and a similarly-shaped pool are facing substantially the same decision, which is the thing the floor is supposed to measure.
- **Drop to pairwise preference.** For each card pair $(A, B)$ that co-occurs in a pack, measure how often $A$ is taken over $B$, conditioned on a coarse pool descriptor. Sample sizes become large, but it measures a different quantity than per-state entropy and the writeup would have to say so plainly.
- **Restrict to early picks and accept a smaller subset.** Keeps the exact-match discipline, but is now measured and ruled out on its own: 1,377 rows at picks 0–8, none before pick 4. Not enough to fit anything stable.

Whichever is chosen, the relaxation gets stated in the writeup as a first-class methodological decision with this measurement attached, not as a footnote.

## 7. Deliverables

- `src/data/` — the ingestion and vocabulary pipeline.
- `src/models/` — the shared front-end, the attention arm, the BDH port.
- `src/training/` — the grid runner, the fitting code, the floor measurement.
- A final writeup carrying the same derivational rigor as the addition-transformer exercise: the fitted curves, the exponent comparison, and the floor validation, stated plainly rather than oversold.

## 8. Risks worth naming up front

- **BDH instability.** Addressed above, but worth repeating: this is not a battle-tested architecture, and the porting stage should be budgeted like a research task, not a translation exercise.
- **Iso-parameter vs. iso-compute.** Reporting only one of them makes the headline claim ambiguous. Both get reported, always.
- **Grid compute budget.** 80 runs is the target; cut seeds before cutting grid coverage if time runs short.
- **Thin matched-state subset — now measured on the full corpus, and worse than "thin".** Exact-match recurring states do not merely turn out to be rare; across all 5,889,954 FIN picks there are *none*, and the obvious relaxation puts 95.2% of its recurrence on picks 11–13 where the pack holds three cards or fewer, against 0.12% across every pick where a real decision exists. See §6a for the numbers and the candidate relaxations. This is the open methodological question in the project, and it is worth settling before the grid consumes real compute, because the floor comparison is half of what makes the study more than a curve fit.

## 9. Build order

Data pipeline first, since it unblocks everything else and is comparatively low-risk. Attention arm second — it reuses more familiar ground and gets a real end-to-end loss number fast. A transformer-only pilot grid third, to validate the fitting procedure itself before BDH is even in the picture. BDH port fourth, sequenced once the harness around it already works, since it's the highest-risk stage and benefits most from not also being the stage where the pipeline itself is still being debugged. Full grid fifth. Floor measurement can run in parallel with the grid, since it's pure data analysis with no dependency on trained checkpoints. Fitting, comparison, and writeup last.
