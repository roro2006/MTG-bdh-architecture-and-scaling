# MTG Drafter: a draft bot built from the kernel up

**Author:** Rohan Reddy
**Status:** Data pipeline and both model arms implemented; nine sets ingested; set-independent feature table landed; pilot runs done; accelerator scaffolding written but not yet exercised on real hardware

## What this is

A Magic: The Gathering draft bot that reads what cards *do* and picks accordingly — including for a set it has never seen before.

Given the pack in front of you and the cards you've already taken, it scores every card physically in the pack and ranks them. What makes that more than a lookup table is where the score comes from: each card is represented by its mechanical behaviour derived from its rules text, and the model's interaction block asks how a candidate card's behaviour fits the behaviour of the cards already in the pool. That is a synergy computation, and it is the thing a draft bot has to get right to be worth using.

Because no part of the representation is keyed to a specific card or a specific set, a set released after training gets a usable representation the moment its cards exist. Drafting an unseen set is the target the whole design is pointed at.

## The other half: it is built from first principles

Every layer of this was written rather than imported, and that is as much the point of the project as the bot is.

- The transformer front end — permutation-invariant set encoders, cross-attention, pointer output head — is written from scratch in JAX/Flax.
- The attention-shaped and neuron-space operations are **hand-written Pallas kernels**, not library calls. Where the memory-hierarchy decision belongs to the implementer, the implementation is ours; LayerNorm, Dense and elementwise ops stay in XLA, which already fuses them well.
- Parameter counts and FLOP counts are **derived term by term** and asserted against the realised model, rather than read off whatever `count_params()` returns.
- The model's size is not guessed. A Chinchilla-style scaling law $L(N, D) = E + A/N^\alpha + B/D^\beta$ is fit on a pilot grid, and the compute-optimal $(N^*, D^*)$ it returns is the configuration the shipped drafter is trained at.

A second architecture rides along: **BDH** ("Dragon Hatchling"), a sparse, Hebbian block proposed in 2025, ported to JAX from scratch since no JAX implementation existed. It is a candidate interaction mechanism, judged the same way the cross-attention block is — by how well it drafts. The comparison is a means of choosing what to ship, not a standalone claim about scaling exponents.

## Why draft-pick prediction

The task has a shape that rewards this treatment:

- **The output space is bounded and real.** A pick is one of at most fourteen cards physically in the pack. The pointer head takes a softmax over exactly those, so the model is structurally incapable of naming a card that isn't there.
- **Order carries no information.** A pack is a set; a pool is a set. The architecture is built to be *incapable* of using order rather than trained to ignore it.
- **The labels are real human decisions**, at volume — 5.9 million of them for a single set.
- **The whole draft history collapses into the pool.** There is no long sequence to attend over; the sufficient statistic is handed to the model at every step.

## How it works

**Cards become behaviour, not identity.** Each card's vector is built from set-independent attributes — colour, mana value, type, rarity, power/toughness — plus structured mechanical features pattern-matched out of its Scryfall oracle text: creates tokens, sacrifice outlet, cares about creatures dying, +1/+1 counters, graveyard recursion, artifacts matter, removal, and so on. Every column means the same thing in every set. Nothing is fitted per-set, and the card's own name is stripped from its text before processing so identity cannot leak back in.

**Sets are encoded without order.** Pack and pool each go through a permutation-invariant encoder — attention blocks with no positional encoding.

**Synergy lives in the interaction block.** Synergy is relational, so it cannot live in a per-card vector. Pack cards are queries, the pool is context, and the learned bilinear form between them is what connects "this candidate is a sacrifice outlet" to "my pool makes tokens." The feature table's job is not to encode synergy but to make it *representable*; the arm's job is to learn it.

**The output is closed to the pack.** One score per pack slot, softmax over just those.

## Where things stand

The data pipeline (`src/data/`) is implemented and tested: vocabulary construction, a streaming ingest that turns the ~9GB raw export into 74MB on disk (253MB in RAM), pool reconstruction, draft-level splitting, and matched-state grouping. Both interaction arms are implemented, both have hand-written Pallas kernels, parameter counts are verified analytically at five widths, and both are now trained end to end at one size. The grid runner, the curve fit, and the floor measurement are still skeletons. Implementation is being built out in stages, each landing as its own commit rather than one dump.

Two pilot runs exist, at $d=64$ and roughly one third of an epoch on FIN:

| | parameters | best val loss | picks 0–8 | decision-pick accuracy | wall clock |
|---|---:|---|---|---|---:|
| cross-attention arm | 260,289 | 0.9402 | 1.1473 | 57.6% | 2,745s |
| BDH arm | 258,177 | **0.9191** | **1.1150** | 58.9% | 3,363s |
| pick-rate prior baseline | — | 1.5662 | — | 45.3% | — |
| uniform over pack | — | 1.7994 | — | — | — |

BDH is ahead per parameter and behind per second, which is the iso-parameter/iso-FLOP ambiguity `docs/ARCHITECTURE.md` exists to keep honest — at one width, one seed, and an untuned shared learning rate, it settles neither. See `docs/RESULTS.md`.

"Decision picks" means picks 0–8, where the pack still holds six or more cards. It is the honest number: 7.1% of all rows are one-card packs whose loss is identically zero, and the all-picks aggregate flatters every model equally.

Three things are known blockers, being worked in order:

1. **Training runs on CPU at 561 examples/second.** One epoch at $d=64$ takes 2.3 hours and the grid is infeasible. `scripts/` now provisions a Colab accelerator and drives a run in resumable segments, but nothing in it has been exercised on real hardware yet, so the throughput it buys is still an estimate.
2. **The grid runner, the curve fit and the floor measurement are still skeletons.** The sizing procedure the project is built around cannot run until they are real.
3. **Cross-set generalisation is not measured yet.** Nine sets are ingested and the feature table is set-independent by construction, which is the precondition — but no run has yet trained on one set and evaluated on another, which is the actual claim.

## Layout

```
docs/
  PROJECT_PLAN.md    — stages, scope, representation design, evaluation, build order
  ARCHITECTURE.md    — the card/pack/pool design, where synergy lives, kernel scope
  DATA.md            — sources, what was verified, and why each is used
  RESULTS.md         — measured numbers, and what they do and do not license
  COLAB.md           — how the grid is made to fit a free Colab session
src/
  data/              — ingestion, vocabulary, feature table
  models/            — front end, both interaction arms, hand-written kernels
  training/          — single-cell runner, grid, scaling fit, floor measurement
  analysis/          — synergy probes and evaluation tooling
  validation.py      — post-ingest gate to run against a newly added set
scripts/             — Colab accelerator scaffolding (remote bootstrap + WSL driver)
tests/               — run against a synthetic export; no download needed
notebooks/           — exploratory analysis, not part of the pipeline
```

## Running on an accelerator

The grid does not exist as a CPU workload. To train a cell on a Colab T4 or
TPU, authenticate once (`colab sessions`, paste the code it prints) and then:

    ./scripts/colab_run.sh --gpu T4 --arm bdh --width 64 --steps 3000

`scripts/README.md` covers the segment loop, data staging and the traps worth
knowing about before the first run.

## Data

Draft picks from [17lands](https://www.17lands.com/public_datasets) — one row per pick, with the full pack contents, across many sets. Card metadata from [Scryfall](https://scryfall.com/docs/api/bulk-data), which supplies the oracle text the mechanical features are derived from. Cube lists from [CubeCobra](https://github.com/dekkerglen/CubeCobraML) as an external synergy signal for checking that the model learned card interaction rather than colour-matching. Full rationale, verified sizes, and the one source deliberately left out are in `docs/DATA.md`.
