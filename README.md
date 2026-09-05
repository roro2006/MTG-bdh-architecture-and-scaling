# MTG Drafter: a draft bot built from the kernel up

**Author:** Rohan Reddy
**Status:** Both arms trained to convergence on a T4, where they tie; ten sets ingested; set-independent feature table, synergy probe, scaling fit, grid runner, inference entry point and every forward-pass kernel written. Cross-set generalisation and the sizing grid are the open work.

## What this is

A Magic: The Gathering draft bot that reads what cards *do* and picks accordingly — including for a set it has never seen before.

Given the pack in front of you and the cards you've already taken, it scores every card physically in the pack and ranks them. What makes that more than a lookup table is where the score comes from: each card is represented by its mechanical behaviour derived from its rules text, and the model's interaction block asks how a candidate card's behaviour fits the behaviour of the cards already in the pool. That is a synergy computation, and it is the thing a draft bot has to get right to be worth using.

Because no part of the representation is keyed to a specific card or a specific set, a set released after training gets a usable representation the moment its cards exist. Drafting an unseen set is the target the whole design is pointed at.

## The other half: it is built from first principles

Every layer of this was written rather than imported, and that is as much the point of the project as the bot is.

- The transformer front end — permutation-invariant set encoders, cross-attention, pointer output head — is written from scratch in JAX/Flax.
- The attention-shaped and neuron-space operations are **hand-written Pallas kernels**, not library calls. As of the set-encoder kernel, that covers every attention-shaped operation in the forward pass rather than only the interaction arm; LayerNorm, Dense and elementwise ops stay in XLA, which already fuses them well.
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

**Cards become behaviour, not identity.** Each card's vector is built from set-independent attributes — colour, mana value, type, rarity, power/toughness — plus structured mechanical features pattern-matched out of its Scryfall oracle text: creates tokens, sacrifice outlet, cares about creatures dying, +1/+1 counters, graveyard recursion, artifacts matter, removal, and so on. 119 columns in all, 15 global keywords and 73 mechanics among them. Every column means the same thing in every set. Nothing is fitted per-set, and the card's own name is stripped from its text before processing so identity cannot leak back in. There is no per-card embedding table anywhere in the model — `CardEmbedding` is an MLP over that feature block — which is what makes scoring an unseen set possible without adding a parameter.

**Sets are encoded without order.** Pack and pool each go through a permutation-invariant encoder — attention blocks with no positional encoding.

**Synergy lives in the interaction block.** Synergy is relational, so it cannot live in a per-card vector. Pack cards are queries, the pool is context, and the learned bilinear form between them is what connects "this candidate is a sacrifice outlet" to "my pool makes tokens." The feature table's job is not to encode synergy but to make it *representable*; the arm's job is to learn it.

**The output is closed to the pack.** One score per pack slot, softmax over just those.

## Where things stand

The data pipeline (`src/data/`) is implemented and tested: vocabulary construction, a streaming ingest that turns the raw exports into a compact on-disk form, pool reconstruction, draft-level splitting, and matched-state grouping. Ten sets are ingested, four of them not on Arena's 3×14 shape. Both interaction arms are implemented, both have hand-written Pallas kernels, parameter counts are verified analytically at five widths, and both have been trained to convergence at one width on a GPU.

Both arms, `--width 64 --steps 92000 --batch-size 512 --learning-rate 3e-4 --seed 0`, on FIN.PremierDraft — 47.1M examples, ten passes over the 4,711,938 training rows, on a Colab T4:

| | BDH | attention |
|---|---|---|
| parameters | 261,633 | 263,745 |
| val loss, picks 0–8 | 1.0037 | 1.0033 |
| val accuracy, picks 0–8 | 0.6248 | 0.6239 |
| val loss, all picks | 0.8214 | 0.8212 |
| best step | 88,250 | 88,250 |
| throughput | 17,479 ex/s | 19,709 ex/s |

Baselines on the same split, picks 0–8: uniform 2.2671, pick-rate prior 1.9474 at 36.30% accuracy.

**The arms tie, and the tie is the result.** 0.0004 apart on the headline slice at 0.8% different parameter counts, peaking at the same step — two runs landing in the same place, well inside what a seed change moves. The pilot that once showed BDH ahead was measured at a third of an epoch, where attention had already peaked and BDH had not; that margin went to zero on the way to convergence and should not be reported as an ordering in either direction. `docs/RESULTS.md` carries the retraction and the numbers behind it.

**The models are real drafters.** 1.0037 against a 1.9474 pick-rate prior is 0.94 nats below a baseline that already knows how good every card is in the abstract, and the synergy probes confirm the margin comes from conditioning on the pool — roughly half the pool effect is genuinely pairwise rather than colour-matching.

"Decision picks" means picks 0–8, where the pack still holds six or more cards. It is the honest number: 7.1% of all rows are one-card packs whose loss is identically zero, and the all-picks aggregate flatters every model equally.

Three things are open, in the order they are being worked:

1. **Cross-set generalisation is not measured yet.** Ten sets are ingested and the feature table is set-independent by construction, which is the precondition — but no run has yet trained on $n-1$ sets and evaluated on the held-out one, which is the actual claim. It needs a multi-set loader; `run.py` takes one `--processed-dir` today.
2. **The sizing grid has not been run.** The grid runner and the curve fit are both implemented and tested, so this is compute rather than code — a neuron probe first, since a grid committed at `neuron_multiplier=4` cannot answer the question §4 reopened, then the grid, then the fit that chooses $(N^*, D^*)$.
3. **The kernels are correct but unbenchmarked.** Every kernel is asserted against a pure-JAX reference on values and on every gradient, but off GPU/TPU the tests run under Pallas's `interpret=True`, which checks semantics and does no fusion. None has been exercised on real hardware yet, and the converged runs above were trained unfused. Re-running the kernel tests with `KERNEL_INTERPRET=0` and benchmarking fused against reference is what closes it.

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
  inference/         — pack + pool in, ranked picks out; the CLI and its metrics
  validation.py      — post-ingest gate to run against a newly added set
scripts/             — Colab accelerator scaffolding (remote bootstrap + WSL driver)
tests/               — run against a synthetic export; no download needed
notebooks/           — exploratory analysis, not part of the pipeline
```

## Ranking a pack

    python -m src.inference.drafter \
        --checkpoint runs/bdh_d64_s92000 \
        --processed-dir data/processed/FIN.PremierDraft \
        --pack "Vivi Ornitier" "Tifa, Martial Artist" --pool "Cloud, Midgar Mercenary"

It takes card *names*, not ids, and refuses the ones it does not recognise — every id in range is a legal card, so an id-taking interface hands a caller who used the wrong `vocab.json` a confident ranking of a pack they never asked about. `src/inference/README.md` covers the other guards and the metrics.

## Running on an accelerator

The grid does not exist as a CPU workload. To train a cell on a Colab T4 or
TPU, authenticate once (`colab sessions`, paste the code it prints) and then:

    ./scripts/colab_run.sh --gpu T4 --arm bdh --width 64 --steps 3000

Use `--epochs` rather than `--steps` for grid cells: at fixed steps a small
`--data-fraction` silently means many passes over a small set, and the fitted
$\beta$ would then be measuring data repetition rather than data scale.
`scripts/README.md` covers the segment loop, data staging and the traps worth
knowing about before the first run.

## Data

Draft picks from [17lands](https://www.17lands.com/public_datasets) — one row per pick, with the full pack contents, across many sets. Card metadata from [Scryfall](https://scryfall.com/docs/api/bulk-data), which supplies the oracle text the mechanical features are derived from. Cube lists from [CubeCobra](https://github.com/dekkerglen/CubeCobraML) as an external synergy signal for checking that the model learned card interaction rather than colour-matching. Full rationale, verified sizes, and the one source deliberately left out are in `docs/DATA.md`.
