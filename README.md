# MTG-BDH: Scaling Laws for Sparse Architectures on a Closed-Vocabulary Decision Task

**Author:** Rohan Reddy
**Status:** Planning / pre-implementation

## What this project is

Most published scaling laws — Kaplan et al., Chinchilla, the various Mixture-of-Experts follow-ups — are fit on open-vocabulary language modeling loss, and every result for the recently proposed BDH ("Dragon Hatchling") architecture lives in that same regime: language and translation benchmarks in the 10M–1B parameter range. Nobody has yet asked whether the scaling exponents a Hebbian, sparsely-activated architecture produces look the same once the task is no longer open-ended text generation.

This project uses Magic: The Gathering draft-pick prediction as that alternate regime. It's a genuinely different shape of problem: a closed vocabulary of a few hundred cards, an output space constrained to whatever's physically in the pack in front of you, and a label that comes from a real human decision rather than a scraped corpus. The plan is to fit $L(N, D) = E + A/N^\alpha + B/D^\beta$ independently for a causal-style architecture and for a from-scratch JAX port of BDH, on identical data and a shared input representation, and see whether $\alpha$ and $\beta$ hold, shift, or fall apart outside language.

There's a second piece worth stating plainly, because it's what separates this from "fit a curve and report it": the constant $E$ (irreducible loss) in almost every published scaling law is a free parameter that nobody checks against anything external. Because 17lands' draft data has many different players facing near-identical pack/pool states, it's possible to independently measure how often two reasonable humans disagree in the same spot — a real Bayes-error floor — and ask whether the fitted $E$ actually converges toward it. If it doesn't, that's not a bug in the writeup; it's a finding.

## Where things stand

The data pipeline (`src/data/`) is implemented and tested: vocabulary construction, a streaming ingest that turns the ~9GB raw export into ~267MB of arrays, pool reconstruction, draft-level splitting, and matched-state grouping. Both model arms are ported, instrumented, and now trained end to end at one size. The grid runner, the curve fit, and the floor measurement are still skeletons. Implementation is being built out in stages, each landing as its own commit rather than one dump.

**First real runs, both arms at `d=64`, 3,000 steps, one seed, LR 3e-4** (see `docs/RESULTS.md` for the numbers and what they do and do not license):

| | attention | BDH |
|---|---|---|
| parameters | 260,289 | 258,177 |
| best val loss | 0.9402 | **0.9191** |
| picks 0–8 only | 1.1473 | **1.1150** |
| wall clock | 2,745s | 3,363s |

Against baselines of 1.7994 (uniform) and 1.5662 (pick-rate prior), so both arms are learning pool-conditional structure rather than card quality alone. BDH is ahead per parameter and behind per second, which is the iso-parameter/iso-FLOP ambiguity `docs/ARCHITECTURE.md` exists to keep honest — at one width, one seed, and an untuned shared learning rate, it settles neither.

The set under study is **FIN.PremierDraft**: 363 cards, 5,889,954 picks across 140,237 drafts, ingested and sitting at 253MB in RAM.

One finding from the data stage is already worth flagging, because it changes the plan rather than just filling it in: the human-disagreement floor described above cannot be measured the way §6 originally specified. Exact recurring `(pack, pool)` states do not exist in this data — not rarely, but *zero across all 5.9M picks* — because a pool of even two cards is already a near-unique fingerprint. The obvious relaxation is worse than it looks: dropping the pool from the state key puts 95% of the resulting recurrence on the last three picks of a pack, where the pack holds three cards or fewer and the "decision" is often a single forced card, while every pick where a real decision exists contributes 0.12% of it. See `docs/PROJECT_PLAN.md` §6a for the measurements and the candidate fixes.

## Layout

```
docs/
  PROJECT_PLAN.md    — stages, scope, grid design, fitting procedure, risks
  ARCHITECTURE.md     — the card/pack/pool encoder design and how BDH plugs into it
  DATA.md             — what data sources are used, what was verified, and why
  RESULTS.md          — measured numbers, and what they do and do not license
src/
  data/               — dataset pipeline (17lands ingestion, vocab construction)
  models/             — the shared front-end, the attention arm, the BDH arm
  training/           — grid runner, curve fitting, floor measurement
tests/                — pipeline tests, run against a synthetic export (no download needed)
notebooks/            — exploratory analysis, not part of the pipeline proper
```

## Data

Draft-pick data from [17lands](https://www.17lands.com/public_datasets) (primary — closed per-set vocabulary, large volume, official public research export), card metadata from [Scryfall](https://scryfall.com/docs/api/bulk-data), and, for a later synergy-focused phase, cube draft/deck data from [CubeCobra's public export](https://github.com/dekkerglen/CubeCobraML). Full rationale, verified file sizes, and the one source that was deliberately left out (EDHREC — technically scrapeable, but its terms of service prohibit automated queries) are in `docs/DATA.md`.
