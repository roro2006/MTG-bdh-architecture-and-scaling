# Analysis

Probes that read a trained checkpoint rather than producing one. Nothing here trains, and nothing here sits on the path of a grid run.

- `synergy.py` — does the model use the pool, and if so, for synergy or for colour?

## Why this runs before the grid, not after

The feature table gained 73 mechanical columns (`docs/DATA.md`, "Card features"). A model that reads none of them and a model that reads all of them produce very similar learning curves, because most of a draft pick is explained by raw card quality and by staying in two colours. Loss alone cannot separate "the text features bought synergy" from "the text features bought a wider table" — and a wider table is not free, since the card embedding is linear in feature width while the arms are quadratic in `hidden_dim`.

So the probe is the thing that decides whether the feature rewrite paid for itself, and it is worth having the answer before spending grid compute rather than after.

    python -m src.analysis.synergy --checkpoint runs/attn_d64_s3000 \
        --processed-dir data/processed/FIN.PremierDraft

## The three probes

**`pool_ablation`** is the headline scalar: re-score real picks with the pool replaced by a decoy of the same size, and report how much worse the model gets. A model that ignores the pool scores identically either way, and no synergy analysis downstream will find anything in it.

Two decoys, because they answer different questions:

- `permuted` borrows another row's *real* pool — right colour concentration, right curve, right card quality, only the pairing with this pack destroyed. This isolates "does the model use **this** pool".
- `random` samples card ids uniformly, destroying pool coherence as well as pairing. A model that has only learned "pools are usually two colours" already beats this decoy, so it is the weaker test — kept because a model that cannot beat it has learned nothing about pools at all.

Rows whose pool is empty (pack 0, pick 0) are excluded: they score identically either way by construction, and including them would drag every number toward zero.

A result quoted without saying which decoy was used is ambiguous. Note also that `permuted` can hurt *more* than `random`, which is not a bug: a coherent pool in the wrong colours actively misleads, where a random pool is closer to uninformative.

**`pool_sensitivity`** holds one pack fixed and scores it against several pools, reporting the candidate ranking, its Spearman correlation against a baseline pool, and the largest log-probability swing on any candidate. This is the qualitative picture — which card the model switches to, and by how much.

**`pairwise_synergy`** gives the log-probability lift a candidate gets from a pool seeded with copies of an anchor card. All candidates sit in one pack so the softmax is over the same options in both scorings; the lift is therefore "how much did this candidate gain *relative to this pack*", not an absolute preference. The pool is deliberately exaggerated (six copies of one card), which no real drafter holds — read the sign and the ordering, not the magnitude.

## Telling synergy from colour matching

This is the part that matters. "The pool is red, so take the red card" is real, useful behaviour and it produces a large lift — but it is not what the mechanical columns were added for, and a probe that cannot separate the two would credit the feature rewrite for something the colour columns already did.

`synergy_summary` splits the lift matrix by whether candidate and anchor share a colour, excluding colourless cards from both halves rather than assigning them to one (they match everything trivially, which would move the comparison without saying anything about it). It reports:

- `colour_gap` — `mean_same − mean_cross`, signed. What sharing a colour is worth on its own.
- `within_colour_spread` — the variation *among* colour-matched pairs. A pure colour matcher treats every same-colour anchor alike and scores near zero here, so this is where card-level synergy has to show up if it exists.

Both are signed quantities on purpose. An earlier version reported a ratio of magnitudes, which turned "off-colour candidates are penalised by 0.63 nats" into a number that read like evidence of colour-independent synergy.

### What a short FIN run actually says

A `d=64`, 1,200-step attention-arm model on FIN (val loss 0.929 against a 1.566 pick-rate prior) reads:

| | |
|---|---|
| pool worth, permuted decoy | **+1.27 nats** |
| pool worth, random decoy | +0.86 nats |
| accuracy, real vs. permuted pool | 0.648 → 0.374 |
| top-1 pick changes | 57% of rows |
| mean lift, shares a colour | +0.75 |
| mean lift, shares no colour | **−0.63** |
| `colour_gap` | 1.39 |
| `within_colour_spread` | 0.97 |

The model leans on the pool hard — the first three rows leave no doubt. But the sign in row six is the finding: a candidate sharing no colour with the pool is *penalised*, which is colour matching, not synergy. The `within_colour_spread` of 0.97 says there is card-level structure inside the colour-matched half too, which is where the mechanical columns would be doing work if they are doing any.

That is one short run on one set and settles nothing on its own. It is the number to watch across the grid, and across sets, which is what it was built for.
