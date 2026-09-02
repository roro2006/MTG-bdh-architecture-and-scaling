# Data

A survey of what is actually reachable, checked against the live endpoints rather than trusted from docs alone, plus the reasoning for using each source where it is used.

## 17lands — the training corpus

17lands publishes per-set draft and game logs at `17lands-public.s3.amazonaws.com/analysis_data/`. Confirmed directly:

- `draft_data/draft_data_public.<SET>.PremierDraft.csv.gz` — one row per pick, with `pack_number`, `pick_number`, `pick`, and a `pack_card_<name>` column for every card in the set showing what was physically in the pack. ~304MB compressed for a single set/format (checked against `OTJ.PremierDraft`).
- `game_data/game_data_public.<SET>.PremierDraft.csv.gz` — one row per game, with `won`, `main_colors`, and `opening_hand_/drawn_/deck_/sideboard_<name>` columns per card. Not used in this version; win-rate prediction is out of scope.

This is the right backbone because it is simultaneously **high-volume** and **honest about the decision context** — every row records what the drafter could actually have taken, not just what they took. The label is a real human choice made under real pressure.

### Several sets, not one

The bot is meant to draft a set it has never seen, so several sets are ingested and at least one is held out entirely for evaluation. Sets are chosen for mechanical variety rather than recency: a held-out set that plays like the training sets tests nothing.

Thirteen sets were confirmed live by direct `HEAD` (bucket-wide listing returns `AccessDenied`), sizes from 141MB (`EOE`) to 304MB (`OTJ`).

### What was verified about the format

Structure was checked by range-fetching and decompressing the first 600KB rather than trusting the format description:

- **740 columns** for FIN: 14 metadata + 363 `pack_card_<name>` + 363 `pool_<name>`. The card count *is* the set's vocabulary size.
- Metadata is `expansion, event_type, draft_id, draft_time, rank, event_match_wins, event_match_losses, pack_number, pick_number, pick, pick_maindeck_rate, pick_sideboard_in_rate, user_n_games_bucket, user_game_win_rate_bucket`.
- `pick` is a card **name**, not an id.
- `pack_card_*` values are **counts, not flags**. FIN's sample is all 0/1, but `OTJ` contains packs with a card at count 2 — a boolean parse would silently drop them.
- Card names contain commas (`pack_card_Zidane, Tantalus Thief`), so the header needs a real CSV parser; a `split(",")` shreds 41 of OTJ's 381 columns.

### Pack geometry varies, and getting it wrong is silent

FIN packs hold **14** selectable cards, and 42 picks make a draft (3 × 14). That is not universal, and the failure mode for a set with different geometry is quiet rather than loud: `PickData` validates $|\text{pool}| = \text{pack\_number} \times \text{picks\_per\_pack} + \text{pick\_number}$ on load, so a wrong constant makes every draft in the set invalid and `on_invalid="drop"` discards the whole corpus without raising. Geometry is therefore detected during ingest and persisted in `ingest_stats.json` rather than hardcoded.

### The pool columns are redundant, and dropping them is what makes this fit

The `pool_*` columns carry no information the pick history does not already have: the pool at any pick is exactly the set of that draft's earlier picks. Verified on the sample — $|\text{pool}| = \text{pack\_number} \times 14 + \text{pick\_number}$ holds on every row checked, and pools reconstructed from the pick prefix match the raw `pool_*` columns exactly on 400 randomly drawn rows.

So ingest reads neither, and stores packs as padded card-id lists rather than length-V count vectors. Measured result: **46 bytes per row**, against roughly 1GB if pools were materialised and ~25× more again for dense count vectors.

### FIN.PremierDraft, as ingested

The first set through the pipeline. 363 cards, **5,889,954 picks across 140,237 drafts**, **253MB in RAM (43.0 bytes/row)** and 73.7MB on disk as compressed `.npz`. The full pass takes 271s at ~21.7k rows/s, gzip decompression included. Unlike `OTJ`, FIN contains **no incomplete drafts** — all 140,237 have their full 42 picks, and none were dropped.

### Fetching it

    curl -o data/raw/draft_data_public.FIN.PremierDraft.csv.gz \
      https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/draft_data_public.FIN.PremierDraft.csv.gz

    python -m src.data.ingest \
      --csv data/raw/draft_data_public.FIN.PremierDraft.csv.gz \
      --out data/processed/FIN.PremierDraft

The gzip is never decompressed to disk; ingest streams it.

## Scryfall — where card behaviour comes from

Official bulk-data API (`api.scryfall.com/bulk-data`), live. Only the cards in each set's vocabulary are fetched, via `/cards/collection` at 75 identifiers per POST, rather than the ~150MB oracle-cards bulk export. Scryfall asks for a descriptive User-Agent and 50–100ms between requests; both are honoured.

**This is the most important source in the project after the picks themselves.** It supplies `oracle_text`, which is what the structured mechanical features are derived from and therefore what makes drafting an unseen set possible at all. A card the model has never encountered is representable because its rules text says what it does.

Two processing requirements, both easy to miss and both consequential:

- **The card's own name is stripped from its oracle text** before anything reads it. Scryfall spells the name out in full ("Whenever Zidane, Tantalus Thief attacks…"), and leaving it in reintroduces card identity as a feature — precisely what the set-independent representation exists to avoid.
- **Parenthetical reminder text is stripped.** It is redundant with the keyword flags and inflates apparent similarity between unrelated cards that happen to share a mechanic.

Never used as a training label.

## CubeCobra — external synergy validation

CubeCobra exports cube lists, draft picks and completed decks to an open, unsigned S3 bucket (`s3://cubecobra-public/export/`), updated quarterly. Verified directly: `indexToOracleMap.json`, `simpleCardDict.json`, `cubes.json`, and the batched `picks/{n}.json` / `cubeInstances/{n}.json` / `decks/{n}.json` files all return live. Pulled and inspected an actual shard (`picks/0.json`): ~1,550 pick records, shaped as `{cube, pack: [card indices], picked, pool: [...]}`. Shard indices run to roughly 4,900–5,000 before 404ing, putting total volume around 7–8 million picks — pooled across every cube on the platform, not one fixed card pool.

**Its role here is validation, not training.** Cube curators hand-pick cards specifically for how they function together, which makes co-occurrence-within-cubes a synergy signal that is *not derivable from 17lands pick data*. That independence is the point: if the model's learned card-to-card affinities correlate with cube co-occurrence, that is evidence it learned interaction rather than colour-matching — evidence from outside its own training distribution.

It is a poor fit as a training corpus for the same reason it is a good validation signal: the vocabulary is pooled across cubes and runs to tens of thousands of cards, and per-cube volume is nowhere near a single 17lands set.

## EDHREC — left out

EDHREC's synergy scores were the obvious first candidate for the validation signal — computed as how far above baseline two cards co-occur across decks, which is exactly the kind of measurement wanted. Their JSON endpoints (`json.edhrec.com/pages/...`) do respond, and robots.txt does not block them, but their terms of service explicitly forbid "automated searches, requests, or queries to the Site" and restrict use to personal, noncommercial use with no redistribution. Being reachable is not the same as being licensed, so it is left out. CubeCobra's own co-occurrence data covers the same need without that problem.
