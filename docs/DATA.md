# Data

This is a survey of what's actually reachable, checked against the live endpoints rather than trusted from docs alone, plus the reasoning for using each source where it's used.

## 17lands — the scaling grid's primary corpus

17lands publishes per-set draft and game logs at `17lands-public.s3.amazonaws.com/analysis_data/`. Confirmed directly:

- `draft_data/draft_data_public.<SET>.PremierDraft.csv.gz` — one row per pick, with `pack_number`, `pick_number`, `pick`, and a `pack_card_<name>` column for every card in the set showing what was physically in the pack. ~304MB compressed for a single set/format (checked against `OTJ.PremierDraft`).
- `game_data/game_data_public.<SET>.PremierDraft.csv.gz` — one row per game, with `won`, `main_colors`, and `opening_hand_/drawn_/deck_/sideboard_<name>` columns per card.

This is the right backbone for the scaling study specifically because it's simultaneously **closed-vocabulary** (one set's card pool, a few hundred cards) and **high-volume** (drafted through the official client by a large player base), which is what a $D$-sweep across multiple orders of magnitude actually needs. Long-standing open dataset in the MTG analytics community; worth a personal glance at their terms page before publishing anything derived from it, but there's no real ambiguity in practice here.

### The set actually being used: FIN.PremierDraft

`FIN` (Final Fantasy) is the set the scaling grid runs on. Its export is 215,700,130 bytes compressed. Bucket-wide listing returns `AccessDenied`, so files were confirmed by direct `HEAD` — thirteen sets checked, all live, sizes from 141MB (`EOE`) to 304MB (`OTJ`).

Structure was verified by range-fetching and decompressing the first 600KB rather than trusting the format description:

- **740 columns**: 14 metadata + 363 `pack_card_<name>` + 363 `pool_<name>`. The card count *is* the vocabulary size.
- Metadata is `expansion, event_type, draft_id, draft_time, rank, event_match_wins, event_match_losses, pack_number, pick_number, pick, pick_maindeck_rate, pick_sideboard_in_rate, user_n_games_bucket, user_game_win_rate_bucket`.
- `pick` is a card **name**, not an id.
- `pack_card_*` values are **counts, not flags**. FIN's sample is all 0/1, but `OTJ` contains packs with a card at count 2 — a boolean parse would silently drop them.
- Packs hold **14** selectable cards, not 15, and 42 picks make a draft (3 x 14).
- Card names contain commas (`pack_card_Zidane, Tantalus Thief`), so the header needs a real CSV parser; a `split(",")` shreds 41 of OTJ's 381 columns.

Extrapolating the observed 42.5x compression ratio put the full file at roughly 9.2GB uncompressed, ~5.8M picks across ~139k drafts. The actual ingest came in at **5,889,954 picks across 140,237 drafts**, so that estimate held. The raw file does not fit in memory as a dataframe, which is what `src/data/ingest.py` exists to deal with.

Ingested, the corpus is **253MB in RAM (43.0 bytes/row)** and 73.7MB on disk as compressed `.npz`. The full pass takes 271s at ~21.7k rows/s, gzip decompression included. Unlike `OTJ`, FIN contains **no incomplete drafts** — all 140,237 have their full 42 picks, and none were dropped.

### The pool columns are redundant, and dropping them is what makes this fit

The 363 `pool_*` columns carry no information the pick history doesn't already have: the pool at any pick is exactly the set of that draft's earlier picks. Verified on the sample — `|pool| == pack_number * 14 + pick_number` holds on every row checked, and pools reconstructed from the pick prefix match the raw `pool_*` columns exactly on 400 randomly drawn rows.

So ingest reads neither, and stores packs as padded card-id lists rather than length-V count vectors. Measured result: **46 bytes per row**, or about **267MB in RAM for the full corpus**, against roughly 1GB if pools were materialized and ~25x more again for dense count vectors. Ingest runs at ~51k rows/s, so a full pass is a few minutes.

### Fetching it

    curl -o data/raw/draft_data_public.FIN.PremierDraft.csv.gz       https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/draft_data_public.FIN.PremierDraft.csv.gz

    python -m src.data.ingest       --csv data/raw/draft_data_public.FIN.PremierDraft.csv.gz       --out data/processed/FIN.PremierDraft

The gzip is never decompressed to disk; ingest streams it.

## Scryfall — card metadata

Official bulk-data API (`api.scryfall.com/bulk-data`), live. Oracle Cards export is ~24MB gzipped, one JSON object per unique card: oracle text, mana cost, types, keywords. Used only to build the composite card embeddings (see `ARCHITECTURE.md`) and the naive text-overlap baseline — never as a training label. Standard, uncontroversial, no access concerns.

## CubeCobra — held for the synergy phase, not the scaling grid

CubeCobra exports cube lists, draft picks, and completed decks to an open, unsigned S3 bucket (`s3://cubecobra-public/export/`), updated quarterly. Verified directly: `indexToOracleMap.json`, `simpleCardDict.json`, `cubes.json`, and the batched `picks/{n}.json` / `cubeInstances/{n}.json` / `decks/{n}.json` files all return live.

Pulled and inspected an actual shard (`picks/0.json`): ~1,550 pick records, shaped as `{cube, pack: [card indices], picked, pool: [...]}`. Shard indices run to roughly 4,900–5,000 before 404ing, putting total volume around 7–8 million picks — but pooled across every cube on the platform, not one fixed card pool. `indexToOracleMap.json` is 1.79MB, which for index→oracle-ID entries implies a global vocabulary in the tens of thousands of cards, since different cubes draw from most of Magic's card pool between them.

That's exactly why this isn't the scaling grid's corpus: restricting to one cube gets the vocabulary back down to a few hundred cards, but at that point the per-cube draft volume is nowhere near what a single 17lands set sees, and the $D$-axis needs range, not just a clean vocabulary. Where CubeCobra earns its place is the follow-on synergy work: cube curators hand-pick cards specifically for how they function together, which makes co-occurrence-within-cubes a cleaner synergy signal than aggregate constructed-deck co-occurrence would be, and volume matters much less for that question than it does for fitting a data-scaling exponent.

## EDHREC — left out

EDHREC's synergy scores were the obvious first candidate for the synergy work — they're computed as how far above baseline two cards co-occur across decks, which is exactly a co-occurrence signal that isn't derivable from card text alone. Their JSON endpoints (`json.edhrec.com/pages/...`) do respond, and robots.txt doesn't block them, but their terms of service explicitly forbid "automated searches, requests, or queries to the Site" and restrict use to personal, noncommercial use with no redistribution. Being reachable isn't the same as being licensed for this, so it's left out. CubeCobra's own co-occurrence data covers the same need without that problem.
