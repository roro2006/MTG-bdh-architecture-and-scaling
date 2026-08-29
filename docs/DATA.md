# Data

This is a survey of what's actually reachable, checked against the live endpoints rather than trusted from docs alone, plus the reasoning for using each source where it's used.

## 17lands — the scaling grid's primary corpus

17lands publishes per-set draft and game logs at `17lands-public.s3.amazonaws.com/analysis_data/`. Confirmed directly:

- `draft_data/draft_data_public.<SET>.PremierDraft.csv.gz` — one row per pick, with `pack_number`, `pick_number`, `pick`, and a `pack_card_<name>` column for every card in the set showing what was physically in the pack. ~304MB compressed for a single set/format (checked against `OTJ.PremierDraft`).
- `game_data/game_data_public.<SET>.PremierDraft.csv.gz` — one row per game, with `won`, `main_colors`, and `opening_hand_/drawn_/deck_/sideboard_<name>` columns per card.

This is the right backbone for the scaling study specifically because it's simultaneously **closed-vocabulary** (one set's card pool, a few hundred cards) and **high-volume** (drafted through the official client by a large player base), which is what a $D$-sweep across multiple orders of magnitude actually needs. Long-standing open dataset in the MTG analytics community; worth a personal glance at their terms page before publishing anything derived from it, but there's no real ambiguity in practice here.

## Scryfall — card metadata

Official bulk-data API (`api.scryfall.com/bulk-data`), live. Oracle Cards export is ~24MB gzipped, one JSON object per unique card: oracle text, mana cost, types, keywords. Used only to build the composite card embeddings (see `ARCHITECTURE.md`) and the naive text-overlap baseline — never as a training label. Standard, uncontroversial, no access concerns.

## CubeCobra — held for the synergy phase, not the scaling grid

CubeCobra exports cube lists, draft picks, and completed decks to an open, unsigned S3 bucket (`s3://cubecobra-public/export/`), updated quarterly. Verified directly: `indexToOracleMap.json`, `simpleCardDict.json`, `cubes.json`, and the batched `picks/{n}.json` / `cubeInstances/{n}.json` / `decks/{n}.json` files all return live.

Pulled and inspected an actual shard (`picks/0.json`): ~1,550 pick records, shaped as `{cube, pack: [card indices], picked, pool: [...]}`. Shard indices run to roughly 4,900–5,000 before 404ing, putting total volume around 7–8 million picks — but pooled across every cube on the platform, not one fixed card pool. `indexToOracleMap.json` is 1.79MB, which for index→oracle-ID entries implies a global vocabulary in the tens of thousands of cards, since different cubes draw from most of Magic's card pool between them.

That's exactly why this isn't the scaling grid's corpus: restricting to one cube gets the vocabulary back down to a few hundred cards, but at that point the per-cube draft volume is nowhere near what a single 17lands set sees, and the $D$-axis needs range, not just a clean vocabulary. Where CubeCobra earns its place is the follow-on synergy work: cube curators hand-pick cards specifically for how they function together, which makes co-occurrence-within-cubes a cleaner synergy signal than aggregate constructed-deck co-occurrence would be, and volume matters much less for that question than it does for fitting a data-scaling exponent.

## EDHREC — left out

EDHREC's synergy scores were the obvious first candidate for the synergy work — they're computed as how far above baseline two cards co-occur across decks, which is exactly a co-occurrence signal that isn't derivable from card text alone. Their JSON endpoints (`json.edhrec.com/pages/...`) do respond, and robots.txt doesn't block them, but their terms of service explicitly forbid "automated searches, requests, or queries to the Site" and restrict use to personal, noncommercial use with no redistribution. Being reachable isn't the same as being licensed for this, so it's left out. CubeCobra's own co-occurrence data covers the same need without that problem.
