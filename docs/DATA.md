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

## Beyond one set: the multi-set corpus

One set cannot show whether a drafter generalises, so nine more were pulled — eight of which turned out to be usable. The point was never the extra volume; it was to find out what a second set does that the first one did not, and the answer turned out to be "quite a lot".

| set | cards | picks | drafts | geometry | pack width | kept after load | dropped drafts |
|---|---:|---:|---:|---|---:|---:|---:|
| `AFR` | 266 | 1,066,415 | 26,177 | 3 x 14 | 14 | **0 — excluded** | 0 |
| `BLB` | 276 | 6,070,278 | 156,706 | 3 x 13 | 13 | 6,028,815 | 2,121 |
| `DSK` | 286 | 7,082,745 | 169,666 | 3 x 14 | 14 | 7,042,224 | 1,994 |
| `EOE` | 321 | 4,070,699 | 104,377 | 3 x 13 | 13 | 4,070,547 | 4 |
| `FIN` | 363 | 5,889,954 | 140,237 | 3 x 14 | 14 | 5,889,954 | 0 |
| `HBG` | 271 | 1,513,835 | 36,330 | 3 x 14 | 14 | 1,497,552 | 674 |
| `LCI` | 291 | 6,121,985 | 137,070 | 3 x 15 | 15 | 6,073,425 | 2,105 |
| `MH3` | 326 | 5,370,272 | 128,647 | 3 x 14 | 14 | 5,340,300 | 1,497 |
| `OTJ` | 381 | 8,249,043 | 197,554 | 3 x 14 | 14 | 8,203,104 | 2,242 |
| `SIR` | 357 | 2,174,412 | 48,740 | 3 x 15 | 15 | 2,154,915 | 853 |

**Nine usable sets, 46,300,836 picks, 2,872 card slots — of which 2,817 are distinct cards.** Only 55 slots are repeats, so the sets share almost nothing: a model that transfers between them has to be reading card attributes, not remembering cards. That is exactly the condition the set-independent feature table was rewritten for, and it is why one set could never have tested it.

Every set validates clean under `python -m src.validation`. `AFR` is excluded for the reason given below; its row is kept in the table so the exclusion is on the record rather than invisible.

### What ingesting a second set actually cost

Everything below was found by trying to ingest these files, not by reading a spec. Each one stopped a set dead, and each is now covered by a test in `tests/test_export_formats.py` or `tests/test_geometry.py`.

**Three different pack geometries turned up across ten sets, and four of the ten are not 14.** `SIR` and `LCI` draft **15** cards per pack; `BLB` and `EOE` draft **13**. Only five of the nine usable sets are Arena's familiar 3×14. This is the case the pack-geometry work exists for, and the old failure mode is worth stating precisely.

For the 15-card sets, `MAX_PACK_SIZE = 14` in `ingest.py` would at least have raised — loudly, which is fine. **The 13-card sets are the ones that should worry anyone**, because 13 ≤ 14 clears that guard entirely and nothing in ingest would have said a word. In every case the dangerous constant was `PICKS_PER_PACK = 14` in `dataset.py`: it makes `pack_number * 14 + pick_number` disagree with every row's position in its draft, so every draft fails the pool-as-prefix identity and the default `on_invalid="drop"` discards the whole corpus — **18.3M of the 46.3M usable picks, across four sets** — and returns an empty `PickData` without raising anything at all.

Geometry is now measured during ingest, written to `ingest_stats.json`, and read back by `PickData`. It is also passed into `ModelConfig`, because it sizes the two `ContextFeatures` embeddings, and JAX clamps an out-of-range `nn.Embed` index rather than raising: a 14-row pick embedding asked for index 14 returns row 13 and trains happily on the wrong vector.

**`AFR` is a gzipped tar, not a gzipped CSV.** Same `.csv.gz` extension as every other set; inside is a tar holding one file. Read as plain gzip nothing raises — the 512-byte tar header runs straight into the first CSV line, so the header parses fine and the first column comes back as archive metadata glued to a real column name. `vocab.open_text` sniffs for the tar magic at byte 257 rather than trusting the filename.

**`AFR` also has different metadata columns.** No `rank`, no `user_n_games_bucket`, no `user_game_win_rate_bucket`; it carries `user_match_win_rate_bucket` and `user_n_matches_bucket` instead. The old code required the union of every set's metadata columns, which rejected a valid export. Only the four columns ingest genuinely needs — `draft_id`, `pack_number`, `pick_number`, `pick` — are required now; the rest are read when present and defaulted when absent.

**`AFR` is nevertheless unusable, and that is a real finding rather than a parsing failure.** Its export omits every draft's very first pick: there is no row anywhere with `pack_number == 0` and `pick_number == 0`, and drafts are 41 rows where 3×14 predicts 42. The taken card survives only in the `pool_*` columns of the next row, which ingest deliberately does not read. So every pool reconstructed from AFR by the prefix rule would be short by exactly one card, throughout the whole draft. Recovering it would mean seeding each draft's pool from the `pool_*` columns of its first surviving row — a change to how pools are represented, not a parsing tweak — so AFR is excluded and the reason recorded here.

**Three sets contain a handful of rows whose recorded pick is not in their recorded pack** — one in `SIR`, five in `LCI`, four in `EOE`, out of 2.2M, 6.1M and 4.1M rows respectively. A single such record used to abort the entire set. A malformed row is now dropped, which shortens its draft, which makes `PickData` drop the rest of that draft — so the bad row can never leak into a neighbouring pool — and `max_bad_row_fraction` (default 0.1%) keeps genuine header misalignment loud.

**Counts, not flags, matters far more outside FIN.** FIN has zero rows with a card at count 2 or more, which makes it a poor set to have validated the parse against. `OTJ` has 250,354, `DSK` 238,768, `BLB` 217,353 and `MH3` 168,691 — around 3% of their rows each. A boolean parse would silently shrink those packs.

**Scryfall's `/cards/collection` refuses the full `"Front // Back"` name that 17lands uses for split, transforming and Room cards.** Verified live: it returns `Crime // Punishment` in `not_found` and resolves the same card when asked for `Crime`. `fetch_scryfall_cards` had a fallback for the opposite direction only, so `DSK` lost 23 of its 286 cards — all of Duskmourn's Rooms — to all-zero feature rows, and `OTJ` lost one. Unresolved names containing `//` are now retried by front face; both sets resolve completely.

**Two of `HBG`'s cards remain unresolved and that is correct** (`A-Baba Lysaga, Night Witch`, `A-Monster Manual`). The `A-` prefix marks an Alchemy *rebalance*, and Scryfall does not index those at all — its search endpoint 404s on the name. Substituting the paper card would fabricate precisely what the rebalance changed, so they keep neutral all-zero rows. `src/validation.py` reports unresolved cards and fails only above 2% of a set, so a genuine source gap is visible without permanently blocking the gate on HBG.

### Making the silent failures loud

The through-line above is that the expensive failures were the quiet ones. `PickData` now refuses to return a corpus that is mostly gone: dropping a few malformed drafts is routine, but dropping nearly all of them means the loader is wrong about the corpus's shape rather than the corpus being broken. The diagnosis reports the modal rows-per-draft against the geometry, which is the single fact that identifies the cause. On AFR it says:

    1,066,415 of 1,066,415 rows (100.0%) belong to drafts that violate the
    pool-as-prefix identity at geometry 3 packs x 14 picks = 42 picks/draft,
    pack width 14. Most drafts have 41 rows where this geometry predicts 42;
    the export is probably missing 1 pick(s) per draft rather than being the
    geometry recorded.

`python -m src.validation --processed-dir <dir>` is the gate to run after ingesting a new set.

### Fetching them

    python -m src.data.ingest --csv data/raw/draft_data_public.<SET>.PremierDraft.csv.gz \
        --out data/processed/<SET>.PremierDraft
    python -m src.data.card_features --processed-dir data/processed/<SET>.PremierDraft --report

The Scryfall JSON for each set is cached under `data/raw/scryfall/<SET>.json`, so rebuilding features does not re-hit the API.

## Card features: one column layout for every set

The feature table feeding the composite embeddings (`src/data/card_features.py`) used to fit its keyword columns to whichever set it was building, keeping any Scryfall keyword carried by at least two of that set's cards. That was the right guard for a per-set fit — a keyword on one card is that card's id in disguise — but the fit itself was the problem: **column *k* was a different keyword in a different set**, so a table built on FIN meant nothing to a model reading BLB. For a project whose stated goal is drafting a set the model has never seen, that is fatal.

FIN's old build shows it plainly. Of its 34 keyword columns, 21 do not exist in the fixed vocabulary, and the clearest cases are FIN's own named mechanics — `Job select` (16 cards) and `Tiered` (6) — which nothing outside the set has at all.

The layout is now two checked-in module constants:

- **`GLOBAL_KEYWORDS`** — the 15 evergreen keywords. A set that never prints one gets an all-zero column; that costs one column and preserves the correspondence, which is the trade being made deliberately.
- **`MECHANICS`** — 73 structured predicates pattern-matched against Scryfall oracle text: creates tokens, sacrifice outlet, cares about creatures dying, +1/+1 counters made and cared about, graveyard recursion, self-mill, discard, draw, lifegain source and payoff, artifacts and enchantments matter, equipment and auras, tribal reference, attack and block triggers, ETB and death triggers, activated abilities with a mana cost, instant speed, the four kinds of removal, counterspells, ramp and fixing, card selection.

The mechanics are separate columns on purpose. The cross-attention arm scores a candidate against the pool through a bilinear interaction, and a bilinear form can only learn "a sacrifice outlet plus a token maker is worth more than either alone" if those are two distinguishable inputs. Collapse them into one aggregate score and the interaction has nothing left to find. That is also why a sentence embedding of the oracle text would be the wrong tool here: embedders encode *similarity*, and synergy is *complementarity*.

### Three things the text pipeline has to get right

1. **Strip the card's own name, in both spellings.** Scryfall uses the full name — *Vivi Ornitier*: "…where X is Vivi Ornitier's power" — and, for a legendary permanent, the pre-comma short name alone: *Zidane, Tantalus Thief*: "When **Zidane** enters, gain control of target creature…". A stripper that only handled full names would leave "Zidane" sitting in the features, which is card identity leaking straight in — the exact failure the old `MIN_KEYWORD_CARDS` threshold existed to prevent. Both forms are replaced with `~`, longest first, so removing the short form cannot leave a dangling `", Tantalus Thief"` behind. Matching is case-sensitive, because card names are capitalised in rules text while the words they collide with are not — a card named *Fire* loses its own name without eating "firebreathing".
2. **Strip reminder text in parentheses.** It restates what a keyword already means, so it is redundant with the keyword flags, and it inflates apparent similarity between two unrelated cards that happen to share one mechanic.
3. **Read both faces of a double-faced card.** Their top-level `oracle_text` is empty and the text lives on `card_faces`; reading only the top level drops the card entirely, and reading only the front face drops half of it.

### Does it actually work across sets?

Built over all nine usable sets, the table is byte-for-byte the same 119 columns everywhere, and **no column is dead in every set** — all 73 mechanics fire somewhere, so none is pure width. The occupancies also move the way they should if the patterns are picking up real mechanics rather than templating noise:

| column | highest | lowest |
|---|---|---|
| `references_creature_type` | SIR 26.3% | EOE 6.2% |
| `tribal_lord` | BLB 1.8% | EOE 0.0% |
| `equipment_matters` | FIN 11.3% | BLB 1.8% |
| `graveyard_size_matters` | LCI 9.3% | EOE 0.0% |
| `artifacts_matter` | EOE 8.7% | DSK 0.0% |
| `make_treasure` | HBG 8.1% | EOE 0.0% |

Shadows over Innistrad and Bloomburrow are the tribal sets; Final Fantasy is the equipment set; Duskmourn prints essentially no artifact payoffs. Those are the right answers, and none of them was fitted — every column is a module constant.

### Width is a real constraint

`dense()` is capped at 120 columns (`MAX_FEATURE_WIDTH`, asserted at build time and in the tests). The card embedding is `_dense(F, embed_hidden)` — linear in the feature width — while the rest of the model is quadratic in `hidden_dim`. A wide table therefore inflates small-*N* models proportionally more than large-*N* ones, which bends exactly the low-*N* corner the Chinchilla Huber fit is most sensitive to. FIN's table goes from 65 columns to 119.

One honest cost of the change: with a global vocabulary, a keyword can legitimately fire on a single card in a given set — `Double strike` does in FIN — where the old per-set threshold would have dropped it. Within a single-set training run that column is still effectively a one-hot for that card. It is 4 such columns out of 119 now, against 84 of 118 under the old scheme, and unlike before the column means the same thing in the next set.


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
