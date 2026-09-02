# Data pipeline

Ingestion and preprocessing for the 17lands draft-pick corpus, and construction of the card feature table the model reads. See `docs/DATA.md` for what was verified about the source format and `docs/PROJECT_PLAN.md` §2–§3 for how this feeds the rest of the project.

- `vocab.py` — builds the card vocabulary for one set from the export's `pack_card_*` columns. Alphabetical, so ids are reproducible across runs and machines. Handles the commas that appear inside real card names.
- `ingest.py` — one chunked streaming pass over the ~9GB uncompressed export, writing `picks.npz` + `vocab.json`. Never reads the `pool_*` columns and stores packs as padded id lists, which is what gets the corpus down to ~46 bytes/row.
- `card_features.py` — fetches Scryfall attributes for the set's cards and builds the feature table. This is where oracle text becomes structured mechanical behaviour.
- `dataset.py` — loads those arrays in canonical order, reconstructs pools as prefix slices, and splits on `draft_id`.

Usage:

    python -m src.data.ingest --csv data/raw/draft_data_public.FIN.PremierDraft.csv.gz --out data/processed/FIN.PremierDraft
    python -m src.data.card_features --processed-dir data/processed/FIN.PremierDraft

Add `--limit-rows N` to work against a truncated prefix of the file while developing.

## The invariant everything rests on

A pick's pool is exactly the set of that draft's earlier picks. `PickData` checks it on load and drops any draft where it fails — incomplete drafts do occur in the real exports, and a missing pick would silently shift every later pool in that draft.

The check is written as $|\text{pool}| = \text{pack\_number} \times \text{picks\_per\_pack} + \text{pick\_number}$, which means **pack geometry has to be right or the whole set is discarded without an error**. Geometry is detected during ingest and persisted rather than assumed; see `docs/PROJECT_PLAN.md` §2.

## The feature table is set-independent by construction

Every column must mean the same thing in every set, because the bot is meant to draft sets it was not trained on. That rules out anything fitted from the set at hand.

One thing worth knowing before touching `card_features.py`: Scryfall's `keywords` field is not just evergreen mechanics. Sets list flavour-named abilities there too, and on FIN **84 of 118 keywords appear on exactly one card**. A feature column set for exactly one card is that card's id in disguise. The earlier version of this file dropped singletons with a `MIN_KEYWORD_CARDS` threshold, which fixed the identity leak but left the surviving columns fitted per-set and therefore useless across sets. The current design replaces them with a fixed global keyword list plus structured mechanical features derived from oracle text — see `docs/PROJECT_PLAN.md` §3.

Two processing steps in there are correctness requirements rather than tidying: the card's own name is stripped from its oracle text before anything reads it (Scryfall spells it out in full, and leaving it in reintroduces card identity as a feature), and parenthetical reminder text is stripped as redundant with the keyword flags.

`CardFeatures.dense()` is the single place the column layout lives. Everything downstream reads the width from the table rather than hardcoding it.
