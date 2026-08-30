# Data pipeline

Ingestion and preprocessing for the 17lands draft-pick corpus. See `docs/DATA.md` for what was verified about the source format and `docs/PROJECT_PLAN.md` §2 and §6a for how this feeds the rest of the study.

- `vocab.py` — builds the closed card vocabulary from the export's `pack_card_*` columns. Alphabetical, so ids are reproducible across runs and machines. Handles the commas that appear inside real card names.
- `ingest.py` — one chunked streaming pass over the ~9GB uncompressed export, writing `picks.npz` + `vocab.json`. Never reads the `pool_*` columns and stores packs as padded id lists, which is what gets the corpus down to ~46 bytes/row.
- `dataset.py` — loads those arrays in canonical order, reconstructs pools as prefix slices, splits on `draft_id`, and groups recurring states for the Bayes-floor measurement.

Usage:

    python -m src.data.ingest --csv data/raw/draft_data_public.FIN.PremierDraft.csv.gz --out data/processed/FIN.PremierDraft

Add `--limit-rows N` to work against a truncated prefix of the file while developing.

The invariant everything else rests on is that a pick's pool is exactly the set of that draft's earlier picks. `PickData` checks it on load and drops any draft where it fails — incomplete drafts do occur in the real exports, and a missing pick would silently shift every later pool in that draft.
