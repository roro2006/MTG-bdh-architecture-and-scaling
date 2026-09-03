# Data pipeline

Ingestion and preprocessing for the 17lands draft-pick corpus, and construction of the card feature table the model reads. See `docs/DATA.md` for what was verified about the source format and `docs/PROJECT_PLAN.md` §2–§3 for how this feeds the rest of the project.

- `vocab.py` — builds the closed card vocabulary from the export's `pack_card_*` columns. Alphabetical, so ids are reproducible across runs and machines. Handles the commas inside real card names, the one set that ships its CSV inside a tar, and the metadata columns that only some sets carry.
- `ingest.py` — one chunked streaming pass over the ~9GB uncompressed export, writing `picks.npz` + `vocab.json` + `ingest_stats.json`. Never reads the `pool_*` columns and stores packs as padded id lists, which is what gets the corpus down to ~46 bytes/row. Measures the pack geometry rather than assuming it.
- `dataset.py` — loads those arrays in canonical order, reconstructs pools as prefix slices, splits on `draft_id`, and groups recurring states for the Bayes-floor measurement.
- `card_features.py` — fetches Scryfall attributes and builds the per-card feature table. This is where oracle text becomes structured mechanical behaviour, and the column layout is a module constant, identical for every set.

Usage:

    python -m src.data.ingest --csv data/raw/draft_data_public.FIN.PremierDraft.csv.gz --out data/processed/FIN.PremierDraft
    python -m src.data.card_features --processed-dir data/processed/FIN.PremierDraft --report

Add `--limit-rows N` to work against a truncated prefix of the file while developing. Note that a truncated pass can only measure the geometry it managed to see.

## Two invariants, and why each is checked rather than assumed

**A pick's pool is exactly the set of that draft's earlier picks.** `PickData` checks this on load and drops any draft where it fails — incomplete drafts do occur in the real exports, and a missing pick would silently shift every later pool in that draft. If *most* of a corpus fails, that is not a corpus of broken drafts, it is a loader that is wrong about the corpus's shape, so it raises with a diagnosis instead of returning an empty `PickData`.

The check is written as $|\text{pool}| = \text{pack\_number} \times \text{picks\_per\_pack} + \text{pick\_number}$, which is why the geometry below has to be right or the whole set is discarded.

**Pack geometry is a property of the set, not a constant.** `PackGeometry` (packs per draft, picks per pack, widest pack) is measured during ingest, written to `ingest_stats.json`, and read back by `PickData`; see `docs/PROJECT_PLAN.md` §2. It is not decoration: it drives the pool padding width, the identity above, and the two `ContextFeatures` embedding sizes. Hardcoding Arena's usual 3×14 fails silently on a set that disagrees — `SIR.PremierDraft` is 3×15 — because the wrong `picks_per_pack` makes every row fail the pool identity and the default `on_invalid="drop"` then discards everything.

Callers that build a `ModelConfig` should pass `data.packs_per_draft` and `data.picks_per_pack`, the same way they already pass `card_feature_dim`. JAX clamps an out-of-range `nn.Embed` index rather than raising, so an undersized context embedding returns the wrong row without complaining.

## Feature columns mean the same thing in every set

Every column must mean the same thing in every set, because the bot is meant to draft sets it was not trained on. That rules out anything fitted from the set at hand — see `docs/PROJECT_PLAN.md` §3.

Scryfall's `keywords` field is what makes this easy to get wrong. It is not just evergreen mechanics; sets list flavour-named abilities there too, and on FIN **84 of 118 keywords appear on exactly one card**. A feature column set for exactly one card is that card's id in disguise. An earlier version dropped singletons with a `MIN_KEYWORD_CARDS` threshold, keeping any keyword on ≥2 of that set's cards. That fixed the identity leak but left the surviving columns fitted per-set: column *k* meant "Job select" in FIN and something unrelated in the next set, which is fatal for a model meant to draft a set it has never seen. The layout is now two checked-in constants — `GLOBAL_KEYWORDS` (evergreen keywords) and `MECHANICS` (structured predicates over oracle text) — so a set that does not print a mechanic gets a zero column and the correspondence survives.

Three things the oracle-text pipeline handles that are easy to miss: the card's own name is stripped in both its full and legendary short forms (Scryfall writes "Whenever Zidane, Tantalus Thief attacks… Zidane deals damage", and leaving that in leaks card identity straight into the features), reminder text in parentheses is removed, and both faces of a double-faced card are read — its top-level `oracle_text` is empty.

`CardFeatures.dense()` is the single place the column layout lives. Everything downstream reads the width from the table rather than hardcoding it.
