# Data pipeline

Ingestion and preprocessing for the 17lands draft-pick corpus: pulling a single set/event's `draft_data_public.*.csv.gz`, building the closed card vocabulary from its `pack_card_*` columns, splitting on `draft_id` to keep every pick from one draft in a single split, and carving out the matched-state subset held back for the Bayes-error floor measurement (see `docs/PROJECT_PLAN.md` §2 and §6).

Nothing here yet — this is the first implementation stage once the plan is settled.
