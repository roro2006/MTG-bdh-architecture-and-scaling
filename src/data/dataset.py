"""Loads, splits, and formats 17lands draft-pick data for training.

Responsibilities (see docs/PROJECT_PLAN.md section 1-2):
  - parse a draft_data_public CSV(.gz) into per-pick examples: pack contents,
    accumulated pool, pack/pick number, and the label (the card taken)
  - split on draft_id so every pick from one draft stays in a single split
  - carve out the matched-state subset (recurring pack/pool combinations
    across distinct drafts) used later for the Bayes-error floor measurement,
    before the train/val/test split is drawn
"""

from __future__ import annotations

from dataclasses import dataclass

from .vocab import Vocabulary


@dataclass(frozen=True)
class DraftExample:
    pack: tuple[int, ...]       # card ids present in the current pack
    pool: tuple[int, ...]       # card ids accumulated so far
    pack_number: int
    pick_number: int
    label: int                  # card id actually taken


def load_examples(draft_csv_path: str, vocab: Vocabulary) -> list[DraftExample]:
    raise NotImplementedError


def split_by_draft(
    examples: list[DraftExample],
    draft_ids: list[str],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[DraftExample], list[DraftExample], list[DraftExample]]:
    """Splits on draft_id, not on individual rows."""
    raise NotImplementedError


def carve_matched_state_subset(
    examples: list[DraftExample],
) -> list[DraftExample]:
    """Pulls out picks whose (pack, pool) state recurs across distinct
    drafts, for the disagreement measurement in PROJECT_PLAN.md section 6.
    This subset is excluded from anything returned by split_by_draft.
    """
    raise NotImplementedError
