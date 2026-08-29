"""Builds the closed card vocabulary for a single 17lands set/event.

The vocabulary is derived from the `pack_card_<name>` columns of a
`draft_data_public.<SET>.<EVENT>.csv.gz` file: one entry per unique card
the set can put in a pack. Sort order is fixed (alphabetical on card name)
so the resulting integer ids are reproducible across runs and machines.

See docs/DATA.md for where the source file comes from and docs/PROJECT_PLAN.md
section 2 for how this fits into the data stage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    """Maps card names to a fixed, reproducible integer id space."""

    card_to_id: dict[str, int]
    id_to_card: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.id_to_card)


def build_vocabulary(draft_csv_path: str) -> Vocabulary:
    """Reads a draft_data_public CSV(.gz) header and builds a Vocabulary
    from its `pack_card_*` columns.

    Not yet implemented — first thing to build once the plan is settled.
    """
    raise NotImplementedError
