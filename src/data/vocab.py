"""Builds the closed card vocabulary for a single 17lands set/event.

The vocabulary is derived from the `pack_card_<name>` columns of a
`draft_data_public.<SET>.<EVENT>.csv.gz` file: one entry per unique card
the set can put in a pack. Sort order is fixed (alphabetical on card name)
so the resulting integer ids are reproducible across runs and machines.

Two things about the real header make the naive approach wrong, both
verified against the live file rather than assumed:

  - card names contain commas ("pack_card_Annie Flash, the Veteran"), so
    the header must go through a real CSV parser, not `line.split(",")`.
  - the file carries both `pack_card_<name>` and `pool_<name>` columns for
    the same card set. They are cross-checked here and must agree; the
    pool columns are then dropped at ingest, since the pool at any pick is
    exactly the prefix of that draft's earlier picks (see ingest.py).

See docs/DATA.md for where the source file comes from and docs/PROJECT_PLAN.md
section 2 for how this fits into the data stage.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

PACK_PREFIX = "pack_card_"
POOL_PREFIX = "pool_"

# Metadata columns carried alongside the per-card columns. Verified present
# in both OTJ and FIN PremierDraft exports.
META_COLUMNS = (
    "expansion",
    "event_type",
    "draft_id",
    "draft_time",
    "rank",
    "event_match_wins",
    "event_match_losses",
    "pack_number",
    "pick_number",
    "pick",
    "pick_maindeck_rate",
    "pick_sideboard_in_rate",
    "user_n_games_bucket",
    "user_game_win_rate_bucket",
)


@dataclass(frozen=True)
class Vocabulary:
    """Maps card names to a fixed, reproducible integer id space."""

    card_to_id: dict[str, int]
    id_to_card: tuple[str, ...]
    expansion: str = ""
    event_type: str = ""

    @property
    def size(self) -> int:
        return len(self.id_to_card)

    def id_of(self, card_name: str) -> int:
        return self.card_to_id[card_name]

    def pack_columns(self) -> list[str]:
        """Header names of the pack_card_* columns, in vocabulary id order.

        Selecting a dataframe by this list makes column position equal card
        id, which is what lets ingest.py skip a per-column remap.
        """
        return [PACK_PREFIX + name for name in self.id_to_card]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "expansion": self.expansion,
            "event_type": self.event_type,
            "size": self.size,
            "id_to_card": list(self.id_to_card),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        id_to_card = tuple(payload["id_to_card"])
        return cls(
            card_to_id={name: i for i, name in enumerate(id_to_card)},
            id_to_card=id_to_card,
            expansion=payload.get("expansion", ""),
            event_type=payload.get("event_type", ""),
        )


def _open_text(path: str | Path):
    """Opens a .csv or .csv.gz uniformly as a text stream."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return open(path, mode="rt", encoding="utf-8", newline="")


def read_header(csv_path: str | Path) -> list[str]:
    """Reads just the header row, without decompressing the whole file."""
    with _open_text(csv_path) as handle:
        return next(csv.reader(handle))


def read_header_and_first_row(csv_path: str | Path) -> tuple[list[str], list[str]]:
    """Header plus one data row — enough to recover expansion/event_type."""
    with _open_text(csv_path) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        try:
            first = next(reader)
        except StopIteration:
            first = []
    return header, first


def build_vocabulary(draft_csv_path: str | Path) -> Vocabulary:
    """Reads a draft_data_public CSV(.gz) header and builds a Vocabulary
    from its `pack_card_*` columns.
    """
    columns, first_row = read_header_and_first_row(draft_csv_path)

    missing = [c for c in META_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"header is missing expected metadata columns: {missing}")

    pack_names = {c[len(PACK_PREFIX):] for c in columns if c.startswith(PACK_PREFIX)}
    pool_names = {c[len(POOL_PREFIX):] for c in columns if c.startswith(POOL_PREFIX)}
    if not pack_names:
        raise ValueError(f"no {PACK_PREFIX}* columns found in {draft_csv_path}")
    if pack_names != pool_names:
        only_pack = sorted(pack_names - pool_names)[:5]
        only_pool = sorted(pool_names - pack_names)[:5]
        raise ValueError(
            "pack_card_* and pool_* columns disagree on the card set "
            f"(pack-only e.g. {only_pack}, pool-only e.g. {only_pool})"
        )

    by_name = dict(zip(columns, first_row))
    id_to_card = tuple(sorted(pack_names))
    return Vocabulary(
        card_to_id={name: i for i, name in enumerate(id_to_card)},
        id_to_card=id_to_card,
        expansion=by_name.get("expansion", ""),
        event_type=by_name.get("event_type", ""),
    )
