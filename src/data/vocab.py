"""Builds the closed card vocabulary for a single 17lands set/event.

The vocabulary is derived from the `pack_card_<name>` columns of a
`draft_data_public.<SET>.<EVENT>.csv.gz` file: one entry per unique card
the set can put in a pack. Sort order is fixed (alphabetical on card name)
so the resulting integer ids are reproducible across runs and machines.

Four things about the real files make the naive approach wrong, all
verified against live exports rather than assumed:

  - card names contain commas ("pack_card_Annie Flash, the Veteran"), so
    the header must go through a real CSV parser, not `line.split(",")`.
  - the file carries both `pack_card_<name>` and `pool_<name>` columns for
    the same card set. They are cross-checked here and must agree; the
    pool columns are then dropped at ingest, since the pool at any pick is
    exactly the prefix of that draft's earlier picks (see ingest.py).
  - not every `.csv.gz` is a gzipped CSV. `AFR.PremierDraft` is a gzipped
    **tar** holding one CSV, under the same file extension as every other
    set. Read as plain gzip its first column comes back as a tar header
    block, and the vocabulary silently comes out wrong. `open_text` sniffs
    for the tar magic rather than trusting the name.
  - the metadata columns are not the same across sets. Older exports carry
    `user_match_win_rate_bucket` and `user_n_matches_bucket` and have no
    `rank` column at all, so only the four columns ingest genuinely needs
    are required; the rest are read when present.

See docs/DATA.md for where the source file comes from and docs/PROJECT_PLAN.md
section 2 for how this fits into the data stage.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

PACK_PREFIX = "pack_card_"
POOL_PREFIX = "pool_"

# What ingest cannot do without. A file missing any of these is not a draft
# export in any useful sense, so this is the only hard requirement.
REQUIRED_META_COLUMNS = (
    "draft_id",
    "pack_number",
    "pick_number",
    "pick",
)

# Read when present, defaulted when absent. AFR.PremierDraft has none of
# the last four; MH3 and FIN have all of them. Treating the union as
# mandatory rejected a perfectly good export.
OPTIONAL_META_COLUMNS = (
    "expansion",
    "event_type",
    "draft_time",
    "rank",
    "event_match_wins",
    "event_match_losses",
    "pick_maindeck_rate",
    "pick_sideboard_in_rate",
    "user_n_games_bucket",
    "user_game_win_rate_bucket",
)

META_COLUMNS = REQUIRED_META_COLUMNS + OPTIONAL_META_COLUMNS


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


TAR_MAGIC_OFFSET = 257
TAR_MAGIC = b"ustar"


def is_gzipped_tar(path: str | Path) -> bool:
    """True if a `.gz` is really a tar archive rather than a bare CSV.

    17lands ships AFR.PremierDraft this way while every other set is a
    plain gzipped CSV, and nothing in the filename says so. The tar magic
    sits at byte 257 of the first 512-byte header block.
    """
    try:
        with gzip.open(path, "rb") as handle:
            head = handle.read(512)
    except OSError:
        return False
    return len(head) >= TAR_MAGIC_OFFSET + len(TAR_MAGIC) and (
        head[TAR_MAGIC_OFFSET : TAR_MAGIC_OFFSET + len(TAR_MAGIC)] == TAR_MAGIC
    )


class _TarBackedText(io.TextIOWrapper):
    """A text view of one member of a tar, which closes the tar with it.

    A plain `TextIOWrapper` over `extractfile()` would leave the archive
    open, and closing the archive first invalidates the member stream. This
    ties their lifetimes together so callers can treat the result as an
    ordinary file object -- used in a `with`, or held open across a chunked
    pandas read, which is what ingest needs.
    """

    def __init__(self, archive: tarfile.TarFile, stream):
        self._archive = archive
        super().__init__(stream, encoding="utf-8", newline="")

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._archive.close()


def open_text(path: str | Path):
    """Opens a 17lands export as a text stream, whatever it actually is.

    Three shapes turn up under the same `.csv.gz` name: a plain gzipped
    CSV, a gzipped tar holding one CSV (AFR), and an uncompressed CSV.
    """
    path = Path(path)
    if path.suffix == ".gz" and is_gzipped_tar(path):
        archive = tarfile.open(path, "r:gz")
        try:
            members = [m for m in archive.getmembers() if m.isfile()]
            if len(members) != 1:
                raise ValueError(
                    f"{path} is a tar holding {len(members)} files; expected "
                    f"exactly one CSV, got {[m.name for m in members][:5]}"
                )
            stream = archive.extractfile(members[0])
            if stream is None:  # pragma: no cover - tarfile contract
                raise ValueError(f"could not read {members[0].name} from {path}")
        except Exception:
            archive.close()
            raise
        return _TarBackedText(archive, stream)

    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return open(path, mode="rt", encoding="utf-8", newline="")


# Kept as the private spelling the rest of this module already used.
_open_text = open_text


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

    missing = [c for c in REQUIRED_META_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"header is missing metadata columns ingest cannot do without: "
            f"{missing}. Optional columns that vary between sets are "
            f"{list(OPTIONAL_META_COLUMNS)} and are not required."
        )

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
