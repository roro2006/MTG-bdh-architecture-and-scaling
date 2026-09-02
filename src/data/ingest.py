"""Streams a 17lands draft_data_public CSV(.gz) into compact numpy arrays.

The raw file cannot be read into memory: a single set/event export is ~740
columns wide and, for FIN.PremierDraft, roughly 9GB uncompressed across
~5.8M picks. This module makes one chunked pass over it and writes a
fixed-width binary form that does fit.

Two decisions do most of the size reduction, both justified by the file's
verified structure:

  - the `pool_*` columns are never read. The pool at any pick is exactly
    the set of that draft's earlier picks, so it is reconstructed at batch
    time (see dataset.py) rather than stored. This halves the parse and
    keeps the on-disk form from tripling in size.
  - a pack is stored as its card ids padded to the widest pack the file
    actually contains, not as a length-V count vector. Packs hold at most
    a dozen or so cards out of a several-hundred-card vocabulary, so the
    dense form would be ~25x larger for no gain.

`pack_card_*` values are counts, not flags -- OTJ.PremierDraft really does
contain packs with a card at count 2 -- so a card at count k is emitted k
times into the pack list.

**Pack geometry is measured, not assumed.** An earlier version hardcoded
14 picks per pack and 3 packs per draft, which is Arena's usual shape but
not a law: Alchemy and remaster events have shipped other geometries, and
a hardcoded 14 does not fail loudly on a set that disagrees -- it makes
every draft violate the pool-as-prefix identity in dataset.py, where the
default `on_invalid="drop"` then discards the entire corpus in silence.
So this pass records what it saw (`PackGeometry`) into ingest_stats.json,
and dataset.py reads it back rather than assuming.

Run directly:
    python -m src.data.ingest --csv data/raw/draft_data_public.FIN.PremierDraft.csv.gz --out data/processed/FIN.PremierDraft
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .vocab import Vocabulary, build_vocabulary, open_text, read_header

# Not a geometry assumption -- a corruption guard. A "pack" wider than this
# means the count columns are not counts any more (or the header no longer
# lines up with the values), and padding every row out to that width would
# quietly blow up the on-disk size instead of complaining.
PACK_WIDTH_LIMIT = 64

# Rows per chunk. 40k x 363 int8 columns is ~15MB of parsed values, which
# keeps peak memory well under control while staying big enough that the
# per-chunk numpy work is not dominated by overhead.
DEFAULT_CHUNK_ROWS = 40_000

_PACK_DTYPE = np.int16  # card ids; vocabularies here are a few hundred entries

_ARRAY_NAMES = (
    "pack",
    "pack_size",
    "label",
    "label_pos",
    "pack_number",
    "pick_number",
    "draft_idx",
    "rank_code",
    "win_rate_bucket",
)


@dataclass(frozen=True)
class PackGeometry:
    """The shape of a draft in one export: how many packs, how many picks
    from each, and the widest pack seen.

    Every downstream shape derives from these three numbers -- the pool
    padding width, the pool-as-prefix identity dataset.py validates rows
    against, and the two `nn.Embed` sizes in ContextFeatures. Carrying them
    as data rather than as constants is what lets one codebase ingest sets
    with different geometries.
    """

    packs_per_draft: int
    picks_per_pack: int
    max_pack_size: int

    @property
    def picks_per_draft(self) -> int:
        """Rows a complete draft contributes."""
        return self.packs_per_draft * self.picks_per_pack

    @property
    def max_pool_size(self) -> int:
        """Largest pool a draft ever presents: everything before the last pick."""
        return self.picks_per_draft - 1

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "PackGeometry":
        return cls(
            packs_per_draft=int(payload["packs_per_draft"]),
            picks_per_pack=int(payload["picks_per_pack"]),
            max_pack_size=int(payload["max_pack_size"]),
        )

    @classmethod
    def from_arrays(
        cls,
        pack_number: np.ndarray,
        pick_number: np.ndarray,
        max_pack_size: int,
    ) -> "PackGeometry":
        """Infers geometry from observed pack/pick numbers.

        Both are 0-based in the export, so the count is the maximum plus
        one. This reads the *observed* extent, which is the right thing for
        a full pass and a slight under-count for a truncated one -- see
        `ingest`'s note on --limit-rows.
        """
        if pack_number.size == 0:
            raise ValueError("cannot infer pack geometry from an empty corpus")
        return cls(
            packs_per_draft=int(pack_number.max()) + 1,
            picks_per_pack=int(pick_number.max()) + 1,
            max_pack_size=int(max_pack_size),
        )

    def describe(self) -> str:
        return (
            f"{self.packs_per_draft} packs x {self.picks_per_pack} picks "
            f"= {self.picks_per_draft} picks/draft, pack width {self.max_pack_size}"
        )


@dataclass(frozen=True)
class IngestStats:
    """What the pass actually saw, for the run log and for sanity checks."""

    rows: int
    drafts: int
    vocab_size: int
    geometry: PackGeometry
    duplicate_card_rows: int
    elapsed_seconds: float
    dropped_rows: int = 0
    missing_meta_columns: tuple[str, ...] = ()

    @property
    def max_pack_seen(self) -> int:
        return self.geometry.max_pack_size

    def summary(self) -> str:
        rate = self.rows / self.elapsed_seconds if self.elapsed_seconds else 0.0
        extra = ""
        if self.dropped_rows:
            extra += f", {self.dropped_rows:,} malformed rows dropped"
        if self.missing_meta_columns:
            extra += f", no {list(self.missing_meta_columns)} in this export"
        return (
            f"{self.rows:,} picks across {self.drafts:,} drafts "
            f"(vocab {self.vocab_size}, {self.geometry.describe()}, "
            f"{self.duplicate_card_rows:,} rows with a duplicated card{extra}) "
            f"in {self.elapsed_seconds:,.1f}s [{rate:,.0f} rows/s]"
        )


def _packs_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Turns an (n_rows, vocab) count matrix into padded card-id lists.

    Column position in `counts` is assumed to already be the card id, which
    is what Vocabulary.pack_columns() guarantees by ordering the selection.
    The returned array is only as wide as this chunk's widest pack; `ingest`
    pads the chunks out to a common width once it knows the file's maximum.
    Returns (padded_packs, pack_sizes, n_rows_with_a_duplicate).
    """
    n_rows = counts.shape[0]
    row_idx, col_idx = np.nonzero(counts)
    reps = counts[row_idx, col_idx].astype(np.int64)

    rows_rep = np.repeat(row_idx, reps)
    cards_rep = np.repeat(col_idx, reps)

    sizes = np.bincount(rows_rep, minlength=n_rows)
    width = int(sizes.max(initial=0))
    if width > PACK_WIDTH_LIMIT:
        raise ValueError(
            f"pack of {width} cards exceeds PACK_WIDTH_LIMIT={PACK_WIDTH_LIMIT}; "
            "the export format has changed and the parse needs revisiting"
        )

    # np.nonzero yields row-major order and np.repeat preserves it, so
    # rows_rep is non-decreasing and each row's slots are contiguous.
    starts = np.zeros(n_rows, dtype=np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    positions = np.arange(rows_rep.size, dtype=np.int64) - starts[rows_rep]

    packs = np.full((n_rows, width), -1, dtype=_PACK_DTYPE)
    packs[rows_rep, positions] = cards_rep.astype(_PACK_DTYPE)

    n_dupes = int((reps > 1).sum())
    return packs, sizes.astype(np.int8), n_dupes


def _pad_to_width(packs: np.ndarray, width: int) -> np.ndarray:
    """Right-pads a chunk's pack array out to the file-wide pack width."""
    if packs.shape[1] == width:
        return packs
    out = np.full((packs.shape[0], width), -1, dtype=packs.dtype)
    out[:, : packs.shape[1]] = packs
    return out


def ingest(
    csv_path: str | Path,
    out_dir: str | Path,
    vocab: Vocabulary | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    limit_rows: int | None = None,
    verbose: bool = True,
    max_bad_row_fraction: float = 1e-3,
) -> IngestStats:
    """One chunked pass over the raw export, writing picks.npz + vocab.json.

    `limit_rows` caps the number of picks read, for developing against a
    truncated sample of the file. Note that a truncated pass can only infer
    the geometry it managed to see: stopping inside the first pack of the
    first draft would record 1 pack per draft. That is fine for development
    and wrong for a real corpus, so the geometry written to ingest_stats.json
    is only as trustworthy as the pass that produced it.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if vocab is None:
        vocab = build_vocabulary(csv_path)

    pack_columns = vocab.pack_columns()
    # `rank` and `user_game_win_rate_bucket` are absent from older exports
    # (AFR carries user_match_win_rate_bucket instead and no rank at all),
    # so they are read when present and defaulted when not. Asking pandas
    # for a column the file does not have is a hard error, hence the check.
    header = set(read_header(csv_path))
    wanted = ["draft_id", "pack_number", "pick_number", "pick"]
    optional = [c for c in ("rank", "user_game_win_rate_bucket") if c in header]
    missing_meta = tuple(
        c for c in ("rank", "user_game_win_rate_bucket") if c not in header
    )
    meta_columns = wanted + optional

    dtypes: dict[str, object] = {c: np.int8 for c in pack_columns}
    dtypes.update({"draft_id": str, "pick": str})
    if "rank" in optional:
        dtypes["rank"] = str

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in _ARRAY_NAMES}
    draft_id_to_idx: dict[str, int] = {}
    rank_to_code: dict[str, int] = {}
    n_rows = 0
    max_pack_seen = 0
    duplicate_card_rows = 0
    dropped_rows = 0
    started = time.monotonic()

    # Read through open_text rather than handing pandas the path: AFR ships
    # its CSV inside a gzipped tar under a plain `.csv.gz` name, which
    # pandas' own decompression does not unwrap. The handle has to outlive
    # the chunk loop, so it is closed in the finally below rather than by a
    # `with` around the read call.
    source = open_text(csv_path)
    reader = pd.read_csv(
        source,
        chunksize=chunk_rows,
        usecols=meta_columns + pack_columns,
        dtype=dtypes,
    )

    try:
        for chunk in reader:
            if limit_rows is not None and n_rows >= limit_rows:
                break
            if limit_rows is not None and n_rows + len(chunk) > limit_rows:
                chunk = chunk.iloc[: limit_rows - n_rows]

            # Selecting in vocabulary order makes column position == card id.
            counts = chunk[pack_columns].to_numpy(dtype=np.int8, copy=False)
            packs, sizes, n_dupes = _packs_from_counts(counts)
            duplicate_card_rows += n_dupes
            max_pack_seen = max(max_pack_seen, int(sizes.max(initial=0)))

            labels = chunk["pick"].map(vocab.card_to_id)
            if labels.isna().any():
                unknown = sorted(set(chunk.loc[labels.isna(), "pick"]))[:5]
                raise ValueError(f"picked cards absent from the vocabulary: {unknown}")
            labels = labels.to_numpy(dtype=_PACK_DTYPE)

            # The label must be a card physically in the pack; that invariant is
            # the whole premise of the pointer head. A row that breaks it is
            # dropped rather than fatal: SIR.PremierDraft contains exactly one
            # such row in 1.6M, and aborting the set over it would cost the
            # whole corpus. Dropping the row shortens its draft, which
            # dataset.py's pool-as-prefix check then removes in full -- so the
            # bad row never contaminates a pool. `max_bad_row_fraction` keeps
            # genuine corruption loud.
            matches = packs == labels[:, None]
            in_pack = matches.any(axis=1)
            if not in_pack.all():
                dropped_rows += int((~in_pack).sum())
                packs = packs[in_pack]
                sizes = sizes[in_pack]
                labels = labels[in_pack]
                matches = matches[in_pack]
                chunk = chunk[in_pack]
            if not len(chunk):
                continue
            label_pos = matches.argmax(axis=1).astype(np.int8)

            for draft_id in chunk["draft_id"]:
                if draft_id not in draft_id_to_idx:
                    draft_id_to_idx[draft_id] = len(draft_id_to_idx)
            draft_idx = chunk["draft_id"].map(draft_id_to_idx).to_numpy(dtype=np.int32)

            if "rank" in optional:
                ranks = chunk["rank"].fillna("unknown")
            else:
                ranks = pd.Series(["unknown"] * len(chunk), index=chunk.index)
            for rank in ranks:
                if rank not in rank_to_code:
                    rank_to_code[rank] = len(rank_to_code)
            rank_code = ranks.map(rank_to_code).to_numpy(dtype=np.int8)

            if "user_game_win_rate_bucket" in optional:
                win_rate = chunk["user_game_win_rate_bucket"].to_numpy(dtype=np.float32)
            else:
                win_rate = np.full(len(chunk), np.nan, dtype=np.float32)

            chunks["pack"].append(packs)
            chunks["pack_size"].append(sizes)
            chunks["label"].append(labels)
            chunks["label_pos"].append(label_pos)
            chunks["pack_number"].append(chunk["pack_number"].to_numpy(dtype=np.int8))
            chunks["pick_number"].append(chunk["pick_number"].to_numpy(dtype=np.int8))
            chunks["draft_idx"].append(draft_idx)
            chunks["rank_code"].append(rank_code)
            chunks["win_rate_bucket"].append(win_rate)

            n_rows += len(chunk)
            if verbose:
                elapsed = time.monotonic() - started
                print(
                    f"  {n_rows:,} picks / {len(draft_id_to_idx):,} drafts "
                    f"({elapsed:,.0f}s, {n_rows / max(elapsed, 1e-9):,.0f} rows/s)",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        source.close()

    if n_rows == 0:
        raise ValueError(f"no rows read from {csv_path}")

    # Chunks were packed at their own widths; a late chunk of last-pick rows
    # is narrower than an early one. Pad to the file-wide maximum on concat.
    chunks["pack"] = [_pad_to_width(p, max_pack_seen) for p in chunks["pack"]]
    arrays = {name: np.concatenate(parts) for name, parts in chunks.items()}

    geometry = PackGeometry.from_arrays(
        arrays["pack_number"], arrays["pick_number"], max_pack_seen
    )
    # A handful of malformed rows is a fact of the corpus; a lot of them
    # means the parse is wrong, and quietly dropping those would turn a
    # broken read into a plausible-looking smaller corpus.
    bad_fraction = dropped_rows / max(n_rows + dropped_rows, 1)
    if bad_fraction > max_bad_row_fraction:
        raise ValueError(
            f"{dropped_rows:,} of {n_rows + dropped_rows:,} rows "
            f"({bad_fraction:.2%}) have a pick that is not in their pack, over "
            f"max_bad_row_fraction={max_bad_row_fraction:.2%}. That is too many "
            "to be data noise -- check the header alignment before raising it."
        )

    stats = IngestStats(
        rows=n_rows,
        drafts=len(draft_id_to_idx),
        vocab_size=vocab.size,
        geometry=geometry,
        duplicate_card_rows=duplicate_card_rows,
        elapsed_seconds=time.monotonic() - started,
        dropped_rows=dropped_rows,
        missing_meta_columns=missing_meta,
    )

    draft_ids = np.array(list(draft_id_to_idx), dtype="U32")
    rank_names = np.array(list(rank_to_code), dtype="U16")

    vocab.save(out_dir / "vocab.json")
    np.savez_compressed(
        out_dir / "picks.npz",
        draft_ids=draft_ids,
        rank_names=rank_names,
        **arrays,
    )
    (out_dir / "ingest_stats.json").write_text(
        json.dumps(asdict(stats), indent=2), encoding="utf-8"
    )
    return stats


def load_geometry(processed_dir: str | Path) -> PackGeometry | None:
    """Reads back the geometry ingest recorded, or None if it is absent.

    Absent means either a corpus ingested before geometry was recorded, or
    a directory that was never ingested. Callers fall back to inferring
    from the arrays; see PickData.
    """
    path = Path(processed_dir) / "ingest_stats.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "geometry" not in payload:
        return None
    return PackGeometry.from_dict(payload["geometry"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a 17lands draft export.")
    parser.add_argument(
        "--csv", required=True, help="path to draft_data_public.<SET>.<EVENT>.csv.gz"
    )
    parser.add_argument(
        "--out", required=True, help="output directory for picks.npz + vocab.json"
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--limit-rows", type=int, default=None, help="stop after N picks (for dev)"
    )
    args = parser.parse_args(argv)

    stats = ingest(
        args.csv, args.out, chunk_rows=args.chunk_rows, limit_rows=args.limit_rows
    )
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
