"""Loads ingested pick arrays, reconstructs pools, and draws the splits.

The ingest pass (ingest.py) deliberately does not store the pool, because
the pool at any pick is exactly the set of that draft's earlier picks. Once
rows are sorted into canonical order -- (draft_idx, pack_number, pick_number)
-- that becomes a slice rather than a computation: row i's pool is
`label[draft_start[i] : i]`. That identity is asserted on load against the
independently known pool size, `pack_number * PICKS_PER_PACK + pick_number`.

Responsibilities (see docs/PROJECT_PLAN.md sections 1-2):
  - parse per-pick examples: pack contents, accumulated pool, pack/pick
    number, and the label (the card taken)
  - split on draft_id so every pick from one draft stays in a single split
  - carve out the matched-state subset (recurring pack/pool combinations
    across distinct drafts) used later for the Bayes-error floor
    measurement, before the train/val/test split is drawn
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ingest import MAX_PACK_SIZE  # noqa: F401  (re-exported: pack width for callers)
from .vocab import Vocabulary

PICKS_PER_PACK = 14
PACKS_PER_DRAFT = 3

# Largest pool a draft ever presents: everything taken before the final pick.
MAX_POOL_SIZE = (PACKS_PER_DRAFT - 1) * PICKS_PER_PACK + (PICKS_PER_PACK - 1)

PAD_ID = -1


@dataclass(frozen=True)
class DraftExample:
    """One decision, in plain Python form. Convenient for tests and for
    eyeballing individual rows; training reads the arrays directly.
    """

    pack: tuple[int, ...]  # card ids present in the current pack
    pool: tuple[int, ...]  # card ids accumulated so far
    pack_number: int
    pick_number: int
    label: int  # card id actually taken


@dataclass(frozen=True)
class Splits:
    """Row indices, not copies of the data."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    matched_state: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "val": int(self.val.size),
            "test": int(self.test.size),
            "matched_state": int(self.matched_state.size),
        }


class PickData:
    """Ingested picks in canonical order, with pools available on demand.

    Rows are sorted by (draft_idx, pack_number, pick_number) at load, so
    every draft occupies one contiguous block in ascending pick order.
    """

    _FIELDS = (
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

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        vocab: Vocabulary,
        on_invalid: str = "drop",
    ):
        if on_invalid not in ("drop", "raise"):
            raise ValueError(f"on_invalid must be 'drop' or 'raise', got {on_invalid!r}")
        order = np.lexsort(
            (arrays["pick_number"], arrays["pack_number"], arrays["draft_idx"])
        )
        self.pack = arrays["pack"][order]
        self.pack_size = arrays["pack_size"][order]
        self.label = arrays["label"][order]
        self.label_pos = arrays["label_pos"][order]
        self.pack_number = arrays["pack_number"][order]
        self.pick_number = arrays["pick_number"][order]
        self.draft_idx = arrays["draft_idx"][order]
        self.rank_code = arrays["rank_code"][order]
        self.win_rate_bucket = arrays["win_rate_bucket"][order]
        self.draft_ids = arrays["draft_ids"]
        self.rank_names = arrays["rank_names"]
        self.vocab = vocab

        self.n_drafts = int(self.draft_idx.max()) + 1 if self.size else 0
        self._reindex()

        bad_drafts = self._invalid_drafts()
        self.dropped_drafts = int(bad_drafts.size)
        self.dropped_rows = 0
        if bad_drafts.size:
            if on_invalid == "raise":
                raise ValueError(
                    f"{bad_drafts.size} drafts violate the pool-as-prefix identity, "
                    f"first is {self.draft_ids[bad_drafts[0]]}. Pass on_invalid='drop' "
                    "to exclude them."
                )
            rows_before = self.size
            observed = np.bincount(self.draft_idx, minlength=self.n_drafts)
            observed = observed[observed > 0]
            keep = ~np.isin(self.draft_idx, bad_drafts)
            self.dropped_rows = int((~keep).sum())
            for field in self._FIELDS:
                setattr(self, field, getattr(self, field)[keep])
            self._reindex()
            if self._invalid_drafts().size:
                raise AssertionError("dropping invalid drafts did not restore the identity")

            # Dropping a few malformed drafts is expected -- OTJ has some.
            # Dropping *every* row is not a data-quality problem, it is a
            # geometry mismatch, and it must not be allowed through: a
            # zero-row dataset trains on nothing, reports NaN for every loss
            # and baseline, and looks like a broken model rather than the
            # wrong set. PICKS_PER_PACK is fixed at FIN's 14 (docs/DATA.md),
            # so any set drafting a different pack size lands here.
            if self.size == 0 and rows_before:
                modal = int(np.bincount(observed).argmax()) if observed.size else 0
                raise ValueError(
                    f"every one of {rows_before:,} rows was dropped as invalid: "
                    f"all {bad_drafts.size:,} drafts violate the expected "
                    f"geometry of {PICKS_PER_PACK * PACKS_PER_DRAFT} rows per "
                    f"draft ({PACKS_PER_DRAFT} packs x {PICKS_PER_PACK} picks), "
                    f"while this export has {modal} rows per draft. "
                    "PICKS_PER_PACK is hardcoded to FIN's pack size; this "
                    "processed directory is from a set that drafts a different "
                    "pack size and cannot be loaded without generalising it."
                )

    def _reindex(self) -> None:
        """First row of each draft, for the pool-as-prefix slice."""
        self._draft_start = np.searchsorted(
            self.draft_idx, np.arange(self.n_drafts), side="left"
        )

    @property
    def size(self) -> int:
        return int(self.label.size)

    @classmethod
    def load(cls, processed_dir: str | Path, on_invalid: str = "drop") -> "PickData":
        processed_dir = Path(processed_dir)
        with np.load(processed_dir / "picks.npz") as handle:
            arrays = {name: handle[name] for name in handle.files}
        vocab = Vocabulary.load(processed_dir / "vocab.json")
        return cls(arrays, vocab, on_invalid=on_invalid)

    def _invalid_drafts(self) -> np.ndarray:
        """Drafts where the pool-as-prefix identity does not hold.

        Row i's distance from its draft's first row must equal the pool size
        implied by its own pack/pick numbers. A draft with a missing or
        duplicated pick in the export breaks that, and every pool derived
        from it afterwards would be silently wrong. Such drafts do occur:
        OTJ.PremierDraft contains at least one draft with fewer than the
        expected 42 rows.
        """
        if self.size == 0:
            return np.empty(0, dtype=np.int64)
        expected = (
            self.pack_number.astype(np.int64) * PICKS_PER_PACK
            + self.pick_number.astype(np.int64)
        )
        actual = np.arange(self.size, dtype=np.int64) - self._draft_start[self.draft_idx]
        bad = expected != actual
        # A draft can also be short without breaking the prefix identity
        # (truncated at the end), which still leaves an incomplete draft.
        counts = np.bincount(self.draft_idx, minlength=self.n_drafts)
        short = np.flatnonzero((counts > 0) & (counts != PICKS_PER_PACK * PACKS_PER_DRAFT))
        return np.union1d(np.unique(self.draft_idx[bad]), short)

    def pool_of(self, i: int) -> np.ndarray:
        """Card ids the drafter had taken before row i, unpadded."""
        return self.label[self._draft_start[self.draft_idx[i]] : i]

    def pools_padded(self, indices: np.ndarray) -> np.ndarray:
        """(len(indices), MAX_POOL_SIZE) int16, PAD_ID-padded, for a batch.

        Built by gathering, not by looping: each row's pool is a fixed
        offset back from its own position, so one broadcast gather covers
        the batch.
        """
        indices = np.asarray(indices, dtype=np.int64)
        starts = self._draft_start[self.draft_idx[indices]]
        lengths = indices - starts
        offsets = np.arange(MAX_POOL_SIZE, dtype=np.int64)[None, :]
        gather = starts[:, None] + offsets
        valid = offsets < lengths[:, None]
        out = np.where(valid, self.label[np.clip(gather, 0, self.size - 1)], PAD_ID)
        return out.astype(np.int16)

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Model inputs for a set of rows, as int32 numpy arrays.

        Ids stay -1-padded rather than being one-hot expanded; the model
        masks on `>= 0` and gathers features itself.
        """
        indices = np.asarray(indices, dtype=np.int64)
        return {
            "pack_ids": self.pack[indices].astype(np.int32),
            "pool_ids": self.pools_padded(indices).astype(np.int32),
            "pack_number": self.pack_number[indices].astype(np.int32),
            "pick_number": self.pick_number[indices].astype(np.int32),
            "label_pos": self.label_pos[indices].astype(np.int32),
            "label": self.label[indices].astype(np.int32),
        }

    def example(self, i: int) -> DraftExample:
        pack = self.pack[i]
        return DraftExample(
            pack=tuple(int(c) for c in pack[pack != PAD_ID]),
            pool=tuple(int(c) for c in self.pool_of(i)),
            pack_number=int(self.pack_number[i]),
            pick_number=int(self.pick_number[i]),
            label=int(self.label[i]),
        )


def _sorted_padded(values: np.ndarray) -> np.ndarray:
    """Sorts each row so that set-equal rows become byte-identical.

    PAD_ID is negative, so a plain ascending sort puts all padding first
    and keeps the real ids in a canonical order behind it.
    """
    return np.sort(values, axis=1)


def matched_state_groups(data: PickData, min_drafts: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Finds picks whose (pack, pool, pack_number, pick_number) state recurs
    across at least `min_drafts` distinct drafts.

    Returns (row_indices, group_ids) where group_ids labels which recurring
    state each returned row belongs to. This is the input to the human
    disagreement measurement in PROJECT_PLAN.md section 6.

    States can only collide within the same (pack_number, pick_number) --
    pool size is determined by those two -- so the comparison is bucketed
    by them. That keeps the temporary key matrix ~40x smaller than hashing
    the whole corpus at once.
    """
    if data.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    all_rows: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    next_group = 0

    bucket_key = (
        data.pack_number.astype(np.int64) * PICKS_PER_PACK + data.pick_number
    )
    for bucket in np.unique(bucket_key):
        rows = np.flatnonzero(bucket_key == bucket)
        if rows.size < min_drafts:
            continue

        pools = data.pools_padded(rows)
        keys = np.concatenate(
            [_sorted_padded(data.pack[rows]), _sorted_padded(pools)], axis=1
        )
        # A contiguous byte view turns whole-row equality into scalar
        # equality, which is what np.unique needs to group them.
        keys = np.ascontiguousarray(keys)
        view = keys.view([("", keys.dtype)] * keys.shape[1]).ravel()
        _, inverse, counts = np.unique(view, return_inverse=True, return_counts=True)

        # Recurrence must be across distinct drafts, not repeated rows of one.
        candidate = counts[inverse] >= min_drafts
        if not candidate.any():
            continue
        cand_rows = rows[candidate]
        cand_groups = inverse[candidate]

        order = np.argsort(cand_groups, kind="stable")
        cand_rows, cand_groups = cand_rows[order], cand_groups[order]
        boundaries = np.flatnonzero(
            np.concatenate([[True], cand_groups[1:] != cand_groups[:-1], [True]])
        )
        keep_rows: list[np.ndarray] = []
        keep_groups: list[np.ndarray] = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            block = cand_rows[start:stop]
            if np.unique(data.draft_idx[block]).size >= min_drafts:
                keep_rows.append(block)
                keep_groups.append(np.full(block.size, next_group, dtype=np.int64))
                next_group += 1
        if keep_rows:
            all_rows.append(np.concatenate(keep_rows))
            all_groups.append(np.concatenate(keep_groups))

    if not all_rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(all_rows), np.concatenate(all_groups)


def split_by_draft(
    data: PickData,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    matched_state_rows: np.ndarray | None = None,
    exclude_granularity: str = "draft",
) -> Splits:
    """Splits on draft_idx, not on individual rows.

    `matched_state_rows`, if given, is held out of all three splits first.
    At `exclude_granularity="draft"` the entire draft containing such a row
    is held out. That is stricter than dropping the row alone, and it is
    the right default: a later pick from the same draft carries the earlier
    pick's label inside its pool, so row-level exclusion would leak the
    held-out answer into training through the pool of a neighbouring row.
    Use "row" only if draft-level exclusion costs too much of the corpus,
    and say so in the writeup if you do.
    """
    if exclude_granularity not in ("draft", "row"):
        raise ValueError(f"exclude_granularity must be 'draft' or 'row', got {exclude_granularity!r}")

    held = np.zeros(data.size, dtype=bool)
    if matched_state_rows is not None and matched_state_rows.size:
        held[matched_state_rows] = True
        if exclude_granularity == "draft":
            held = np.isin(data.draft_idx, np.unique(data.draft_idx[matched_state_rows]))

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(data.n_drafts)
    n_val = int(round(val_frac * data.n_drafts))
    n_test = int(round(test_frac * data.n_drafts))
    assignment = np.empty(data.n_drafts, dtype=np.int8)
    assignment[shuffled[:n_val]] = 1
    assignment[shuffled[n_val : n_val + n_test]] = 2
    assignment[shuffled[n_val + n_test :]] = 0

    row_split = assignment[data.draft_idx]
    return Splits(
        train=np.flatnonzero((row_split == 0) & ~held),
        val=np.flatnonzero((row_split == 1) & ~held),
        test=np.flatnonzero((row_split == 2) & ~held),
        matched_state=(
            np.sort(matched_state_rows)
            if matched_state_rows is not None
            else np.empty(0, dtype=np.int64)
        ),
    )


def decision_rows(data: PickData, indices: np.ndarray) -> np.ndarray:
    """`indices` with the forced picks removed.

    A pack holding one card admits one answer, so its cross-entropy is
    identically zero for *every* parameter value and its gradient is
    identically zero with it. Those rows cannot teach the model anything;
    they only occupy a slot in a batch. On FIN that is 336,567 of the
    4,711,938 training rows -- 7.14%, one row in fourteen, because each of
    the three packs ends in exactly one forced pick.

    This is for the training stream only. Evaluation keeps them: they are
    part of the task, `evaluate_by_pick` reports them separately, and
    dropping them there would change what the reported loss means. See
    docs/RESULTS.md on why the two populations have to stay distinguishable.
    """
    return indices[data.pack_size[indices] > 1]
