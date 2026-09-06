"""Concatenates several ingested sets into one corpus over a shared card index.

Everything else in this project is single-set: `run.py` takes one
`--processed-dir`, and each processed directory carries its own
`Vocabulary` whose ids start at 0. That is fine for the scaling grid, where
every cell trains and evaluates on the same set, and useless for the claim
the project is actually built on -- that a drafter trained on nine sets can
draft a tenth it has never seen. This module is the plumbing for that.

Why the transfer is architecturally free
----------------------------------------
`CardEmbedding` (src/models/embeddings.py) is a two-layer MLP over the
feature block from `CardFeatures.dense()`, not a per-card lookup table, and
`PointerHead` scores *pack slots* rather than vocabulary entries. So there
is no parameter anywhere in `PickModel` whose shape depends on the
vocabulary, and none whose meaning is tied to one set's ids. A model
trained on nine sets can score a tenth set's cards with no new parameters
at all. The only thing missing was a corpus that spans sets, which is what
this file builds.

Reprints: one global id per distinct card NAME
-----------------------------------------------
87 of the 3,010 distinct cards across the ten ingested sets are printed in
more than one of them (five -- the basic lands -- in all ten). They get a
single global id.

That is the right call here, and not merely the convenient one:

  - It is invisible to the model either way. `CardEmbedding` is a pure
    function of the feature row, so two ids carrying identical feature rows
    cannot be told apart in the forward pass. Splitting a reprint into
    per-set ids would be a distinction with no representation behind it.

  - The feature rows genuinely are identical. `card_features.py` fetches
    Scryfall by card *name*, not by (name, set), so every attribute it
    records -- including rarity, which is a property of a printing -- comes
    from one canonical printing and is therefore printing-independent by
    construction. `build_card_index` re-checks this rather than assuming
    it: if any reprint's feature rows disagree between sets it raises and
    names the differing columns, because that would mean the two sets are
    describing different cards under one name and merging them would be a
    silent corruption. Measured on the ten sets in data/processed: zero
    disagreements.

  - It keeps the zero-shot claim honest. A card in the held-out set that
    was also printed in a training set really is the same card, and the
    model really has seen it. Merging makes that overlap a number that can
    be reported (`CardIndex.overlap`) instead of hiding it behind an id
    space that pretends every set is disjoint. The held-out evaluation is
    "a set the model has never drafted", not "cards the model has never
    seen", and the two are different claims.

Mixed pack geometry
-------------------
Four of the ten sets are not 3x14: BLB and EOE draft 13 cards a pack, LCI
and SIR 15. `PackGeometry` already carries this per set, and this module
concatenates rather than flattens it:

  - the concatenated `pack` matrix is padded on the right to the widest
    pack across the sets, with PAD_ID, so a 13-card set's rows keep
    `label_pos` unchanged and the pad columns mask out;
  - `pools_padded` uses the ENVELOPE geometry's `max_pool_size`, so a
    3x13 set's rows simply leave the last six pool columns padded;
  - the pool-as-prefix identity is checked per set against that set's own
    `picks_per_pack` (`_invalid_drafts` below), never against the
    envelope. Checking it against the envelope would flag every row of
    every non-3x14 set and the loader would throw the corpus away.

The envelope is also what sizes `ContextFeatures`: `picks_per_pack` becomes
the maximum across the corpus, so pick 14 has an embedding that only LCI
and SIR ever train. A held-out 3x14 set never indexes it.

Splits come from the single-set loader
--------------------------------------
`load_multiset` splits each set with `split_by_draft` on that set's own
`PickData` *before* concatenating, then offsets the row indices into the
merged frame. That is deliberate: it makes the held-out set's val split
bit-identical to the split a single-set run of the same seed would draw, so
a zero-shot number is comparable to the same-set number on the same rows.

See docs/PROJECT_PLAN.md section 10 step 6, and src/training/transfer.py
for the entry point that runs the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .card_features import CardFeatures
from .dataset import PAD_ID, PickData, Splits, split_by_draft
from .ingest import PackGeometry
from .vocab import Vocabulary


def set_code(name: str) -> str:
    """"FIN.PremierDraft" -> "FIN". The bit anyone actually says out loud."""
    return name.split(".")[0]


@dataclass(frozen=True)
class CardIndex:
    """One id space shared by every set in a multi-set corpus.

    Ids are positions in `id_to_card`, which is the alphabetically sorted
    union of the member sets' vocabularies. `features` is the (V, F) table
    those ids index, assembled from the per-set tables and checked to agree
    wherever they overlap.
    """

    id_to_card: tuple[str, ...]
    card_to_id: dict[str, int]
    features: np.ndarray               # (V, F) float32
    column_names: tuple[str, ...]      # F labels, identical in every member set
    printings: dict[str, tuple[str, ...]]   # card -> the sets that print it

    @property
    def size(self) -> int:
        return len(self.id_to_card)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def remap_for(self, vocab: Vocabulary) -> np.ndarray:
        """(V_set,) array taking one set's ids to global ids.

        Indexing it with a padded id array is wrong -- PAD_ID is -1 and
        would wrap to the last card -- so callers go through `remap_ids`.
        """
        return np.array(
            [self.card_to_id[name] for name in vocab.id_to_card], dtype=np.int64
        )

    def as_vocabulary(self, expansion: str = "", event_type: str = "") -> Vocabulary:
        return Vocabulary(
            card_to_id=dict(self.card_to_id),
            id_to_card=self.id_to_card,
            expansion=expansion,
            event_type=event_type,
        )

    def reprints(self) -> tuple[str, ...]:
        """Cards printed in more than one member set."""
        return tuple(
            name for name in self.id_to_card if len(self.printings[name]) > 1
        )

    def overlap(self, held_out: str, training: Sequence[str]) -> dict:
        """How much of `held_out`'s card pool the training sets already print.

        The number that keeps the zero-shot claim honest: the held-out
        *set* is unseen, but some of its cards are not, and this says how
        many.
        """
        train = set(training)
        pool = [n for n in self.id_to_card if held_out in self.printings[n]]
        seen = [n for n in pool if train & set(self.printings[n])]
        return {
            "held_out_cards": len(pool),
            "cards_also_in_training_sets": len(seen),
            "fraction_seen": len(seen) / max(len(pool), 1),
            "examples": sorted(seen)[:10],
        }


def remap_ids(ids: np.ndarray, remap: np.ndarray, dtype=np.int16) -> np.ndarray:
    """Applies a per-set -> global id map, leaving PAD_ID as PAD_ID."""
    valid = ids >= 0
    mapped = remap[np.where(valid, ids, 0)]
    return np.where(valid, mapped, PAD_ID).astype(dtype)


def build_card_index(
    parts: Sequence[tuple[str, Vocabulary, CardFeatures]]
) -> CardIndex:
    """Unions several sets' vocabularies into one id space and one table.

    Raises rather than reconciling on every disagreement it can detect.
    Each of these would otherwise be a silent corruption:

      - a feature table of a different width, or the same width with
        different columns. Column k has to be the same attribute in every
        set or the concatenated table is meaningless (see
        card_features.py, "Every column means the same thing in every
        set");
      - a features file whose card order does not match its set's
        vocabulary, which would misattribute every row;
      - a reprint whose feature rows differ between sets, which means the
        two sets do not agree on what the card is.
    """
    if not parts:
        raise ValueError("build_card_index needs at least one set")

    base_name, _, base_features = parts[0]
    base_columns = base_features.column_names()
    base_width = base_features.dense().shape[1]

    for name, vocab, features in parts:
        if tuple(features.card_names) != tuple(vocab.id_to_card):
            raise ValueError(
                f"{name}: card_features.npz lists {len(features.card_names)} cards "
                f"in a different order from vocab.json's {vocab.size}. Every row of "
                "the feature table would be attributed to the wrong card. Rerun "
                "`python -m src.data.card_features --processed-dir <dir>`."
            )
        width = features.dense().shape[1]
        if width != base_width:
            raise ValueError(
                f"feature tables have different widths: {base_name} has "
                f"{base_width} columns, {name} has {width}. These cannot be "
                "concatenated -- rebuild both with the same card_features.py."
            )
        columns = features.column_names()
        if columns != base_columns:
            differing = [
                (i, a, b)
                for i, (a, b) in enumerate(zip(base_columns, columns))
                if a != b
            ]
            first = differing[0] if differing else None
            raise ValueError(
                f"feature tables are the same width but not the same columns: "
                f"{base_name} vs {name} first differ at column "
                f"{first[0]} ({first[1]!r} vs {first[2]!r}); "
                f"{len(differing)} columns differ in all. Column k must mean the "
                "same attribute in every set."
            )

    printings: dict[str, list[str]] = {}
    rows: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    disagreements: list[str] = []

    for name, _, features in parts:
        dense = features.dense()
        for i, card in enumerate(features.card_names):
            row = dense[i]
            if card in rows:
                if not np.array_equal(rows[card], row):
                    differing = np.flatnonzero(rows[card] != row)
                    disagreements.append(
                        f"{card!r} differs between {sources[card]} and {name} "
                        f"in {[base_columns[c] for c in differing[:6]]}"
                    )
                printings[card].append(name)
                continue
            rows[card] = row
            sources[card] = name
            printings[card] = [name]

    if disagreements:
        raise ValueError(
            "reprinted cards have different features in different sets, so they "
            "cannot share a global id:\n  "
            + "\n  ".join(disagreements[:10])
            + (
                f"\n  ... and {len(disagreements) - 10} more"
                if len(disagreements) > 10
                else ""
            )
            + "\nThis should be impossible: card_features.py fetches Scryfall by "
            "card name, not by printing. Rebuild the feature tables before "
            "training on the union."
        )

    id_to_card = tuple(sorted(rows))
    return CardIndex(
        id_to_card=id_to_card,
        card_to_id={name: i for i, name in enumerate(id_to_card)},
        features=np.stack([rows[name] for name in id_to_card]).astype(np.float32),
        column_names=base_columns,
        printings={name: tuple(printings[name]) for name in id_to_card},
    )


@dataclass(frozen=True)
class SetSlice:
    """Where one set's rows and drafts live in the concatenated corpus."""

    name: str
    position: int
    row_start: int
    row_stop: int
    draft_offset: int
    n_drafts: int
    geometry: PackGeometry
    vocab_size: int
    rank_names: tuple[str, ...]

    @property
    def code(self) -> str:
        return set_code(self.name)

    @property
    def rows(self) -> int:
        return self.row_stop - self.row_start

    def to_global(self, local_indices: np.ndarray) -> np.ndarray:
        """Row indices in this set's own frame -> the concatenated frame."""
        return np.asarray(local_indices, dtype=np.int64) + self.row_start


class MultiSetPickData(PickData):
    """Several sets' picks in one array, over a shared card index.

    Deliberately a `PickData` by inheritance: `train_model`,
    `evaluate_by_pick`, `frequency_baseline` and `decision_rows` all take
    one, and none of them should have to know whether the corpus spans sets.
    Everything they touch -- `batch`, `pools_padded`, `pack_size`, `vocab`,
    `geometry` -- means the same thing here.

    `PickData.__init__` is NOT called. It sorts, infers one geometry, and
    validates the pool-as-prefix identity against it; on a mixed-geometry
    corpus that last step condemns every row of every set that is not the
    modal shape, and the default `on_invalid="drop"` would then quietly
    return an empty corpus. The parts are individually loaded, sorted and
    validated already, so this constructor concatenates them in order and
    re-checks the identity per set instead.
    """

    def __init__(
        self,
        parts: Sequence[tuple[str, PickData]],
        card_index: CardIndex,
    ):
        if not parts:
            raise ValueError("MultiSetPickData needs at least one set")

        total_rows = sum(part.size for part in (p for _, p in parts))
        pack_width = max(int(part.pack.shape[1]) for _, part in parts)
        # The envelope: widest of everything, so no member set's rows are
        # truncated and every member's ContextFeatures index is in range.
        geometry = PackGeometry(
            packs_per_draft=max(part.geometry.packs_per_draft for _, part in parts),
            picks_per_pack=max(part.geometry.picks_per_pack for _, part in parts),
            max_pack_size=max(
                max(part.geometry.max_pack_size for _, part in parts), pack_width
            ),
        )

        id_dtype = np.int16 if card_index.size <= np.iinfo(np.int16).max else np.int32
        self.pack = np.full((total_rows, pack_width), PAD_ID, dtype=id_dtype)
        self.pack_size = np.empty(total_rows, dtype=np.int8)
        self.label = np.empty(total_rows, dtype=id_dtype)
        self.label_pos = np.empty(total_rows, dtype=np.int8)
        self.pack_number = np.empty(total_rows, dtype=np.int8)
        self.pick_number = np.empty(total_rows, dtype=np.int8)
        self.draft_idx = np.empty(total_rows, dtype=np.int32)
        self.rank_code = np.empty(total_rows, dtype=np.int8)
        self.win_rate_bucket = np.empty(total_rows, dtype=np.float32)
        self.set_idx = np.empty(total_rows, dtype=np.int8)

        slices: list[SetSlice] = []
        draft_ids: list[np.ndarray] = []
        row_cursor = 0
        draft_cursor = 0

        for position, (name, part) in enumerate(parts):
            start, stop = row_cursor, row_cursor + part.size
            remap = card_index.remap_for(part.vocab)
            width = int(part.pack.shape[1])

            self.pack[start:stop, :width] = remap_ids(part.pack, remap, id_dtype)
            self.label[start:stop] = remap_ids(part.label, remap, id_dtype)
            self.pack_size[start:stop] = part.pack_size
            self.label_pos[start:stop] = part.label_pos
            self.pack_number[start:stop] = part.pack_number
            self.pick_number[start:stop] = part.pick_number
            self.draft_idx[start:stop] = part.draft_idx.astype(np.int64) + draft_cursor
            self.rank_code[start:stop] = part.rank_code
            self.win_rate_bucket[start:stop] = part.win_rate_bucket
            self.set_idx[start:stop] = position

            # Prefixed so a row can be traced back to its export: draft ids
            # are only unique within a set's own file.
            code = set_code(name)
            draft_ids.append(
                np.char.add(f"{code}:", part.draft_ids.astype(str))
                if part.draft_ids.size
                else np.empty(0, dtype="U40")
            )

            slices.append(
                SetSlice(
                    name=name,
                    position=position,
                    row_start=start,
                    row_stop=stop,
                    draft_offset=draft_cursor,
                    n_drafts=part.n_drafts,
                    geometry=part.geometry,
                    vocab_size=part.vocab.size,
                    rank_names=tuple(np.asarray(part.rank_names).tolist()),
                )
            )
            row_cursor = stop
            draft_cursor += part.n_drafts

        self.draft_ids = (
            np.concatenate(draft_ids) if draft_ids else np.empty(0, dtype="U40")
        )
        # Rank codes index each SET's own rank_names, which differ between
        # exports (AFR has no rank column at all). There is no corpus-wide
        # table they all index, so there is none here; read them off the
        # SetSlice if you ever need them. Nothing in this project does.
        self.rank_names = np.empty(0, dtype="U16")

        self.sets = tuple(slices)
        self.card_index = card_index
        self.vocab = card_index.as_vocabulary(
            expansion="+".join(s.code for s in slices), event_type="PremierDraft"
        )
        self.geometry = geometry
        self.n_drafts = draft_cursor
        self.dropped_drafts = sum(part.dropped_drafts for _, part in parts)
        self.dropped_rows = sum(part.dropped_rows for _, part in parts)

        # Per-row picks_per_pack, so the prefix identity is checked against
        # each row's OWN geometry rather than the envelope's.
        self._row_picks_per_pack = np.array(
            [s.geometry.picks_per_pack for s in slices], dtype=np.int64
        )[self.set_idx]
        self._row_picks_per_draft = np.array(
            [s.geometry.picks_per_draft for s in slices], dtype=np.int64
        )[self.set_idx]

        self._reindex()

        bad = self._invalid_drafts()
        if bad.size:
            raise AssertionError(
                f"{bad.size} drafts violate the pool-as-prefix identity after "
                f"concatenation (first is {self.draft_ids[bad[0]]}), although "
                "every part validated on its own. The concatenation reordered "
                "rows or mis-offset a draft index."
            )

    # -- geometry-aware overrides -------------------------------------------

    def _invalid_drafts(self) -> np.ndarray:
        """As `PickData._invalid_drafts`, but per set rather than per corpus.

        The base implementation multiplies `pack_number` by one
        `picks_per_pack` for the whole corpus. Here that constant is a
        per-row quantity: a BLB row's pool after pack 1 pick 0 is 13 cards,
        an LCI row's is 15. Using the envelope's 15 for both marks every
        BLB and EOE row invalid.
        """
        if self.size == 0:
            return np.empty(0, dtype=np.int64)
        expected = (
            self.pack_number.astype(np.int64) * self._row_picks_per_pack
            + self.pick_number.astype(np.int64)
        )
        actual = np.arange(self.size, dtype=np.int64) - self._draft_start[self.draft_idx]
        bad = expected != actual

        counts = np.bincount(self.draft_idx, minlength=self.n_drafts)
        # A draft's expected length is its own set's, and an absent draft
        # (dropped by its part's loader) is expected to have no rows at all.
        per_draft_expected = np.zeros(self.n_drafts, dtype=np.int64)
        for s in self.sets:
            span = slice(s.draft_offset, s.draft_offset + s.n_drafts)
            per_draft_expected[span] = s.geometry.picks_per_draft
        short = np.flatnonzero((counts > 0) & (counts != per_draft_expected))
        return np.union1d(np.unique(self.draft_idx[bad]), short)

    # -- convenience ---------------------------------------------------------

    def slice_of(self, name: str) -> SetSlice:
        for s in self.sets:
            if s.name == name or s.code == set_code(name):
                return s
        raise KeyError(
            f"{name!r} is not in this corpus; it holds "
            f"{[s.code for s in self.sets]}"
        )

    def rows_of(self, name: str) -> np.ndarray:
        s = self.slice_of(name)
        return np.arange(s.row_start, s.row_stop, dtype=np.int64)

    def describe(self) -> str:
        lines = [
            f"{'set':<6} {'geometry':<34} {'cards':>6} {'drafts':>9} {'rows':>11}"
        ]
        for s in self.sets:
            lines.append(
                f"{s.code:<6} {s.geometry.describe():<34} {s.vocab_size:>6} "
                f"{s.n_drafts:>9,} {s.rows:>11,}"
            )
        lines.append(
            f"{'ALL':<6} {self.geometry.describe():<34} {self.vocab.size:>6} "
            f"{self.n_drafts:>9,} {self.size:>11,}"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class MultiSetCorpus:
    """A concatenated corpus plus the per-set splits, in global row indices."""

    data: MultiSetPickData
    card_index: CardIndex
    splits: dict[str, Splits]
    split_seed: int
    # (set name, why) for directories `load_multiset` was asked to include
    # and could not. Empty unless skip_unloadable was set.
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def feature_table(self) -> np.ndarray:
        return self.card_index.features

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.data.sets)

    def rows(self, names: Iterable[str], which: str = "train") -> np.ndarray:
        """Concatenated `which` split over the named sets, in global indices."""
        chunks = [getattr(self.splits[n], which) for n in names]
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.concatenate(chunks))


def processed_dirs(root: str | Path, names: Sequence[str] | None = None) -> list[Path]:
    """The processed directories under `root`, resolved from set codes.

    Accepts either full directory names ("FIN.PremierDraft") or bare set
    codes ("FIN"), because nobody types the event suffix.
    """
    root = Path(root)
    available = sorted(p for p in root.iterdir() if (p / "picks.npz").exists())
    if names is None:
        return available
    by_code = {set_code(p.name): p for p in available}
    by_name = {p.name: p for p in available}
    resolved = []
    for name in names:
        path = by_name.get(name) or by_code.get(set_code(name))
        if path is None:
            raise FileNotFoundError(
                f"no ingested set {name!r} under {root} "
                f"(have {sorted(by_code)})"
            )
        resolved.append(path)
    return resolved


def load_multiset(
    dirs: Sequence[str | Path],
    split_seed: int = 0,
    on_invalid: str = "drop",
    verbose: bool = True,
    skip_unloadable: bool = False,
) -> MultiSetCorpus:
    """Loads several processed directories into one corpus.

    Each set is loaded, split and validated on its own first, exactly as a
    single-set run would; only then are the arrays concatenated and the ids
    remapped. That ordering is what makes a held-out set's val split the
    same rows a single-set run of the same seed would evaluate on.

    `skip_unloadable` is for callers that pass "everything ingested" rather
    than a chosen list -- AFR is ingested and cannot be loaded, because its
    export omits every draft's first pick (docs/DATA.md). It skips such a
    set instead of failing, but it does not do it quietly: the reason is
    printed and carried on the corpus as `skipped`. A set named explicitly
    still fails explicitly, which is the point of the flag being off by
    default.
    """
    dirs = [Path(d) for d in dirs]
    parts: list[tuple[str, PickData]] = []
    index_inputs: list[tuple[str, Vocabulary, CardFeatures]] = []
    skipped: list[tuple[str, str]] = []

    for path in dirs:
        try:
            part = PickData.load(path, on_invalid=on_invalid)
        except ValueError as error:
            if not skip_unloadable:
                raise
            skipped.append((path.name, str(error)))
            print(
                f"  SKIPPED {set_code(path.name):<4} it does not load: {error}",
                flush=True,
            )
            continue
        features = CardFeatures.load(path / "card_features.npz")
        parts.append((path.name, part))
        index_inputs.append((path.name, part.vocab, features))
        if verbose:
            print(
                f"  loaded {set_code(path.name):<4} {part.size:>10,} rows  "
                f"{part.vocab.size:>4} cards  {part.geometry.describe()}",
                flush=True,
            )

    card_index = build_card_index(index_inputs)
    splits: dict[str, Splits] = {}
    for (name, part), offset in zip(parts, _row_offsets(parts)):
        local = split_by_draft(part, seed=split_seed)
        splits[name] = Splits(
            train=local.train + offset,
            val=local.val + offset,
            test=local.test + offset,
            matched_state=np.empty(0, dtype=np.int64),
        )
    data = MultiSetPickData(parts, card_index)
    return MultiSetCorpus(
        data=data,
        card_index=card_index,
        splits=splits,
        split_seed=split_seed,
        skipped=tuple(skipped),
    )


def _row_offsets(parts: Sequence[tuple[str, PickData]]) -> list[int]:
    offsets, cursor = [], 0
    for _, part in parts:
        offsets.append(cursor)
        cursor += part.size
    return offsets
