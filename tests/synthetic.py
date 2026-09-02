"""Synthetic 17lands exports and feature tables for tests.

Tests run against generated data rather than the real 200MB export, so the
suite needs no network, no download, and no ingested corpus. The generator
mirrors the real format's awkward parts deliberately -- card names with
commas, `pack_card_*` as counts rather than flags, the redundant `pool_*`
columns -- because those are exactly the things a parser gets wrong.

Kept separate from conftest.py so it can be imported explicitly: a test
that builds a deliberately malformed export wants the builder, not a
fixture that has already made choices for it.
"""

from __future__ import annotations

import csv
import gzip

import numpy as np

PICKS_PER_PACK = 14
PACKS_PER_DRAFT = 3
FEATURE_DIM = 65

# One name carries a comma, as real cards do ("Zidane, Tantalus Thief").
CARDS = sorted([f"Card {i:02d}" for i in range(20)] + ["Zidane, Tantalus Thief"])

COLUMNS = (
    ["expansion", "event_type", "draft_id", "draft_time", "rank",
     "event_match_wins", "event_match_losses", "pack_number", "pick_number",
     "pick", "pick_maindeck_rate", "pick_sideboard_in_rate"]
    + [f"pack_card_{c}" for c in CARDS]
    + [f"pool_{c}" for c in CARDS]
    + ["user_n_games_bucket", "user_game_win_rate_bucket"]
)


def write_export(path, drafts, gzipped: bool = True, drop_columns=()) -> None:
    """Writes a format-faithful draft_data_public CSV.

    `drafts` is a list of (draft_id, packs), where packs holds
    PACKS_PER_DRAFT lists of card names in the order that drafter took them.

    `drop_columns` omits named metadata columns, because the real exports
    disagree about which ones they carry: AFR.PremierDraft has no `rank`
    and no `user_game_win_rate_bucket` at all.
    """
    drop = set(drop_columns)
    keep = [i for i, c in enumerate(COLUMNS) if c not in drop]

    opener = gzip.open if gzipped else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        raw = csv.writer(handle)

        class _RowWriter:
            def writerow(self, row):
                raw.writerow(row if not drop else [row[i] for i in keep])

        writer = _RowWriter()
        writer.writerow(COLUMNS)
        for draft_id, packs in drafts:
            pool: list[str] = []
            for pack_number, taken in enumerate(packs):
                for pick_number in range(len(taken)):
                    pack_counts = {c: 0 for c in CARDS}
                    for card in taken[pick_number:]:
                        pack_counts[card] += 1
                    pool_counts = {c: 0 for c in CARDS}
                    for card in pool:
                        pool_counts[card] += 1
                    writer.writerow(
                        ["TST", "PremierDraft", draft_id, "2025-01-01 00:00:00",
                         "gold", 3, 2, pack_number, pick_number,
                         taken[pick_number], 1.0, 0.0]
                        + [pack_counts[c] for c in CARDS]
                        + [pool_counts[c] for c in CARDS]
                        + [3, 0.55]
                    )
                    pool.append(taken[pick_number])


def make_draft(
    rng,
    draft_id: str,
    picks_per_pack: int = PICKS_PER_PACK,
    packs_per_draft: int = PACKS_PER_DRAFT,
):
    """One well-formed draft: `packs_per_draft` packs of distinct cards.

    The geometry is a parameter rather than a constant because the ingest
    pass is supposed to measure it instead of assuming Arena's usual 3x14
    -- see src/data/ingest.py::PackGeometry. A test that can only build
    3x14 exports cannot tell a working detector from a hardcoded 14.
    """
    packs = [
        list(rng.choice(CARDS, size=picks_per_pack, replace=False))
        for _ in range(packs_per_draft)
    ]
    return (draft_id, packs)


def make_drafts(
    rng,
    count: int,
    prefix: str = "draft",
    picks_per_pack: int = PICKS_PER_PACK,
    packs_per_draft: int = PACKS_PER_DRAFT,
):
    return [
        make_draft(rng, f"{prefix}{i:04d}", picks_per_pack, packs_per_draft)
        for i in range(count)
    ]


def make_matched_pair(rng):
    """Two drafts sharing an opening pack and their first three picks.

    The only way to get a genuine (pack, pool) recurrence across distinct
    drafts on purpose, which is what the matched-state grouping needs to be
    tested against.
    """
    _, packs_a = make_draft(rng, "a")
    packs_b = [list(p) for p in packs_a]
    packs_b[0][3:] = list(reversed(packs_a[0][3:]))
    packs_b[1] = list(rng.choice(CARDS, size=PICKS_PER_PACK, replace=False))
    packs_b[2] = list(rng.choice(CARDS, size=PICKS_PER_PACK, replace=False))
    return [("draftaaa", packs_a), ("draftbbb", packs_b)]


def make_feature_table(vocab_size: int = 40, seed: int = 0) -> np.ndarray:
    """A stand-in for CardFeatures.dense() with the same width."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(vocab_size, FEATURE_DIM)).astype(np.float32)


def make_model_batch(rng, batch_size=6, vocab_size=40, pack_sizes=None, pool_sizes=None):
    """Padded pack/pool id arrays plus the scalar features, as the model takes them."""
    pack = np.full((batch_size, 14), -1, dtype=np.int32)
    pool = np.full((batch_size, 41), -1, dtype=np.int32)
    pack_sizes = pack_sizes if pack_sizes is not None else rng.integers(1, 15, batch_size)
    pool_sizes = pool_sizes if pool_sizes is not None else rng.integers(0, 42, batch_size)
    for i in range(batch_size):
        pack[i, : pack_sizes[i]] = rng.integers(0, vocab_size, size=pack_sizes[i])
        if pool_sizes[i]:
            pool[i, : pool_sizes[i]] = rng.integers(0, vocab_size, size=pool_sizes[i])
    return {
        "pack_ids": pack,
        "pool_ids": pool,
        "pack_number": rng.integers(0, 3, size=batch_size).astype(np.int32),
        "pick_number": rng.integers(0, 14, size=batch_size).astype(np.int32),
    }
