"""Tests for the data pipeline.

These run against a synthetic export written to a temp directory rather
than against a real 17lands file, so they need no network and no 200MB
download. The synthetic writer mirrors the real format's awkward parts on
purpose: card names containing commas, `pack_card_*` as counts rather than
flags, and the redundant `pool_*` columns that ingest is supposed to ignore
and dataset is supposed to be able to reconstruct.
"""

from __future__ import annotations

import csv
import gzip

import numpy as np
import pytest

from src.data.dataset import (
    PACKS_PER_DRAFT,
    PICKS_PER_PACK,
    MAX_POOL_SIZE,
    PickData,
    matched_state_groups,
    split_by_draft,
)
from src.data.ingest import ingest
from src.data.vocab import build_vocabulary

# One name carries a comma, as real cards do ("Zidane, Tantalus Thief").
CARDS = [f"Card {i:02d}" for i in range(20)] + ["Zidane, Tantalus Thief"]
CARDS = sorted(CARDS)


def _write_export(path, drafts, gzipped=True):
    """Writes a minimal but format-faithful draft_data_public CSV.

    `drafts` is a list of (draft_id, packs) where packs is a list of
    PACKS_PER_DRAFT lists, each holding PICKS_PER_PACK card names in the
    order that drafter took them.
    """
    columns = (
        ["expansion", "event_type", "draft_id", "draft_time", "rank",
         "event_match_wins", "event_match_losses", "pack_number", "pick_number",
         "pick", "pick_maindeck_rate", "pick_sideboard_in_rate"]
        + [f"pack_card_{c}" for c in CARDS]
        + [f"pool_{c}" for c in CARDS]
        + ["user_n_games_bucket", "user_game_win_rate_bucket"]
    )
    opener = gzip.open if gzipped else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for draft_id, packs in drafts:
            pool: list[str] = []
            for pack_number, taken in enumerate(packs):
                for pick_number in range(len(taken)):
                    remaining = taken[pick_number:]
                    pack_counts = {c: 0 for c in CARDS}
                    for c in remaining:
                        pack_counts[c] += 1
                    pool_counts = {c: 0 for c in CARDS}
                    for c in pool:
                        pool_counts[c] += 1
                    writer.writerow(
                        ["TST", "PremierDraft", draft_id, "2025-01-01 00:00:00",
                         "gold", 3, 2, pack_number, pick_number, taken[pick_number],
                         1.0, 0.0]
                        + [pack_counts[c] for c in CARDS]
                        + [pool_counts[c] for c in CARDS]
                        + [3, 0.55]
                    )
                    pool.append(taken[pick_number])


def _draft(rng, draft_id):
    packs = [list(rng.choice(CARDS, size=PICKS_PER_PACK, replace=False))
             for _ in range(PACKS_PER_DRAFT)]
    return (draft_id, packs)


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.default_rng(0)
    drafts = [_draft(rng, f"draft{i:04d}") for i in range(40)]
    csv_path = tmp_path / "draft_data_public.TST.PremierDraft.csv.gz"
    _write_export(csv_path, drafts)
    out = tmp_path / "processed"
    stats = ingest(csv_path, out, verbose=False)
    return csv_path, out, stats, drafts


def test_vocabulary_handles_commas_in_card_names(corpus):
    csv_path, _, _, _ = corpus
    vocab = build_vocabulary(csv_path)
    assert vocab.size == len(CARDS)
    assert "Zidane, Tantalus Thief" in vocab.card_to_id
    # Ids must be alphabetical and stable across rebuilds.
    assert list(vocab.id_to_card) == sorted(CARDS)
    assert build_vocabulary(csv_path).id_to_card == vocab.id_to_card
    # Column position must equal card id, which ingest relies on.
    assert vocab.pack_columns()[vocab.id_of("Zidane, Tantalus Thief")] == (
        "pack_card_Zidane, Tantalus Thief"
    )


def test_ingest_counts_and_label_position(corpus):
    _, out, stats, drafts = corpus
    assert stats.rows == len(drafts) * PACKS_PER_DRAFT * PICKS_PER_PACK
    assert stats.drafts == len(drafts)
    assert stats.max_pack_seen == PICKS_PER_PACK

    data = PickData.load(out)
    assert data.dropped_drafts == 0
    # The label is always a card in the pack, at the recorded position.
    assert (data.pack[np.arange(data.size), data.label_pos] == data.label).all()
    assert ((data.pack >= 0).sum(axis=1) == data.pack_size).all()
    # Pack shrinks by one card per pick within a round.
    assert (data.pack_size == PICKS_PER_PACK - data.pick_number).all()


def test_ingest_preserves_duplicate_cards_in_a_pack(tmp_path):
    """`pack_card_*` values are counts; OTJ really does contain a pack with
    a card at count 2. A card at count k must appear k times in the pack.
    """
    rng = np.random.default_rng(1)
    draft_id, packs = _draft(rng, "dupe0000")
    # Force a duplicate: the drafter takes the same card twice in one round.
    packs[0][1] = packs[0][0]
    csv_path = tmp_path / "dupe.csv.gz"
    _write_export(csv_path, [(draft_id, packs)])
    stats = ingest(csv_path, tmp_path / "processed", verbose=False)
    assert stats.duplicate_card_rows > 0

    data = PickData.load(tmp_path / "processed")
    first = data.pack[0]
    dup_id = data.vocab.id_of(packs[0][0])
    assert (first == dup_id).sum() == 2


def test_pool_reconstruction_matches_the_raw_pool_columns(corpus):
    """The pool_* columns are dropped at ingest on the claim that the pool
    is exactly the prefix of earlier picks. That claim is the load-bearing
    one for the on-disk size, so it is checked against the raw columns.
    """
    csv_path, out, _, _ = corpus
    data = PickData.load(out)

    with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        index = {c: i for i, c in enumerate(columns)}
        raw = {}
        for row in reader:
            key = (row[index["draft_id"]], int(row[index["pack_number"]]),
                   int(row[index["pick_number"]]))
            raw[key] = row

    pools = data.pools_padded(np.arange(data.size))
    assert pools.shape == (data.size, MAX_POOL_SIZE)
    for i in range(data.size):
        key = (str(data.draft_ids[data.draft_idx[i]]),
               int(data.pack_number[i]), int(data.pick_number[i]))
        row = raw[key]
        expected = sorted(
            data.vocab.id_of(c)
            for c in CARDS
            for _ in range(int(row[index[f"pool_{c}"]]))
        )
        assert sorted(int(x) for x in pools[i] if x >= 0) == expected


def test_incomplete_drafts_are_dropped_not_silently_misaligned(tmp_path):
    """A short draft would shift every later pool by the missing picks."""
    rng = np.random.default_rng(2)
    good = _draft(rng, "good0000")
    short_id, short_packs = _draft(rng, "short000")
    short_packs[2] = short_packs[2][:5]  # truncated final round
    csv_path = tmp_path / "short.csv.gz"
    _write_export(csv_path, [good, (short_id, short_packs)])
    ingest(csv_path, tmp_path / "processed", verbose=False)

    data = PickData.load(tmp_path / "processed")
    assert data.dropped_drafts == 1
    # The whole short draft goes, not just its missing picks: its surviving
    # rows are the ones whose pools would be wrong.
    assert data.dropped_rows == 2 * PICKS_PER_PACK + 5
    assert data.size == PACKS_PER_DRAFT * PICKS_PER_PACK

    with pytest.raises(ValueError, match="pool-as-prefix"):
        PickData.load(tmp_path / "processed", on_invalid="raise")


def test_split_keeps_every_pick_of_a_draft_together(corpus):
    _, out, _, _ = corpus
    data = PickData.load(out)
    splits = split_by_draft(data, val_frac=0.2, test_frac=0.2, seed=0)

    drafts = [set(np.unique(data.draft_idx[part]).tolist())
              for part in (splits.train, splits.val, splits.test)]
    assert drafts[0] & drafts[1] == set()
    assert drafts[0] & drafts[2] == set()
    assert drafts[1] & drafts[2] == set()
    assert splits.train.size + splits.val.size + splits.test.size == data.size
    # Reproducible for a given seed, different for a different one.
    assert (split_by_draft(data, seed=0).train == split_by_draft(data, seed=0).train).all()
    assert split_by_draft(data, seed=1).train.size != 0


def test_matched_states_are_found_across_distinct_drafts(tmp_path):
    """Two drafters facing the same pack with the same pool is the whole
    input to the Bayes-floor measurement, so grouping them is tested on a
    case constructed to contain exactly one such recurrence.
    """
    rng = np.random.default_rng(3)
    _, packs_a = _draft(rng, "a")
    packs_b = [list(p) for p in packs_a]
    # Same opening pack and same first three picks, diverging afterwards.
    packs_b[0][3:] = list(reversed(packs_a[0][3:]))
    packs_b[1] = list(rng.choice(CARDS, size=PICKS_PER_PACK, replace=False))
    packs_b[2] = list(rng.choice(CARDS, size=PICKS_PER_PACK, replace=False))
    csv_path = tmp_path / "matched.csv.gz"
    _write_export(csv_path, [("draftaaa", packs_a), ("draftbbb", packs_b)])
    ingest(csv_path, tmp_path / "processed", verbose=False)

    data = PickData.load(tmp_path / "processed")
    rows, groups = matched_state_groups(data)
    assert rows.size > 0
    for group in np.unique(groups):
        block = rows[groups == group]
        # Every group must span at least two distinct drafts...
        assert np.unique(data.draft_idx[block]).size >= 2
        # ...and every member must genuinely share the same state.
        packs = {tuple(sorted(data.pack[i].tolist())) for i in block}
        pools = {tuple(sorted(data.pools_padded([i])[0].tolist())) for i in block}
        assert len(packs) == 1 and len(pools) == 1
        assert len({int(data.pick_number[i]) for i in block}) == 1


def test_matched_state_rows_are_held_out_of_every_split(tmp_path):
    rng = np.random.default_rng(4)
    _, packs_a = _draft(rng, "a")
    packs_b = [list(p) for p in packs_a]
    packs_b[0][3:] = list(reversed(packs_a[0][3:]))
    packs_b[1] = list(rng.choice(CARDS, size=PICKS_PER_PACK, replace=False))
    packs_b[2] = list(rng.choice(CARDS, size=PICKS_PER_PACK, replace=False))
    others = [_draft(rng, f"other{i:03d}") for i in range(10)]
    csv_path = tmp_path / "held.csv.gz"
    _write_export(csv_path, [("draftaaa", packs_a), ("draftbbb", packs_b)] + others)
    ingest(csv_path, tmp_path / "processed", verbose=False)

    data = PickData.load(tmp_path / "processed")
    rows, _ = matched_state_groups(data)
    splits = split_by_draft(data, matched_state_rows=rows)

    trained_on = np.concatenate([splits.train, splits.val, splits.test])
    assert np.intersect1d(trained_on, rows).size == 0
    # At draft granularity the whole contributing draft is withheld, because
    # a later pick's pool would otherwise carry the held-out label.
    contributing = np.unique(data.draft_idx[rows])
    assert np.intersect1d(data.draft_idx[trained_on], contributing).size == 0

    row_level = split_by_draft(data, matched_state_rows=rows, exclude_granularity="row")
    assert row_level.train.size > splits.train.size
