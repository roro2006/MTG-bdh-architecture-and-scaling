"""Concatenating sets must not quietly change what a row means.

Every failure this module guards against is silent. A card id that means
one card in one set and a different card in another still trains; a feature
column that is "Flying" in one table and "Ward" in the next still trains; a
pool-as-prefix check run at the wrong geometry drops a whole set and the
corpus that comes back is simply smaller. None of them raises on its own,
and all of them would land as a zero-shot number that is wrong rather than
as an error.

So the tests here are mostly identity checks against the single-set loader:
after concatenation, row i of set S must present the same cards, the same
label, the same label position and the same pool as it did before -- and
the split it lands in must be the same split too, because a held-out
number is only comparable to a same-set number if both are measured on the
same rows.

Geometry is varied on purpose. Three of the fixtures' four sets are not the
default 3x14, mirroring the real corpus where BLB and EOE draft 13 cards a
pack and LCI and SIR 15.
"""

from __future__ import annotations

import zlib
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from src.data.card_features import CardFeatures
from src.data.dataset import PickData, split_by_draft
from src.data.ingest import ingest
from src.data.multiset import (
    build_card_index,
    load_multiset,
    processed_dirs,
    remap_ids,
    set_code,
)

from .synthetic import make_drafts, write_export

RAW_COLUMNS = 12
# What `CardFeatures.dense()` makes of those: colour identity (5), castable
# colours (5), the rarity one-hot (6) and the scalar block (6).
DENSE_COLUMNS = 22

# Four "sets" with deliberately different card pools and geometries. ONE
# card ("Shared Relic") is printed in all four and one more in two of them,
# so the reprint path is exercised rather than assumed away.
SHARED = ["Shared Relic", "Basic Wastes"]
SET_SPECS = {
    "AAA": dict(
        cards=sorted([f"A {i:02d}" for i in range(16)] + SHARED),
        picks_per_pack=14, packs_per_draft=3, drafts=14,
    ),
    "BBB": dict(
        cards=sorted([f"B {i:02d}" for i in range(16)] + SHARED),
        picks_per_pack=10, packs_per_draft=3, drafts=12,
    ),
    "CCC": dict(
        # "Basic Wastes" only; no second shared card, so the printings map
        # has to distinguish "in all four" from "in two".
        cards=sorted([f"C {i:02d}" for i in range(18)] + ["Shared Relic"]),
        picks_per_pack=15, packs_per_draft=3, drafts=10,
    ),
    "DDD": dict(
        cards=sorted([f"D {i:02d}" for i in range(16)] + SHARED),
        picks_per_pack=8, packs_per_draft=4, drafts=9,
    ),
}


def _features_for(names) -> CardFeatures:
    """A CardFeatures whose rows are a pure function of the card NAME.

    That is the invariant the real builder has by construction -- it
    fetches Scryfall by name, not by printing -- and the one
    `build_card_index` re-checks. Deriving the row from a hash of the name
    reproduces it here without a network call.
    """
    names = tuple(names)
    size = len(names)
    rows = np.stack(
        [
            np.array(
                [
                    (zlib.crc32(f"{name}:{c}".encode()) % 1000) / 1000.0
                    for c in range(RAW_COLUMNS)
                ],
                dtype=np.float32,
            )
            for name in names
        ]
    ) if names else np.zeros((0, RAW_COLUMNS), dtype=np.float32)
    return CardFeatures(
        color_identity=rows[:, :5],
        colors=rows[:, 5:10],
        mana_value=rows[:, 10] * 6,
        type_flags=np.zeros((size, 0), dtype=np.float32),
        keyword_flags=np.zeros((size, 0), dtype=np.float32),
        mechanic_flags=np.zeros((size, 0), dtype=np.float32),
        rarity=np.zeros(size, dtype=np.int32),
        power=rows[:, 11] * 5,
        toughness=rows[:, 11] * 4,
        is_creature=np.zeros(size, dtype=np.float32),
        keyword_names=(),
        mechanic_names=(),
        card_names=names,
    )


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory):
    """Four ingested sets under one root, ready for `load_multiset`."""
    root = tmp_path_factory.mktemp("processed")
    rng = np.random.default_rng(0)
    for code, spec in SET_SPECS.items():
        drafts = make_drafts(
            rng, spec["drafts"], prefix=f"{code.lower()}",
            picks_per_pack=spec["picks_per_pack"],
            packs_per_draft=spec["packs_per_draft"],
            cards=spec["cards"],
        )
        csv_path = root / f"draft_data_public.{code}.PremierDraft.csv.gz"
        write_export(csv_path, drafts, cards=spec["cards"], expansion=code)
        out = root / f"{code}.PremierDraft"
        ingest(csv_path, out, verbose=False)
        _features_for(spec["cards"]).save(out / "card_features.npz")
        csv_path.unlink()
    return root


@pytest.fixture(scope="module")
def corpus(corpus_root):
    return load_multiset(processed_dirs(corpus_root), verbose=False)


# --------------------------------------------------------------------------
# The shared card index
# --------------------------------------------------------------------------


def test_index_is_the_sorted_union_and_reprints_collapse(corpus, corpus_root):
    index = corpus.card_index
    expected = sorted({c for spec in SET_SPECS.values() for c in spec["cards"]})

    assert list(index.id_to_card) == expected
    assert index.size == len(expected)
    assert index.feature_dim == DENSE_COLUMNS

    # Two cards are printed more than once, and they are the two intended.
    assert set(index.reprints()) == {"Shared Relic", "Basic Wastes"}
    assert len(index.printings["Shared Relic"]) == 4
    assert len(index.printings["Basic Wastes"]) == 3
    assert len(index.printings["C 00"]) == 1


def test_remap_takes_each_set_id_to_the_same_card(corpus, corpus_root):
    """The whole point of the index: id k in set S names the same card as
    remap[k] does globally."""
    for name in corpus.names:
        part = PickData.load(corpus_root / name)
        remap = corpus.card_index.remap_for(part.vocab)
        for local_id, card in enumerate(part.vocab.id_to_card):
            assert corpus.card_index.id_to_card[remap[local_id]] == card


def test_remap_leaves_padding_alone():
    """PAD_ID is -1, which a bare gather would wrap to the last card."""
    remap = np.array([7, 3, 9], dtype=np.int64)
    ids = np.array([[0, 2, -1], [-1, -1, 1]], dtype=np.int16)
    out = remap_ids(ids, remap)
    assert out.tolist() == [[7, 9, -1], [-1, -1, 3]]


def test_overlap_reports_which_held_out_cards_were_seen(corpus):
    train = [n for n in corpus.names if not n.startswith("CCC")]
    overlap = corpus.card_index.overlap("CCC.PremierDraft", train)

    assert overlap["held_out_cards"] == len(SET_SPECS["CCC"]["cards"])
    # CCC shares exactly "Shared Relic" with the others.
    assert overlap["cards_also_in_training_sets"] == 1
    assert overlap["examples"] == ["Shared Relic"]


def test_feature_width_mismatch_is_refused(corpus_root):
    a = PickData.load(corpus_root / "AAA.PremierDraft")
    b = PickData.load(corpus_root / "BBB.PremierDraft")
    # Drop a colour column so dense() comes out one narrower.
    narrow = replace(
        _features_for(b.vocab.id_to_card),
        colors=_features_for(b.vocab.id_to_card).colors[:, :4],
    )

    with pytest.raises(ValueError, match="different widths"):
        build_card_index(
            [
                ("AAA", a.vocab, _features_for(a.vocab.id_to_card)),
                ("BBB", b.vocab, narrow),
            ]
        )


def test_same_width_but_different_columns_is_refused(corpus_root):
    a = PickData.load(corpus_root / "AAA.PremierDraft")
    b = PickData.load(corpus_root / "BBB.PremierDraft")
    features_b = _features_for(b.vocab.id_to_card)
    # Same total width, but one keyword column instead of one colour: column
    # k now means something different from column k in A.
    shifted = replace(
        features_b,
        colors=features_b.colors[:, :4],
        keyword_flags=np.zeros((features_b.size, 1), dtype=np.float32),
        keyword_names=("Flying",),
    )
    assert shifted.dense().shape[1] == features_b.dense().shape[1]

    with pytest.raises(ValueError, match="not the same columns"):
        build_card_index(
            [
                ("AAA", a.vocab, _features_for(a.vocab.id_to_card)),
                ("BBB", b.vocab, shifted),
            ]
        )


def test_disagreeing_reprint_is_refused(corpus_root):
    a = PickData.load(corpus_root / "AAA.PremierDraft")
    b = PickData.load(corpus_root / "BBB.PremierDraft")
    features_b = _features_for(b.vocab.id_to_card)
    tampered = features_b.mana_value.copy()
    tampered[b.vocab.id_of("Shared Relic")] += 1.0
    features_b = replace(features_b, mana_value=tampered)

    with pytest.raises(ValueError, match="Shared Relic"):
        build_card_index(
            [
                ("AAA", a.vocab, _features_for(a.vocab.id_to_card)),
                ("BBB", b.vocab, features_b),
            ]
        )


def test_features_out_of_vocabulary_order_is_refused(corpus_root):
    a = PickData.load(corpus_root / "AAA.PremierDraft")
    scrambled = _features_for(list(reversed(a.vocab.id_to_card)))

    with pytest.raises(ValueError, match="different order"):
        build_card_index([("AAA", a.vocab, scrambled)])


# --------------------------------------------------------------------------
# The concatenated corpus
# --------------------------------------------------------------------------


def test_corpus_totals_and_envelope_geometry(corpus, corpus_root):
    data = corpus.data
    parts = [PickData.load(corpus_root / n) for n in corpus.names]

    assert data.size == sum(p.size for p in parts)
    assert data.n_drafts == sum(p.n_drafts for p in parts)
    # The envelope is the widest of each dimension, taken independently:
    # DDD has the most packs (4), CCC the most picks per pack (15).
    assert data.packs_per_draft == 4
    assert data.picks_per_pack == 15
    assert data.pack.shape[1] == 15
    assert data.max_pool_size == 4 * 15 - 1
    # And each set keeps its own, so the identity can be checked per set.
    assert {s.code: s.geometry.picks_per_pack for s in data.sets} == {
        "AAA": 14, "BBB": 10, "CCC": 15, "DDD": 8
    }


def test_rows_keep_their_cards_labels_and_positions(corpus, corpus_root):
    for name in corpus.names:
        part = PickData.load(corpus_root / name)
        s = corpus.data.slice_of(name)
        assert s.rows == part.size

        for i in range(0, part.size, max(part.size // 17, 1)):
            g = s.row_start + i
            local = [part.vocab.id_to_card[c] for c in part.pack[i] if c >= 0]
            merged = [
                corpus.data.vocab.id_to_card[c]
                for c in corpus.data.pack[g]
                if c >= 0
            ]
            assert local == merged
            assert (
                part.vocab.id_to_card[part.label[i]]
                == corpus.data.vocab.id_to_card[corpus.data.label[g]]
            )
            assert part.label_pos[i] == corpus.data.label_pos[g]
            assert part.pack_size[i] == corpus.data.pack_size[g]
            assert part.pack_number[i] == corpus.data.pack_number[g]
            assert part.pick_number[i] == corpus.data.pick_number[g]


def test_pools_survive_the_wider_envelope(corpus, corpus_root):
    """A narrow set's pool must be its own pool, then padding -- not the
    envelope's width filled with something."""
    for name in corpus.names:
        part = PickData.load(corpus_root / name)
        s = corpus.data.slice_of(name)
        rows = np.arange(part.size)
        remap = corpus.card_index.remap_for(part.vocab)

        local = part.pools_padded(rows)
        merged = corpus.data.pools_padded(rows + s.row_start)
        expected = np.where(local >= 0, remap[np.clip(local, 0, None)], -1)

        assert merged.shape[1] == corpus.data.max_pool_size
        assert np.array_equal(merged[:, : local.shape[1]], expected)
        assert (merged[:, local.shape[1]:] == -1).all()


def test_the_identity_is_checked_per_set_not_per_corpus(corpus):
    """The regression this module exists for.

    `PickData._invalid_drafts` multiplies pack_number by ONE
    picks_per_pack. Run over a mixed corpus it condemns every row of every
    set that is not the envelope's shape -- and `on_invalid="drop"` would
    then hand back a corpus that is quietly missing them. The override must
    find nothing where the base implementation finds most of the corpus.
    """
    assert corpus.data._invalid_drafts().size == 0

    # What the envelope's single picks_per_pack would have concluded: the
    # prefix identity computed with 15 for every row, not each row's own.
    data = corpus.data
    envelope_expected = (
        data.pack_number.astype(np.int64) * data.geometry.picks_per_pack
        + data.pick_number.astype(np.int64)
    )
    actual = np.arange(data.size, dtype=np.int64) - data._draft_start[data.draft_idx]
    misjudged = envelope_expected != actual

    # CCC alone drafts 15 a pack, so it is the only set the envelope gets
    # right; every other set has rows it would have thrown away.
    for s in data.sets:
        rows = slice(s.row_start, s.row_stop)
        if s.geometry.picks_per_pack == data.geometry.picks_per_pack:
            assert not misjudged[rows].any(), s.code
        else:
            assert misjudged[rows].any(), s.code
    assert misjudged.sum() > 0.5 * data.size


def test_batch_is_shaped_for_the_envelope(corpus):
    batch = corpus.data.batch(np.array([0, corpus.data.size // 2, corpus.data.size - 1]))
    assert batch["pack_ids"].shape == (3, 15)
    assert batch["pool_ids"].shape == (3, corpus.data.max_pool_size)
    assert batch["pack_ids"].max() < corpus.card_index.size
    assert batch["label"].max() < corpus.card_index.size


def test_draft_ids_stay_traceable_to_their_set(corpus):
    for s in corpus.data.sets:
        ids = corpus.data.draft_ids[s.draft_offset : s.draft_offset + s.n_drafts]
        assert all(i.startswith(f"{s.code}:") for i in ids)
    assert len(set(corpus.data.draft_ids)) == corpus.data.n_drafts


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def test_splits_are_the_single_set_splits_offset(corpus, corpus_root):
    """What makes a held-out number comparable to a same-set number."""
    for name in corpus.names:
        part = PickData.load(corpus_root / name)
        local = split_by_draft(part, seed=0)
        s = corpus.data.slice_of(name)
        for which in ("train", "val", "test"):
            assert np.array_equal(
                getattr(corpus.splits[name], which) - s.row_start,
                getattr(local, which),
            )


def test_held_out_rows_never_appear_in_the_training_pool(corpus):
    held = corpus.data.slice_of("CCC")
    train_names = [n for n in corpus.names if not n.startswith("CCC")]
    train = corpus.rows(train_names, "train")

    assert train.size > 0
    assert not ((train >= held.row_start) & (train < held.row_stop)).any()

    val = corpus.splits["CCC.PremierDraft"].val
    assert val.size > 0
    assert ((val >= held.row_start) & (val < held.row_stop)).all()
    # And a draft is never split across the two.
    assert not np.intersect1d(
        corpus.data.draft_idx[train], corpus.data.draft_idx[val]
    ).size


def test_rows_of_and_slice_of_agree(corpus):
    for s in corpus.data.sets:
        rows = corpus.data.rows_of(s.name)
        assert rows[0] == s.row_start and rows[-1] == s.row_stop - 1
        assert (corpus.data.set_idx[rows] == s.position).all()
    with pytest.raises(KeyError):
        corpus.data.slice_of("ZZZ")


def test_set_code_strips_the_event_suffix():
    assert set_code("FIN.PremierDraft") == "FIN"
    assert set_code("FIN") == "FIN"


# --------------------------------------------------------------------------
# The model reads it
# --------------------------------------------------------------------------


def test_a_model_scores_a_held_out_set_with_no_new_parameters(corpus):
    """The architectural claim, as a test.

    A model built for the merged corpus has a parameter count that depends
    on the feature width and the envelope geometry, and on nothing about
    which sets are in it. Feeding it a set's rows is a gather, not a
    lookup, so the held-out set needs no parameter of its own.
    """
    import jax

    from src.models.pick_model import ModelConfig, count_params_analytic, init_model

    config = ModelConfig(
        hidden_dim=16,
        card_feature_dim=corpus.card_index.feature_dim,
        packs_per_draft=corpus.data.packs_per_draft,
        picks_per_pack=corpus.data.picks_per_pack,
    )
    table = jnp.asarray(corpus.feature_table)
    model, params = init_model(config, table, arm="attention", seed=0)
    expected = count_params_analytic(config, arm="attention")["total"]

    held = corpus.data.slice_of("CCC")
    batch = corpus.data.batch(np.arange(held.row_start, held.row_start + 8))
    logits = model.apply(
        params, table, jnp.asarray(batch["pack_ids"]), jnp.asarray(batch["pool_ids"]),
        jnp.asarray(batch["pack_number"]), jnp.asarray(batch["pick_number"]),
    )

    assert logits.shape == (8, corpus.data.pack.shape[1])
    realised = sum(int(p.size) for p in jax.tree_util.tree_leaves(params))
    assert realised == expected
    assert bool(jnp.isfinite(logits[batch["pack_ids"] >= 0]).all())
