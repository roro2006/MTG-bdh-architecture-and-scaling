"""Tests for the synergy probes.

These run against a randomly initialised model rather than a trained one:
what is under test is that the probes measure what they claim to measure,
not that any particular model scores well. Where a claim needs a model with
known behaviour, the test builds a stub whose response to the pool is fixed
by construction, so the expected number can be written down in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.synergy import (
    AblationResult,
    PickProbe,
    decoy_pools,
    pairwise_synergy,
    pool_ablation,
    pool_sensitivity,
    strongest_pairs,
    synergy_summary,
)
from src.data.card_features import build_features
from src.data.dataset import PAD_ID, PickData
from src.data.vocab import Vocabulary


@pytest.fixture
def corpus(ingested):
    data, stats, out = ingested(count=60)
    return data


def _probe(data, hidden_dim=16, feature_dim=9, seed=0):
    """A randomly initialised PickModel wrapped in a PickProbe."""
    import jax
    import jax.numpy as jnp

    from src.models.pick_model import ModelConfig, PickModel

    config = ModelConfig(
        hidden_dim=hidden_dim,
        card_feature_dim=feature_dim,
        packs_per_draft=data.packs_per_draft,
        picks_per_pack=data.picks_per_pack,
    )
    model = PickModel(config=config, arm="attention")
    table = jax.random.normal(
        jax.random.PRNGKey(seed), (data.vocab.size, feature_dim)
    )
    batch = data.batch(np.arange(4))
    params = model.init(
        jax.random.PRNGKey(seed),
        table,
        jnp.asarray(batch["pack_ids"]),
        jnp.asarray(batch["pool_ids"]),
        jnp.asarray(batch["pack_number"]),
        jnp.asarray(batch["pick_number"]),
    )
    return PickProbe(model, params, table, data.geometry, data.vocab)


# --------------------------------------------------------------------------
# Shaping
# --------------------------------------------------------------------------

def test_probe_shapes_follow_the_corpus_geometry(corpus):
    probe = _probe(corpus)
    assert probe.pad_pack([[1, 2, 3]]).shape == (1, corpus.geometry.max_pack_size)
    assert probe.pad_pool([[1, 2]]).shape == (1, corpus.geometry.max_pool_size)
    padded = probe.pad_pool([[1, 2]])
    assert padded[0, 0] == 1 and padded[0, 2] == PAD_ID


def test_probabilities_are_a_distribution_over_the_pack(corpus):
    probe = _probe(corpus)
    probs = probe.probabilities(
        probe.pad_pack([[0, 1, 2, 3]]),
        probe.pad_pool([[4, 5]]),
        np.array([1]),
        np.array([2]),
    )
    assert probs.shape == (1, corpus.geometry.max_pack_size)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    # Padded slots are masked out of the softmax, not merely small.
    assert probs[0, 4:].sum() < 1e-6


# --------------------------------------------------------------------------
# Decoy pools
# --------------------------------------------------------------------------

def test_decoy_pools_preserve_pool_size_exactly(corpus):
    rows = np.arange(200, 400)
    real = corpus.pools_padded(rows)
    sizes = (real >= 0).sum(axis=1)
    for mode in ("permuted", "random"):
        decoy = decoy_pools(corpus, rows, mode=mode, seed=0)
        assert ((decoy >= 0).sum(axis=1) == sizes).all(), mode
        assert decoy.shape == real.shape


def test_a_permuted_decoy_is_someone_elses_real_pool(corpus):
    """The point of "permuted" over "random": the decoy has to be a
    plausible draft pool, so the probe measures "does the model use *this*
    pool" rather than "can the model tell a pool from noise".
    """
    rows = np.arange(200, 400)
    decoy = decoy_pools(corpus, rows, mode="permuted", seed=0)
    real = corpus.pools_padded(rows)

    # Every decoy card is a card someone actually drafted...
    assert set(np.unique(decoy[decoy >= 0])).issubset(set(np.unique(corpus.label)))
    # ...and the decoys differ from the real pools on most rows.
    differs = [
        not np.array_equal(np.sort(real[i]), np.sort(decoy[i]))
        for i in range(rows.size)
        if (real[i] >= 0).sum() > 2
    ]
    assert sum(differs) > 0.8 * len(differs)


def test_random_decoys_are_drawn_from_the_whole_vocabulary(corpus):
    rows = np.arange(200, 600)
    decoy = decoy_pools(corpus, rows, mode="random", seed=0)
    assert (decoy[decoy >= 0] < corpus.vocab.size).all()
    assert (decoy[decoy >= 0] >= 0).all()


def test_decoy_mode_is_validated(corpus):
    with pytest.raises(ValueError, match="permuted"):
        decoy_pools(corpus, np.arange(100), mode="nonsense")


# --------------------------------------------------------------------------
# The ablation
# --------------------------------------------------------------------------

def test_ablation_excludes_empty_pools(corpus):
    """A pick with no pool scores identically both ways, so including those
    rows would drag every number toward zero for no reason.
    """
    probe = _probe(corpus)
    rows = np.arange(corpus.size)
    result = pool_ablation(probe, corpus, rows, seed=0)
    empty = ((corpus.pools_padded(rows) >= 0).sum(axis=1) == 0).sum()
    assert result.rows == corpus.size - empty
    assert empty > 0


def test_ablation_reports_zero_for_a_model_that_ignores_the_pool():
    """The headline scalar has to read zero when it should.

    A model whose logits do not depend on pool_ids must produce identical
    real and decoy losses -- otherwise the probe is measuring its own noise
    and every synergy number downstream is meaningless.
    """
    import jax.numpy as jnp

    from src.data.ingest import PackGeometry

    class PoolBlindModel:
        def apply(self, params, table, pack_ids, pool_ids, pack_number, pick_number):
            # Score each pack slot from its card id alone. No pool anywhere.
            scores = jnp.where(pack_ids >= 0, pack_ids.astype(jnp.float32), -1e9)
            return scores

    rng = np.random.default_rng(0)
    data = _tiny_corpus(rng)
    probe = PickProbe(
        PoolBlindModel(), {}, np.zeros((data.vocab.size, 3), dtype=np.float32),
        data.geometry, data.vocab,
    )
    result = pool_ablation(probe, data, np.arange(data.size), seed=0)
    assert result.pool_utilisation == pytest.approx(0.0, abs=1e-6)
    assert result.mean_total_variation == pytest.approx(0.0, abs=1e-6)
    assert result.top1_flip_rate == 0.0
    assert result.real_loss == pytest.approx(result.decoy_loss, abs=1e-6)


def test_ablation_is_nonzero_for_a_model_that_reads_the_pool():
    """And the converse: a model that does depend on the pool must show up."""
    import jax.numpy as jnp

    class PoolReadingModel:
        def apply(self, params, table, pack_ids, pool_ids, pack_number, pick_number):
            # A pack card scores high when it is already in the pool.
            match = (pack_ids[:, :, None] == pool_ids[:, None, :]).sum(axis=-1)
            return jnp.where(pack_ids >= 0, 5.0 * match.astype(jnp.float32), -1e9)

    rng = np.random.default_rng(1)
    data = _tiny_corpus(rng)
    probe = PickProbe(
        PoolReadingModel(), {}, np.zeros((data.vocab.size, 3), dtype=np.float32),
        data.geometry, data.vocab,
    )
    result = pool_ablation(probe, data, np.arange(data.size), mode="random", seed=0)
    assert result.mean_total_variation > 0.0
    assert isinstance(result, AblationResult)
    assert "pool is worth" in result.summary()


def test_ablation_refuses_a_sample_too_small_to_mean_anything(corpus):
    probe = _probe(corpus)
    with pytest.raises(ValueError, match="too few"):
        pool_ablation(probe, corpus, np.arange(60, 70), seed=0)


def _tiny_corpus(rng):
    """A small PickData built straight from arrays, no CSV round trip."""
    from src.data.ingest import PackGeometry

    n_drafts, packs, picks, vocab_size = 12, 2, 6, 20
    geometry = PackGeometry(packs_per_draft=packs, picks_per_pack=picks,
                            max_pack_size=picks)
    rows = n_drafts * packs * picks
    pack = np.full((rows, picks), PAD_ID, dtype=np.int16)
    pack_size = np.zeros(rows, dtype=np.int8)
    label = np.zeros(rows, dtype=np.int16)
    label_pos = np.zeros(rows, dtype=np.int8)
    pack_number = np.zeros(rows, dtype=np.int8)
    pick_number = np.zeros(rows, dtype=np.int8)
    draft_idx = np.zeros(rows, dtype=np.int32)

    i = 0
    for d in range(n_drafts):
        for p in range(packs):
            contents = list(rng.choice(vocab_size, size=picks, replace=False))
            for k in range(picks):
                remaining = contents[k:]
                pack[i, : len(remaining)] = remaining
                pack_size[i] = len(remaining)
                label[i] = remaining[0]
                label_pos[i] = 0
                pack_number[i], pick_number[i], draft_idx[i] = p, k, d
                i += 1

    names = tuple(f"Card {j:02d}" for j in range(vocab_size))
    arrays = dict(
        pack=pack, pack_size=pack_size, label=label, label_pos=label_pos,
        pack_number=pack_number, pick_number=pick_number, draft_idx=draft_idx,
        rank_code=np.zeros(rows, dtype=np.int8),
        win_rate_bucket=np.zeros(rows, dtype=np.float32),
        draft_ids=np.array([f"d{j}" for j in range(n_drafts)], dtype="U32"),
        rank_names=np.array(["unknown"], dtype="U16"),
    )
    vocab = Vocabulary(card_to_id={n: j for j, n in enumerate(names)}, id_to_card=names)
    return PickData(arrays, vocab, geometry=geometry)


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------

def test_pool_sensitivity_returns_one_row_per_pool(corpus):
    probe = _probe(corpus)
    pack = [int(c) for c in corpus.pack[100] if c >= 0]
    result = pool_sensitivity(
        probe,
        pack,
        pools=[[], [1, 2, 3], [4, 5, 6, 7]],
        pool_labels=["empty", "a", "b"],
    )
    assert result.probabilities.shape == (3, len(pack))
    assert np.allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert len(result.top_pick) == 3
    assert set(result.top_pick).issubset(set(result.pack))
    # Correlation against the first pool is 1.0 with itself, by definition.
    assert result.rank_correlation[0] == pytest.approx(1.0)
    assert "rank correlation" in result.summary()


def test_pool_sensitivity_names_cards_not_ids(corpus):
    probe = _probe(corpus)
    pack = [int(c) for c in corpus.pack[100] if c >= 0]
    result = pool_sensitivity(probe, pack, pools=[[], [1]])
    assert result.pack == tuple(corpus.vocab.id_to_card[c] for c in pack)


# --------------------------------------------------------------------------
# Pairwise synergy and the colour control
# --------------------------------------------------------------------------

def test_pairwise_synergy_shape_and_baseline(corpus):
    probe = _probe(corpus)
    candidates = [0, 1, 2, 3, 4]
    anchors = [6, 7, 8]
    lift = pairwise_synergy(probe, candidates, anchors, pool_copies=4)
    assert lift.shape == (len(candidates), len(anchors))
    assert np.isfinite(lift).all()


def test_pairwise_synergy_refuses_more_candidates_than_a_pack_holds(corpus):
    probe = _probe(corpus)
    too_many = list(range(corpus.geometry.max_pack_size + 1))
    with pytest.raises(ValueError, match="will not fit in a pack"):
        pairwise_synergy(probe, too_many, [1, 2])


def test_synergy_summary_splits_on_colour_and_ignores_colourless():
    """The colour split is the whole point of the probe, so it is checked
    against a hand-built feature table where the answer is known.
    """
    names = ("Red One", "Blue One", "Red Two", "Colourless One")
    vocab = Vocabulary(card_to_id={n: i for i, n in enumerate(names)}, id_to_card=names)
    cards = {
        "Red One": _card("Red One", ["R"]),
        "Blue One": _card("Blue One", ["U"]),
        "Red Two": _card("Red Two", ["R"]),
        "Colourless One": _card("Colourless One", []),
    }
    features = build_features(vocab, cards)

    candidates = [0, 1, 3]           # red, blue, colourless
    anchors = [2, 1]                 # red, blue
    # Rows are candidates, columns anchors.
    lift = np.array(
        [
            [2.0, 0.0],   # red candidate: big lift from the red anchor
            [0.0, 2.0],   # blue candidate: big lift from the blue anchor
            [9.0, 9.0],   # colourless: must be excluded from both halves
        ]
    )
    summary = synergy_summary(None, features, lift, candidates, anchors)
    assert summary["same_colour_pairs"] == 2
    assert summary["cross_colour_pairs"] == 2
    assert summary["mean_lift_same_colour"] == pytest.approx(2.0)
    assert summary["mean_lift_cross_colour"] == pytest.approx(0.0)
    # The colourless card's huge lift is in neither half.
    assert summary["pairs"] == 6


def test_synergy_summary_flags_a_pure_colour_matcher():
    """A model whose whole pool effect is colour should report a
    cross-colour share near zero -- that is the read that says "this is
    colour matching, not synergy"."""
    names = ("Red One", "Blue One", "Red Two", "Blue Two")
    vocab = Vocabulary(card_to_id={n: i for i, n in enumerate(names)}, id_to_card=names)
    cards = {
        "Red One": _card("Red One", ["R"]), "Red Two": _card("Red Two", ["R"]),
        "Blue One": _card("Blue One", ["U"]), "Blue Two": _card("Blue Two", ["U"]),
    }
    features = build_features(vocab, cards)
    candidates, anchors = [0, 1], [2, 3]
    colour_only = np.array([[3.0, 0.0], [0.0, 3.0]])
    assert synergy_summary(None, features, colour_only, candidates, anchors)[
        "cross_colour_share"
    ] == pytest.approx(0.0, abs=1e-6)

    even = np.array([[3.0, 3.0], [3.0, 3.0]])
    assert synergy_summary(None, features, even, candidates, anchors)[
        "cross_colour_share"
    ] == pytest.approx(1.0, rel=1e-4)


def test_strongest_pairs_reports_names_in_descending_order(corpus):
    probe = _probe(corpus)
    candidates, anchors = [0, 1, 2], [5, 6]
    lift = np.array([[0.1, 0.5], [0.9, 0.2], [0.3, 0.4]])
    pairs = strongest_pairs(probe, lift, candidates, anchors, top=3)
    assert [round(v, 3) for _, _, v in pairs] == [0.9, 0.5, 0.4]
    assert pairs[0][0] == corpus.vocab.id_to_card[1]
    assert pairs[0][1] == corpus.vocab.id_to_card[5]


def _card(name, colors):
    return {
        "name": name,
        "oracle_text": "",
        "type_line": "Creature — Human",
        "cmc": 2.0,
        "color_identity": colors,
        "colors": colors,
        "keywords": [],
        "rarity": "common",
        "power": "2",
        "toughness": "2",
    }
