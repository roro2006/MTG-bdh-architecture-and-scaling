"""Tests for the shared front-end and the attention arm.

The properties checked here are the ones docs/ARCHITECTURE.md asserts as
structural rather than learned -- order-blindness over the pack and pool,
an output space closed to the pack -- plus the parameter-count derivation
that docs/PROJECT_PLAN.md section 3b requires be done by hand. If any of
these silently stopped holding, the scaling curves would still fit, and
would still be wrong.

No data download is needed: features and ids are synthesised here.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.embeddings import gather_card_features
from src.models.pick_model import (
    MASK_SCORE,
    ModelConfig,
    PickModel,
    count_params_actual,
    count_params_analytic,
    cross_entropy_loss,
    init_model,
)

from .conftest import VOCAB_SIZE
from .synthetic import FEATURE_DIM

MAX_PACK = 14
MAX_POOL = 41

# `feature_table` comes from conftest.py, session-scoped: model tests all
# want the same table and rebuilding it per module bought nothing.


def _config(hidden_dim: int = 32) -> ModelConfig:
    return ModelConfig(
        hidden_dim=hidden_dim,
        num_heads=4,
        pool_encoder_layers=2,
        pack_encoder_layers=1,
        arm_layers=2,
        card_feature_dim=FEATURE_DIM,
    )


def _batch(rng, batch_size=6, pack_sizes=None, pool_sizes=None):
    pack = np.full((batch_size, MAX_PACK), -1, dtype=np.int32)
    pool = np.full((batch_size, MAX_POOL), -1, dtype=np.int32)
    pack_sizes = pack_sizes or rng.integers(1, MAX_PACK + 1, size=batch_size)
    pool_sizes = pool_sizes or rng.integers(0, MAX_POOL + 1, size=batch_size)
    for i in range(batch_size):
        pack[i, : pack_sizes[i]] = rng.integers(0, VOCAB_SIZE, size=pack_sizes[i])
        pool[i, : pool_sizes[i]] = rng.integers(0, VOCAB_SIZE, size=pool_sizes[i])
    return {
        "pack_ids": pack,
        "pool_ids": pool,
        "pack_number": rng.integers(0, 3, size=batch_size).astype(np.int32),
        "pick_number": rng.integers(0, MAX_PACK, size=batch_size).astype(np.int32),
    }


def _apply(model, params, table, batch):
    return model.apply(
        params, table, jnp.asarray(batch["pack_ids"]), jnp.asarray(batch["pool_ids"]),
        jnp.asarray(batch["pack_number"]), jnp.asarray(batch["pick_number"]),
    )


def test_gather_zeroes_padding_rather_than_reading_card_zero(feature_table):
    ids = jnp.asarray([[0, 5, -1, -1]])
    gathered = gather_card_features(feature_table, ids)
    assert gathered.shape == (1, 4, FEATURE_DIM)
    assert jnp.allclose(gathered[0, 0], feature_table[0])
    assert jnp.allclose(gathered[0, 1], feature_table[5])
    # Padding must be zero, not card 0's features -- -1 would otherwise wrap.
    assert jnp.abs(gathered[0, 2]).max() == 0.0
    assert jnp.abs(gathered[0, 3]).max() == 0.0


@pytest.mark.parametrize("hidden_dim", [16, 32, 64, 128])
def test_analytic_parameter_count_matches_realised_tree(feature_table, hidden_dim):
    """PROJECT_PLAN section 3b: N is derived, not read off the pytree.

    Parameterised over width because the grid sweeps width, and a
    derivation that is only right at one size is not a derivation.
    """
    config = _config(hidden_dim)
    _, params = init_model(config, feature_table, seed=0)
    analytic = count_params_analytic(config)
    assert analytic["total"] == count_params_actual(params)
    assert sum(v for k, v in analytic.items() if k != "total") == analytic["total"]


def test_analytic_count_matches_for_the_mean_pooling_encoder(feature_table):
    config = _config(32).__class__(
        **{**_config(32).__dict__, "set_encoder_mode": "mean"}
    )
    _, params = init_model(config, feature_table, seed=0)
    assert count_params_analytic(config)["total"] == count_params_actual(params)


def test_output_is_closed_to_the_pack(feature_table):
    """The pointer head cannot express a pick outside the pack."""
    rng = np.random.default_rng(1)
    model, params = init_model(_config(), feature_table, seed=0)
    batch = _batch(rng)
    logits = _apply(model, params, feature_table, batch)

    pack_mask = batch["pack_ids"] >= 0
    assert logits.shape == (len(batch["pack_ids"]), MAX_PACK)
    assert bool((logits[~pack_mask] == MASK_SCORE).all())
    probs = jax.nn.softmax(logits, axis=-1)
    assert float(probs[~pack_mask].sum()) == 0.0
    assert bool(jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5))


def test_a_single_card_pack_is_certain(feature_table):
    """14 - 13 = 1 card left means no decision; loss must be exactly 0."""
    rng = np.random.default_rng(2)
    model, params = init_model(_config(), feature_table, seed=0)
    batch = _batch(rng, batch_size=4, pack_sizes=[1, 1, 1, 1])
    logits = _apply(model, params, feature_table, batch)
    loss, accuracy = cross_entropy_loss(logits, jnp.zeros(4, dtype=jnp.int32))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert float(accuracy) == 1.0


def test_pack_order_permutes_logits_and_nothing_else(feature_table):
    """Order within a pack carries no signal, so the architecture must be
    equivariant to it rather than merely trained to ignore it.
    """
    rng = np.random.default_rng(3)
    model, params = init_model(_config(), feature_table, seed=0)
    batch = _batch(rng, batch_size=5, pack_sizes=[MAX_PACK] * 5)
    logits = np.asarray(_apply(model, params, feature_table, batch))

    perm = rng.permutation(MAX_PACK)
    shuffled = dict(batch, pack_ids=batch["pack_ids"][:, perm])
    logits_perm = np.asarray(_apply(model, params, feature_table, shuffled))
    assert np.allclose(logits[:, perm], logits_perm, atol=1e-4)


def test_pool_order_changes_nothing(feature_table):
    rng = np.random.default_rng(4)
    model, params = init_model(_config(), feature_table, seed=0)
    batch = _batch(rng, batch_size=5, pool_sizes=[MAX_POOL] * 5)
    logits = np.asarray(_apply(model, params, feature_table, batch))

    perm = rng.permutation(MAX_POOL)
    shuffled = dict(batch, pool_ids=batch["pool_ids"][:, perm])
    logits_perm = np.asarray(_apply(model, params, feature_table, shuffled))
    assert np.allclose(logits, logits_perm, atol=1e-4)


def test_an_empty_pool_does_not_produce_nans(feature_table):
    """Pack 0 pick 0 has no pool at all. Without the arm's learned null key
    every attention key would be masked and the softmax would divide by
    zero -- 140,237 rows of the FIN corpus, one in every 42.
    """
    rng = np.random.default_rng(5)
    model, params = init_model(_config(), feature_table, seed=0)
    batch = _batch(rng, batch_size=4, pack_sizes=[MAX_PACK] * 4, pool_sizes=[0, 0, 0, 0])
    assert (batch["pool_ids"] < 0).all()

    logits = _apply(model, params, feature_table, batch)
    assert bool(jnp.isfinite(logits).all())

    def loss_fn(p):
        out = model.apply(
            p, feature_table, jnp.asarray(batch["pack_ids"]), jnp.asarray(batch["pool_ids"]),
            jnp.asarray(batch["pack_number"]), jnp.asarray(batch["pick_number"]),
        )
        return cross_entropy_loss(out, jnp.zeros(4, dtype=jnp.int32))[0]

    grads = jax.grad(loss_fn)(params)
    assert all(
        bool(jnp.isfinite(leaf).all()) for leaf in jax.tree_util.tree_leaves(grads)
    )


def test_padding_in_the_pool_does_not_change_the_answer(feature_table):
    """A pool of 5 cards must score identically whether the array is padded
    to 41 or the cards sit in different slots.
    """
    rng = np.random.default_rng(6)
    model, params = init_model(_config(), feature_table, seed=0)
    batch = _batch(rng, batch_size=3, pack_sizes=[8, 8, 8], pool_sizes=[5, 5, 5])
    logits = np.asarray(_apply(model, params, feature_table, batch))

    moved = np.full_like(batch["pool_ids"], -1)
    moved[:, 10:15] = batch["pool_ids"][:, 0:5]
    logits_moved = np.asarray(_apply(model, params, feature_table, dict(batch, pool_ids=moved)))
    assert np.allclose(logits, logits_moved, atol=1e-4)


def test_unknown_arm_is_rejected(feature_table):
    with pytest.raises(ValueError, match="unknown arm"):
        init_model(_config(), feature_table, arm="mamba", seed=0)


def test_both_arms_build_and_share_everything_but_the_arm(feature_table):
    """The grid's premise: swapping the arm changes the arm and nothing else.

    Checked on the parameter tree rather than the total, because a total
    can coincide while the front-end quietly differs.
    """
    config = _config()
    shared = {}
    for arm in ("attention", "bdh"):
        _, params = init_model(config, feature_table, arm=arm, seed=0)
        tree = params["params"]
        assert set(tree) == {
            "card_embedding", "pack_encoder", "pool_encoder", "context", "arm", "pointer"
        }
        shared[arm] = {k: jax.tree_util.tree_structure(v)
                       for k, v in tree.items() if k != "arm"}
    assert shared["attention"] == shared["bdh"]


def test_neuron_multiplier_four_is_iso_parameter_with_attention(feature_table):
    """Why the default is 4 and not the reference's 128.

    A BDH layer costs 3*multiplier*D^2 against a cross-attention block's
    12*D^2 + 15*D, so multiplier=4 matches to within a term linear in D.
    The reference's own default would be 32x larger and no iso-parameter
    grid would be possible at all.
    """
    config = _config(hidden_dim=256)
    attention = count_params_analytic(config, "attention")["arm"]
    bdh = count_params_analytic(config, "bdh")["arm"]
    assert bdh == pytest.approx(attention, rel=0.01)

    fat = count_params_analytic(
        replace(config, neuron_multiplier=128), "bdh"
    )["arm"]
    assert fat > 30 * attention


def test_loss_matches_a_hand_computed_softmax(feature_table):
    logits = jnp.asarray([[1.0, 2.0, MASK_SCORE], [0.5, 0.5, 0.5]])
    # Row 1 is a three-way tie, so its label is chosen to be one argmax does
    # not return: accuracy must not depend on how ties break.
    label_pos = jnp.asarray([1, 1], dtype=jnp.int32)
    loss, accuracy = cross_entropy_loss(logits, label_pos)

    # Row 0: the masked slot must contribute nothing to the partition function.
    expected_0 = -(2.0 - np.log(np.exp(1.0) + np.exp(2.0)))
    expected_1 = -np.log(1.0 / 3.0)
    assert float(loss) == pytest.approx((expected_0 + expected_1) / 2, abs=1e-5)
    assert float(accuracy) == pytest.approx(0.5)
