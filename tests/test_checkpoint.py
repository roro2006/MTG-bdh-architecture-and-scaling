"""Tests for checkpointing and evaluation.

The property that matters is bit-for-bit agreement: a restored checkpoint
must produce exactly the logits the saved model produced. A checkpoint that
loads without error but into subtly wrong shapes or a permuted tree would
still train, still evaluate, and silently report the wrong loss for a grid
cell -- which is the sort of error a scaling fit absorbs rather than reveals.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.pick_model import ModelConfig, count_params_actual, init_model
from src.training.checkpoint import (
    METADATA_FILE,
    PARAMS_FILE,
    load_metadata,
    restore,
    save_checkpoint,
)
from src.training.evaluate import summarise_by_pick
from src.training.train import TrainConfig

from .conftest import VOCAB_SIZE
from .synthetic import FEATURE_DIM


def _config(hidden_dim=32):
    return ModelConfig(hidden_dim=hidden_dim, card_feature_dim=FEATURE_DIM)


def _inputs(rng, batch_size=4):
    pack = np.full((batch_size, 14), -1, dtype=np.int32)
    pool = np.full((batch_size, 41), -1, dtype=np.int32)
    for i in range(batch_size):
        pack[i, : 6 + i] = rng.integers(0, VOCAB_SIZE, size=6 + i)
        pool[i, : 3 * i] = rng.integers(0, VOCAB_SIZE, size=3 * i)
    return (
        jnp.asarray(pack),
        jnp.asarray(pool),
        jnp.asarray(rng.integers(0, 3, size=batch_size).astype(np.int32)),
        jnp.asarray(rng.integers(0, 14, size=batch_size).astype(np.int32)),
    )


def test_restored_params_reproduce_the_logits_exactly(tmp_path, feature_table):
    config = _config()
    model, params = init_model(config, feature_table, seed=3)
    rng = np.random.default_rng(1)
    inputs = _inputs(rng)
    before = model.apply(params, feature_table, *inputs)

    save_checkpoint(
        tmp_path / "cell", params, model_config=config, arm="attention",
        train_config=TrainConfig(total_steps=1),
        metrics={"val_loss": 1.234},
    )
    restored_model, restored_params, metadata = restore(tmp_path / "cell")
    after = restored_model.apply(restored_params, feature_table, *inputs)

    assert np.array_equal(np.asarray(before), np.asarray(after))
    assert metadata["arm"] == "attention"
    assert metadata["num_params"] == count_params_actual(params)
    assert metadata["metrics"]["val_loss"] == 1.234


def test_restored_tree_matches_structure_and_dtypes(tmp_path, feature_table):
    config = _config(64)
    _, params = init_model(config, feature_table, seed=0)
    save_checkpoint(tmp_path / "c", params, model_config=config, arm="attention")
    _, restored, _ = restore(tmp_path / "c")

    original_leaves = jax.tree_util.tree_leaves(params)
    restored_leaves = jax.tree_util.tree_leaves(restored)
    assert len(original_leaves) == len(restored_leaves)
    for a, b in zip(original_leaves, restored_leaves):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(restored)


def test_checkpoint_writes_both_files_and_metadata_is_json(tmp_path, feature_table):
    config = _config()
    _, params = init_model(config, feature_table, seed=0)
    directory = save_checkpoint(
        tmp_path / "cell", params, model_config=config, arm="attention",
        train_config=TrainConfig(),
        metrics={"numpy_float": np.float32(0.5), "numpy_int": np.int64(7)},
    )
    assert (directory / PARAMS_FILE).exists()
    assert (directory / METADATA_FILE).exists()

    # Must be plain JSON -- numpy scalars would otherwise raise on dump.
    raw = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    assert raw["metrics"]["numpy_float"] == pytest.approx(0.5)
    assert raw["metrics"]["numpy_int"] == 7
    assert raw["model_config"]["hidden_dim"] == config.hidden_dim
    assert load_metadata(directory) == raw


def test_a_checkpoint_from_a_different_width_does_not_load_silently(
    tmp_path, feature_table
):
    """Architecture drift must surface, not load into the wrong shapes.

    flax's `from_bytes` does not enforce this: handed a d=64 template and
    d=32 bytes it returns the d=32 arrays and raises nothing. `restore`
    checks the tree itself, which is what this pins down.
    """
    _, params = init_model(_config(32), feature_table, seed=0)
    save_checkpoint(tmp_path / "c", params, model_config=_config(32), arm="attention")

    metadata = json.loads((tmp_path / "c" / METADATA_FILE).read_text(encoding="utf-8"))
    metadata["model_config"]["hidden_dim"] = 64  # pretend the model changed
    (tmp_path / "c" / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="architecture changed|shape"):
        restore(tmp_path / "c")


def test_from_bytes_alone_would_not_have_caught_it(tmp_path, feature_table):
    """Documents the flax behaviour the guard above exists to compensate for.

    If a future flax version starts validating, this test fails and the
    hand-rolled check in checkpoint.py can be reconsidered.
    """
    from flax import serialization

    _, small = init_model(_config(32), feature_table, seed=0)
    _, large = init_model(_config(64), feature_table, seed=0)
    restored = serialization.from_bytes(large, serialization.to_bytes(small))

    assert count_params_actual(restored) == count_params_actual(small)
    assert count_params_actual(restored) != count_params_actual(large)


def test_summarise_separates_forced_picks_from_real_decisions():
    rows = [
        {"pick_number": 0, "rows": 100, "loss": 2.0, "accuracy": 0.4,
         "uniform_loss": 2.6, "mean_pack_size": 14.0},
        {"pick_number": 8, "rows": 100, "loss": 1.4, "accuracy": 0.5,
         "uniform_loss": 1.8, "mean_pack_size": 6.0},
        {"pick_number": 12, "rows": 100, "loss": 0.5, "accuracy": 0.8,
         "uniform_loss": 0.69, "mean_pack_size": 2.0},
        {"pick_number": 13, "rows": 100, "loss": 0.0, "accuracy": 1.0,
         "uniform_loss": 0.0, "mean_pack_size": 1.0},
    ]
    summary = summarise_by_pick(rows)

    assert summary["all_picks"]["rows"] == 400
    assert summary["all_picks"]["loss"] == pytest.approx((2.0 + 1.4 + 0.5 + 0.0) / 4)
    # Picks 0-8 only, so the two easy picks drop out and the loss rises.
    assert summary["decision_picks"]["rows"] == 200
    assert summary["decision_picks"]["loss"] == pytest.approx(1.7)
    # A one-card pack is a forced pick, not a decision.
    assert summary["forced_rows"] == 100
    assert summary["forced_fraction"] == pytest.approx(0.25)
