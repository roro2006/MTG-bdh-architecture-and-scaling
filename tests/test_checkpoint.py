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


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------
#
# The property under test is the same one the rest of this file is about:
# agreement, not merely "it ran". A resume that restarts Adam from zero, or
# replays a batch the run had already consumed, still produces a smooth
# plausible loss curve -- it just produces the wrong one, and a scaling fit
# absorbs that rather than revealing it. So the test is that a run chopped
# into segments lands on exactly the parameters an uninterrupted run does.


def _tiny_train_config(**overrides):
    defaults = dict(
        batch_size=8, learning_rate=1e-3, warmup_steps=2, total_steps=20,
        eval_every=5, seed=0, eval_batches=2,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _train_inputs(ingested, feature_table):
    """A PickData small enough to train on in a test, plus its splits."""
    from src.data.dataset import split_by_draft

    data, _, _ = ingested(count=60)
    splits = split_by_draft(data, seed=0)
    config = ModelConfig(
        hidden_dim=16, num_heads=2, pool_encoder_layers=1, pack_encoder_layers=1,
        arm_layers=1, card_feature_dim=data.vocab.size and FEATURE_DIM,
    )
    return data, splits, config


def test_resume_reproduces_an_uninterrupted_run(tmp_path, ingested, feature_table):
    """Segmented training must land on the same parameters as one long run."""
    from src.training.train import train_model

    data, splits, config = _train_inputs(ingested, feature_table)
    table = jnp.asarray(np.zeros((data.vocab.size, FEATURE_DIM), dtype=np.float32))
    train_config = _tiny_train_config()

    straight = train_model(
        data, table, splits.train, splits.val, config, train_config,
        arm="attention", verbose=False, checkpoint_dir=tmp_path / "straight",
    )
    assert straight["completed"] is True
    assert straight["stopped_at_step"] == train_config.total_steps

    # max_seconds=0 stops at the very first evaluation boundary every time,
    # so this chops the same run into four segments deterministically.
    segmented_dir = tmp_path / "segmented"
    segments = 0
    while True:
        result = train_model(
            data, table, splits.train, splits.val, config, train_config,
            arm="attention", verbose=False, checkpoint_dir=segmented_dir,
            resume=True, max_seconds=0.0,
        )
        segments += 1
        if result["completed"]:
            break
        assert segments < 10, "segmented run is not making progress"

    assert segments > 1, "the budget never fired; this would not test resume"
    assert result["stopped_at_step"] == train_config.total_steps

    straight_leaves = jax.tree_util.tree_leaves(straight["params"])
    resumed_leaves = jax.tree_util.tree_leaves(result["params"])
    assert len(straight_leaves) == len(resumed_leaves)
    for a, b in zip(straight_leaves, resumed_leaves):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    # The curve must be continuous too, not just the endpoint: a resume that
    # replayed data would still converge, to a different place along the way.
    assert [h["step"] for h in result["history"]] == [
        h["step"] for h in straight["history"]
    ]
    for got, want in zip(result["history"], straight["history"]):
        assert got["train_loss"] == pytest.approx(want["train_loss"], abs=1e-6)
        assert got["val_loss"] == pytest.approx(want["val_loss"], abs=1e-6)


def test_batch_stream_restores_its_exact_position(ingested):
    """The stream is the part of resume that has no shape check to catch it."""
    from src.training.train import BatchStream

    data, _, _ = ingested(count=60)
    indices = np.arange(data.size)

    reference = BatchStream(data, indices, batch_size=8, seed=3)
    consumed = [next(reference) for _ in range(11)]
    saved = reference.state()
    expected = [next(reference) for _ in range(5)]

    restored = BatchStream(data, indices, batch_size=8, seed=3)
    restored.restore(saved)
    got = [next(restored) for _ in range(5)]

    for want, have in zip(expected, got):
        for key in want:
            np.testing.assert_array_equal(want[key], have[key])
    assert len(consumed) == 11  # the stream really did advance past one epoch


def test_resume_refuses_a_run_with_a_different_config(tmp_path, ingested):
    """A width change must fail loudly rather than continue the loss curve."""
    from src.training.train import train_model

    data, splits, config = _train_inputs(ingested, None)
    table = jnp.asarray(np.zeros((data.vocab.size, FEATURE_DIM), dtype=np.float32))
    out = tmp_path / "cell"

    train_model(
        data, table, splits.train, splits.val, config, _tiny_train_config(),
        arm="attention", verbose=False, checkpoint_dir=out, max_seconds=0.0,
    )

    wider = ModelConfig(**{**config.__dict__, "hidden_dim": 32})
    with pytest.raises(ValueError, match="different configuration"):
        train_model(
            data, table, splits.train, splits.val, wider, _tiny_train_config(),
            arm="attention", verbose=False, checkpoint_dir=out, resume=True,
        )


def test_finished_run_leaves_no_resume_state(tmp_path, ingested):
    """Otherwise re-running a completed cell silently does nothing."""
    from src.training.checkpoint import RESUME_PARAMS_FILE, RESUME_STATE_FILE
    from src.training.train import train_model

    data, splits, config = _train_inputs(ingested, None)
    table = jnp.asarray(np.zeros((data.vocab.size, FEATURE_DIM), dtype=np.float32))
    out = tmp_path / "cell"

    partial = train_model(
        data, table, splits.train, splits.val, config, _tiny_train_config(),
        arm="attention", verbose=False, checkpoint_dir=out, max_seconds=0.0,
    )
    assert partial["completed"] is False
    assert (out / RESUME_PARAMS_FILE).exists()
    assert (out / RESUME_STATE_FILE).exists()

    while not train_model(
        data, table, splits.train, splits.val, config, _tiny_train_config(),
        arm="attention", verbose=False, checkpoint_dir=out,
        resume=True, max_seconds=0.0,
    )["completed"]:
        pass

    assert not (out / RESUME_PARAMS_FILE).exists()
    assert not (out / RESUME_STATE_FILE).exists()


def test_segment_budget_is_per_segment_not_cumulative(tmp_path, ingested, feature_table):
    """A resumed segment gets the full budget again, not what is left of it.

    `elapsed_s` in the history is deliberately cumulative so the curve stays
    monotonic across an interruption. Charging the budget against that same
    figure is what turns `max_seconds` into a whole-run cap: once the total
    passes it, every later segment stops at its first evaluation boundary and
    a long cell advances one eval interval per Colab round trip, exhausting
    `--max-segments` far short of the run.

    The other resume tests all use `max_seconds=0.0`, which fires whichever
    clock it reads, so none of them can tell the two apart. This one forces
    the accumulated offset to dwarf the budget and asserts the segment runs
    to completion anyway.
    """
    from src.training.train import train_model

    data, splits, config = _train_inputs(ingested, feature_table)
    table = jnp.asarray(np.zeros((data.vocab.size, FEATURE_DIM), dtype=np.float32))
    train_config = _tiny_train_config()
    out = tmp_path / "budgeted"

    first = train_model(
        data, table, splits.train, splits.val, config, train_config,
        arm="attention", verbose=False, checkpoint_dir=out, max_seconds=0.0,
    )
    assert first["completed"] is False
    assert first["stopped_at_step"] < train_config.total_steps

    # Backdate the run: pretend earlier segments already burned ten hours.
    # Nothing else about the state changes, so a correct resume is unaffected.
    state_path = out / "resume.json"
    state = json.loads(state_path.read_text())
    state["history"][-1]["elapsed_s"] = 36_000.0
    state_path.write_text(json.dumps(state))

    # A budget far below the accumulated offset, but far above anything this
    # 20-step run can spend. Cumulative accounting stops at the next boundary;
    # per-segment accounting finishes the run.
    resumed = train_model(
        data, table, splits.train, splits.val, config, train_config,
        arm="attention", verbose=False, checkpoint_dir=out,
        resume=True, max_seconds=600.0,
    )

    assert resumed["completed"] is True, (
        "the segment stopped early despite a 600s budget it could not have "
        "spent; the budget is being charged the previous segments' time"
    )
    assert resumed["stopped_at_step"] == train_config.total_steps
    # The history stays cumulative -- the fix must not reset that too.
    assert resumed["history"][-1]["elapsed_s"] > 36_000.0
