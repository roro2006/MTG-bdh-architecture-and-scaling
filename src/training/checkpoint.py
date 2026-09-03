"""Saving and restoring a trained cell.

One directory per run holding `params.msgpack` and `metadata.json`. The
metadata carries everything needed to rebuild the model without the data
pipeline, so a checkpoint can be reloaded and evaluated on its own.

flax's msgpack rather than orbax: a grid cell is 1MB to 64MB, so there is
nothing to shard, and a single self-contained file per cell is easier to
copy off a training machine, diff, and reason about than a directory with
async writers.

One thing to know about `flax.serialization.from_bytes`: despite taking a
template tree, it does **not** validate against it. Restoring a d=32
checkpoint into a freshly initialised d=64 tree returns the d=32 arrays and
raises nothing. Verified directly against flax 0.12.9. Nor does comparing
the restored parameter count to the metadata catch it, since both come from
the same file and agree with each other.

So `restore` checks the tree explicitly -- structure, then per-leaf shape
and dtype -- against a freshly initialised model. Without that, a
checkpoint written before an architecture change would load silently into
the wrong shapes, evaluate without complaint, and report a plausible but
wrong loss for a grid cell. A scaling fit absorbs that kind of error rather
than revealing it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from ..models.pick_model import ModelConfig, PickModel, count_params_actual, init_model

PARAMS_FILE = "params.msgpack"
METADATA_FILE = "metadata.json"


def save_checkpoint(
    directory: str | Path,
    params: Any,
    *,
    model_config: ModelConfig,
    arm: str,
    train_config: Any = None,
    metrics: dict | None = None,
) -> Path:
    """Writes params + metadata. Returns the directory written to."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / PARAMS_FILE).write_bytes(serialization.to_bytes(params))
    metadata = {
        "arm": arm,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config) if train_config is not None else None,
        "num_params": count_params_actual(params),
        "metrics": metrics or {},
    }
    (directory / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    return directory


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def load_metadata(directory: str | Path) -> dict:
    return json.loads((Path(directory) / METADATA_FILE).read_text(encoding="utf-8"))


def restore(directory: str | Path) -> tuple[PickModel, Any, dict]:
    """Rebuilds the model and its trained parameters from a checkpoint.

    Does not need the real feature table: the model is initialised against
    a zero table of the right width purely to obtain the parameter tree's
    structure, which the saved bytes are then restored into.
    """
    directory = Path(directory)
    metadata = load_metadata(directory)
    config = ModelConfig(**metadata["model_config"])

    dummy_table = jnp.zeros((1, config.card_feature_dim), dtype=jnp.float32)
    model, template = init_model(config, dummy_table, arm=metadata["arm"], seed=0)

    restored = serialization.from_bytes(
        template, (directory / PARAMS_FILE).read_bytes()
    )
    _check_matches_template(restored, template, directory)

    # from_bytes hands back numpy arrays; convert so jit sees the same types
    # it saw during training.
    restored = jax.tree_util.tree_map(jnp.asarray, restored)

    actual = count_params_actual(restored)
    if actual != metadata["num_params"]:
        raise ValueError(
            f"checkpoint at {directory} holds {actual:,} parameters but its "
            f"metadata records {metadata['num_params']:,}"
        )
    return model, restored, metadata


def _check_matches_template(restored: Any, template: Any, directory: Path) -> None:
    """Raises unless `restored` has exactly the template's structure and shapes.

    `from_bytes` will not do this itself -- see the module docstring.
    """
    restored_structure = jax.tree_util.tree_structure(restored)
    template_structure = jax.tree_util.tree_structure(template)
    if restored_structure != template_structure:
        raise ValueError(
            f"checkpoint at {directory} does not match the current model "
            f"definition: parameter tree structure differs. "
            f"checkpoint={restored_structure} model={template_structure}"
        )

    restored_leaves = jax.tree_util.tree_leaves_with_path(restored)
    template_leaves = jax.tree_util.tree_leaves(template)
    for (path, saved), expected in zip(restored_leaves, template_leaves):
        if saved.shape != expected.shape:
            name = "/".join(str(k.key) for k in path if hasattr(k, "key"))
            raise ValueError(
                f"checkpoint at {directory} does not match the current model "
                f"definition: parameter {name!r} has shape {saved.shape}, but "
                f"the model expects {expected.shape}. The architecture changed "
                "since this checkpoint was written."
            )
        if np.dtype(saved.dtype) != np.dtype(expected.dtype):
            name = "/".join(str(k.key) for k in path if hasattr(k, "key"))
            raise ValueError(
                f"checkpoint at {directory}: parameter {name!r} has dtype "
                f"{saved.dtype}, model expects {expected.dtype}"
            )


# --------------------------------------------------------------------------
# Resume state
# --------------------------------------------------------------------------
#
# `save_checkpoint` above writes the *result* of a run: the best-val parameters
# and enough metadata to rebuild and evaluate the model. That is the right
# artefact to keep, and the wrong thing to restart from -- it carries no
# optimiser moments, no step counter, and no position in the shuffled batch
# stream, so reloading it would restart Adam from zero on a decayed learning
# rate and replay data the run had already seen.
#
# A Colab session caps at 12h and dies after 90 minutes idle, so a long cell
# will be interrupted. These functions persist everything needed to continue a
# run exactly where it stopped.
#
# Two things this deliberately does not do:
#
#   - resume across a config change. The fingerprint below is checked on load
#     and a mismatch refuses rather than adapts. Silently continuing a d=64 run
#     into a d=128 tree is the failure mode checkpoint.py already guards
#     against for parameters, and it is worse here: the loss curve would be
#     continuous and the result meaningless.
#   - resume approximately. The batch stream's position is restored exactly
#     (see BatchStream in train.py), so a resumed run sees the same examples in
#     the same order as an uninterrupted one. tests/test_checkpoint.py asserts
#     the two agree step for step.

RESUME_PARAMS_FILE = "resume.msgpack"
RESUME_STATE_FILE = "resume.json"


def resume_fingerprint(
    *,
    model_config: ModelConfig,
    train_config: Any,
    arm: str,
    train_rows: int,
) -> dict:
    """The set of things that must match for a resume to be meaningful.

    Deliberately includes `train_rows`: --data-fraction and --seed change
    which drafts are in the training set, and resuming across that would
    mix two different datasets into one loss curve.
    """
    return {
        "arm": arm,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "train_rows": int(train_rows),
    }


def save_resume(
    directory: str | Path,
    *,
    params: Any,
    opt_state: Any,
    best_params: Any,
    step: int,
    best_val: float,
    best_step: int,
    history: list,
    stream_state: dict,
    fingerprint: dict,
) -> Path:
    """Persists everything needed to continue this run.

    Written atomically (temp file + os.replace) because the process may be
    killed at any moment -- that is the whole reason this exists. A
    half-written resume file that still parses would be worse than none.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    blob = serialization.to_bytes(
        {"params": params, "opt_state": opt_state, "best_params": best_params}
    )
    _atomic_write_bytes(directory / RESUME_PARAMS_FILE, blob)

    state = {
        "step": int(step),
        "best_val": float(best_val),
        "best_step": int(best_step),
        "history": history,
        "stream_state": stream_state,
        "fingerprint": fingerprint,
    }
    _atomic_write_text(
        directory / RESUME_STATE_FILE,
        json.dumps(state, indent=2, default=_json_default),
    )
    return directory


def load_resume(
    directory: str | Path,
    *,
    params_template: Any,
    opt_state_template: Any,
    fingerprint: dict,
) -> dict | None:
    """Returns the saved run state, or None if there is nothing to resume.

    Raises if a resume file exists but was written by a differently
    configured run -- see the module note above.
    """
    directory = Path(directory)
    params_path = directory / RESUME_PARAMS_FILE
    state_path = directory / RESUME_STATE_FILE
    if not (params_path.exists() and state_path.exists()):
        return None

    state = json.loads(state_path.read_text(encoding="utf-8"))
    saved = state.get("fingerprint", {})
    if saved != fingerprint:
        raise ValueError(
            f"cannot resume from {directory}: it was written by a run with a "
            f"different configuration.\n"
            f"  saved:   {_fingerprint_diff(saved, fingerprint)[0]}\n"
            f"  current: {_fingerprint_diff(saved, fingerprint)[1]}\n"
            "Delete the directory to start fresh, or point --out-dir elsewhere."
        )

    template = {
        "params": params_template,
        "opt_state": opt_state_template,
        "best_params": params_template,
    }
    restored = serialization.from_bytes(template, params_path.read_bytes())

    return {
        "params": _restore_into_template(restored["params"], params_template),
        "opt_state": _restore_into_template(restored["opt_state"], opt_state_template),
        "best_params": _restore_into_template(restored["best_params"], params_template),
        "step": int(state["step"]),
        "best_val": float(state["best_val"]),
        "best_step": int(state["best_step"]),
        "history": state["history"],
        "stream_state": state["stream_state"],
    }


def _restore_into_template(restored: Any, template: Any) -> Any:
    """Rebuilds `restored` with the template's exact pytree node types.

    `serialization.from_bytes` hands optax's NamedTuple states back as plain
    tuples and every leaf as a numpy array. optax happens to tolerate the
    former today because its combinators index positionally, but anything
    reading `state.count` by attribute would break, and jit would retrace on
    the numpy leaves. Reflattening onto the template's treedef restores the
    real types; it also fails loudly if the two structures disagree.
    """
    leaves = jax.tree_util.tree_leaves(restored)
    treedef = jax.tree_util.tree_structure(template)
    expected = treedef.num_leaves
    if len(leaves) != expected:
        raise ValueError(
            f"resume state has {len(leaves)} arrays but the current model "
            f"expects {expected}; the architecture changed since it was written"
        )
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in leaves])


def _fingerprint_diff(saved: dict, current: dict) -> tuple[str, str]:
    """The differing keys only -- a full config dump buries the one that moved."""
    keys = sorted(set(saved) | set(current))
    differing = [k for k in keys if saved.get(k) != current.get(k)]
    if not differing:
        return (repr(saved), repr(current))
    return (
        ", ".join(f"{k}={saved.get(k)!r}" for k in differing),
        ", ".join(f"{k}={current.get(k)!r}" for k in differing),
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def clear_resume(directory: str | Path) -> None:
    """Removes resume state once a run has finished.

    Leaving it behind would make a re-run of a completed cell a no-op that
    silently reports the old numbers.
    """
    directory = Path(directory)
    for name in (RESUME_PARAMS_FILE, RESUME_STATE_FILE):
        (directory / name).unlink(missing_ok=True)
