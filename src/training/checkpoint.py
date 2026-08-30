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
