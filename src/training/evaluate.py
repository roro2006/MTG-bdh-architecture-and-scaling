"""Evaluation over a whole split, and broken down by pick number.

The aggregate loss on this task averages fourteen quite different problems.
A pack of one card has loss identically zero and makes up 7.1% of every
split; a pack of two or three is nearly as forced. Reporting only the
aggregate hides that, and -- more importantly for the scaling study -- an
easy subset that saturates at small N while the hard subset keeps improving
bends the aggregate curve in a way that is easy to misread as an exponent.

So the per-pick table is a first-class output here, not a diagnostic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..data.dataset import PickData
from ..models.pick_model import cross_entropy_loss

DEFAULT_EVAL_BATCH = 1024


def _iter_batches(data: PickData, indices: np.ndarray, batch_size: int):
    for start in range(0, indices.size, batch_size):
        yield data.batch(indices[start : start + batch_size])


def evaluate_split(
    model,
    params,
    feature_table: jnp.ndarray,
    data: PickData,
    indices: np.ndarray,
    batch_size: int = DEFAULT_EVAL_BATCH,
) -> dict:
    """Exact loss and accuracy over every row in `indices`.

    Batches are weighted by their true size rather than averaged, so a
    short final batch cannot skew the result.
    """

    @jax.jit
    def step(params, pack_ids, pool_ids, pack_number, pick_number, label_pos):
        logits = model.apply(
            params, feature_table, pack_ids, pool_ids, pack_number, pick_number
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        chosen = jnp.take_along_axis(log_probs, label_pos[:, None], axis=-1)[:, 0]
        correct = (logits.argmax(axis=-1) == label_pos).astype(jnp.float32)
        return -chosen.sum(), correct.sum()

    total_loss = 0.0
    total_correct = 0.0
    total_rows = 0
    for batch in _iter_batches(data, indices, batch_size):
        loss_sum, correct_sum = step(
            params,
            jnp.asarray(batch["pack_ids"]),
            jnp.asarray(batch["pool_ids"]),
            jnp.asarray(batch["pack_number"]),
            jnp.asarray(batch["pick_number"]),
            jnp.asarray(batch["label_pos"]),
        )
        total_loss += float(loss_sum)
        total_correct += float(correct_sum)
        total_rows += len(batch["label_pos"])

    return {
        "loss": total_loss / max(total_rows, 1),
        "accuracy": total_correct / max(total_rows, 1),
        "rows": total_rows,
    }


def evaluate_by_pick(
    model,
    params,
    feature_table: jnp.ndarray,
    data: PickData,
    indices: np.ndarray,
    batch_size: int = DEFAULT_EVAL_BATCH,
) -> list[dict]:
    """One row per pick number, plus the mean pack size at that pick."""
    results = []
    for pick_number in range(int(data.pick_number.max()) + 1):
        selected = indices[data.pick_number[indices] == pick_number]
        if selected.size == 0:
            continue
        metrics = evaluate_split(
            model, params, feature_table, data, selected, batch_size
        )
        metrics["pick_number"] = pick_number
        metrics["mean_pack_size"] = float(data.pack_size[selected].mean())
        metrics["uniform_loss"] = float(
            np.log(data.pack_size[selected].astype(np.float64)).mean()
        )
        results.append(metrics)
    return results


def summarise_by_pick(rows: list[dict], decision_picks: int = 9) -> dict:
    """Aggregates the per-pick table two ways: over everything, and over the
    picks where a real decision exists.

    `decision_picks` defaults to 9, i.e. picks 0-8, where the pack still
    holds six or more cards.
    """

    def weighted(subset, key):
        total = sum(r["rows"] for r in subset)
        if total == 0:
            return 0.0
        return sum(r[key] * r["rows"] for r in subset) / total

    decisions = [r for r in rows if r["pick_number"] < decision_picks]
    trivial = [r for r in rows if r["mean_pack_size"] <= 1.0]
    total_rows = sum(r["rows"] for r in rows)
    return {
        "all_picks": {
            "loss": weighted(rows, "loss"),
            "accuracy": weighted(rows, "accuracy"),
            "uniform_loss": weighted(rows, "uniform_loss"),
            "rows": total_rows,
        },
        "decision_picks": {
            "loss": weighted(decisions, "loss"),
            "accuracy": weighted(decisions, "accuracy"),
            "uniform_loss": weighted(decisions, "uniform_loss"),
            "rows": sum(r["rows"] for r in decisions),
            "picks": f"0-{decision_picks - 1}",
        },
        "forced_rows": sum(r["rows"] for r in trivial),
        "forced_fraction": sum(r["rows"] for r in trivial) / max(total_rows, 1),
    }


def format_by_pick(rows: list[dict]) -> str:
    lines = [
        f"{'pick':>4} {'pack':>5} {'rows':>9} {'loss':>8} {'uniform':>8} {'acc':>7}"
    ]
    for row in rows:
        lines.append(
            f"{row['pick_number']:>4} {row['mean_pack_size']:>5.1f} {row['rows']:>9,} "
            f"{row['loss']:>8.4f} {row['uniform_loss']:>8.4f} {row['accuracy']:>7.4f}"
        )
    return "\n".join(lines)
