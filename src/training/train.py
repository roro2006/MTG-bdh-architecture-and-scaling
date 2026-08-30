"""Trains one model: the unit of work a grid cell runs.

Kept deliberately plain -- one optimiser, one schedule, no tricks -- because
the scaling grid needs every cell to differ only in the axes being swept
(architecture, width, data fraction, seed). Anything clever that helps one
cell more than another shows up as a bend in the fitted curve and gets
misread as an exponent.

See docs/PROJECT_PLAN.md sections 4-5.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..data.dataset import PickData
from ..models.pick_model import (
    ModelConfig,
    count_params_actual,
    count_params_analytic,
    cross_entropy_loss,
    init_model,
)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 512
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    total_steps: int = 2_000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    eval_every: int = 200
    eval_batches: int = 20


def uniform_baseline(data: PickData, indices: np.ndarray) -> float:
    """Loss of guessing uniformly among the cards in the pack.

    The floor any model must beat to have learned anything at all. Note it
    is not 0 even for a perfect model, and not comparable across pick
    numbers, since pack size shrinks through a round.
    """
    return float(np.log(data.pack_size[indices].astype(np.float64)).mean())


def frequency_baseline(
    data: PickData, train_indices: np.ndarray, eval_indices: np.ndarray
) -> tuple[float, float]:
    """Score every card by how often it is taken when present, ignoring pool.

    A real baseline rather than a strawman: card quality alone explains a
    lot of drafting, and any claim that the model has learned *synergy*
    has to be measured against this, not against the uniform guess.
    """
    vocab_size = data.vocab.size
    taken = np.bincount(data.label[train_indices], minlength=vocab_size).astype(np.float64)
    seen = np.bincount(
        data.pack[train_indices][data.pack[train_indices] >= 0], minlength=vocab_size
    ).astype(np.float64)
    # Laplace smoothing keeps a card that never appears from scoring -inf.
    rate = (taken + 1.0) / (seen + 2.0)
    scores = np.log(rate)

    packs = data.pack[eval_indices]
    mask = packs >= 0
    card_scores = np.where(mask, scores[np.clip(packs, 0, None)], -1e9)
    shifted = card_scores - card_scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted) * mask
    log_probs = shifted - np.log(exp.sum(axis=1, keepdims=True))
    label_pos = data.label_pos[eval_indices]
    loss = -log_probs[np.arange(len(eval_indices)), label_pos].mean()
    accuracy = (card_scores.argmax(axis=1) == label_pos).mean()
    return float(loss), float(accuracy)


def _batch_stream(
    data: PickData, indices: np.ndarray, batch_size: int, seed: int
):
    """Infinite shuffled stream of batches drawn from `indices`."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(indices)
    cursor = 0
    while True:
        if cursor + batch_size > order.size:
            order = rng.permutation(indices)
            cursor = 0
        chunk = order[cursor : cursor + batch_size]
        cursor += batch_size
        yield data.batch(chunk)


def train_model(
    data: PickData,
    feature_table: jnp.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    arm: str = "attention",
    verbose: bool = True,
) -> dict:
    """Trains one cell and returns its metrics plus the metadata the fit needs."""
    model, params = init_model(model_config, feature_table, arm=arm, seed=train_config.seed)
    analytic = count_params_analytic(model_config, arm=arm)
    actual = count_params_actual(params)
    if analytic["total"] != actual:
        raise AssertionError(
            f"analytic parameter count {analytic['total']:,} != realised {actual:,}; "
            "the derivation in pick_model.py is stale"
        )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=train_config.learning_rate * 0.05,
        peak_value=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        decay_steps=max(train_config.total_steps, train_config.warmup_steps + 1),
        end_value=train_config.learning_rate * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_config.grad_clip),
        optax.adamw(schedule, weight_decay=train_config.weight_decay),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            logits = model.apply(
                p, feature_table, batch["pack_ids"], batch["pool_ids"],
                batch["pack_number"], batch["pick_number"],
            )
            return cross_entropy_loss(logits, batch["label_pos"])

        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, accuracy

    @jax.jit
    def eval_step(params, batch):
        logits = model.apply(
            params, feature_table, batch["pack_ids"], batch["pool_ids"],
            batch["pack_number"], batch["pick_number"],
        )
        return cross_entropy_loss(logits, batch["label_pos"])

    def evaluate() -> tuple[float, float]:
        stream = _batch_stream(
            data, val_indices, train_config.batch_size, seed=train_config.seed + 9999
        )
        losses, accuracies = [], []
        for _ in range(train_config.eval_batches):
            batch = {k: jnp.asarray(v) for k, v in next(stream).items()}
            loss, accuracy = eval_step(params, batch)
            losses.append(float(loss))
            accuracies.append(float(accuracy))
        return float(np.mean(losses)), float(np.mean(accuracies))

    stream = _batch_stream(data, train_indices, train_config.batch_size, train_config.seed)
    history: list[dict] = []
    started = time.monotonic()

    for step in range(1, train_config.total_steps + 1):
        batch = {k: jnp.asarray(v) for k, v in next(stream).items()}
        params, opt_state, loss, accuracy = train_step(params, opt_state, batch)

        if step % train_config.eval_every == 0 or step == train_config.total_steps:
            val_loss, val_accuracy = evaluate()
            elapsed = time.monotonic() - started
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss),
                    "train_accuracy": float(accuracy),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "elapsed_s": elapsed,
                }
            )
            if verbose:
                print(
                    f"  step {step:>6,}  train {float(loss):.4f}  val {val_loss:.4f}  "
                    f"val_acc {val_accuracy:.4f}  ({elapsed:,.0f}s, "
                    f"{step * train_config.batch_size / elapsed:,.0f} ex/s)",
                    flush=True,
                )

    val_loss, val_accuracy = evaluate()
    return {
        "arm": arm,
        "hidden_dim": model_config.hidden_dim,
        "num_params": actual,
        "param_breakdown": analytic,
        "train_examples": int(train_indices.size),
        "steps": train_config.total_steps,
        "tokens_seen": int(train_config.total_steps * train_config.batch_size),
        "final_val_loss": val_loss,
        "final_val_accuracy": val_accuracy,
        "uniform_baseline": uniform_baseline(data, val_indices),
        "history": history,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "elapsed_s": time.monotonic() - started,
        "params": params,
    }
