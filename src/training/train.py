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
from pathlib import Path

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


class BatchStream:
    """Infinite shuffled stream of batches drawn from `indices`.

    A class rather than a generator so its position can be saved and
    restored. Resuming a run that replayed data it had already seen -- or
    skipped data it had not -- would put a discontinuity in the loss curve
    at every interruption, and the scaling fit would read that as
    structure rather than as an artefact of the 12h session cap.

    The position is `(reshuffles, cursor)` rather than the permuted order
    itself: `order` is a few million int64s, far too large to sit in a JSON
    resume file, and it is anyway a pure function of the seed and the
    number of reshuffles so far. Restoring replays that many permutations,
    which costs milliseconds and reproduces the order exactly.
    """

    def __init__(
        self, data: PickData, indices: np.ndarray, batch_size: int, seed: int
    ):
        self._data = data
        self._indices = indices
        self._batch_size = batch_size
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._reshuffles = 0
        self._cursor = 0
        self._order = self._reshuffle()

    def _reshuffle(self) -> np.ndarray:
        self._reshuffles += 1
        self._cursor = 0
        return self._rng.permutation(self._indices)

    def __iter__(self) -> "BatchStream":
        return self

    def __next__(self) -> dict:
        if self._cursor + self._batch_size > self._order.size:
            self._order = self._reshuffle()
        chunk = self._order[self._cursor : self._cursor + self._batch_size]
        self._cursor += self._batch_size
        return self._data.batch(chunk)

    def state(self) -> dict:
        return {"reshuffles": self._reshuffles, "cursor": self._cursor}

    def restore(self, state: dict) -> None:
        """Winds the stream forward to a previously saved position."""
        target = int(state["reshuffles"])
        self._rng = np.random.default_rng(self._seed)
        self._reshuffles = 0
        self._order = self._indices
        for _ in range(target):
            self._order = self._rng.permutation(self._indices)
            self._reshuffles += 1
        self._cursor = int(state["cursor"])


def _batch_stream(
    data: PickData, indices: np.ndarray, batch_size: int, seed: int
) -> BatchStream:
    return BatchStream(data, indices, batch_size, seed)


def train_model(
    data: PickData,
    feature_table: jnp.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    arm: str = "attention",
    verbose: bool = True,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    max_seconds: float | None = None,
) -> dict:
    """Trains one cell and returns its metrics plus the metadata the fit needs.

    If `checkpoint_dir` is given, the parameters are written there whenever
    validation loss improves. A grid cell is tens of minutes of compute;
    losing all of it to a crash at step 2,900 is avoidable, and the
    best-val parameters are what any later analysis wants anyway.

    Alongside that best-val artefact, a full resume state (optimiser
    moments, step counter, batch-stream position) is written at every
    evaluation boundary. `resume=True` picks it up and continues; see the
    note in checkpoint.py for why the two are separate files.

    `max_seconds` stops the run cleanly at the next evaluation boundary once
    that much wall clock has elapsed, leaving a resumable state behind. A
    Colab session dies at 12h and after 90 minutes idle, so a long cell has
    to be run as a sequence of bounded segments; the returned `completed`
    flag says whether this was the last one.
    """
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

    stream = BatchStream(
        data, train_indices, train_config.batch_size, train_config.seed
    )
    history: list[dict] = []
    best_val = float("inf")
    best_params = params
    best_step = 0
    start_step = 0

    # Imported here for the same reason the save below is: checkpoint.py
    # imports from pick_model, and keeping these local leaves it free to
    # import from this module later without a cycle.
    from .checkpoint import (
        clear_resume,
        load_resume,
        resume_fingerprint,
        save_checkpoint,
        save_resume,
    )

    fingerprint = resume_fingerprint(
        model_config=model_config,
        train_config=train_config,
        arm=arm,
        train_rows=int(train_indices.size),
    )
    if resume and checkpoint_dir is not None:
        saved = load_resume(
            checkpoint_dir,
            params_template=params,
            opt_state_template=opt_state,
            fingerprint=fingerprint,
        )
        if saved is not None:
            params = saved["params"]
            opt_state = saved["opt_state"]
            best_params = saved["best_params"]
            best_val = saved["best_val"]
            best_step = saved["best_step"]
            history = saved["history"]
            start_step = saved["step"]
            stream.restore(saved["stream_state"])
            if verbose:
                print(
                    f"  resuming at step {start_step:,} of "
                    f"{train_config.total_steps:,} "
                    f"(best val {best_val:.4f} at step {best_step:,})",
                    flush=True,
                )

    if start_step >= train_config.total_steps:
        # Nothing left to do. Fall through so the caller still gets a full
        # result dict rather than having to special-case a finished run.
        if verbose:
            print("  already complete; nothing to train", flush=True)

    # Wall clock accumulated by earlier segments, so `elapsed_s` in the
    # history stays monotonic across an interruption instead of resetting.
    elapsed_offset = history[-1]["elapsed_s"] if history else 0.0
    started = time.monotonic()
    completed = True

    def _elapsed() -> float:
        return elapsed_offset + (time.monotonic() - started)

    # `max_seconds` is the budget for THIS segment, so it is measured from
    # this process's start and must not carry the offset. Charging a segment
    # for time earlier segments already spent turns the budget into a
    # whole-run cap: once the cumulative total passes it, every later segment
    # stops at its first evaluation boundary, advancing one eval interval per
    # Colab round trip and exhausting --max-segments far short of the run.
    def _segment_elapsed() -> float:
        return time.monotonic() - started

    def _write_resume(step: int) -> None:
        if checkpoint_dir is None:
            return
        save_resume(
            checkpoint_dir,
            params=params,
            opt_state=opt_state,
            best_params=best_params,
            step=step,
            best_val=best_val,
            best_step=best_step,
            history=history,
            stream_state=stream.state(),
            fingerprint=fingerprint,
        )

    # Seeded so `stopped_at` is well defined even when the loop body never
    # runs, which is what an already-finished run being re-resumed looks like.
    step = start_step
    for step in range(start_step + 1, train_config.total_steps + 1):
        batch = {k: jnp.asarray(v) for k, v in next(stream).items()}
        params, opt_state, loss, accuracy = train_step(params, opt_state, batch)

        if step % train_config.eval_every == 0 or step == train_config.total_steps:
            val_loss, val_accuracy = evaluate()
            elapsed = _elapsed()
            improved = val_loss < best_val
            if improved:
                best_val, best_params, best_step = val_loss, params, step
                if checkpoint_dir is not None:
                    save_checkpoint(
                        checkpoint_dir,
                        params,
                        model_config=model_config,
                        arm=arm,
                        train_config=train_config,
                        metrics={"step": step, "val_loss": val_loss,
                                 "val_accuracy": val_accuracy},
                    )
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
                    f"{step * train_config.batch_size / elapsed:,.0f} ex/s)"
                    f"{'  *' if improved else ''}",
                    flush=True,
                )

            # Resume state is written at every evaluation boundary, not only
            # on improvement: a run killed at step 2,900 must restart from
            # 2,750, not from whichever earlier step last happened to be a
            # new best.
            _write_resume(step)

            # `step < total_steps` so the budget can never fire on the last
            # step: the run is finished at that point, and reporting it as
            # incomplete would send the caller into a resume that has nothing
            # left to do.
            if (
                max_seconds is not None
                and step < train_config.total_steps
                and _segment_elapsed() >= max_seconds
            ):
                completed = False
                if verbose:
                    print(
                        f"  segment budget reached ({_segment_elapsed():,.0f}s >= "
                        f"{max_seconds:,.0f}s) at step {step:,} of "
                        f"{train_config.total_steps:,}, {_elapsed():,.0f}s total; "
                        f"state saved, resumable",
                        flush=True,
                    )
                break

    stopped_at = min(step, train_config.total_steps)
    if completed and checkpoint_dir is not None:
        # A finished run must not leave resume state behind: re-running the
        # cell would otherwise find it, decide there is nothing to do, and
        # silently re-report the old numbers.
        clear_resume(checkpoint_dir)

    val_loss, val_accuracy = evaluate()
    return {
        "completed": completed,
        "stopped_at_step": int(stopped_at),
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
        "best_val_loss": best_val,
        "best_step": best_step,
        # The whole run, not this segment: a resumed cell would otherwise
        # report the last segment's slice as if it were the total, and the
        # throughput derived from it -- examples over elapsed -- is the
        # number the grid in docs/PROJECT_PLAN.md is planned against.
        "elapsed_s": _elapsed(),
        "model": model,
        "params": params,
        "best_params": best_params,
    }
