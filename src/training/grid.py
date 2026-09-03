"""Runs the (architecture x model size x data fraction x seed) sweep
described in docs/PROJECT_PLAN.md section 4.

Two things shape this module beyond the plan's original description, both
of them consequences of measurement rather than preference.

**The grid is L-shaped, not a full Cartesian product.** A full product
spends most of its compute in the corner where N and D are both large, and
that corner is the least informative per FLOP: alpha is fit by varying N at
fixed D, and beta by varying D at fixed N, so neither exponent needs the
expensive intersection. Measured against the FLOP accounting, the full
product is 0.53 EFLOP and the L-shape is 0.29 -- a 46% saving that costs
neither a seed nor the top of the size range.

What it does cost is stated plainly: the design assumes the surface really
is separable, which is what `E + A/N^alpha + B/D^beta` asserts by having no
interaction term. That makes the design consistent with the model being fit
but less able to *detect* that the model is wrong, so `full_grid` adds a
few interior points whose only job is to sit off the fitted surface if the
functional form is inadequate. They are cheap and they are the difference
between assuming separability and having checked it.

**Cells are resumable.** Each writes its own result file and a cell whose
file already exists is skipped, so an interrupted sweep continues instead
of restarting. On a free Colab runtime -- which is where this is designed
to run, and which disconnects on a timer -- that is the difference between
a grid that finishes and one that does not. Cells are ordered largest
first for the same reason: the expensive ones should be attempted while the
session is fresh, not discovered at the end of it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..data.dataset import PickData, Splits, decision_rows
from ..models.flops import count_flops_analytic
from ..models.pick_model import ModelConfig, count_params_analytic
from .run import subsample_by_draft
from .train import TrainConfig, train_model

# Widths chosen so N lands on a log-spaced ladder from ~0.5M to ~50M, the
# range PROJECT_PLAN.md section 4 asks for. Derived once with
# count_params_analytic rather than recomputed, so the ladder is a fixed
# property of the experiment and not something that drifts with a config
# default; `test_grid.py` asserts they still hit their targets.
LADDER = (90, 142, 226, 360, 570, 904)

# The D axis. One epoch over the subsampled set, so a fraction is a real
# reduction in data seen rather than the same data seen fewer times.
FRACTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625)

# Where the D sweep is run. Cheap enough that all five fractions cost less
# than one cell at the top of the ladder, which is the whole point.
D_ANCHORS = (142, 226)

ARMS = ("attention", "bdh")

# FIN's training split after `decision_rows` removes the forced picks, which
# is what a cell actually iterates over since `run_cell` drops them by
# default. Using the pre-drop 4,711,938 here would overstate every estimate
# by 7.14%, and an estimate that does not describe the run it is estimating
# is worse than no estimate.
DEFAULT_TRAIN_ROWS = 4_375_371


@dataclass(frozen=True)
class GridCell:
    architecture: str  # "attention" or "bdh"
    hidden_dim: int
    data_fraction: float
    seed: int
    role: str = "N"  # "N", "D", "interior", or "pilot" -- why the cell exists

    @property
    def name(self) -> str:
        """Filename-safe identity. Two cells with the same name are the
        same experiment, which is what makes skipping completed work safe.
        """
        frac = f"{self.data_fraction:g}".replace(".", "p")
        return f"{self.architecture}_d{self.hidden_dim}_f{frac}_s{self.seed}"

    def config(self, card_feature_dim: int = 65, **overrides) -> ModelConfig:
        return ModelConfig(
            hidden_dim=self.hidden_dim,
            card_feature_dim=card_feature_dim,
            **overrides,
        )

    @property
    def num_params(self) -> int:
        """N for the fit, derived rather than measured.

        `train_model` asserts this against the realised tree on every cell,
        so a drift between the derivation and the model fails the run
        instead of quietly shifting the x-axis of the scaling law.
        """
        return count_params_analytic(self.config(), self.architecture)["total"]


def pilot_grid() -> list[GridCell]:
    """Small and fast, and its only job is to fail.

    PROJECT_PLAN.md section 9 puts a pilot before the real grid so pipeline
    bugs surface on cheap cells. It sweeps both arms, two widths and two
    fractions -- enough that a broken D axis or a broken arm switch shows
    up -- at widths whose cells run in minutes.
    """
    return [
        GridCell(arm, d, f, seed=0, role="pilot")
        for d in (90, 142)
        for f in (1.0, 0.25)
        for arm in ARMS
    ]


def full_grid(seeds: tuple[int, ...] = (0, 1)) -> list[GridCell]:
    """The L-shaped design. See the module docstring for why.

    Ordered largest-cell-first, so an interrupted run has done the
    expensive work rather than saved it for last.
    """
    cells: list[GridCell] = []

    # The N arm of the L: every size, at full data.
    cells += [
        GridCell(arm, d, 1.0, s, role="N")
        for d in LADDER
        for arm in ARMS
        for s in seeds
    ]

    # The D arm of the L: every fraction, at the cheap anchors. Fraction
    # 1.0 is already covered above, so it is not repeated.
    cells += [
        GridCell(arm, d, f, s, role="D")
        for d in D_ANCHORS
        for f in FRACTIONS
        if f != 1.0
        for arm in ARMS
        for s in seeds
    ]

    # Interior points. Not needed to fit either exponent -- they exist to
    # be compared against the surface fitted without them, and a systematic
    # residual here means the separable form is wrong. One seed each: this
    # is a specification check, not an estimate.
    cells += [
        GridCell(arm, d, f, seeds[0], role="interior")
        for d, f in ((360, 0.25), (570, 0.5), (360, 0.0625))
        for arm in ARMS
    ]

    return order_by_cost(cells)


def cell_flops(cell: GridCell, train_rows: int) -> float:
    """Training FLOPs for one cell: one epoch over its subsampled split.

    Used for ordering and for budgeting a session, so it counts the same
    dense arithmetic the hardware actually runs -- `flops.py` deliberately
    counts padding, because a dense pass pays for it.
    """
    per_example = count_flops_analytic(
        cell.config(), arm=cell.architecture, include_backward=True
    )["total"]
    return per_example * cell.data_fraction * train_rows


def order_by_cost(cells: list[GridCell], train_rows: int = DEFAULT_TRAIN_ROWS) -> list[GridCell]:
    return sorted(cells, key=lambda c: -cell_flops(c, train_rows))


def estimate(cells: list[GridCell], tflops: float, train_rows: int = DEFAULT_TRAIN_ROWS) -> dict:
    """What this grid costs at an assumed achieved throughput.

    `tflops` is *achieved*, not peak, and it is the soft number in any such
    estimate: a small cell is bound by launch overhead and memory traffic
    rather than arithmetic, so a pure FLOP roofline will misprice it. Treat
    the result as a budget, and measure a real cell before trusting it.
    """
    total = sum(cell_flops(c, train_rows) for c in cells)
    return {
        "cells": len(cells),
        "total_flops": total,
        "hours": total / (tflops * 1e12) / 3600,
        "largest_cell_hours": (
            max(cell_flops(c, train_rows) for c in cells) / (tflops * 1e12) / 3600
            if cells else 0.0
        ),
    }


def run_cell(
    cell: GridCell,
    data: PickData,
    feature_table: jnp.ndarray,
    splits: Splits,
    *,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    epochs: float = 1.0,
    eval_every: int = 250,
    drop_forced: bool = True,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Trains one grid cell and returns what the curve fit needs.

    `drop_forced` removes the rows whose gradient is identically zero -- a
    one-card pack has one admissible answer, so it cannot teach the model
    anything and only occupies a slot in a batch. That is 7.14% of the FIN
    training split. It is on by default because it is not a trade: the
    rows carry no signal to lose.

    It does mean D counts *decisions*, not rows, and the returned
    `train_rows` says which. That distinction matters downstream: the
    fitted E is compared against a human-disagreement floor that section 6a
    has already concluded must be measured where a real decision exists, so
    both sides of that comparison have to be the same population.
    """
    train_indices = subsample_by_draft(
        data, splits.train, cell.data_fraction, seed=cell.seed
    )
    if drop_forced:
        train_indices = decision_rows(data, train_indices)

    steps = max(1, int(round(epochs * train_indices.size / batch_size)))
    model_config = cell.config(card_feature_dim=int(feature_table.shape[1]))
    train_config = TrainConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        total_steps=steps,
        seed=cell.seed,
        eval_every=eval_every,
    )

    started = time.time()
    result = train_model(
        data, feature_table, train_indices, splits.val,
        model_config, train_config, arm=cell.architecture,
        verbose=True, checkpoint_dir=checkpoint_dir,
    )

    return {
        "cell": asdict(cell),
        "name": cell.name,
        "role": cell.role,
        # The two axes of the fit, both recorded rather than inferred.
        "num_params": cell.num_params,
        "train_rows": int(train_indices.size),
        "examples_seen": int(steps * batch_size),
        "forced_rows_dropped": bool(drop_forced),
        # Dense FLOPs: what the hardware runs. See ARCHITECTURE.md.
        "flops_per_example": count_flops_analytic(
            model_config, arm=cell.architecture, include_backward=True
        )["total"],
        "steps": steps,
        "final_val_loss": result["final_val_loss"],
        "best_val_loss": result["best_val_loss"],
        "best_step": result["best_step"],
        "history": result["history"],
        "elapsed_s": time.time() - started,
    }


def run_grid(
    cells: list[GridCell],
    data: PickData,
    feature_table: jnp.ndarray,
    splits: Splits,
    out_dir: str | Path,
    *,
    skip_completed: bool = True,
    **cell_kwargs,
) -> list[dict]:
    """Runs cells in order, one result file each, skipping what is done.

    Writing per cell rather than at the end is what makes an interrupted
    session recoverable, and it is not a hypothetical: a free Colab runtime
    disconnects on a timer, and the grid is longer than the timer. Point
    `out_dir` at Drive rather than local disk, which is wiped on
    disconnect.

    A cell is skipped on the presence of its result file alone. That makes
    a rerun idempotent and cheap to restart, and it means deleting a
    result file is how you ask for a cell to be run again.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for i, cell in enumerate(cells, 1):
        target = out_dir / f"{cell.name}.json"
        if skip_completed and target.exists():
            print(f"[{i}/{len(cells)}] {cell.name}: done, skipping")
            results.append(json.loads(target.read_text(encoding="utf-8")))
            continue

        print(f"[{i}/{len(cells)}] {cell.name} ({cell.role}) starting")
        result = run_cell(
            cell, data, feature_table, splits,
            checkpoint_dir=out_dir / cell.name,
            **cell_kwargs,
        )
        # Written only after the cell completes, so a file's existence
        # always means a finished cell rather than an interrupted one.
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        print(f"[{i}/{len(cells)}] {cell.name}: best val {result['best_val_loss']:.4f}")

    return results


def load_results(out_dir: str | Path) -> list[dict]:
    """Every completed cell in `out_dir`, for the fit to consume."""
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(out_dir).glob("*.json"))
    ]
