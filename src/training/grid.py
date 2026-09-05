"""Runs the (architecture x model size x data fraction x seed) sweep
described in docs/PROJECT_PLAN.md section 6.

Three things shape this module beyond the plan's original description, all
of them consequences of measurement rather than preference.

**The ladder starts at a measured point and climbs.** See `LADDER` and
`DEFAULT_EPOCHS`: the 92,000-step runs at d=64 put both arms 0.0004 apart
on the headline slice, flat over their last 20,000 steps, with train loss
0.04 below val. That is a capacity ceiling. So the grid spends its compute
going up in width rather than sideways, and it spends three epochs per cell
rather than ten, because the tenth epoch bought 0.001 and the fourth
through tenth together bought 0.012.

**The grid is L-shaped, not a full Cartesian product.** A full product
spends most of its compute in the corner where N and D are both large, and
that corner is the least informative per FLOP: alpha is fit by varying N at
fixed D, and beta by varying D at fixed N, so neither exponent needs the
expensive intersection. Measured against the FLOP accounting at three
epochs, the full product is 0.187 EFLOP and the L-shape is 0.136 -- a 27%
saving that costs neither a seed nor the top of the size range. That margin
was 46% on the previous six-rung ladder and is smaller here for a
structural reason worth knowing: on four rungs the product is not much
bigger than the L, so the L-shape is now bought mostly for the top rung it
keeps rather than for the corner it drops.

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

**It is not an afternoon on a free T4.** Section 6 says "an afternoon on one
rented GPU" and that was written before d=512 was on the ladder. At the
measured T4 throughput the grid is 40 hours and its largest single cell is
11.7 -- inside a 12-hour session only by luck, on a runtime reclaimed after
90 idle minutes. That figure is a floor: `MEASURED_T4_TFLOPS` comes from a
d=64 cell reaching 12% of the device's fp32 peak, and the wide rungs
utilise it far better, which puts the realistic T4 total nearer 13 hours.
On an A100 the same grid is about 3 hours and section 6's sentence is true
again -- but an A100 does not allocate on a free Colab account, so that
option costs money rather than patience. On the free tier the grid is a T4
job measured in sessions, which the segment loop survives and a single
sitting does not.

What *is* comfortable on a free T4 is everything below the top rung.
`neuron_probe` is three cells at 13, 17 and 25 minutes, each inside one
default 30-minute segment, so the question section 4 reopened can be
answered before any of this is committed to.
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
from .evaluate import DEFAULT_EVAL_BATCH, evaluate_by_pick, summarise_by_pick
from .run import subsample_by_draft
from .train import TrainConfig, train_model

# The card feature table's width. It is a property of the representation
# rather than of a corpus -- PROJECT_PLAN.md section 3a requires every
# column to mean the same thing in every set -- so it is a constant here.
# It was 65 before the rebuild and is 119 after it, and the difference is
# not cosmetic: `GridCell.num_params` derives N from this, so a cell that
# derived N at 65 while its model trained at 119 would place the fit's
# x-axis wrong on every point and fail nothing while doing it.
CARD_FEATURE_DIM = 119

# The N axis, starting from a measured point rather than a chosen one.
# d=64 is the width both arms were trained at for 92,000 steps
# (docs/RESULTS.md), so the bottom rung has a known loss, throughput and
# parameter count. Each rung doubles d, which is 4x in N:
#
#     d=64   ->    261,633 (BDH) /    263,745 (attention)
#     d=128  ->  1,022,977       /  1,027,201
#     d=256  ->  4,044,801       /  4,053,249
#     d=512  -> 16,084,993       / 16,101,889
#
# 61x on N across four rungs, which is what alpha is fit over.
#
# The previous ladder (90, 142, 226, 360, 570, 904) was sized for a 65-column
# table and, more importantly, went *sideways* from the anchor rather than up
# from it. The 92,000-step runs are what settle the direction: at d=64 both
# arms flatten by epoch 8, train loss sits 0.04 below val, and the last
# 20,000 steps move inside a 0.004 band. That is a capacity ceiling, not a
# step budget, and a ladder whose bottom rung is already at the ceiling
# cannot measure where the ceiling goes.
LADDER = (64, 128, 256, 512)

# The D axis. Four fractions, which is the top of the 3-4 that
# PROJECT_PLAN.md section 6 asks for. A fraction subsamples *drafts*, so it
# is a real reduction in data rather than the same data seen fewer times --
# see DEFAULT_EPOCHS for the other half of that guarantee.
FRACTIONS = (1.0, 0.5, 0.25, 0.125)

# Where the D sweep is run: the two cheap rungs, so the whole D arm costs
# less than one cell at the top of the ladder.
D_ANCHORS = (64, 128)

# Passes over each cell's training subsample. Three, not ten, and the
# 92,000-step runs are why. Best-val by epoch at d=64, BDH / attention:
#
#     epoch  1   0.8657 / 0.8699
#     epoch  3   0.8407 / 0.8421
#     epoch  4   0.8345 / 0.8367
#     epoch 10   0.8222 / 0.8211
#
# So three epochs costs 0.0185 nats against ten and buys back 3.3x the
# compute -- and it costs it almost identically in both arms (0.0014 apart
# at epoch 3, against 0.0002 at epoch 10), which is what matters for a
# comparison. Ten epochs is buying memorisation: train loss ends 0.04 below
# val and both arms peak at step 88,250 of 92,000.
#
# This is a budget, not a convergence claim, and it has a cost worth stating
# rather than burying: every cell is truncated, larger cells are further
# from their own converged loss than smaller ones, and that biases the
# fitted alpha slightly optimistic and inflates E. Holding the pass count
# *constant* across the grid is what keeps it from biasing beta -- with
# fixed steps instead, a small fraction would silently mean many passes and
# beta would be measuring repetition, which section 6 names explicitly.
DEFAULT_EPOCHS = 3.0

# Achieved TFLOP/s on a T4 at d=64, from the 92,000-step runs: BDH ran
# 17,479 ex/s at 54.76 MFLOP/example, attention 19,709 at 46.83. Both land
# near 0.94, which is 12% of the T4's fp32 peak -- a d=64 cell is bound by
# launch overhead and memory traffic, not arithmetic. It is therefore a
# *floor* for the wider rungs, which utilise the device far better, and
# `estimate` says so.
MEASURED_T4_TFLOPS = 0.94

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
    role: str = "N"  # "N", "D", "interior", "pilot", "neuron" -- why it exists
    # BDH's neuron width is neuron_multiplier * hidden_dim. It was pinned at
    # 4 to make an iso-parameter comparison against attention possible;
    # PROJECT_PLAN.md section 4 has since dropped that comparison, so the
    # multiplier is a free axis again and `neuron_probe` sweeps it. The
    # default stays 4 because every measured run so far used 4, and moving
    # the default would silently renumber the anchor. Attention ignores it.
    neuron_multiplier: int = 4

    @property
    def name(self) -> str:
        """Filename-safe identity. Two cells with the same name are the
        same experiment, which is what makes skipping completed work safe.

        The multiplier is in the name only when it is not the default, so
        every result file written before it became an axis still refers to
        the cell it was written for.
        """
        frac = f"{self.data_fraction:g}".replace(".", "p")
        stem = f"{self.architecture}_d{self.hidden_dim}_f{frac}_s{self.seed}"
        return stem if self.neuron_multiplier == 4 else f"{stem}_n{self.neuron_multiplier}"

    def config(self, card_feature_dim: int = CARD_FEATURE_DIM, **overrides) -> ModelConfig:
        overrides.setdefault("neuron_multiplier", self.neuron_multiplier)
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

    A naming collision worth stating once: PROJECT_PLAN.md section 6 calls
    the law-fitting sweep "a pilot grid", meaning small relative to a
    publication. That sweep is `full_grid` below. *This* function is the
    pre-flight smoke test that section 10's build order puts in front of it,
    so that a broken D axis or a broken arm switch surfaces on cells costing
    minutes rather than on the first expensive rung.

    Deliberately below the ladder's bottom rung: these cells exist to
    exercise the plumbing, not to contribute a point to the fit.
    """
    return [
        GridCell(arm, d, f, seed=0, role="pilot")
        for d in (32, 48)
        for f in (1.0, 0.25)
        for arm in ARMS
    ]


def full_grid(seeds: tuple[int, ...] = (0,)) -> list[GridCell]:
    """Section 6's grid: the L-shaped design. See the module docstring.

    One seed by default, which is what section 6 specifies and what the
    92,000-step runs argue for. Those two runs put the arms 0.0004 apart on
    the headline slice at d=64, which is not a result -- it is two runs
    landing in the same place. A seed sweep at the bottom rung would buy an
    error bar on a gap that has no width; the ladder buys the thing actually
    being asked, which is whether the arms separate at a size worth
    shipping. Pass more seeds if a *fitted exponent* needs an interval.

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
    #
    # Chosen for spread per FLOP, which on a four-rung ladder is a real
    # constraint rather than a preference: (512, 0.5) is the obvious far
    # corner and would be the second most expensive cell in the entire
    # grid, more than the whole D arm put together. (512, 0.25) sits just
    # as far off both arms for half of that, and the two d=256 points cover
    # the middle of the fraction range cheaply.
    cells += [
        GridCell(arm, d, f, seeds[0], role="interior")
        for d, f in ((256, 0.5), (256, 0.125), (512, 0.25))
        for arm in ARMS
    ]

    return order_by_cost(cells)


def neuron_probe(
    hidden_dim: int = 64,
    multipliers: tuple[int, ...] = (4, 8, 16),
) -> list[GridCell]:
    """Two or three BDH cells that ask whether the neuron axis is worth one.

    PROJECT_PLAN.md section 4 unpinned `neuron_multiplier` when it dropped
    the iso-parameter comparison, but every run measured so far still used
    4, so "larger neuron widths are back on the table" is a permission and
    not yet a finding. Section 5 is the reason to care: BDH's kernel earns
    its place in the wide-neuron regime, and a grid committed at
    multiplier 4 would never enter it.

    This runs at the ladder's bottom rung, where a cell is minutes, and
    against the measured d=64 anchor. The multiplier moves N as well as the
    neuron width -- 261,633 at 4, 359,937 at 8, 556,545 at 16 -- so the
    comparison to read is not multiplier against multiplier but each cell
    against the width-ladder point of the same N. If 8 does no better than
    a plain d=64 does per parameter, the axis is not worth a column in the
    grid and the grid stays at 4.
    """
    return [
        GridCell("bdh", hidden_dim, 1.0, seed=0, role="neuron", neuron_multiplier=m)
        for m in multipliers
    ]


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


def estimate(
    cells: list[GridCell],
    tflops: float = MEASURED_T4_TFLOPS,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    epochs: float = DEFAULT_EPOCHS,
) -> dict:
    """What this grid costs at an assumed achieved throughput.

    `tflops` is *achieved*, not peak, and it is the soft number in any such
    estimate: a small cell is bound by launch overhead and memory traffic
    rather than arithmetic, so a pure FLOP roofline will misprice it. The
    default is `MEASURED_T4_TFLOPS`, taken from a real d=64 cell, and it is
    a **floor** rather than a central estimate -- d=64 reaches 12% of the
    T4's fp32 peak and the wider rungs will do considerably better. Measure
    a cell at the width you care about before trusting a total.
    """
    total = epochs * sum(cell_flops(c, train_rows) for c in cells)
    return {
        "cells": len(cells),
        "epochs": epochs,
        "tflops": tflops,
        "total_flops": total,
        "hours": total / (tflops * 1e12) / 3600,
        "largest_cell_hours": (
            epochs * max(cell_flops(c, train_rows) for c in cells)
            / (tflops * 1e12) / 3600
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
    epochs: float = DEFAULT_EPOCHS,
    eval_every: int = 250,
    drop_forced: bool = True,
    checkpoint_dir: str | Path | None = None,
    by_pick_eval: bool = True,
    eval_batch_size: int = DEFAULT_EVAL_BATCH,
) -> dict:
    """Trains one grid cell and returns what the curve fit needs.

    `drop_forced` removes the rows whose gradient is identically zero -- a
    one-card pack has one admissible answer, so it cannot teach the model
    anything and only occupies a slot in a batch. That is 7.14% of the FIN
    training split. It is on by default because it is not a trade: the
    rows carry no signal to lose.

    It does mean D counts *decisions*, not rows, and the returned
    `train_rows` says which. That distinction matters downstream: the
    fitted E is compared against a human-disagreement floor that section 7
    has already concluded must be measured where a real decision exists, so
    both sides of that comparison have to be the same population.

    `by_pick_eval` is what puts the fit's y-axis in the result file, and it
    is the same exact full-split evaluation `run.py` writes -- same
    function, same `summary` key -- so a cell run here and a cell run by
    hand are comparable. Without it the only loss recorded is
    `best_val_loss`, which is *sampled* over `eval_batches` and covers all
    fourteen picks; docs/RESULTS.md fits on the exact picks-0-8 number
    instead, and neither of those substitutions is visible in a curve.
    Turning it off saves one pass over the val split and costs the cell its
    place in the fit.
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

    record = {
        "cell": asdict(cell),
        "name": cell.name,
        "role": cell.role,
        # The two axes of the fit, both recorded rather than inferred.
        #
        # N is derived from `model_config` -- the config the model was
        # actually built from, whose feature width came off the table in
        # front of it -- and not from `cell.num_params`, which assumes
        # CARD_FEATURE_DIM. The two agree on a correctly staged corpus and
        # `design_num_params` records the design-time figure so a
        # disagreement is visible rather than silent. They diverged once
        # already, when the rebuild took the table from 65 columns to 119,
        # and a fit whose x-axis is wrong does not fail -- it just reports a
        # different exponent.
        "num_params": count_params_analytic(
            model_config, cell.architecture
        )["total"],
        "design_num_params": cell.num_params,
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

    if by_pick_eval:
        # Evaluated at the best-val parameters, which are what the
        # checkpoint holds and what any later analysis loads. Scoring the
        # final ones instead would put a different model's loss on the
        # curve than the one this cell ships.
        by_pick = evaluate_by_pick(
            result["model"], result["best_params"], feature_table,
            data, splits.val, eval_batch_size,
        )
        record["by_pick"] = by_pick
        record["summary"] = summarise_by_pick(by_pick)

    return record


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
        # The picks-0-8 number is the one the fit uses and the one
        # docs/RESULTS.md quotes, so it is what a session's log should show;
        # best_val_loss is sampled and covers the forced picks too.
        headline = result.get("summary", {}).get("decision_picks", {}).get("loss")
        measured = "" if headline is None else f" | picks 0-8 {headline:.4f}"
        print(
            f"[{i}/{len(cells)}] {cell.name}: best val "
            f"{result['best_val_loss']:.4f}{measured}"
        )

    return results


def load_results(out_dir: str | Path) -> list[dict]:
    """Every completed cell in `out_dir`, for the fit to consume."""
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(out_dir).glob("*.json"))
    ]
