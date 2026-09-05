"""Tests for the grid design and its resumability.

The properties here are the ones that would not announce themselves if they
broke. A grid that silently reran completed cells still produces correct
results, just never within a Colab session; a ladder that drifted off its
target parameter counts still fits a curve, just not the one the plan
specifies. Both would look like everything was fine.

No data download is needed: the sweep-construction tests are pure, and the
resumability test drives a synthetic corpus.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from src.data.dataset import PickData, decision_rows, split_by_draft
from src.models.pick_model import ModelConfig, count_params_analytic
from src.training.grid import (
    ARMS,
    CARD_FEATURE_DIM,
    D_ANCHORS,
    DEFAULT_EPOCHS,
    FRACTIONS,
    LADDER,
    GridCell,
    cell_flops,
    estimate,
    full_grid,
    load_results,
    neuron_probe,
    pilot_grid,
    run_grid,
)


def test_ladder_hits_its_target_parameter_counts():
    """PROJECT_PLAN section 6 asks for 4-5 log-spaced widths.

    The widths are hardcoded, so nothing would complain if a config default
    changed underneath them and the ladder quietly stopped spanning its
    range. That would not break the fit -- it would just narrow the range
    alpha is estimated over, which is exactly the kind of silent damage
    section 3b's derive-don't-read discipline exists to prevent.

    The bottom rung is not a target but a *measurement*: d=64 is the width
    both arms were trained at for 92,000 steps, so 261,633 is a number the
    ladder inherits rather than chooses.
    """
    assert 4 <= len(LADDER) <= 5
    counts = [
        count_params_analytic(
            ModelConfig(hidden_dim=d, card_feature_dim=CARD_FEATURE_DIM)
        )["total"]
        for d in LADDER
    ]
    assert counts == sorted(counts)
    assert counts[0] == pytest.approx(263_745, rel=0.01)   # the measured anchor
    assert counts[-1] / counts[0] > 50                     # enough span to fit alpha

    # Log-spaced: consecutive ratios should be close to constant.
    ratios = [b / a for a, b in zip(counts, counts[1:])]
    assert max(ratios) / min(ratios) < 1.35


def test_the_ladder_starts_at_the_measured_anchor():
    """Section 6's ladder has to include the point we already know.

    The whole argument for climbing rather than moving sideways rests on
    the d=64 result, and it only rests on it if d=64 is on the ladder --
    otherwise the grid's cheapest point is an extrapolation from a run it
    does not contain.
    """
    assert LADDER[0] == 64
    assert 64 in D_ANCHORS


def test_default_epochs_is_the_measured_budget_not_convergence():
    """Three, from the 92,000-step curve. Ten is memorisation.

    Guarded because it is the parameter most likely to be 'tidied' back to
    1.0 by someone reading `cell_flops`, whose docstring says one epoch.
    """
    assert DEFAULT_EPOCHS == 3.0


def test_the_grid_is_l_shaped_not_cartesian():
    """The saving is the point, so assert the shape that produces it.

    A full product over the same axes would be 120 cells; the L-shape is
    fewer, and specifically it must not contain the expensive corner --
    every large size at every data fraction.
    """
    cells = full_grid()
    cartesian = len(LADDER) * len(FRACTIONS) * len(ARMS)
    assert len(cells) < cartesian

    # The largest sizes appear only at full data (plus the deliberate
    # interior probes, which are single-seed specification checks).
    for c in cells:
        if c.hidden_dim > max(D_ANCHORS) and c.data_fraction != 1.0:
            assert c.role == "interior", f"{c.name} is an unintended corner cell"

    # Both exponents still have something to be fit from.
    n_axis = {c.hidden_dim for c in cells if c.role == "N"}
    d_axis = {c.data_fraction for c in cells if c.role == "D"}
    assert n_axis == set(LADDER)
    assert d_axis == set(FRACTIONS) - {1.0}


def test_interior_points_exist_and_are_off_the_arms():
    """They are the only check that the separable form is adequate.

    Without them the design assumes what the fit asserts, so if they ever
    got dropped as 'redundant' the grid would lose its ability to detect a
    wrong functional form while looking cheaper.
    """
    interior = [c for c in full_grid() if c.role == "interior"]
    assert interior
    for c in interior:
        assert c.data_fraction != 1.0          # not on the N arm
        assert c.hidden_dim not in D_ANCHORS   # not on the D arm


def test_interior_points_are_not_the_expensive_corner():
    """They are a specification check, so they must not dominate the bill.

    On a four-rung ladder the far corner (512, 0.5) is the second most
    expensive cell in the whole grid -- more than the entire D arm. An
    interior point that costs that much stops being a cheap check and
    starts being a reason to skip the check, which is how separability
    quietly goes unverified.
    """
    cells = full_grid()
    interior = [c for c in cells if c.role == "interior"]
    n_arm = [c for c in cells if c.role == "N"]
    assert interior and n_arm

    interior_cost = sum(cell_flops(c, 1_000) for c in interior)
    n_cost = sum(cell_flops(c, 1_000) for c in n_arm)
    assert interior_cost < 0.5 * n_cost


def test_the_neuron_probe_moves_the_neuron_axis_and_says_so_in_the_name():
    """Unpinning `neuron_multiplier` only helps if a cell records it.

    Two failures this guards. A multiplier that never reaches ModelConfig
    would produce three identical runs reported as a sweep; and a name that
    ignored the multiplier would make the second cell look already-done and
    silently skip it, since `run_grid` resumes on filename alone.
    """
    cells = neuron_probe()
    assert len({c.name for c in cells}) == len(cells)
    assert all(c.architecture == "bdh" for c in cells)
    assert all(c.role == "neuron" for c in cells)

    multipliers = [c.neuron_multiplier for c in cells]
    assert multipliers == sorted(multipliers)
    assert all(c.config().neuron_multiplier == c.neuron_multiplier for c in cells)

    # A wider neuron space is a bigger model, which is the reason the probe
    # is read per parameter rather than per multiplier.
    counts = [c.num_params for c in cells]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_the_default_multiplier_leaves_existing_cell_names_untouched():
    """Result files written before the axis existed must still resume.

    `run_grid` skips a cell when its file is present, so a name change
    would rerun every completed cell in a Drive directory -- the exact
    failure the resume logic exists to prevent, arriving as a silent
    rerun rather than an error.
    """
    assert GridCell("bdh", 64, 1.0, 0).name == "bdh_d64_f1_s0"
    assert GridCell("bdh", 64, 1.0, 0, neuron_multiplier=4).name == "bdh_d64_f1_s0"
    assert (
        GridCell("bdh", 64, 1.0, 0, neuron_multiplier=8).name == "bdh_d64_f1_s0_n8"
    )


def test_cells_are_ordered_most_expensive_first():
    """A session that dies mid-grid should have done the expensive work."""
    cells = full_grid()
    costs = [cell_flops(c, 1_000) for c in cells]
    assert costs == sorted(costs, reverse=True)


def test_cell_names_are_unique_and_filename_safe():
    """Names are the resume key, so a collision would silently skip a cell
    that had never run, and a '.' in a fraction would fragment the file
    stem.
    """
    for grid in (pilot_grid(), full_grid(), neuron_probe()):
        names = [c.name for c in grid]
        assert len(names) == len(set(names))
        assert all(set(n) <= set("abcdefghijklmnopqrstuvwxyz0123456789_") for n in names)


def test_estimate_scales_with_throughput_and_reports_the_largest_cell():
    cells = full_grid()
    slow = estimate(cells, tflops=1.0)
    fast = estimate(cells, tflops=10.0)
    assert slow["cells"] == fast["cells"] == len(cells)
    assert slow["hours"] == pytest.approx(10 * fast["hours"])
    # The largest single cell has to fit inside one Colab session.
    assert 0 < slow["largest_cell_hours"] < slow["hours"]


def test_pilot_is_far_cheaper_than_the_full_grid():
    """It exists to be run first, which only works if it is cheap."""
    assert estimate(pilot_grid(), 1.0)["hours"] < 0.1 * estimate(full_grid(), 1.0)["hours"]


def test_completed_cells_are_skipped_and_results_reloaded(ingested, tmp_path):
    """Resumability, driven end to end on a synthetic corpus.

    This is the property a free Colab runtime depends on, and the failure
    mode is expensive rather than loud: without it every disconnect throws
    away the session's work and the grid never finishes, while each
    individual run still looks correct.
    """
    data, _, _ = ingested(count=60)
    splits = split_by_draft(data, seed=0)
    table = jnp.zeros((data.vocab.size, 8), dtype=jnp.float32)
    out = tmp_path / "grid"

    cells = [GridCell("attention", 16, 1.0, 0, role="pilot")]
    kwargs = dict(batch_size=8, epochs=0.05, eval_every=1000)

    first = run_grid(cells, data, table, splits, out, **kwargs)
    assert len(first) == 1
    result_file = out / f"{cells[0].name}.json"
    assert result_file.exists()

    # A file's presence must mean a *finished* cell, so it carries the
    # numbers the fit needs rather than a partial record.
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    assert payload["num_params"] > 0
    assert payload["train_rows"] > 0
    assert payload["flops_per_example"] > 0

    # Rerunning must not retrain: mtime is the evidence, since a rerun that
    # silently retrained would return identical-looking results.
    before = result_file.stat().st_mtime_ns
    second = run_grid(cells, data, table, splits, out, **kwargs)
    assert result_file.stat().st_mtime_ns == before
    assert second[0]["name"] == first[0]["name"]

    assert [r["name"] for r in load_results(out)] == [cells[0].name]


def test_run_cell_drops_forced_rows_from_training_only(ingested):
    """D counts decisions, and the split it is drawn from is untouched.

    Dropping the rows from `splits.train` itself would be a different and
    worse change -- evaluation and the floor comparison both need the full
    population -- so the filter has to live in the training stream.
    """
    data, _, _ = ingested(count=60)
    splits = split_by_draft(data, seed=0)
    kept = decision_rows(data, splits.train)

    assert kept.size < splits.train.size
    assert (data.pack_size[kept] > 1).all()
    # The split itself is unchanged: decision_rows returns a view, not a
    # mutation, so evaluation still sees every row.
    assert (data.pack_size[splits.train] == 1).any()
