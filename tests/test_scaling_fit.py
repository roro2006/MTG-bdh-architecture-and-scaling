"""Tests for the curve fit, on planted laws rather than on grid results.

A scaling fit is the hardest thing in this repository to test against
reality, because reality here costs about thirteen GPU-hours and arrives
once. What can be tested is the property that matters: handed data drawn
from a known L(N, D), the fit has to return that law's exponents rather
than a plausible-looking pair of numbers. Every failure mode below produces
a fit -- a converged optimiser, a finite residual, a printable table -- and
differs from a correct one only in the value it reports.

So the design is: plant (E, A, B, alpha, beta), generate the L-shaped grid
the real sweep will run, and assert recovery. No GPU, no training, and the
one test that does drive `run_cell` uses the synthetic corpus.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.data.dataset import split_by_draft
from src.training.grid import D_ANCHORS, FRACTIONS, LADDER, GridCell, run_cell
from src.training.scaling_fit import (
    collect_points,
    compare_at,
    fit_by_architecture,
    fit_scaling_law,
    format_fit,
)

# A law whose terms land where the measured d=64 runs did: about 0.85 nats
# on the picks-0-8 slice, an E a little under that, and both terms
# contributing a tenth of a nat or two across the ladder. Recovery is
# easier on a law whose terms are large, so the planted one is sized to be
# realistic rather than convenient.
TRUE = dict(E=0.62, A=13.9, alpha=0.34, B=11.4, beta=0.31)

# The ladder's parameter counts, from grid.py's own comment.
PARAMS = {64: 261_633, 128: 1_022_977, 256: 4_044_801, 512: 16_084_993}
FULL_ROWS = 4_375_371

# Multiplicative, because the fit works on log-residuals: 0.2% is roughly
# the seed-to-seed spread the 92,000-step pair showed on the headline slice.
NOISE = 0.002


def law(N, D, E, A, alpha, B, beta, interaction=0.0):
    """The fitted form, plus an optional interaction term it cannot express.

    The interaction is what the interior cells exist to detect: it is
    invisible along either arm of the L, because holding one axis fixed
    folds it into that axis's coefficient, and it only shows up where both
    axes are off their anchors at once.
    """
    return (
        E
        + A * N**-alpha
        + B * D**-beta
        + interaction * (N**-alpha) * (D**-beta)
    )


def synthetic_grid(
    architecture="bdh", *, seed=0, noise=NOISE, interaction=0.0, include_pilot=False, **planted
):
    """Result records in the shape `grid.run_cell` writes, on the L-shape
    `grid.full_grid` builds."""
    values = dict(TRUE)
    values.update(planted)
    rng = np.random.default_rng(seed)

    cells = [(d, 1.0, "N") for d in LADDER]
    cells += [(d, f, "D") for d in D_ANCHORS for f in FRACTIONS if f != 1.0]
    cells += [(d, f, "interior") for d, f in ((256, 0.5), (256, 0.125), (512, 0.25))]
    if include_pilot:
        # Below the ladder, and given a loss no law would produce, so that a
        # fit which used them could not possibly go unnoticed.
        cells += [(32, 1.0, "pilot"), (48, 0.25, "pilot")]

    results = []
    for hidden_dim, fraction, role in cells:
        num_params = PARAMS.get(hidden_dim, 40_000 + 300 * hidden_dim**2)
        train_rows = FULL_ROWS * fraction
        loss = law(num_params, train_rows, interaction=interaction, **values)
        if role == "pilot":
            loss = 4.0
        else:
            loss *= float(np.exp(rng.normal(0.0, noise)))
        results.append(
            {
                "cell": {
                    "architecture": architecture,
                    "hidden_dim": hidden_dim,
                    "data_fraction": fraction,
                    "seed": 0,
                },
                "name": f"{architecture}_d{hidden_dim}_f{fraction:g}".replace(".", "p"),
                "role": role,
                "num_params": num_params,
                "train_rows": train_rows,
                "examples_seen": 3 * train_rows,
                "flops_per_example": 54.76e6 * (hidden_dim / 64) ** 2,
                "best_val_loss": loss,
                "summary": {
                    "decision_picks": {"loss": loss, "rows": 100_000, "picks": "0-8"},
                    "all_picks": {"loss": 0.9 * loss, "rows": 140_000},
                },
            }
        )
    return results


def in_sample_rmse(fit):
    """Residual RMSE over the cells that were fitted, in nats.

    `ScalingFit.rmse_log` is a log residual and the held-out ones are in
    nats, so comparing the two directly would be comparing units.
    """
    residuals = [p.loss - fit.predict(p.N, p.D) for p in fit.points]
    return float(np.sqrt(np.mean(np.square(residuals))))


def test_recovers_a_planted_law():
    """The whole point, and the one thing no grid run can check afterwards.

    A fit that returned the wrong exponent would still size the drafter,
    still print a table and still be believed; nothing downstream of it
    knows what the right answer was.
    """
    fit = fit_scaling_law(synthetic_grid(), bootstrap=0)

    assert fit.alpha == pytest.approx(TRUE["alpha"], abs=0.03)
    assert fit.beta == pytest.approx(TRUE["beta"], abs=0.05)
    assert fit.E == pytest.approx(TRUE["E"], abs=0.03)
    # The residual has to be the noise that was planted and nothing else.
    assert fit.rmse_log < 3 * NOISE
    assert format_fit(fit)


def test_bootstrap_intervals_bracket_the_planted_values():
    """An interval that excluded the truth would be worse than no interval.

    This is the parameter set the sizing decision is argued from, so a
    confidently narrow and wrong interval is the failure to avoid --
    reporting alpha to three digits from eleven noisy cells is a claim the
    data cannot support, and the interval is what stops it being made.
    """
    fit = fit_scaling_law(synthetic_grid(), bootstrap=120, seed=1)

    assert fit.bootstrap_samples > 0
    for value, (low, high) in (
        (TRUE["alpha"], fit.alpha_ci),
        (TRUE["beta"], fit.beta_ci),
        (TRUE["E"], fit.E_ci),
    ):
        assert low <= value <= high
        assert low < high  # a degenerate interval is not an interval


def test_bootstrap_is_skipped_rather_than_faked_when_asked_for_zero():
    fit = fit_scaling_law(synthetic_grid(), bootstrap=0)
    assert fit.bootstrap_samples == 0
    assert np.isnan(fit.alpha_ci[0]) and np.isnan(fit.beta_ci[1])


def test_huber_survives_an_outlier_that_least_squares_does_not():
    """Chinchilla's reason for the robust loss, on a planted stall.

    One cell in a grid this size can land badly -- a seed that stalls, a
    session that resumed wrong -- and squared error on raw loss lets that
    single cell set the exponent. The comparator differs from the default
    only in its loss function, so what this measures is robustness and not
    a difference in parameterisation.
    """
    results = synthetic_grid()
    stalled = next(r for r in results if r["name"] == "bdh_d128_f0p25")
    stalled["summary"]["decision_picks"]["loss"] *= 1.35

    robust = fit_scaling_law(results, bootstrap=0)
    naive = fit_scaling_law(results, bootstrap=0, objective="lsq")

    robust_error = abs(robust.alpha - TRUE["alpha"])
    naive_error = abs(naive.alpha - TRUE["alpha"])
    assert robust_error < 0.05, "the outlier should not move the robust fit much"
    assert naive_error > 5 * robust_error, (
        f"least squares was supposed to be the fragile one: "
        f"huber {robust_error:.4f} vs lsq {naive_error:.4f}"
    )


def test_interior_cells_sit_on_a_surface_fitted_without_them():
    """They are held out, so their residual is a measurement rather than a
    consequence of having been fitted.

    If they were folded into the fit this assertion would pass on any data
    at all, which is exactly why the API holds them out.
    """
    fit = fit_scaling_law(synthetic_grid(), bootstrap=0)

    assert len(fit.held_out) == 3
    assert {p.role for p in fit.points} == {"N", "D"}
    assert all(h.role == "interior" for h in fit.held_out)
    assert abs(fit.held_out_bias) < 0.005
    assert fit.held_out_rmse < 0.01


def test_interior_cells_expose_an_interaction_term():
    """The only check in the whole design that the separable form is right.

    grid.py spends three cells on this. They earn it only if a surface with
    an interaction term actually shows up as a systematic offset here --
    a scatter that grew would be noise, a mean that moved is a wrong
    functional form.
    """
    clean = fit_scaling_law(synthetic_grid(), bootstrap=0)
    warped = fit_scaling_law(synthetic_grid(interaction=900.0), bootstrap=0)

    assert abs(warped.held_out_bias) > 0.01
    assert abs(warped.held_out_bias) > 5 * abs(clean.held_out_bias)

    # And they show it more loudly than the fitted cells do. Each arm of
    # the L holds one axis fixed, which folds the interaction into that
    # arm's own coefficient and hides most of it; the interior cells are
    # the only ones with both axes off their anchors, so the part that was
    # absorbed shows up there and nowhere else.
    assert warped.held_out_rmse > 1.5 * in_sample_rmse(warped)
    # Where the form is right, the held-out cells are no worse than the
    # fitted ones -- which is what makes the comparison above a signal
    # rather than a property of having been held out.
    assert clean.held_out_rmse < 3 * in_sample_rmse(clean)


def test_pilot_cells_never_enter_a_fit():
    """`pilot` runs below the ladder to exercise plumbing.

    Its cells are chosen for being cheap, not for being informative, and an
    exponent extrapolated through them is fitted partly to the smoke test.
    """
    with_pilot = fit_scaling_law(synthetic_grid(include_pilot=True), bootstrap=0)
    without = fit_scaling_law(synthetic_grid(), bootstrap=0)

    assert all(p.role != "pilot" for p in with_pilot.points)
    assert with_pilot.alpha == pytest.approx(without.alpha, abs=1e-6)


def test_one_surface_through_both_arms_is_refused():
    """The grid exists to compare two arms, so silently averaging them is
    the one wrong answer that would look entirely normal."""
    both = synthetic_grid("bdh") + synthetic_grid("attention", seed=1)

    with pytest.raises(ValueError, match="fit_by_architecture"):
        fit_scaling_law(both)

    named = fit_scaling_law(both, architecture="bdh", bootstrap=0)
    assert named.architecture == "bdh"
    assert {p.architecture for p in named.points} == {"bdh"}


def test_fit_by_architecture_compares_the_arms_at_one_budget():
    """The sizing decision is 'which arm at the budget we have', which is a
    question about two curves evaluated at the same point."""
    both = synthetic_grid("bdh") + synthetic_grid("attention", seed=1, E=0.58)
    fits = fit_by_architecture(both, bootstrap=0)

    assert set(fits) == {"attention", "bdh"}
    assert fits["attention"].E < fits["bdh"].E

    verdict = compare_at(fits, PARAMS[512], FULL_ROWS)
    assert verdict["best"] == "attention"
    assert verdict["gap"] > 0
    assert verdict["loss"]["attention"] < verdict["loss"]["bdh"]


def test_curves_fitted_on_different_cost_axes_are_not_compared():
    """'The same N' means something different on each axis, so a comparison
    across them answers neither question."""
    results = synthetic_grid("bdh") + synthetic_grid("attention", seed=1)
    mixed = {
        "bdh": fit_scaling_law(results, "params", architecture="bdh", bootstrap=0),
        "attention": fit_scaling_law(
            results, "flops", architecture="attention", bootstrap=0
        ),
    }
    with pytest.raises(ValueError, match="cost axes"):
        compare_at(mixed, 1e6, 1e6)


def test_both_compute_axes_fit_and_they_are_not_the_same_fit():
    """ARCHITECTURE.md's fairness note: quality per parameter and quality
    per dense FLOP are different questions, and BDH's arm costs 1.53x
    attention's at iso-parameter sizing, so the two can rank the arms
    differently. Both have to be available."""
    results = synthetic_grid()
    by_params = fit_scaling_law(results, "params", bootstrap=0)
    by_flops = fit_scaling_law(results, "flops", bootstrap=0)

    assert by_params.compute_axis == "params"
    assert by_flops.compute_axis == "flops"
    assert by_params.points[0].N != by_flops.points[0].N
    # Total training FLOPs travels with every point regardless of axis: it
    # is the budget the curves get compared at, not the axis they are fit
    # against.
    assert all(p.total_flops > 0 for p in by_params.points)


def test_a_cell_without_the_by_pick_summary_is_an_error():
    """The y-axis is the exact picks-0-8 loss. `best_val_loss` is a sampled
    all-picks number, and quietly substituting it for cells that predate
    the by-pick record would put two different quantities on one curve --
    an error the resulting fit absorbs rather than reports.
    """
    results = synthetic_grid()
    del results[0]["summary"]

    with pytest.raises(ValueError, match="by-pick summary"):
        fit_scaling_law(results)

    # But the sampled axis is available to anyone who asks for it by name.
    legacy = fit_scaling_law(results, loss_key="best_val_loss", bootstrap=0)
    assert legacy.loss_key == "best_val_loss"


def test_a_grid_that_moved_along_one_axis_only_is_refused():
    """With D constant, beta is unidentified -- and an unconstrained
    optimiser reports it anyway, at whatever value its start had."""
    n_arm_only = [r for r in synthetic_grid() if r["role"] == "N"]
    with pytest.raises(ValueError, match="one axis"):
        fit_scaling_law(n_arm_only + n_arm_only[:1])


def test_too_few_cells_to_fit_five_parameters_is_refused():
    results = [r for r in synthetic_grid() if r["role"] != "interior"][:4]
    with pytest.raises(ValueError, match="interpolate"):
        fit_scaling_law(results)


def test_the_data_axis_choice_moves_B_and_leaves_beta_alone():
    """`train_rows` and `examples_seen` differ by the (constant) pass count,
    so the exponent cannot depend on which is used. If it ever does, the
    grid stopped holding epochs fixed, and beta started measuring
    repetition -- the failure section 6 names explicitly.
    """
    results = synthetic_grid()
    unique = fit_scaling_law(results, data_axis="train_rows", bootstrap=0)
    seen = fit_scaling_law(results, data_axis="examples_seen", bootstrap=0)

    assert seen.beta == pytest.approx(unique.beta, abs=1e-3)
    assert seen.B > unique.B  # same curve, D rescaled by the pass count


def test_collect_points_keeps_the_identity_of_each_cell():
    """An outlier is only actionable if the fit can say which cell it was."""
    points = collect_points(synthetic_grid(include_pilot=True))
    assert all(p.name for p in points)
    assert all(p.role != "pilot" for p in points)
    assert {p.role for p in points} == {"N", "D", "interior"}


def test_run_cell_records_the_by_pick_summary_the_fit_reads(ingested):
    """The gap this closed: without it the grid's 26 result files hold no
    picks-0-8 loss at all, and the fit's y-axis does not exist.

    Driven end to end on the synthetic corpus, because the failure is not
    that the number is wrong -- it is that the key is absent, which only a
    real `run_cell` call can demonstrate.
    """
    data, _, _ = ingested(count=60)
    splits = split_by_draft(data, seed=0)
    table = jnp.zeros((data.vocab.size, 8), dtype=jnp.float32)
    cell = GridCell("attention", 16, 1.0, 0, role="pilot")

    record = run_cell(
        cell, data, table, splits,
        batch_size=8, epochs=0.05, eval_every=1000, eval_batch_size=64,
    )

    assert record["by_pick"], "the per-pick table itself is a first-class output"
    summary = record["summary"]
    # Both slices, so nothing is lost by fitting on one of them.
    assert summary["decision_picks"]["loss"] > 0
    assert summary["all_picks"]["loss"] > 0
    assert summary["decision_picks"]["rows"] <= summary["all_picks"]["rows"]
    assert summary["decision_picks"]["picks"] == "0-8"

    # The record is exactly what the fit's extractor expects to read.
    record["name"] = cell.name
    [point] = collect_points([record], excluded_roles=())
    assert point.loss == pytest.approx(summary["decision_picks"]["loss"])
    assert point.N == record["num_params"]
    assert point.D == record["train_rows"]


def test_run_cell_can_skip_the_by_pick_pass_and_says_so_by_omission(ingested):
    """Turning it off costs the cell its place in the fit, and the fit says
    so loudly rather than substituting the sampled loss."""
    data, _, _ = ingested(count=60)
    splits = split_by_draft(data, seed=0)
    table = jnp.zeros((data.vocab.size, 8), dtype=jnp.float32)

    record = run_cell(
        GridCell("attention", 16, 1.0, 0, role="pilot"), data, table, splits,
        batch_size=8, epochs=0.05, eval_every=1000, by_pick_eval=False,
    )
    assert "summary" not in record
    with pytest.raises(ValueError, match="by-pick summary"):
        collect_points([record], excluded_roles=())
