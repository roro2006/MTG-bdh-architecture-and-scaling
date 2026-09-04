"""Fits L(N, D) = E + A / N^alpha + B / D^beta to grid results, following
the Chinchilla paper's robust-fit approach (Huber loss on log-residuals,
not naive least squares) so the small-N, high-loss corner of the grid
doesn't dominate the fit. See docs/PROJECT_PLAN.md section 5.

Both an iso-parameter and an iso-FLOP fit are produced for each
architecture -- see docs/ARCHITECTURE.md's fairness note on why these are
genuinely different comparisons here.

Three things about this fit are not free choices, and each of them is a way
it could otherwise report a confident wrong number.

**The y-axis is the exact picks-0-8 loss, not the aggregate and not the
sampled one.** docs/RESULTS.md says so, and the reason is downstream: the
fitted E is compared against a human-disagreement floor that section 7
concluded must be measured where a real decision exists. 7.1% of val rows
are one-card packs contributing identically zero, so an all-picks E and a
decision-picks floor are not the same quantity, and their difference is not
a measurement of anything. `all_picks` is recorded alongside and available
by name, but it is not the default.

**Under compute_axis="flops" the cost axis is per-example dense FLOPs, not
total training FLOPs.** Total FLOPs is `flops_per_example x examples_seen`;
it is carried on every point and it is the budget at which two arms' curves
are compared (`compare_at`). It cannot also be the N axis. Along the D arm
of the L-shaped grid `flops_per_example` is constant, so total FLOPs there
is exactly proportional to D, and `A/N^alpha` and `B/D^beta` collapse into
two power laws in the same variable. The optimiser still returns five
numbers; two of them are just not alpha and beta.

**Interior cells are held out of the fit rather than fitted.** grid.py puts
them there to sit off a surface fitted without them, because the separable
form asserts there is no interaction term and nothing else in the design
can detect that assertion failing. Fitting them would spend them: the
residual would be pulled toward zero by construction and the check would
pass on any data at all. `pilot` cells are excluded outright -- they run
below the ladder's bottom rung to exercise plumbing, and a fit that
included them would extrapolate an exponent from points chosen for being
cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import logsumexp

# Chinchilla's delta, and small for a reason worth keeping: the residuals
# it sees are already in log space, so anything much larger leaves the
# Huber quadratic across the whole range actually observed -- which is the
# least squares this module exists to avoid, wearing a robust name.
HUBER_DELTA = 1e-3

# Resamples for the confidence intervals. A few hundred is enough for a
# percentile interval on five parameters fit from ~11 points; the cost is
# one refit each, warm-started from the full-data solution.
DEFAULT_BOOTSTRAP = 200

# Roles from grid.GridCell. `pilot` never enters a fit; `interior` is held
# out of one so it can be compared against it.
EXCLUDED_ROLES = ("pilot",)
HELD_OUT_ROLES = ("interior",)

# A negative exponent asserts that a bigger model, or more data, makes the
# drafter worse. That is not a scaling law in need of fitting, it is a
# broken grid, and an unbounded optimiser will report one rather than fail.
EXPONENT_BOUNDS = (0.0, 3.0)

# Seed values for the (alpha, beta) grid the starting points are built on.
# Spread over the range a language-model-shaped fit lands in, log-spaced,
# because the objective is multi-modal and a single start finds whichever
# basin it was dropped in.
EXPONENT_SEEDS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)

# Starting points kept from that grid and refined. Eight is well past the
# point where the best refined solution stops changing on a grid this size.
DEFAULT_STARTS = 8

LOSS_KEYS = ("decision_picks", "all_picks", "best_val_loss", "final_val_loss")
COMPUTE_AXES = ("params", "flops")
DATA_AXES = ("train_rows", "examples_seen")

# Keeps log() finite when the non-negative least squares that seeds a start
# puts a term at exactly zero. It is a starting point, not a result.
_FLOOR = 1e-12


@dataclass(frozen=True)
class FitPoint:
    """One grid cell as the fit sees it: two axes, one loss, and enough
    identity left attached to say which cell an outlier was."""

    name: str
    role: str
    architecture: str
    N: float
    D: float
    loss: float
    total_flops: float


@dataclass(frozen=True)
class HeldOut:
    """An interior cell against the surface fitted without it."""

    name: str
    role: str
    N: float
    D: float
    observed: float
    predicted: float
    residual: float  # observed - predicted, in nats


@dataclass(frozen=True)
class ScalingFit:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    # bootstrapped confidence intervals, not just point estimates
    alpha_ci: tuple[float, float]
    beta_ci: tuple[float, float]
    # E gets one too, because it is the parameter with a consumer: section 7
    # compares it against a measured human-disagreement floor, and "0.72
    # against a floor of 0.75" says nothing until you know whether 0.72
    # could as easily have been 0.78.
    E_ci: tuple[float, float]

    architecture: str
    compute_axis: str
    data_axis: str
    loss_key: str
    objective: str

    points: tuple[FitPoint, ...]
    held_out: tuple[HeldOut, ...]

    rmse_log: float
    # Signed mean of the held-out residuals. The scatter is not the
    # question -- a *systematic* offset is what "the separable form is
    # wrong" looks like, and an interaction term shows up here long before
    # it shows up in the RMSE.
    held_out_bias: float
    held_out_rmse: float
    bootstrap_samples: int

    def predict(self, N, D):
        """The fitted surface at (N, D). Accepts scalars or arrays."""
        cost = np.asarray(N, dtype=float)
        data = np.asarray(D, dtype=float)
        out = self.E + self.A * cost**-self.alpha + self.B * data**-self.beta
        return float(out) if np.ndim(out) == 0 else out

    @property
    def n_points(self) -> int:
        return len(self.points)


def _architecture(result: dict) -> str:
    """The arm a result came from, wherever the writer put it.

    Grid cells nest it under `cell`; a hand-run cell's metrics.json calls it
    `arm`. Guessing wrong here would silently fit one surface across both
    architectures, which is the one comparison this module exists to keep
    separate.
    """
    cell = result.get("cell")
    if isinstance(cell, dict) and "architecture" in cell:
        return str(cell["architecture"])
    for key in ("architecture", "arm"):
        if key in result:
            return str(result[key])
    raise ValueError(
        f"result {result.get('name', '<unnamed>')!r} names no architecture; "
        "a fit over mixed arms reports an exponent for neither"
    )


def _observed_loss(result: dict, loss_key: str) -> float:
    """The y-axis value, refusing to substitute a different quantity for it."""
    if loss_key in ("best_val_loss", "final_val_loss"):
        return float(result[loss_key])

    summary = result.get("summary")
    if not isinstance(summary, dict) or loss_key not in summary:
        raise ValueError(
            f"{result.get('name', '<unnamed>')!r} carries no by-pick summary, so "
            f"the {loss_key!r} loss the fit wants does not exist in it. Grid "
            "results written before run_cell recorded `summary` hold only a "
            "sampled all-picks val loss; fitting some cells on one quantity and "
            "some on another would put two y-axes on one curve. Re-run the cell, "
            "or pass loss_key='best_val_loss' to fit the sampled number knowingly."
        )
    return float(summary[loss_key]["loss"])


def collect_points(
    grid_results: list[dict],
    *,
    compute_axis: str = "params",
    data_axis: str = "train_rows",
    loss_key: str = "decision_picks",
    excluded_roles: tuple[str, ...] = EXCLUDED_ROLES,
) -> list[FitPoint]:
    """Grid result JSONs to the (N, D, L) triples the fit consumes.

    D counts *unique decisions* by default rather than examples seen. With
    the pass count held constant across the grid the two differ by exactly
    that constant, so beta is identical either way and only B moves; the
    default is the one that matches what the grid's D axis actually
    subsamples, which is drafts.
    """
    if compute_axis not in COMPUTE_AXES:
        raise ValueError(f"compute_axis must be one of {COMPUTE_AXES}, got {compute_axis!r}")
    if data_axis not in DATA_AXES:
        raise ValueError(f"data_axis must be one of {DATA_AXES}, got {data_axis!r}")
    if loss_key not in LOSS_KEYS:
        raise ValueError(f"loss_key must be one of {LOSS_KEYS}, got {loss_key!r}")

    points: list[FitPoint] = []
    for result in grid_results:
        role = str(result.get("role", "N"))
        if role in excluded_roles:
            continue
        cost = (
            float(result["num_params"])
            if compute_axis == "params"
            else float(result["flops_per_example"])
        )
        points.append(
            FitPoint(
                name=str(result.get("name", "<unnamed>")),
                role=role,
                architecture=_architecture(result),
                N=cost,
                D=float(result[data_axis]),
                loss=_observed_loss(result, loss_key),
                total_flops=float(result["flops_per_example"])
                * float(result["examples_seen"]),
            )
        )
    return points


def _huber(residual: np.ndarray, delta: float) -> np.ndarray:
    magnitude = np.abs(residual)
    return np.where(
        magnitude <= delta, 0.5 * residual**2, delta * (magnitude - 0.5 * delta)
    )


def _make_objective(
    log_cost: np.ndarray,
    log_data: np.ndarray,
    loss: np.ndarray,
    objective: str,
    delta: float,
):
    """The thing minimised, over theta = (log E, log A, log B, alpha, beta),
    returned with its analytic gradient.

    Fitting the three scale terms in logs is what keeps E, A and B positive
    without a constraint the optimiser has to be told about; a fit that
    wandered to a negative E would report a loss floor below zero and
    nothing downstream would notice.

    The gradient is derived rather than differenced because the Huber
    objective is very nearly L1 over the residuals actually seen -- delta is
    1e-3 -- and a finite-difference L-BFGS-B stops on such a surface while
    still sitting on its starting exponent. That failure is silent and looks
    exactly like a fit: it returns a plausible alpha, which is whichever one
    it was handed.

    `objective="lsq"` is the naive comparator the module docstring warns
    about -- squared residuals on raw loss -- and it differs from the
    default in the loss function *only*, so a comparison between them
    measures robustness rather than parameterisation.
    """
    if objective not in ("huber", "lsq"):
        raise ValueError(f"objective must be 'huber' or 'lsq', got {objective!r}")
    log_loss = np.log(loss)

    def evaluate(theta: np.ndarray) -> tuple[float, np.ndarray]:
        log_e, log_a, log_b, alpha, beta = theta
        terms = np.stack(
            [
                np.full_like(log_cost, log_e),
                log_a - alpha * log_cost,
                log_b - beta * log_data,
            ]
        )
        predicted_log = logsumexp(terms, axis=0)
        # Softmax weights over the three terms: each term's share of the
        # prediction is exactly its share of the derivative.
        share = np.exp(terms - predicted_log)
        jacobian = np.stack(
            [
                share[0],
                share[1],
                share[2],
                -share[1] * log_cost,
                -share[2] * log_data,
            ]
        )

        if objective == "huber":
            residual = predicted_log - log_loss
            value = _huber(residual, delta).mean()
            outer = np.clip(residual, -delta, delta)
        else:
            predicted = np.exp(predicted_log)
            residual = predicted - loss
            value = (residual**2).mean()
            # d/dtheta of the raw prediction is the log-space jacobian
            # scaled by the prediction itself.
            outer = 2.0 * residual * predicted

        return float(value), jacobian @ outer / residual.size

    return evaluate


def _seed_starts(
    log_cost: np.ndarray,
    log_data: np.ndarray,
    loss: np.ndarray,
    evaluate,
    n_starts: int,
) -> list[np.ndarray]:
    """Starting points, from a grid over the exponents only.

    At fixed (alpha, beta) the form is linear in (E, A, B), so the scale
    terms need not be guessed at all -- a non-negative least squares solves
    them exactly for each pair on the grid. That leaves a two-dimensional
    search where a naive multi-start does five, and it is what keeps the
    bootstrap affordable: without it, enough starts to reliably clear the
    objective's local minima costs more per resample than the resample is
    worth.
    """
    ones = np.ones_like(loss)
    scored: list[tuple[float, np.ndarray]] = []
    for alpha in EXPONENT_SEEDS:
        for beta in EXPONENT_SEEDS:
            design = np.stack(
                [ones, np.exp(-alpha * log_cost), np.exp(-beta * log_data)], axis=1
            )
            coefficients, _ = nnls(design, loss)
            theta = np.array(
                [
                    np.log(max(coefficients[0], _FLOOR)),
                    np.log(max(coefficients[1], _FLOOR)),
                    np.log(max(coefficients[2], _FLOOR)),
                    alpha,
                    beta,
                ]
            )
            scored.append((evaluate(theta)[0], theta))

    scored.sort(key=lambda pair: pair[0])
    return [theta for _, theta in scored[:n_starts]]


def _solve(
    cost: np.ndarray,
    data: np.ndarray,
    loss: np.ndarray,
    *,
    objective: str = "huber",
    delta: float = HUBER_DELTA,
    starts: list[np.ndarray] | None = None,
    n_starts: int = DEFAULT_STARTS,
) -> tuple[np.ndarray, float]:
    log_cost = np.log(cost)
    log_data = np.log(data)
    evaluate = _make_objective(log_cost, log_data, loss, objective, delta)

    if starts is None:
        starts = _seed_starts(log_cost, log_data, loss, evaluate, n_starts)

    bounds = [(-60.0, 60.0)] * 3 + [EXPONENT_BOUNDS] * 2
    best_theta, best_value = None, np.inf
    for start in starts:
        result = minimize(
            evaluate,
            np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds]),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            # Default ftol stops when the objective moves less than ~2e-9,
            # which on a mean Huber of order 1e-6 is a 0.1% relative change
            # -- i.e. before the exponents have moved off their seed.
            options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 2000},
        )
        if result.fun < best_value:
            best_theta, best_value = result.x, float(result.fun)

    if best_theta is None:  # pragma: no cover - only reachable with zero starts
        raise RuntimeError("no starting point produced a solution")
    return best_theta, best_value


def _spanning_resample(
    rng: np.random.Generator, cost: np.ndarray, data: np.ndarray, attempts: int = 50
) -> np.ndarray | None:
    """A bootstrap draw that can still estimate both exponents.

    A resample holding N constant cannot say anything about alpha, and the
    optimiser will return whatever the bound or the starting point gives it.
    Letting those into the percentile interval widens it for a reason that
    has nothing to do with the data, which is the opposite of what the
    interval is for.
    """
    size = cost.size
    for _ in range(attempts):
        index = rng.integers(0, size, size)
        if np.unique(cost[index]).size >= 2 and np.unique(data[index]).size >= 2:
            return index
    return None


def _bootstrap_intervals(
    cost: np.ndarray,
    data: np.ndarray,
    loss: np.ndarray,
    theta: np.ndarray,
    *,
    objective: str,
    delta: float,
    samples: int,
    ci: float,
    seed: int,
) -> tuple[dict[str, tuple[float, float]], int]:
    if samples <= 0:
        nan = (float("nan"), float("nan"))
        return {"alpha": nan, "beta": nan, "E": nan}, 0

    rng = np.random.default_rng(seed)
    # Warm-started from the full-data solution plus two grid seeds: a
    # resample is a small perturbation of the data it came from, so the
    # basin is almost always the same one, and paying for a full multi-start
    # per draw would make the interval cost more than the fit.
    log_cost, log_data = np.log(cost), np.log(data)
    evaluate = _make_objective(log_cost, log_data, loss, objective, delta)
    warm = [theta] + _seed_starts(log_cost, log_data, loss, evaluate, 2)

    draws: list[np.ndarray] = []
    for _ in range(samples):
        index = _spanning_resample(rng, cost, data)
        if index is None:
            continue
        drawn, _ = _solve(
            cost[index], data[index], loss[index],
            objective=objective, delta=delta, starts=warm,
        )
        draws.append(drawn)

    if not draws:
        nan = (float("nan"), float("nan"))
        return {"alpha": nan, "beta": nan, "E": nan}, 0

    stacked = np.stack(draws)
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2

    def interval(values: np.ndarray) -> tuple[float, float]:
        return (float(np.percentile(values, lo)), float(np.percentile(values, hi)))

    return (
        {
            "alpha": interval(stacked[:, 3]),
            "beta": interval(stacked[:, 4]),
            "E": interval(np.exp(stacked[:, 0])),
        },
        len(draws),
    )


def _predict(theta: np.ndarray, cost: np.ndarray, data: np.ndarray) -> np.ndarray:
    log_e, log_a, log_b, alpha, beta = theta
    return np.exp(log_e) + np.exp(log_a) * cost**-alpha + np.exp(log_b) * data**-beta


def fit_scaling_law(
    grid_results: list[dict],
    compute_axis: str = "params",
    *,
    data_axis: str = "train_rows",
    loss_key: str = "decision_picks",
    architecture: str | None = None,
    objective: str = "huber",
    huber_delta: float = HUBER_DELTA,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = 0,
    held_out_roles: tuple[str, ...] = HELD_OUT_ROLES,
    excluded_roles: tuple[str, ...] = EXCLUDED_ROLES,
) -> ScalingFit:
    """compute_axis: "params" for the iso-parameter fit, "flops" for the
    iso-FLOP fit.

    One architecture per fit. Passing both arms' results without naming one
    raises rather than fitting a single surface through the pair, because
    such a surface describes neither arm and the whole point of the grid is
    the difference between them; use `fit_by_architecture` for both.

    `objective="lsq"` fits squared residuals on raw loss instead. It exists
    so the robustness claim in this module's docstring can be tested rather
    than asserted, and is not a supported way to produce a number.
    """
    points = collect_points(
        grid_results,
        compute_axis=compute_axis,
        data_axis=data_axis,
        loss_key=loss_key,
        excluded_roles=excluded_roles,
    )
    if architecture is not None:
        points = [p for p in points if p.architecture == architecture]

    arms = sorted({p.architecture for p in points})
    if not arms:
        raise ValueError("no cells left to fit after filtering")
    if len(arms) > 1:
        raise ValueError(
            f"results span {arms}; one surface through both arms is a curve for "
            "neither. Pass architecture=, or call fit_by_architecture()."
        )

    fitted = [p for p in points if p.role not in held_out_roles]
    held = [p for p in points if p.role in held_out_roles]

    # Five free parameters. Fewer points than that is an interpolation
    # reported as a law, and the residual it reports will be zero.
    if len(fitted) < 5:
        raise ValueError(
            f"{len(fitted)} cells for 5 parameters; the fit would interpolate. "
            "Run more of the grid, or fold the interior points in with "
            "held_out_roles=()."
        )
    cost = np.array([p.N for p in fitted], dtype=float)
    data = np.array([p.D for p in fitted], dtype=float)
    loss = np.array([p.loss for p in fitted], dtype=float)
    if np.unique(cost).size < 2 or np.unique(data).size < 2:
        raise ValueError(
            "the fitted cells vary along only one axis, so one of the two "
            "exponents is unidentified and would be reported anyway"
        )

    theta, _ = _solve(cost, data, loss, objective=objective, delta=huber_delta)
    intervals, draws = _bootstrap_intervals(
        cost, data, loss, theta,
        objective=objective, delta=huber_delta,
        samples=bootstrap, ci=ci, seed=seed,
    )

    log_residual = np.log(_predict(theta, cost, data)) - np.log(loss)
    residuals = tuple(
        HeldOut(
            name=p.name,
            role=p.role,
            N=p.N,
            D=p.D,
            observed=p.loss,
            predicted=float(_predict(theta, p.N, p.D)),
            residual=p.loss - float(_predict(theta, p.N, p.D)),
        )
        for p in held
    )
    held_values = np.array([r.residual for r in residuals], dtype=float)

    log_e, log_a, log_b, alpha, beta = theta
    return ScalingFit(
        E=float(np.exp(log_e)),
        A=float(np.exp(log_a)),
        B=float(np.exp(log_b)),
        alpha=float(alpha),
        beta=float(beta),
        alpha_ci=intervals["alpha"],
        beta_ci=intervals["beta"],
        E_ci=intervals["E"],
        architecture=arms[0],
        compute_axis=compute_axis,
        data_axis=data_axis,
        loss_key=loss_key,
        objective=objective,
        points=tuple(fitted),
        held_out=residuals,
        rmse_log=float(np.sqrt(np.mean(log_residual**2))),
        held_out_bias=float(held_values.mean()) if residuals else float("nan"),
        held_out_rmse=(
            float(np.sqrt((held_values**2).mean())) if residuals else float("nan")
        ),
        bootstrap_samples=draws,
    )


def fit_by_architecture(
    grid_results: list[dict], compute_axis: str = "params", **kwargs
) -> dict[str, ScalingFit]:
    """One curve per arm, which is the form the sizing decision needs.

    The grid exists to say which arm is better at the budget the final run
    will have, and that question is asked of two curves rather than of one
    surface with an architecture column.
    """
    arms = sorted(
        {
            _architecture(r)
            for r in grid_results
            if str(r.get("role", "N")) not in kwargs.get("excluded_roles", EXCLUDED_ROLES)
        }
    )
    return {
        arm: fit_scaling_law(grid_results, compute_axis, architecture=arm, **kwargs)
        for arm in arms
    }


def compare_at(fits: dict[str, ScalingFit], N: float, D: float) -> dict:
    """Both arms' fitted curves at one (N, D), and the gap between them.

    Refuses to compare curves fitted on different cost axes: "the same N" is
    a different sentence under compute_axis="params" than under "flops", and
    a comparison that mixed them would silently be answering neither
    question.
    """
    axes = {f.compute_axis for f in fits.values()}
    if len(axes) > 1:
        raise ValueError(f"fits use different cost axes {sorted(axes)}; not comparable")

    predicted = {arm: fit.predict(N, D) for arm, fit in fits.items()}
    ordered = sorted(predicted.items(), key=lambda kv: kv[1])
    return {
        "N": float(N),
        "D": float(D),
        "compute_axis": axes.pop() if axes else None,
        "loss": predicted,
        "best": ordered[0][0] if ordered else None,
        "gap": (ordered[1][1] - ordered[0][1]) if len(ordered) > 1 else 0.0,
    }


def format_fit(fit: ScalingFit) -> str:
    def interval(bounds: tuple[float, float]) -> str:
        lo, hi = bounds
        return "        --      " if np.isnan(lo) else f"[{lo:.4f}, {hi:.4f}]"

    lines = [
        f"{fit.architecture} | {fit.compute_axis} axis | {fit.loss_key} loss | "
        f"{fit.n_points} cells, {fit.bootstrap_samples} bootstrap draws",
        f"  E     {fit.E:>10.4f}  {interval(fit.E_ci)}",
        f"  alpha {fit.alpha:>10.4f}  {interval(fit.alpha_ci)}",
        f"  beta  {fit.beta:>10.4f}  {interval(fit.beta_ci)}",
        f"  A     {fit.A:>10.4g}      B {fit.B:>10.4g}",
        f"  rmse (log residual) {fit.rmse_log:.5f}",
    ]
    if fit.held_out:
        lines.append(
            f"  held out ({len(fit.held_out)} cells): bias {fit.held_out_bias:+.4f} "
            f"rmse {fit.held_out_rmse:.4f}"
        )
        for point in fit.held_out:
            lines.append(
                f"    {point.name:<28} observed {point.observed:.4f} "
                f"predicted {point.predicted:.4f} residual {point.residual:+.4f}"
            )
    return "\n".join(lines)
