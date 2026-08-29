"""Fits L(N, D) = E + A / N^alpha + B / D^beta to grid results, following
the Chinchilla paper's robust-fit approach (Huber loss on log-residuals,
not naive least squares) so the small-N, high-loss corner of the grid
doesn't dominate the fit. See docs/PROJECT_PLAN.md section 5.

Both an iso-parameter and an iso-FLOP fit are produced for each
architecture — see docs/ARCHITECTURE.md's fairness note on why these are
genuinely different comparisons here.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def fit_scaling_law(grid_results: list[dict], compute_axis: str = "params") -> ScalingFit:
    """compute_axis: "params" for the iso-parameter fit, "flops" for the
    iso-FLOP fit.
    """
    raise NotImplementedError
