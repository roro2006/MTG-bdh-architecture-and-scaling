"""Runs the (architecture x model size x data fraction x seed) sweep
described in docs/PROJECT_PLAN.md section 4.

Pilot grid: a handful of model sizes x a couple of data fractions x one
seed x both architectures, to catch pipeline bugs before spending real
compute. Full grid: five or six log-spaced model sizes, four or five
log-spaced data fractions, two seeds, both architectures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridCell:
    architecture: str  # "attention" or "bdh"
    num_params: int
    data_fraction: float
    seed: int


def pilot_grid() -> list[GridCell]:
    raise NotImplementedError


def full_grid() -> list[GridCell]:
    raise NotImplementedError


def run_cell(cell: GridCell) -> dict:
    """Trains one grid cell and returns its final loss plus metadata
    needed for the curve fit (actual parameter count, actual FLOPs/step).
    """
    raise NotImplementedError
