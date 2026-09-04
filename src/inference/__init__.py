"""Running a trained drafter: pack and pool in, ranked picks out.

`Drafter` is the entry point (PROJECT_PLAN section 8). `PickProbe` is the
layer under it, shared with `src/analysis/`, which speaks card ids rather
than names and makes none of the guards `Drafter` does -- use it for
deliberately synthetic states, and `Drafter` for real ones.
"""

from .drafter import Drafter, PickRanking, RankedPick, UnknownCardError
from .metrics import calibration, ranking_report
from .probe import PickProbe

__all__ = [
    "Drafter",
    "PickProbe",
    "PickRanking",
    "RankedPick",
    "UnknownCardError",
    "calibration",
    "ranking_report",
]
