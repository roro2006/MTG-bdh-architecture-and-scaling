"""Running a trained drafter: pack and pool in, ranked picks out.

`Drafter` is the entry point (PROJECT_PLAN section 8). `PickProbe` is the
layer under it, shared with `src/analysis/`, which speaks card ids rather
than names and makes none of the guards `Drafter` does -- use it for
deliberately synthetic states, and `Drafter` for real ones.

The re-exports below are resolved on first use rather than at import.
Importing `drafter` here eagerly would put it in `sys.modules` before
`python -m src.inference.drafter` -- the invocation this repo's CLIs all
use -- got to execute it, and runpy would then run a *second* copy of the
module under the name `__main__`, warning about it on every run.
"""

_LAZY = {
    "Drafter": "drafter",
    "PickRanking": "drafter",
    "RankedPick": "drafter",
    "UnknownCardError": "drafter",
    "calibration": "metrics",
    "ranking_report": "metrics",
    "PickProbe": "probe",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{_LAZY[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # resolved once; later lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
