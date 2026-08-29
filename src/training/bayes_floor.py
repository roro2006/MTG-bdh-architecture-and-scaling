"""Measures the human disagreement floor from the matched-state subset
carved out in src/data/dataset.py, and checks whether a ScalingFit's E
converges toward it. See docs/PROJECT_PLAN.md section 6.

A mismatch is a reportable result, not just a failed sanity check: if the
fitted E comes in below the measured floor, that points to the model (or
the split) exploiting something it shouldn't have access to.
"""

from __future__ import annotations


def measure_disagreement_floor(matched_state_examples: list) -> float:
    """Returns the cross-entropy of the empirical human pick distribution
    against itself, aggregated over all recurring (pack, pool) states.
    """
    raise NotImplementedError
