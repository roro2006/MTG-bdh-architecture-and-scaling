"""Tests for the analytic FLOP derivation.

Same discipline as the parameter counts: the number that goes into an
iso-FLOP comparison has to be derived and then checked against something
independent. Here that something is XLA's own cost model for the compiled
forward pass.

The two will not agree exactly, and should not. The analytic count is
matmuls only; XLA counts the layer norms, GELUs and softmaxes too. Those
scale as d while matmuls scale as d^2, so the gap has to *shrink* with
width -- which is itself a testable property, and a stronger check than any
single tolerance.
"""

from __future__ import annotations

import pytest

from src.models.flops import (
    BACKWARD_MULTIPLIER,
    arm_flop_share,
    bdh_ideal_flops,
    count_flops_analytic,
    flops_per_step,
    measure_flops_xla,
    total_training_flops,
)
from src.models.pick_model import init_model

from .synthetic import FEATURE_DIM


def test_components_sum_to_the_total(model_config):
    counts = count_flops_analytic(model_config(64))
    assert sum(v for k, v in counts.items() if k != "total") == pytest.approx(
        counts["total"]
    )


def test_flops_scale_quadratically_in_width(model_config):
    """Matmul cost is dominated by d^2 terms, so doubling width should
    roughly quadruple FLOPs -- and approach 4x as the linear terms fade.
    """
    ratios = []
    for width in (64, 128, 256, 512):
        small = count_flops_analytic(model_config(width))["total"]
        large = count_flops_analytic(model_config(2 * width))["total"]
        ratios.append(large / small)
    assert all(3.5 < r < 4.0 for r in ratios), ratios
    # Later ratios must be closer to 4 than earlier ones.
    assert ratios == sorted(ratios)


@pytest.mark.slow
@pytest.mark.parametrize("width", [32, 64, 128])
def test_analytic_agrees_with_xla_cost_model(feature_table, model_config, width):
    config = model_config(width)
    model, params = init_model(config, feature_table, seed=0)
    analytic = count_flops_analytic(config)["total"]
    measured = measure_flops_xla(model, params, feature_table)
    if measured is None:
        pytest.skip("backend provides no cost model")

    # XLA counts elementwise work the derivation omits, so it reads higher,
    # never lower.
    assert measured >= analytic
    assert measured / analytic < 1.15


@pytest.mark.slow
def test_the_gap_to_xla_shrinks_with_width(feature_table, model_config):
    """The excluded terms are linear in d; the counted ones are quadratic.

    If this ever stopped holding it would mean the derivation is missing
    something that scales like the matmuls, not just the elementwise tail.
    """
    ratios = []
    for width in (32, 128, 384):
        config = model_config(width)
        model, params = init_model(config, feature_table, seed=0)
        measured = measure_flops_xla(model, params, feature_table)
        if measured is None:
            pytest.skip("backend provides no cost model")
        ratios.append(measured / count_flops_analytic(config)["total"])

    assert ratios == sorted(ratios, reverse=True), ratios
    assert ratios[-1] < 1.02


def test_backward_pass_is_three_times_forward(model_config):
    config = model_config(64)
    forward = count_flops_analytic(config)["total"]
    training = count_flops_analytic(config, include_backward=True)["total"]
    assert training == pytest.approx(forward * (1 + BACKWARD_MULTIPLIER))

    assert flops_per_step(config, batch_size=8) == pytest.approx(training * 8)
    assert total_training_flops(config, batch_size=8, steps=10) == pytest.approx(
        training * 80
    )


def test_padding_is_counted_because_a_dense_pass_pays_for_it(model_config):
    """A shorter pool must cost strictly less, which is what makes the
    padded default a deliberate choice rather than an oversight.
    """
    config = model_config(64)
    padded = count_flops_analytic(config, max_pool=41)["total"]
    short = count_flops_analytic(config, max_pool=10)["total"]
    assert short < padded


def test_arm_share_is_a_minority_of_the_budget_by_default(model_config):
    """The arm is the only component that differs between architectures.

    With the default shape it is about a quarter of the forward pass, which
    is what makes the iso-FLOP comparison weaker than the fairness note in
    docs/ARCHITECTURE.md assumes. Pinned here so a shape change that fixes
    it -- or makes it worse -- is visible rather than silent.
    """
    share = arm_flop_share(model_config(256))
    assert 0.2 < share < 0.35

    # Deepening the arm moves the budget toward the mechanism under study.
    deeper = arm_flop_share(model_config(256, arm_layers=8))
    assert deeper > 0.5
    assert deeper > share


def test_bdh_arm_is_counted_separately_from_attention(model_config):
    """It must not silently inherit the attention arm's count: BDH's whole
    claim is that it does different arithmetic per weight.
    """
    config = model_config(256)
    attention = count_flops_analytic(config, arm="attention")
    bdh = count_flops_analytic(config, arm="bdh")

    assert bdh["arm"] != attention["arm"]
    # Everything outside the arm is the shared front-end and must be
    # untouched by which arm is selected -- that is the premise of the whole
    # comparison.
    for component in ("card_embedding", "pack_encoder", "pool_encoder", "pointer"):
        assert bdh[component] == attention[component]


def test_bdh_ideal_flops_are_bounded_by_the_unskippable_encodes(model_config):
    """Sparsity cannot make BDH free, and the accounting has to show why.

    Only the score and decode reductions shrink with density. The three
    D->N encodes are paid in full at any density, because a ReLU's output
    is not known until its input has been computed. So the ideal count has
    a floor well above zero, and reporting an iso-FLOP comparison as though
    it did not is the exact error the fairness note warns about.
    """
    config = model_config(256)
    dense = bdh_ideal_flops(config, score_density=1.0, gate_density=1.0)
    assert dense["total"] == pytest.approx(dense["dense_total"])

    vanishing = bdh_ideal_flops(config, score_density=0.0, gate_density=0.0)
    assert vanishing["scores"] == 0.0
    assert vanishing["decode"] == 0.0
    # Even with every neuron dead, the encodes survive and dominate.
    assert vanishing["total"] > 0.5 * dense["dense_total"]

    sparse = bdh_ideal_flops(config, score_density=0.05, gate_density=0.05)
    assert vanishing["total"] < sparse["total"] < dense["total"]


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_bdh_ideal_flops_rejects_impossible_densities(model_config, bad):
    """A density outside [0, 1] means the measurement is wrong; silently
    scaling by it would produce a plausible number from a broken input."""
    with pytest.raises(ValueError, match="fraction"):
        bdh_ideal_flops(model_config(64), score_density=bad, gate_density=0.5)


def test_feature_dim_enters_only_through_the_embedding(model_config):
    wide = count_flops_analytic(model_config(64, card_feature_dim=FEATURE_DIM * 2))
    narrow = count_flops_analytic(model_config(64, card_feature_dim=FEATURE_DIM))
    for component in ("pack_encoder", "pool_encoder", "arm", "pointer"):
        assert wide[component] == narrow[component]
    assert wide["card_embedding"] > narrow["card_embedding"]
