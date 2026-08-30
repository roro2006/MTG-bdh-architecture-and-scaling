"""Analytic FLOP counts, derived term by term like the parameter counts.

`docs/PROJECT_PLAN.md` section 5 and the fairness note in
`docs/ARCHITECTURE.md` both require an iso-FLOP comparison alongside the
iso-parameter one, because BDH's activations are sparse by design and a
forward pass can do meaningfully less arithmetic than a dense block holding
the same number of weights. Reporting only iso-parameter would let "which
architecture scales better" mean two different things depending on which
axis is held fixed.

Conventions, stated because FLOP counts are only comparable if the
convention is:

  - a matmul (m, k) @ (k, n) costs 2*m*n*k -- one multiply and one add per
    accumulation. This is the Kaplan/Chinchilla convention.
  - **matmuls only.** Layer norms, GELUs, softmaxes and masking are real
    arithmetic but they scale linearly in d while matmuls scale as d^2, so
    they vanish as a fraction as the grid grows. Measured against XLA's own
    cost model they account for about 5% at d=64 and less above it. They
    are excluded here rather than estimated with invented constants, and
    `measure_flops_xla` exists to keep that gap honest and visible.
  - **padded lengths, not true set sizes.** A dense implementation pays for
    all 14 pack slots and all 41 pool slots regardless of how many hold a
    real card. That is the arithmetic actually spent, so it is what an
    iso-FLOP comparison should hold fixed. Pass smaller `max_pack`/`max_pool`
    to ask the different question of what the real cards alone would cost.
  - the backward pass is taken as 2x the forward, the standard estimate,
    so a training step is 3x forward.

Cross-checked against XLA's compiled cost analysis in tests/test_flops.py,
which is the same discipline the parameter counts get: a derivation nobody
checks is a number nobody should trust.
"""

from __future__ import annotations

from .pick_model import ModelConfig

# Padded widths the model always computes over. These match
# src/data/dataset.py's MAX_POOL_SIZE and ingest.py's MAX_PACK_SIZE.
DEFAULT_MAX_PACK = 14
DEFAULT_MAX_POOL = 41

BACKWARD_MULTIPLIER = 2.0  # backward ~ 2x forward; a step is 3x


def _matmul(m: int, k: int, n: int) -> int:
    """(m, k) @ (k, n): one multiply and one add per accumulation."""
    return 2 * m * k * n


def _dense(tokens: int, fan_in: int, fan_out: int) -> int:
    return _matmul(tokens, fan_in, fan_out)


def _self_attention(length: int, d: int) -> int:
    """Q, K, V and output projections, plus the two length x length matmuls."""
    projections = 4 * _dense(length, d, d)          # q, k, v, out
    scores = _matmul(length, d, length)             # Q @ K^T
    values = _matmul(length, length, d)             # weights @ V
    return projections + scores + values


def _cross_attention(query_length: int, context_length: int, d: int) -> int:
    """Queries come from the pack, keys and values from the pool."""
    query_projection = _dense(query_length, d, d)
    kv_projections = 2 * _dense(context_length, d, d)
    output_projection = _dense(query_length, d, d)
    scores = _matmul(query_length, d, context_length)
    values = _matmul(query_length, context_length, d)
    return query_projection + kv_projections + output_projection + scores + values


def _mlp(tokens: int, d: int, ratio: int) -> int:
    return _dense(tokens, d, ratio * d) + _dense(tokens, ratio * d, d)


def count_flops_analytic(
    config: ModelConfig,
    arm: str = "attention",
    max_pack: int = DEFAULT_MAX_PACK,
    max_pool: int = DEFAULT_MAX_POOL,
    include_backward: bool = False,
) -> dict[str, float]:
    """Forward FLOPs for one example, per component, with the total.

    `include_backward` scales by 3 to give the cost of a full training step.
    """
    d = config.hidden_dim
    r = config.mlp_ratio
    embed_hidden = config.embed_hidden

    # Every card in the pack and the pool goes through the same embedding.
    cards = max_pack + max_pool
    embedding = _dense(cards, config.card_feature_dim, embed_hidden) + _dense(
        cards, embed_hidden, d
    )

    if config.set_encoder_mode == "attention":
        def block(length: int) -> int:
            return _self_attention(length, d) + _mlp(length, d, r)
    else:
        def block(length: int) -> int:
            return _mlp(length, d, r)

    pack_encoder = config.pack_encoder_layers * block(max_pack)
    pool_encoder = config.pool_encoder_layers * block(max_pool)

    counts = {
        "card_embedding": float(embedding),
        "pack_encoder": float(pack_encoder),
        "pool_encoder": float(pool_encoder),
        "pointer": float(_dense(max_pack, d, 1)),
    }

    if arm == "attention":
        # The arm prepends a learned null key, so the pool it attends over
        # is one longer than the padded pool.
        context_length = max_pool + 1
        per_layer = _cross_attention(max_pack, context_length, d) + _mlp(max_pack, d, r)
        counts["arm"] = float(config.arm_layers * per_layer)
    else:
        raise NotImplementedError(
            f"no analytic FLOP derivation for arm {arm!r}; the BDH arm's count "
            "has to account for its activation sparsity explicitly"
        )

    total = sum(counts.values())
    if include_backward:
        scale = 1.0 + BACKWARD_MULTIPLIER
        counts = {k: v * scale for k, v in counts.items()}
        total *= scale
    counts["total"] = total
    return counts


def arm_flop_share(config: ModelConfig, arm: str = "attention", **kwargs) -> float:
    """Fraction of forward FLOPs spent in the interaction arm.

    Worth checking before trusting an iso-FLOP comparison. The arm is the
    only component that differs between the two architectures, but with the
    default shape it is about 27% of the forward pass -- the shared
    front-end, and the pool encoder in particular, carries the rest. If the
    BDH arm turned out to be free, total FLOPs would fall by only that 27%,
    so the iso-FLOP and iso-parameter curves would be nearly the same
    experiment and the fairness note in docs/ARCHITECTURE.md would not be
    doing the work it is meant to.

    Raising `arm_layers` relative to the encoder depths moves the budget
    toward the mechanism under study: at d=256, arm_layers=4 puts it at 42%
    and arm_layers=8 at 59%.
    """
    counts = count_flops_analytic(config, arm=arm, **kwargs)
    return counts["arm"] / counts["total"]


def flops_per_step(
    config: ModelConfig,
    batch_size: int,
    arm: str = "attention",
    **kwargs,
) -> float:
    """Training FLOPs for one optimiser step over a full batch."""
    per_example = count_flops_analytic(
        config, arm=arm, include_backward=True, **kwargs
    )["total"]
    return per_example * batch_size


def total_training_flops(
    config: ModelConfig,
    batch_size: int,
    steps: int,
    arm: str = "attention",
    **kwargs,
) -> float:
    """C for a whole run -- the x-axis of the iso-FLOP comparison."""
    return flops_per_step(config, batch_size, arm=arm, **kwargs) * steps


def measure_flops_xla(
    model,
    params,
    feature_table,
    batch_size: int = 8,
    max_pack: int = DEFAULT_MAX_PACK,
    max_pool: int = DEFAULT_MAX_POOL,
) -> float | None:
    """XLA's own forward-pass FLOP estimate for one example.

    An independent check on the derivation above rather than a replacement
    for it: XLA counts elementwise work the analytic count deliberately
    omits, so it reads a few percent higher and the gap should shrink with
    width. Returns None if the backend does not provide a cost model.
    """
    import jax
    import jax.numpy as jnp

    pack = jnp.zeros((batch_size, max_pack), jnp.int32)
    pool = jnp.zeros((batch_size, max_pool), jnp.int32)
    scalars = jnp.zeros((batch_size,), jnp.int32)

    compiled = jax.jit(
        lambda p: model.apply(p, feature_table, pack, pool, scalars, scalars)
    ).lower(params).compile()
    try:
        analysis = compiled.cost_analysis()
    except Exception:
        return None
    if isinstance(analysis, (list, tuple)):
        analysis = analysis[0]
    flops = analysis.get("flops")
    return None if flops is None else float(flops) / batch_size
