"""The Pallas kernels must compute exactly what the reference computes.

This is the test that makes the kernels usable. A kernel that is merely
fast is worthless here: the attention arm is the control every BDH result
is measured against, so a silently wrong kernel would not produce a visible
failure, it would produce a plausible scaling curve that happens to be
false. Parameter-count tests cannot catch that -- the shapes stay right
while the numbers go wrong.

So every kernel is asserted against a pure-JAX reference on values *and* on
every gradient it produces, and the fused Flax blocks are asserted against
the reference blocks with one shared set of parameters.

Everything runs under `interpret=True` so the suite passes on a CPU-only
machine. That checks the kernels' semantics, not their lowering; a GPU or
TPU box should run the same file with `interpret=False`, which is what
`KERNEL_INTERPRET` is for.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import pytest

from src.models.attention_arm import CrossAttentionArm, CrossAttentionBlock
from src.models.bdh_arm import BDHArm, BDHBlock, measure_density
from src.models.kernels.bdh import (
    bdh_gate_decode,
    bdh_scores,
    reference_gate_decode,
    reference_scores,
)
from src.models.kernels.cross_attention import (
    FusedCrossAttentionBlock,
    fused_attention,
    reference_attention,
)

# float32 matmuls reassociated by a different schedule; ~1e-6 relative is
# the expected disagreement, and anything larger is a real bug.
ATOL = 2e-4
RTOL = 2e-5

INTERPRET = os.environ.get("KERNEL_INTERPRET", "1") == "1"
KERNEL_KW = dict(interpret=INTERPRET)
BDH_KW = dict(interpret=INTERPRET, block_n=16)

D, HEADS, BATCH, PACK, POOL = 32, 4, 3, 14, 41
# Lengths chosen to exercise the awkward cases: a full pack, a part-full
# one, a single card, and -- for the pool -- the empty pool at pick zero.
PACK_LENGTHS = jnp.array([14, 7, 1])
POOL_LENGTHS = jnp.array([41, 20, 0])


def close(a, b):
    return jnp.allclose(a, b, atol=ATOL, rtol=RTOL)


def grad_of(fn, argnums):
    return jax.grad(lambda *a: fn(*a), argnums=argnums)


@pytest.fixture
def masks():
    pack_mask = jnp.arange(PACK)[None, :] < PACK_LENGTHS[:, None]
    pool_mask = jnp.arange(POOL)[None, :] < POOL_LENGTHS[:, None]
    return pack_mask, pool_mask


@pytest.fixture
def arm_inputs(masks):
    pack_mask, pool_mask = masks
    keys = jax.random.split(jax.random.key(0), 3)
    return dict(
        pack=jax.random.normal(keys[0], (BATCH, PACK, D)),
        pool=jax.random.normal(keys[1], (BATCH, POOL, D)),
        context=jax.random.normal(keys[2], (BATCH, D)),
        pack_mask=pack_mask,
        pool_mask=pool_mask,
    )


def param_paths(tree, prefix=""):
    """Flattened 'name shape' strings, for comparing two parameter trees."""
    out = []
    for key, value in sorted(tree.items()):
        if hasattr(value, "shape"):
            out.append(f"{prefix}{key} {value.shape}")
        else:
            out.extend(param_paths(value, f"{prefix}{key}/"))
    return out


# --------------------------------------------------------------------------
# Cross-attention kernel
# --------------------------------------------------------------------------

class TestFusedAttention:
    @pytest.fixture
    def qkv(self):
        keys = jax.random.split(jax.random.key(1), 3)
        head_dim = D // HEADS
        return (
            jax.random.normal(keys[0], (BATCH, HEADS, PACK, head_dim)),
            jax.random.normal(keys[1], (BATCH, HEADS, POOL, head_dim)),
            jax.random.normal(keys[2], (BATCH, HEADS, POOL, head_dim)),
        )

    @pytest.fixture
    def mask(self, masks):
        pack_mask, pool_mask = masks
        return pack_mask[:, None, :, None] & pool_mask[:, None, None, :]

    def test_forward_matches_reference(self, qkv, mask):
        q, k, v = qkv
        assert close(
            reference_attention(q, k, v, mask), fused_attention(q, k, v, mask, **KERNEL_KW)
        )

    def test_gradients_match_reference(self, qkv, mask):
        q, k, v = qkv
        weight = jax.random.normal(jax.random.key(2), (BATCH, HEADS, PACK, D // HEADS))
        ref = lambda *a: (reference_attention(*a, mask) * weight).sum()
        fused = lambda *a: (fused_attention(*a, mask, **KERNEL_KW) * weight).sum()

        for expected, actual in zip(
            grad_of(ref, (0, 1, 2))(q, k, v), grad_of(fused, (0, 1, 2))(q, k, v)
        ):
            assert close(expected, actual)

    def test_empty_query_row_is_zero(self, qkv, mask):
        """A pack slot with no visible key contributes nothing.

        Flax would hand such a row a uniform softmax over every key, whose
        value depends on how far the key axis happens to be padded. Both
        paths here define it as zero instead; see EMPTY_ROW_NOTE.
        """
        q, k, v = qkv
        out = fused_attention(q, k, v, mask, **KERNEL_KW)
        empty = ~mask.any(axis=-1)  # (B, 1, Lq)
        assert jnp.all(out[jnp.broadcast_to(empty, out.shape[:3])] == 0.0)

    def test_block_matches_reference_block(self, masks):
        """Same parameters through both blocks give the same answer.

        The parameter trees must match first -- otherwise this would be
        comparing two different models and passing for the wrong reason.
        """
        pack_mask, pool_mask = masks
        keys = jax.random.split(jax.random.key(3), 3)
        queries = jax.random.normal(keys[0], (BATCH, PACK, D))
        context = jax.random.normal(keys[1], (BATCH, POOL, D))

        ref = CrossAttentionBlock(hidden_dim=D, num_heads=HEADS)
        fused = FusedCrossAttentionBlock(hidden_dim=D, num_heads=HEADS, **KERNEL_KW)
        args = (queries, context, pack_mask, pool_mask)

        params = ref.init(keys[2], *args)
        assert param_paths(params["params"]) == param_paths(
            fused.init(keys[2], *args)["params"]
        )

        # Compared on rows that carry a decision: a real pack slot whose
        # pool is non-empty. The other rows are exactly the ones the two
        # paths deliberately disagree on (EMPTY_ROW_NOTE) and exactly the
        # ones PointerHead throws away. A bare block has no null key, so at
        # pick zero every row is an empty row.
        comparable = (pack_mask & pool_mask.any(axis=-1)[:, None])[..., None]
        a, b = ref.apply(params, *args), fused.apply(params, *args)
        assert close(jnp.where(comparable, a, 0.0), jnp.where(comparable, b, 0.0))


# --------------------------------------------------------------------------
# BDH kernels
# --------------------------------------------------------------------------

class TestBDHKernels:
    NEURONS = 64

    @pytest.fixture
    def tensors(self):
        keys = jax.random.split(jax.random.key(4), 6)
        return dict(
            xq=jax.random.normal(keys[0], (BATCH, PACK, D)),
            xk=jax.random.normal(keys[1], (BATCH, POOL, D)),
            enc=jax.random.normal(keys[2], (HEADS, D, self.NEURONS)) * 0.1,
            encv=jax.random.normal(keys[3], (HEADS, D, self.NEURONS)) * 0.1,
            dec=jax.random.normal(keys[4], (HEADS, self.NEURONS, D)) * 0.1,
            yn=jax.random.normal(keys[5], (BATCH, HEADS, PACK, D)),
        )

    def test_scores_forward(self, tensors):
        args = (tensors["xq"], tensors["xk"], tensors["enc"])
        assert close(reference_scores(*args), bdh_scores(*args, **BDH_KW))

    def test_scores_gradients(self, tensors):
        args = (tensors["xq"], tensors["xk"], tensors["enc"])
        weight = jax.random.normal(jax.random.key(5), (BATCH, HEADS, PACK, POOL))
        ref = lambda *a: (reference_scores(*a) * weight).sum()
        fused = lambda *a: (bdh_scores(*a, **BDH_KW) * weight).sum()

        for expected, actual in zip(
            grad_of(ref, (0, 1, 2))(*args), grad_of(fused, (0, 1, 2))(*args)
        ):
            assert close(expected, actual)

    def test_gate_decode_forward(self, tensors):
        args = tuple(tensors[k] for k in ("xq", "yn", "enc", "encv", "dec"))
        assert close(reference_gate_decode(*args), bdh_gate_decode(*args, **BDH_KW))

    def test_gate_decode_gradients(self, tensors):
        args = tuple(tensors[k] for k in ("xq", "yn", "enc", "encv", "dec"))
        weight = jax.random.normal(jax.random.key(6), (BATCH, HEADS, PACK, D))
        ref = lambda *a: (reference_gate_decode(*a) * weight).sum()
        fused = lambda *a: (bdh_gate_decode(*a, **BDH_KW) * weight).sum()

        for expected, actual in zip(
            grad_of(ref, (0, 1, 2, 3, 4))(*args), grad_of(fused, (0, 1, 2, 3, 4))(*args)
        ):
            assert close(expected, actual)

    @pytest.mark.parametrize("block_n", [16, 32, 64])
    def test_neuron_tiling_does_not_change_the_answer(self, tensors, block_n):
        """The neuron axis is tiled for SRAM, not for semantics."""
        args = (tensors["xq"], tensors["xk"], tensors["enc"])
        assert close(
            reference_scores(*args),
            bdh_scores(*args, block_n=block_n, interpret=INTERPRET),
        )


# --------------------------------------------------------------------------
# Whole arms
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "arm_cls,kwargs",
    [
        (CrossAttentionArm, {}),
        (BDHArm, {"neuron_multiplier": 4}),
    ],
    ids=["attention", "bdh"],
)
def test_fused_arm_matches_reference_arm(arm_inputs, arm_cls, kwargs):
    """End to end: one parameter set, two execution paths, same answer.

    Gradients matter more than values here. A kernel can get the forward
    pass right and still train to a different model, which is the failure
    that would quietly corrupt a scaling curve.
    """
    build = lambda **extra: arm_cls(
        hidden_dim=D, num_heads=HEADS, num_layers=2, **kwargs, **extra
    )
    reference, fused = build(), build(fused=True)
    args = (
        arm_inputs["pack"],
        arm_inputs["pack_mask"],
        arm_inputs["pool"],
        arm_inputs["pool_mask"],
        arm_inputs["context"],
    )

    params = reference.init(jax.random.key(7), *args)
    assert param_paths(params["params"]) == param_paths(
        fused.init(jax.random.key(7), *args)["params"]
    )

    # Padded pack slots are compared out. The attention arm's null key
    # guarantees every real slot still sees a key even at pick zero, so
    # nothing else needs excluding; BDH zeroes its padded slots already.
    valid = arm_inputs["pack_mask"][..., None]
    keep = lambda x: jnp.where(valid, x, 0.0)
    assert close(
        keep(reference.apply(params, *args)), keep(fused.apply(params, *args))
    )

    weight = jax.random.normal(jax.random.key(8), (BATCH, PACK, D)) * valid
    loss = lambda module, p: (module.apply(p, *args) * weight).sum()
    flatten = lambda g: jnp.concatenate(
        [leaf.ravel() for leaf in jax.tree_util.tree_leaves(g)]
    )
    assert close(
        flatten(jax.grad(loss, argnums=1)(reference, params)),
        flatten(jax.grad(loss, argnums=1)(fused, params)),
    )


def test_bdh_arm_survives_an_empty_pool(arm_inputs):
    """Pack 0 pick 0 has nothing in the pool, on 1 row in 42 of the corpus.

    The attention arm needs a null key to keep its softmax denominator away
    from zero. BDH has no denominator, so an empty pool should simply
    produce a zero score -- but "should" is why there is a test.
    """
    arm = BDHArm(hidden_dim=D, num_heads=HEADS, num_layers=2)
    args = (
        arm_inputs["pack"],
        arm_inputs["pack_mask"],
        arm_inputs["pool"],
        arm_inputs["pool_mask"],
        arm_inputs["context"],
    )
    params = arm.init(jax.random.key(9), *args)
    out = arm.apply(params, *args)

    assert POOL_LENGTHS[2] == 0, "this test needs a row with an empty pool"
    assert jnp.isfinite(out[2]).all()

    grads = jax.grad(lambda p: arm.apply(p, *args).sum())(params)
    assert all(
        jnp.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(grads)
    )


def test_bdh_arm_ignores_pool_order(arm_inputs):
    """The pool is a set. Permuting it must change nothing at all.

    This is the property docs/ARCHITECTURE.md claims the architecture has
    *structurally*, and it is the reason the port drops the reference's
    RoPE and causal mask. If this ever fails, the port has reintroduced a
    notion of order.
    """
    arm = BDHArm(hidden_dim=D, num_heads=HEADS, num_layers=2)
    args = (
        arm_inputs["pack"],
        arm_inputs["pack_mask"],
        arm_inputs["pool"],
        arm_inputs["pool_mask"],
        arm_inputs["context"],
    )
    params = arm.init(jax.random.key(10), *args)

    order = jax.random.permutation(jax.random.key(11), POOL)
    shuffled = arm.apply(
        params,
        arm_inputs["pack"],
        arm_inputs["pack_mask"],
        arm_inputs["pool"][:, order],
        arm_inputs["pool_mask"][:, order],
        arm_inputs["context"],
    )
    assert close(arm.apply(params, *args), shuffled)


def test_measured_density_is_a_fraction(arm_inputs):
    """Density feeds `bdh_ideal_flops`, so it has to be a real fraction.

    At initialisation the encoder is symmetric noise, so roughly half the
    neurons fire and the gate -- a product of two such -- fires on roughly a
    quarter. Asserting the ordering rather than the values keeps this a
    test of the measurement rather than of the initialiser.
    """
    arm = BDHArm(hidden_dim=D, num_heads=HEADS, num_layers=2)
    args = (
        arm_inputs["pack"],
        arm_inputs["pack_mask"],
        arm_inputs["pool"],
        arm_inputs["pool_mask"],
        arm_inputs["context"],
    )
    params = arm.init(jax.random.key(12), *args)
    density = measure_density(arm, params, *args)

    assert set(density) == {"query", "gate", "score"}
    assert all(0.0 <= v <= 1.0 for v in density.values())
    # The gate is a product of two activations, so it cannot be denser than
    # either one of them.
    assert density["gate"] <= density["query"] + 1e-6
