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
from dataclasses import replace

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
from src.models.kernels.set_encoder import (
    FusedSetAttentionBlock,
    fused_set_attention,
    reference_set_attention,
)
from src.models.pick_model import ModelConfig, PickModel
from src.models.set_encoder import SetAttentionBlock, SetEncoder

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
# Set-encoder kernel
#
# The set encoders are 63-74% of forward FLOPs (docs/PROJECT_PLAN.md section
# 5), so this kernel is on the majority of the arithmetic in every run. It
# is also on both arms' inputs, which means a wrong one would corrupt the
# BDH side and the control side identically -- a bias no A/B comparison
# could detect.
# --------------------------------------------------------------------------

class TestFusedSetAttention:
    @pytest.fixture(params=["pack", "pool"])
    def qkv_valid(self, request, masks):
        """Both sets, because they exercise different things.

        The pack is 14 long, padded to 16, and is never empty. The pool is
        41, padded to 64 -- a 56% pad, the worst case in the model -- and is
        empty at pick zero, which is the row that makes the softmax
        denominator a question.
        """
        pack_mask, pool_mask = masks
        length, valid = (
            (PACK, pack_mask) if request.param == "pack" else (POOL, pool_mask)
        )
        keys = jax.random.split(jax.random.key(20), 3)
        shape = (BATCH, HEADS, length, D // HEADS)
        return (
            tuple(jax.random.normal(keys[i], shape) for i in range(3)),
            valid,
        )

    def test_forward_matches_reference(self, qkv_valid):
        (q, k, v), valid = qkv_valid
        assert close(
            reference_set_attention(q, k, v, valid),
            fused_set_attention(q, k, v, valid, **KERNEL_KW),
        )

    def test_gradients_match_reference(self, qkv_valid):
        (q, k, v), valid = qkv_valid
        weight = jax.random.normal(jax.random.key(21), q.shape)
        ref = lambda *a: (reference_set_attention(*a, valid) * weight).sum()
        fused = lambda *a: (fused_set_attention(*a, valid, **KERNEL_KW) * weight).sum()

        for expected, actual in zip(
            grad_of(ref, (0, 1, 2))(q, k, v), grad_of(fused, (0, 1, 2))(q, k, v)
        ):
            assert close(expected, actual)

    def test_padded_rows_are_zero(self, qkv_valid):
        """A PAD_ID slot attends to nothing and is defined as zero.

        The mask is symmetric here, so this covers the empty pool at pick
        zero as the same case rather than a special one.
        """
        (q, k, v), valid = qkv_valid
        out = fused_set_attention(q, k, v, valid, **KERNEL_KW)
        padded = jnp.broadcast_to(~valid[:, None, :], out.shape[:3])
        assert jnp.all(out[padded] == 0.0)
        assert jnp.isfinite(out).all()

    def test_padding_cannot_reach_a_real_row(self, qkv_valid):
        """Changing a padded card must change no real card's output.

        This is the failure the ragged masking exists to prevent, and it is
        the one that would be invisible: the shapes stay right, the loss
        stays plausible, and the model quietly conditions on whatever
        garbage sits in the padding region of the batch.
        """
        (q, k, v), valid = qkv_valid
        pad = ~valid[:, None, :, None]
        noise = jax.random.normal(jax.random.key(22), q.shape) * 10.0
        perturb = lambda t: jnp.where(pad, t + noise, t)

        base = fused_set_attention(q, k, v, valid, **KERNEL_KW)
        moved = fused_set_attention(
            perturb(q), perturb(k), perturb(v), valid, **KERNEL_KW
        )
        real = valid[:, None, :, None]
        assert close(jnp.where(real, base, 0.0), jnp.where(real, moved, 0.0))

    def test_padded_slots_receive_no_gradient(self, qkv_valid):
        """The backward half of the same guarantee.

        A padded slot that collected a gradient would train the shared
        per-card projections on the padding value, which is a leak that
        only shows up as slightly wrong weights.
        """
        (q, k, v), valid = qkv_valid
        weight = jax.random.normal(jax.random.key(23), q.shape)
        loss = lambda *a: (fused_set_attention(*a, valid, **KERNEL_KW) * weight).sum()
        pad = ~valid[:, None, :, None]

        for grad in grad_of(loss, (0, 1, 2))(q, k, v):
            assert jnp.all(jnp.where(pad, grad, 0.0) == 0.0)

    def test_block_matches_reference_block(self, masks):
        """One parameter set, two execution paths.

        Compared on real rows only: padded rows are where the two paths
        deliberately disagree (EMPTY_ROW_NOTE), and `SetEncoder` zeroes them
        before anything downstream can see them.
        """
        pack_mask, _ = masks
        keys = jax.random.split(jax.random.key(24), 2)
        x = jax.random.normal(keys[0], (BATCH, PACK, D))

        ref = SetAttentionBlock(hidden_dim=D, num_heads=HEADS)
        fused = FusedSetAttentionBlock(hidden_dim=D, num_heads=HEADS, **KERNEL_KW)

        params = ref.init(keys[1], x, pack_mask)
        assert param_paths(params["params"]) == param_paths(
            fused.init(keys[1], x, pack_mask)["params"]
        )

        keep = lambda t: jnp.where(pack_mask[..., None], t, 0.0)
        assert close(
            keep(ref.apply(params, x, pack_mask)),
            keep(fused.apply(params, x, pack_mask)),
        )


@pytest.mark.parametrize("length,lengths", [(PACK, PACK_LENGTHS), (POOL, POOL_LENGTHS)])
def test_fused_set_encoder_matches_reference_exactly(length, lengths):
    """Whole encoder, no rows excluded.

    The block-level test has to compare on real rows only. This one does
    not, and that is the point: `SetEncoder` multiplies by the mask before
    returning, so the deliberate divergence on padded rows is erased and
    exact agreement is the right bar. If the two paths ever differ *here*,
    something reached a real card.
    """
    mask = jnp.arange(length)[None, :] < lengths[:, None]
    x = jax.random.normal(jax.random.key(25), (BATCH, length, D))
    build = lambda **extra: SetEncoder(
        hidden_dim=D, num_heads=HEADS, num_layers=2, **extra
    )
    ref, fused = build(), build(fused=True)

    params = ref.init(jax.random.key(26), x, mask)
    assert param_paths(params["params"]) == param_paths(
        fused.init(jax.random.key(26), x, mask)["params"]
    )

    for expected, actual in zip(
        ref.apply(params, x, mask), fused.apply(params, x, mask)
    ):
        assert close(expected, actual)

    weight = jax.random.normal(jax.random.key(27), (BATCH, length, D))
    loss = lambda module, p: (module.apply(p, x, mask)[0] * weight).sum()
    flatten = lambda g: jnp.concatenate(
        [leaf.ravel() for leaf in jax.tree_util.tree_leaves(g)]
    )
    assert close(
        flatten(jax.grad(loss, argnums=1)(ref, params)),
        flatten(jax.grad(loss, argnums=1)(fused, params)),
    )


def test_fused_set_encoder_is_permutation_equivariant():
    """Reordering the set permutes the outputs and changes nothing else.

    docs/PROJECT_PLAN.md section 4 makes order *structurally* unusable, not
    merely ignored. A kernel is the easiest place to break that by accident
    -- a block index leaking into the arithmetic would be a positional
    encoding nobody chose. This is the assertion that keeps it closed.
    """
    mask = jnp.arange(POOL)[None, :] < POOL_LENGTHS[:, None]
    x = jax.random.normal(jax.random.key(28), (BATCH, POOL, D))
    encoder = SetEncoder(hidden_dim=D, num_heads=HEADS, num_layers=2, fused=True)
    params = encoder.init(jax.random.key(29), x, mask)

    order = jax.random.permutation(jax.random.key(30), POOL)
    per_card, pooled = encoder.apply(params, x, mask)
    shuffled_cards, shuffled_pooled = encoder.apply(
        params, x[:, order], mask[:, order]
    )
    assert close(per_card[:, order], shuffled_cards)
    assert close(pooled, shuffled_pooled)


def test_fused_set_encoder_works_under_jit():
    """Training runs jitted, so eager agreement is not enough on its own.

    Asserting `pallas_call` actually reaches the jaxpr matters as much as
    the values: a silent fallback to the reference block would make every
    other test in this section pass for the wrong reason.
    """
    mask = jnp.arange(POOL)[None, :] < POOL_LENGTHS[:, None]
    x = jax.random.normal(jax.random.key(31), (BATCH, POOL, D))
    build = lambda **extra: SetEncoder(
        hidden_dim=D, num_heads=HEADS, num_layers=2, **extra
    )
    ref, fused = build(), build(fused=True)
    params = ref.init(jax.random.key(32), x, mask)

    assert "pallas_call" in str(
        jax.make_jaxpr(lambda p: fused.apply(p, x, mask)[0])(params)
    )
    forward = jax.jit(lambda module, p: module.apply(p, x, mask)[0], static_argnums=0)
    assert close(forward(ref, params), forward(fused, params))

    weight = jax.random.normal(jax.random.key(33), (BATCH, POOL, D))
    backward = jax.jit(
        jax.grad(lambda p, module: (module.apply(p, x, mask)[0] * weight).sum()),
        static_argnums=1,
    )
    flatten = lambda g: jnp.concatenate(
        [leaf.ravel() for leaf in jax.tree_util.tree_leaves(g)]
    )
    assert close(
        flatten(backward(params, ref)), flatten(backward(params, fused))
    )


def test_fused_kernels_flag_routes_the_encoders_too():
    """`ModelConfig(fused_kernels=True)` is one switch for the whole model.

    Before this kernel the flag only reached the arm, which is 26-37% of
    forward FLOPs. A flag that silently left the majority of the model on
    flax would make any wall-clock claim about "the fused model" false.
    """
    feature_table = jax.random.normal(jax.random.key(34), (30, 65))
    pack_ids = jnp.array([[0, 1, 2, 3], [4, 5, -1, -1], [6, -1, -1, -1]])
    pool_ids = jnp.array([[7, 8, -1], [9, -1, -1], [-1, -1, -1]])
    scalar = jnp.zeros((3,), dtype=jnp.int32)
    args = (feature_table, pack_ids, pool_ids, scalar, scalar)

    base = ModelConfig(hidden_dim=D, num_heads=HEADS, card_feature_dim=65)
    reference = PickModel(config=base, arm="attention")
    fused = PickModel(config=replace(base, fused_kernels=True), arm="attention")

    params = reference.init(jax.random.key(35), *args)
    assert param_paths(params["params"]) == param_paths(
        fused.init(jax.random.key(35), *args)["params"]
    )
    assert "pallas_call" in str(
        jax.make_jaxpr(lambda p: fused.apply(p, *args))(params)
    )
    assert close(reference.apply(params, *args), fused.apply(params, *args))


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


@pytest.mark.parametrize(
    "arm_cls,kwargs",
    [(CrossAttentionArm, {}), (BDHArm, {"neuron_multiplier": 4})],
    ids=["attention", "bdh"],
)
def test_fused_arm_works_under_jit(arm_inputs, arm_cls, kwargs):
    """Training runs jitted, so eager agreement is not enough on its own.

    This exists because it caught a real bug: the head-dimension scale was
    computed as `float(jnp.sqrt(dh))`, which is fine eagerly -- the sqrt
    evaluates to a concrete array -- but under a trace it stages out to a
    tracer, and `float()` on a tracer raises ConcretizationTypeError. Every
    other test in this file ran eagerly and passed while the kernel was
    unusable in the only mode that matters.

    Asserting `pallas_call` actually appears in the jaxpr matters just as
    much: without it a silent fallback to the reference path would make
    every other test here pass for the wrong reason.
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
    params = reference.init(jax.random.key(13), *args)

    # Padded pack slots are compared out for the same reason as in
    # test_fused_arm_matches_reference_arm: the two paths deliberately
    # disagree there, and PointerHead discards those slots.
    valid = arm_inputs["pack_mask"][..., None]
    forward = jax.jit(
        lambda module, p: jnp.where(valid, module.apply(p, *args), 0.0),
        static_argnums=0,
    )
    assert "pallas_call" in str(jax.make_jaxpr(lambda p: fused.apply(p, *args))(params))
    assert close(forward(reference, params), forward(fused, params))

    weight = jax.random.normal(jax.random.key(14), (BATCH, PACK, D)) * valid
    backward = jax.jit(
        jax.grad(lambda p, module: (module.apply(p, *args) * weight).sum()),
        static_argnums=1,
    )
    flatten = lambda g: jnp.concatenate(
        [leaf.ravel() for leaf in jax.tree_util.tree_leaves(g)]
    )
    assert close(
        flatten(backward(params, reference)), flatten(backward(params, fused))
    )


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
