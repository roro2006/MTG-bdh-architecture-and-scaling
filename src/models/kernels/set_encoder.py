"""Fused masked self-attention for the pack and pool set encoders.

Why this is the largest remaining kernel win
--------------------------------------------
`docs/PROJECT_PLAN.md` section 5 puts the set encoders at 58% of parameters
and 63-74% of forward FLOPs, all of it still running on
`nn.MultiHeadDotProductAttention`. The project's stated commitment is that
attention-shaped operations are hand-written; until this kernel existed the
*majority* of the arithmetic was flax's, and the commitment described the
arm rather than the model.

Why a second kernel rather than a call into `cross_attention`
------------------------------------------------------------
The cross-attention kernel takes a materialised `(B, 1, Lq, Lk)` boolean
mask, because a pack query and a pool key have independent validity. Set
attention does not: both axes are the *same* set, so the mask is the outer
product of one `(B, L)` validity vector with itself and carries no
information the vector does not.

Passing the matrix anyway costs twice. `nn.make_attention_mask` builds an
`L x L` bool array in HBM before the kernel starts, and the kernel then
reads it back -- 4,096 bytes per batch element at the pool's padded length
against 64 for the vector, on an operation whose entire justification is
that it wins memory traffic rather than FLOPs. So this kernel takes the
vector and reconstructs the mask in registers, and the `L x L` array is
never built at all, in either the forward or the backward pass.

Everything else is deliberately the cross-attention kernel's structure:
one `(batch, head)` slice per grid program, no key loop, no online
rescaling, log-sum-exp as the only forward residual. A pack holds at most
14 cards and a pool at most 41, so the whole score matrix fits in
registers and FlashAttention's tiling would buy nothing.

No positional encoding
----------------------
There is none here and none must be added. `docs/PROJECT_PLAN.md` section 4
makes order structurally unusable rather than merely ignored, and a set
encoder that learned position would be a silent violation of the property
`tests/test_models.py` asserts about the whole model. The kernel reads only
`q`, `k`, `v` and the validity vector; there is nowhere for an index to
enter. `test_fused_encoder_is_permutation_equivariant` holds that closed.

Correctness contract
--------------------
`fused_set_attention` is numerically equivalent to
`reference_set_attention`, which is `flax.linen.attention` transcribed with
the outer-product mask -- same scale, same `finfo.min` fill, same softmax --
except on fully-padded rows, where both paths here return zero. See
`cross_attention.EMPTY_ROW_NOTE`; the argument is the same one, and it is
if anything stronger here because `SetEncoder` already zeroes padded rows
before returning them.

`tests/test_kernels.py` asserts values and all three gradients against the
reference, asserts that padding cannot reach a real row's output *or* its
gradient, and asserts the whole `SetEncoder` agrees exactly -- not merely on
comparable rows -- because that final zeroing makes exact agreement the
right bar.
"""

from __future__ import annotations

import functools
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Shared rather than redefined: `_mask_fill` is the flax constant the
# equivalence claim rests on, and two copies of it could drift apart
# without any test noticing which one was wrong.
from .cross_attention import _mask_fill, _pow2, default_interpret


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------

def reference_set_attention(
    q: jnp.ndarray,      # (B, H, L, Dh)
    k: jnp.ndarray,      # (B, H, L, Dh)
    v: jnp.ndarray,      # (B, H, L, Dh)
    valid: jnp.ndarray,  # (B, L) bool -- real card, not PAD_ID
) -> jnp.ndarray:
    """Pure-JAX masked set self-attention, matching flax's
    `dot_product_attention` under `nn.make_attention_mask(valid, valid)`,
    except on rows of an all-padding set -- see `cross_attention`'s
    `EMPTY_ROW_NOTE`."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    keep = valid[:, None, None, :]  # (B, 1, 1, L): which keys are real
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    scores = jnp.where(keep, scores, _mask_fill(scores.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    # The mask is symmetric, so a padded query row and a row of an
    # all-padding set collapse to one case: `valid[q]` is False in both, and
    # a row that is valid necessarily has at least itself as a visible key.
    # That is why the empty pool at pick zero needs no separate handling.
    return jnp.where(valid[:, None, :, None], out, 0.0)


# --------------------------------------------------------------------------
# Kernels
#
# The mask is rebuilt from the validity vector inside each program. That is
# an `L x L` bool in registers, which costs nothing, against an `L x L` bool
# in HBM read once per program, which is the traffic this kernel exists to
# remove.
# --------------------------------------------------------------------------

def _fwd_kernel(q_ref, k_ref, v_ref, valid_ref, o_ref, lse_ref, *, scale, fill):
    q = q_ref[...].astype(jnp.float32)   # (BL, BD)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    valid = valid_ref[...].astype(jnp.bool_)  # (BL,)

    row_valid = valid[:, None]
    col_valid = valid[None, :]

    scores = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ()))) * scale
    scores = jnp.where(col_valid, scores, fill)

    peak = jnp.max(scores, axis=-1, keepdims=True)
    exp = jnp.exp(scores - peak)
    denom = jnp.sum(exp, axis=-1, keepdims=True)

    # Padded rows are zero, not the mean of every value. Flax's value for
    # them depends on how far the key axis happened to be padded to reach a
    # power of two, so the same model would answer differently at different
    # block sizes. A *valid* row always sees at least itself, so `denom` is
    # never near zero on any row whose output survives.
    o_ref[...] = jnp.where(row_valid, jnp.dot(exp, v) / denom, 0.0).astype(o_ref.dtype)
    lse_ref[...] = (peak[:, 0] + jnp.log(denom[:, 0])).astype(lse_ref.dtype)


def _bwd_kernel(
    q_ref, k_ref, v_ref, valid_ref, do_ref, lse_ref, dq_ref, dk_ref, dv_ref,
    *, scale, fill,
):
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    valid = valid_ref[...].astype(jnp.bool_)
    lse = lse_ref[...].astype(jnp.float32)[:, None]

    row_valid = valid[:, None]
    col_valid = valid[None, :]

    # Cotangent of the forward pass's zeroing: a row that contributed
    # nothing receives nothing. Without this a padded card would collect a
    # gradient and train the shared per-card projections on noise.
    do = jnp.where(row_valid, do_ref[...].astype(jnp.float32), 0.0)

    scores = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ()))) * scale
    scores = jnp.where(col_valid, scores, fill)
    probs = jnp.exp(scores - lse)
    # `exp(finfo.min - lse)` is already zero, but this forbids a padded
    # column contributing even a denormal to dk/dv -- the exact leak the
    # ragged-set masking exists to prevent.
    probs = jnp.where(col_valid & row_valid, probs, 0.0)

    dv = jax.lax.dot_general(probs, do, (((0,), (0,)), ((), ())))
    dprobs = jax.lax.dot_general(do, v, (((1,), (1,)), ((), ())))
    rowsum = jnp.sum(dprobs * probs, axis=-1, keepdims=True)
    dscores = probs * (dprobs - rowsum)

    dq_ref[...] = (jnp.dot(dscores, k) * scale).astype(dq_ref.dtype)
    dk_ref[...] = (
        jax.lax.dot_general(dscores, q, (((0,), (0,)), ((), ()))) * scale
    ).astype(dk_ref.dtype)
    dv_ref[...] = dv.astype(dv_ref.dtype)


# --------------------------------------------------------------------------
# custom_vjp wrapper
#
# One (batch, head) slice per program, so dq/dk/dv are written disjointly
# and no atomics are needed. q, k and v are separate cotangents even though
# they come from the same `x`: the projections are outside the kernel, so
# autodiff sums the three contributions there.
# --------------------------------------------------------------------------

def _specs(bl, bd):
    qkv_spec = pl.BlockSpec((None, None, bl, bd), lambda i, j: (i, j, 0, 0))
    # The validity vector is shared across heads, so its index map ignores
    # the head axis of the grid entirely.
    valid_spec = pl.BlockSpec((None, bl), lambda i, j: (i, 0))
    lse_spec = pl.BlockSpec((None, None, bl), lambda i, j: (i, j, 0))
    return qkv_spec, valid_spec, lse_spec


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _set_attention(q, k, v, valid, interpret):
    return _set_attention_fwd(q, k, v, valid, interpret)[0]


def _set_attention_fwd(q, k, v, valid, interpret):
    b, h, l, dh = q.shape
    scale = 1.0 / math.sqrt(dh)
    fill = _mask_fill(q.dtype)
    qkv_spec, valid_spec, lse_spec = _specs(l, dh)

    out, lse = pl.pallas_call(
        functools.partial(_fwd_kernel, scale=scale, fill=fill),
        grid=(b, h),
        in_specs=[qkv_spec, qkv_spec, qkv_spec, valid_spec],
        out_specs=[qkv_spec, lse_spec],
        out_shape=[
            jax.ShapeDtypeStruct((b, h, l, dh), q.dtype),
            jax.ShapeDtypeStruct((b, h, l), jnp.float32),
        ],
        interpret=interpret,
    )(q, k, v, valid)
    return out, (q, k, v, valid, lse)


def _set_attention_bwd(interpret, res, do):
    q, k, v, valid, lse = res
    b, h, l, dh = q.shape
    scale = 1.0 / math.sqrt(dh)
    fill = _mask_fill(q.dtype)
    qkv_spec, valid_spec, lse_spec = _specs(l, dh)

    dq, dk, dv = pl.pallas_call(
        functools.partial(_bwd_kernel, scale=scale, fill=fill),
        grid=(b, h),
        in_specs=[qkv_spec, qkv_spec, qkv_spec, valid_spec, qkv_spec, lse_spec],
        out_specs=[qkv_spec, qkv_spec, qkv_spec],
        out_shape=[jax.ShapeDtypeStruct((b, h, l, dh), t.dtype) for t in (q, k, v)],
        interpret=interpret,
    )(q, k, v, valid, do, lse)
    # `valid` is bool and carries no cotangent.
    return dq, dk, dv, None


_set_attention.defvjp(_set_attention_fwd, _set_attention_bwd)


def fused_set_attention(
    q: jnp.ndarray,      # (B, H, L, Dh)
    k: jnp.ndarray,      # (B, H, L, Dh)
    v: jnp.ndarray,      # (B, H, L, Dh)
    valid: jnp.ndarray,  # (B, L) bool
    interpret: bool | None = None,
) -> jnp.ndarray:
    """Fused set self-attention. Equivalent to `reference_set_attention`.

    Padding to a power of two happens here rather than in the kernel,
    because Triton block dimensions must be powers of two and 14 and 41 are
    not. The padding is a single `False` region appended to `valid`, which
    is what makes it safe: a padded slot is indistinguishable from a
    PAD_ID card, so it can neither be attended to nor receive a gradient,
    and the answer does not depend on how far the axis was padded.
    """
    if interpret is None:
        interpret = default_interpret()

    b, h, l, dh = q.shape
    bl, bd = _pow2(l), _pow2(dh)

    def pad_len(x, axis, target):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (0, target - x.shape[axis])
        return jnp.pad(x, pad)

    project = lambda t: pad_len(pad_len(t, 2, bl), 3, bd)
    out = _set_attention(
        project(q), project(k), project(v), pad_len(valid, 1, bl), interpret
    )
    return out[:, :, :l, :dh]


# --------------------------------------------------------------------------
# Flax blocks
# --------------------------------------------------------------------------

class _FusedSelfMHA(nn.Module):
    """Drop-in for `nn.MultiHeadDotProductAttention` in the self-attention
    case, kernel in the middle.

    Submodule names (`query`, `key`, `value`, `out`) and their shapes are
    flax's, so this nests at the same path and produces the same parameter
    tree -- which is what lets one checkpoint run down either path and makes
    the equivalence test a test rather than a coincidence.
    """

    hidden_dim: int
    num_heads: int
    interpret: bool | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
        d, nh = self.hidden_dim, self.num_heads
        head_dim = d // nh
        proj = functools.partial(
            nn.DenseGeneral, features=(nh, head_dim), axis=-1, use_bias=True
        )
        to_heads = lambda t: jnp.transpose(t, (0, 2, 1, 3))  # (B,L,H,Dh)->(B,H,L,Dh)

        out = fused_set_attention(
            to_heads(proj(name="query")(x)),
            to_heads(proj(name="key")(x)),
            to_heads(proj(name="value")(x)),
            valid,
            interpret=self.interpret,
        )
        out = jnp.transpose(out, (0, 2, 1, 3))
        return nn.DenseGeneral(features=d, axis=(-2, -1), use_bias=True, name="out")(out)


class FusedSetAttentionBlock(nn.Module):
    """`SetAttentionBlock` with the attention core replaced by the kernel.

    The projections, norms and MLP stay in XLA: they are plain GEMMs and
    elementwise ops that XLA already emits well. Only the part that would
    materialise an `(L, L)` intermediate moves into Pallas.

    Parameter tree is identical to `SetAttentionBlock`'s, name for name and
    shape for shape.
    """

    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4
    interpret: bool | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        h = nn.LayerNorm(name="norm_attn")(x)
        h = _FusedSelfMHA(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            interpret=self.interpret,
            name="attn",
        )(h, mask.astype(bool))
        x = x + h

        h = nn.LayerNorm(name="norm_mlp")(x)
        h = nn.Dense(self.hidden_dim * self.mlp_ratio, name="mlp_in")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim, name="mlp_out")(h)
        return x + h


# --------------------------------------------------------------------------
# Benchmark harness
#
# UNRUN. `default_interpret()` is True on every machine this project has
# been developed on, and interpret mode executes the kernel's semantics in
# pure JAX with no fusion whatsoever -- it is reliably *slower* than the
# reference and says nothing at all about the kernel's speed. This function
# exists so that the first GPU session has nothing left to write; any
# number it prints off GPU should be discarded, and it refuses to pretend
# otherwise.
# --------------------------------------------------------------------------

def benchmark_set_attention(
    widths: tuple[int, ...] = (64, 128, 256, 512),
    num_heads: int = 4,
    batch: int = 256,
    length: int = 41,
    repeats: int = 20,
    interpret: bool | None = None,
) -> list[dict[str, float]]:
    """Times reference against fused set attention, forward and backward.

    Backward is timed as well as forward because backward is roughly two
    thirds of training compute, so a forward-only speedup would be a
    misleading headline. Lengths default to the pool's 41 rather than the
    pack's 14: the pool encoder is two layers to the pack's one and the
    longer axis is where the `L x L` traffic actually is.
    """
    import time

    if interpret is None:
        interpret = default_interpret()

    rows = []
    for d in widths:
        head_dim = d // num_heads
        keys = jax.random.split(jax.random.key(0), 4)
        shape = (batch, num_heads, length, head_dim)
        q, k, v = (jax.random.normal(keys[i], shape) for i in range(3))
        # A realistic ragged mix rather than full packs: half-full pools are
        # the common case, and a kernel that is only fast on dense input
        # would be fast on input this model never sees.
        lengths = jax.random.randint(keys[3], (batch,), 0, length + 1)
        valid = jnp.arange(length)[None, :] < lengths[:, None]
        cotangent = jnp.ones(shape)

        def timed(fn, *args):
            jax.block_until_ready(fn(*args))
            start = time.perf_counter()
            for _ in range(repeats):
                jax.block_until_ready(fn(*args))
            return 1e3 * (time.perf_counter() - start) / repeats

        ref_f = jax.jit(lambda a, b, c: reference_set_attention(a, b, c, valid))
        fus_f = jax.jit(
            lambda a, b, c: fused_set_attention(a, b, c, valid, interpret=interpret)
        )
        ref_b = jax.jit(jax.grad(lambda a, b, c: (ref_f(a, b, c) * cotangent).sum(),
                                 argnums=(0, 1, 2)))
        fus_b = jax.jit(jax.grad(lambda a, b, c: (fus_f(a, b, c) * cotangent).sum(),
                                 argnums=(0, 1, 2)))

        rows.append(
            {
                "hidden_dim": d,
                "reference_fwd_ms": timed(ref_f, q, k, v),
                "fused_fwd_ms": timed(fus_f, q, k, v),
                "reference_bwd_ms": timed(ref_b, q, k, v),
                "fused_bwd_ms": timed(fus_b, q, k, v),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Time the set-encoder kernel against its pure-JAX reference."
    )
    parser.add_argument(
        "--widths", type=int, nargs="+", default=[64, 128, 256, 512],
        help="hidden dims to sweep; the grid's ladder by default",
    )
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--length", type=int, default=41, help="41 = pool, 14 = pack")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)

    interpret = default_interpret()
    if interpret:
        print(
            "REFUSING TO REPORT TIMINGS: no GPU or TPU backend, so Pallas is in\n"
            "interpret mode, which runs the kernel's semantics in pure JAX with no\n"
            "fusion. Any number printed here would measure the interpreter, not the\n"
            f"kernel. Backend is {jax.default_backend()!r}.",
        )
        return 1

    rows = benchmark_set_attention(
        widths=tuple(args.widths),
        num_heads=args.num_heads,
        batch=args.batch,
        length=args.length,
        repeats=args.repeats,
        interpret=False,
    )
    header = f"{'d':>6} {'ref fwd':>10} {'fused fwd':>10} {'ref bwd':>10} {'fused bwd':>10} {'speedup':>9}"
    print(f"batch={args.batch} L={args.length} heads={args.num_heads}")
    print(header)
    for row in rows:
        total_ref = row["reference_fwd_ms"] + row["reference_bwd_ms"]
        total_fused = row["fused_fwd_ms"] + row["fused_bwd_ms"]
        print(
            f"{row['hidden_dim']:>6} {row['reference_fwd_ms']:>10.3f} "
            f"{row['fused_fwd_ms']:>10.3f} {row['reference_bwd_ms']:>10.3f} "
            f"{row['fused_bwd_ms']:>10.3f} {total_ref / total_fused:>8.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
