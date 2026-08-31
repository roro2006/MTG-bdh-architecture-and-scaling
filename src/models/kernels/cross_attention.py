"""Fused pack-to-pool cross-attention as a single Pallas kernel.

Why this is not FlashAttention
------------------------------
FlashAttention exists because a sequence of thousands of tokens cannot fit
its score matrix in SRAM, so it tiles the key axis and carries a running
softmax. None of that applies here. A pack holds at most 14 cards and a
pool at most 41 (42 with the arm's null key), both fixed by the rules of
the format, so one (batch, head) slice of the score matrix is at most
14x42 floats. The whole attention -- scores, mask, softmax, and the value
matmul -- fits in registers.

That makes the right kernel *simpler* than the generic one, not harder: no
key loop, no online rescaling, no recomputation. It also makes the backward
pass tractable in one block, which matters because backward is roughly two
thirds of training compute and a kernel that only accelerates forward would
leave most of the win on the table.

What is actually saved
----------------------
XLA already fuses the elementwise parts of attention. What it cannot avoid
is a round trip to HBM for the (B, H, L_pack, L_pool) score array and again
for the softmax probabilities. This kernel never writes either one: the
scores are produced, masked, softmaxed and consumed inside a single
program. The forward pass keeps only the log-sum-exp (one float per query)
so the backward pass can reconstruct the probabilities exactly rather than
storing them.

Correctness contract
--------------------
`fused_attention` is numerically equivalent to `reference_attention`, which
is a transcription of `flax.linen.attention.dot_product_attention` -- same
scale, same `finfo.min` mask fill, same softmax. tests/test_kernels.py
asserts agreement on values and on all three gradients. This matters more
here than for the BDH kernel: the attention arm is the control that every
BDH result is measured against.
"""

from __future__ import annotations

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def default_interpret() -> bool:
    """True when Pallas has no real backend to lower to.

    Interpret mode runs the kernel's semantics in pure JAX: correct, and
    useful for testing on a machine with no GPU, but it does no fusion, so
    it is slower than just calling the reference. Never leave it on for a
    real training run.
    """
    return jax.default_backend() not in ("gpu", "tpu")


def _pow2(n: int) -> int:
    """Triton wants power-of-two block dimensions."""
    return 1 << (n - 1).bit_length()


def _mask_fill(dtype) -> float:
    """The value flax substitutes for masked logits."""
    return float(jnp.finfo(dtype).min)


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------

def reference_attention(
    q: jnp.ndarray,     # (B, H, Lq, Dh)
    k: jnp.ndarray,     # (B, H, Lk, Dh)
    v: jnp.ndarray,     # (B, H, Lk, Dh)
    mask: jnp.ndarray,  # (B, 1, Lq, Lk) or (B, H, Lq, Lk), bool
) -> jnp.ndarray:
    """Pure-JAX attention, matching flax's dot_product_attention, except on
    query rows with no visible key at all -- see `EMPTY_ROW_NOTE`."""
    scale = 1.0 / jnp.sqrt(q.shape[-1]).astype(jnp.float32)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    scores = jnp.where(mask, scores, _mask_fill(scores.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    return jnp.where(mask.any(axis=-1)[..., None], out, 0.0)


EMPTY_ROW_NOTE = """
A query row whose mask is entirely False is a padded pack slot. Flax's
softmax gives such a row a uniform distribution over every key, so its
output is the mean of all values -- an artefact of `finfo.min` being a
finite number, not a designed behaviour. It is invisible in the model
because PointerHead's `jnp.where(pack_mask, ...)` discards those slots and
blocks their gradient.

It stops being invisible in a kernel: padding the key axis to a power of
two changes how many values that mean is taken over, so the "same" model
would give different numbers at different block sizes. Both paths here
therefore define an empty row as zero, which is well-defined at any padding
and matches what the rest of the model already assumes. The divergence from
stock flax is confined to rows whose values are thrown away.
""".strip()


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

def _fwd_kernel(q_ref, k_ref, v_ref, m_ref, o_ref, lse_ref, *, scale, fill):
    q = q_ref[...].astype(jnp.float32)   # (BQ, BD)
    k = k_ref[...].astype(jnp.float32)   # (BK, BD)
    v = v_ref[...].astype(jnp.float32)   # (BK, BD)
    m = m_ref[...]                       # (BQ, BK) bool

    scores = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ()))) * scale
    scores = jnp.where(m, scores, fill)

    # Plain stable softmax. There is no running rescale because there is no
    # key loop to rescale across -- the whole key axis is already resident.
    peak = jnp.max(scores, axis=-1, keepdims=True)
    exp = jnp.exp(scores - peak)
    denom = jnp.sum(exp, axis=-1, keepdims=True)

    # An all-masked row is a padded pack slot: defined as zero, not as the
    # mean of every value. See EMPTY_ROW_NOTE.
    row_valid = jnp.any(m, axis=-1, keepdims=True)
    o_ref[...] = jnp.where(row_valid, jnp.dot(exp, v) / denom, 0.0).astype(o_ref.dtype)
    # Storing the log-sum-exp rather than the probabilities is what lets the
    # backward pass be exact without a (BQ, BK) residual in HBM.
    lse_ref[...] = (peak[:, 0] + jnp.log(denom[:, 0])).astype(lse_ref.dtype)


def _bwd_kernel(
    q_ref, k_ref, v_ref, m_ref, do_ref, lse_ref, dq_ref, dk_ref, dv_ref, *, scale, fill
):
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    m = m_ref[...]
    lse = lse_ref[...].astype(jnp.float32)[:, None]

    # Cotangent of the forward pass's empty-row zeroing: an all-masked row
    # contributed nothing, so it receives nothing.
    row_valid = jnp.any(m, axis=-1, keepdims=True)
    do = jnp.where(row_valid, do_ref[...].astype(jnp.float32), 0.0)

    scores = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ()))) * scale
    scores = jnp.where(m, scores, fill)
    probs = jnp.exp(scores - lse)
    # exp(finfo.min - lse) is already zero, but a masked slot must not be
    # able to contribute a denormal to dk/dv.
    probs = jnp.where(m, probs, 0.0)

    # dv = P^T dO; the softmax Jacobian collapses to the usual
    # P * (dP - rowsum(dP * P)) because rows of P sum to one.
    dv = jax.lax.dot_general(probs, do, (((0,), (0,)), ((), ())))
    dprobs = jax.lax.dot_general(do, v, (((1,), (1,)), ((), ())))
    rowsum = jnp.sum(dprobs * probs, axis=-1, keepdims=True)
    dscores = probs * (dprobs - rowsum)

    dq = jnp.dot(dscores, k) * scale
    dk = jax.lax.dot_general(dscores, q, (((0,), (0,)), ((), ()))) * scale

    dq_ref[...] = dq.astype(dq_ref.dtype)
    dk_ref[...] = dk.astype(dk_ref.dtype)
    dv_ref[...] = dv.astype(dv_ref.dtype)


# --------------------------------------------------------------------------
# custom_vjp wrapper
#
# Each grid program owns one (batch, head) slice of every input and every
# output, so dq/dk/dv are written disjointly and no atomics are needed.
# --------------------------------------------------------------------------

def _specs(b, h, bq, bk, bd, mask_heads):
    q_spec = pl.BlockSpec((None, None, bq, bd), lambda i, j: (i, j, 0, 0))
    k_spec = pl.BlockSpec((None, None, bk, bd), lambda i, j: (i, j, 0, 0))
    # A (B, 1, Lq, Lk) mask is shared across heads, so its head index is
    # pinned to 0 rather than following the grid.
    m_spec = pl.BlockSpec(
        (None, None, bq, bk),
        (lambda i, j: (i, j, 0, 0)) if mask_heads else (lambda i, j: (i, 0, 0, 0)),
    )
    lse_spec = pl.BlockSpec((None, None, bq), lambda i, j: (i, j, 0))
    return q_spec, k_spec, m_spec, lse_spec


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _attention(q, k, v, mask, interpret):
    return _attention_fwd(q, k, v, mask, interpret)[0]


def _attention_fwd(q, k, v, mask, interpret):
    b, h, lq, dh = q.shape
    lk = k.shape[2]
    scale = 1.0 / float(jnp.sqrt(dh))
    fill = _mask_fill(q.dtype)
    q_spec, k_spec, m_spec, lse_spec = _specs(
        b, h, lq, lk, dh, mask_heads=mask.shape[1] != 1
    )

    out, lse = pl.pallas_call(
        functools.partial(_fwd_kernel, scale=scale, fill=fill),
        grid=(b, h),
        in_specs=[q_spec, k_spec, k_spec, m_spec],
        out_specs=[q_spec, lse_spec],
        out_shape=[
            jax.ShapeDtypeStruct((b, h, lq, dh), q.dtype),
            jax.ShapeDtypeStruct((b, h, lq), jnp.float32),
        ],
        interpret=interpret,
    )(q, k, v, mask)
    return out, (q, k, v, mask, lse)


def _attention_bwd(interpret, res, do):
    q, k, v, mask, lse = res
    b, h, lq, dh = q.shape
    lk = k.shape[2]
    scale = 1.0 / float(jnp.sqrt(dh))
    fill = _mask_fill(q.dtype)
    q_spec, k_spec, m_spec, lse_spec = _specs(
        b, h, lq, lk, dh, mask_heads=mask.shape[1] != 1
    )

    dq, dk, dv = pl.pallas_call(
        functools.partial(_bwd_kernel, scale=scale, fill=fill),
        grid=(b, h),
        in_specs=[q_spec, k_spec, k_spec, m_spec, q_spec, lse_spec],
        out_specs=[q_spec, k_spec, k_spec],
        out_shape=[
            jax.ShapeDtypeStruct((b, h, lq, dh), q.dtype),
            jax.ShapeDtypeStruct((b, h, lk, dh), k.dtype),
            jax.ShapeDtypeStruct((b, h, lk, dh), v.dtype),
        ],
        interpret=interpret,
    )(q, k, v, mask, do, lse)
    # `mask` is a bool input and carries no cotangent.
    return dq, dk, dv, None


_attention.defvjp(_attention_fwd, _attention_bwd)


def fused_attention(
    q: jnp.ndarray,     # (B, H, Lq, Dh)
    k: jnp.ndarray,     # (B, H, Lk, Dh)
    v: jnp.ndarray,     # (B, H, Lk, Dh)
    mask: jnp.ndarray,  # (B, 1, Lq, Lk) or (B, H, Lq, Lk), bool
    interpret: bool | None = None,
) -> jnp.ndarray:
    """Fused cross-attention. Equivalent to `reference_attention`.

    Lengths are padded up to powers of two here rather than inside the
    kernel, because Triton block dimensions must be powers of two and the
    arrays involved are small enough that the padding is cheap. Padded key
    columns are masked off; padded query rows are computed and discarded.
    """
    if interpret is None:
        interpret = default_interpret()

    b, h, lq, dh = q.shape
    lk = k.shape[2]
    bq, bk, bd = _pow2(lq), _pow2(lk), _pow2(dh)

    def pad_len(x, axis, target):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (0, target - x.shape[axis])
        return jnp.pad(x, pad)

    qp = pad_len(pad_len(q, 2, bq), 3, bd)
    kp = pad_len(pad_len(k, 2, bk), 3, bd)
    vp = pad_len(pad_len(v, 2, bk), 3, bd)
    # False in the padded region, so padded keys can never be attended to.
    mp = pad_len(pad_len(mask, 2, bq), 3, bk)

    out = _attention(qp, kp, vp, mp, interpret)
    return out[:, :, :lq, :dh]


# --------------------------------------------------------------------------
# Flax block
# --------------------------------------------------------------------------

class _FusedMHA(nn.Module):
    """Drop-in for `nn.MultiHeadDotProductAttention`, kernel in the middle.

    Submodule names (`query`, `key`, `value`, `out`) and their shapes are
    flax's, so this nests under the same path and produces the same tree.
    """

    hidden_dim: int
    num_heads: int
    interpret: bool | None = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        ctx: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        d, nh = self.hidden_dim, self.num_heads
        head_dim = d // nh
        proj = functools.partial(
            nn.DenseGeneral, features=(nh, head_dim), axis=-1, use_bias=True
        )
        q = proj(name="query")(x)
        k = proj(name="key")(ctx)
        v = proj(name="value")(ctx)

        to_heads = lambda t: jnp.transpose(t, (0, 2, 1, 3))  # (B,L,H,Dh)->(B,H,L,Dh)
        out = fused_attention(
            to_heads(q), to_heads(k), to_heads(v), mask, interpret=self.interpret
        )
        out = jnp.transpose(out, (0, 2, 1, 3))
        return nn.DenseGeneral(features=d, axis=(-2, -1), use_bias=True, name="out")(out)


class FusedCrossAttentionBlock(nn.Module):
    """`CrossAttentionBlock` with the attention core replaced by the kernel.

    The projections stay in XLA: they are plain GEMMs, which XLA already
    emits well, and fusing them in would buy nothing. Only the part that
    materialises an (Lq, Lk) intermediate moves into Pallas.

    Parameter tree is identical to `CrossAttentionBlock`'s, name for name
    and shape for shape, so the same parameters can be applied to both --
    which is what makes the equivalence test in tests/test_kernels.py a
    real test rather than a coincidence of initialisation.
    """

    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4
    interpret: bool | None = None

    @nn.compact
    def __call__(
        self,
        queries: jnp.ndarray,
        context: jnp.ndarray,
        query_mask: jnp.ndarray,
        context_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        d, nh = self.hidden_dim, self.num_heads
        if d % nh:
            raise ValueError(f"hidden_dim {d} must be divisible by num_heads {nh}")

        mask = nn.make_attention_mask(query_mask, context_mask).astype(bool)

        x = nn.LayerNorm(name="norm_attn")(queries)
        ctx = nn.LayerNorm(name="norm_context")(context)

        h = _FusedMHA(
            hidden_dim=d,
            num_heads=nh,
            interpret=self.interpret,
            name="cross_attn",
        )(x, ctx, mask)
        queries = queries + h

        h = nn.LayerNorm(name="norm_mlp")(queries)
        h = nn.Dense(d * self.mlp_ratio, name="mlp_in")(h)
        h = nn.gelu(h)
        h = nn.Dense(d, name="mlp_out")(h)
        return queries + h
