"""Fused Pallas kernels for the BDH arm's neuron space.

The problem this solves
-----------------------
BDH's cost is not in its output. It is in the four `(B, nh, L, N)` tensors
it builds on the way -- `q_sparse`, `k_sparse`, `y_sparse` and their
product -- where `N` is the per-head neuron width. None of them is needed
outside the block. Under XLA every one of them is written to HBM and read
back, and at the reference's own sizing (`mlp_internal_dim_multiplier=128`,
so `N = 32*D`) a single one of them is larger than the entire rest of the
model's activations put together.

So the neuron axis is what has to stay off HBM, and that is exactly what
these kernels do: `N` is a *sequential grid dimension*, tiled and
accumulated into an output block that never leaves SRAM. What crosses HBM
is only what the next stage genuinely needs -- an `(Lq, Lk)` score matrix,
or an `(Lq, D)` decoded vector.

Why two kernels rather than one
-------------------------------
The block is not a single chain: between the scores and the gate sits a
mask, a value matmul, and a LayerNorm, all of them over small `(Lq, Lk)`
or `(Lq, D)` tensors that XLA handles perfectly well and differentiates
correctly. Fusing them in would mean hand-deriving a LayerNorm backward
inside a kernel to save nothing. So the split is:

    bdh_scores        xq, xk, enc          -> S   (B, nh, Lq, Lk)
    [XLA]             mask, S @ xk, LN     -> Yn  (B, nh, Lq, D)
    bdh_gate_decode   xq, Yn, enc, encv,   -> O   (B, nh, Lq, D)
                      dec

Both kernels tile `N`, so neither ever materialises a neuron-space tensor.
Each carries its own `custom_vjp`, so the composition differentiates
correctly without either kernel knowing about the other.

A note on determinism
---------------------
Activation gradients and weight gradients reduce along opposite axes:
`dxq` sums over heads and neuron tiles, `d_enc` sums over the batch. Doing
both in one kernel would need atomics, and float atomics accumulate in
nondeterministic order -- which is not acceptable when the whole point of
the project is comparing training runs against each other. So the backward
is two kernels with different grids, each reducing over a *sequential*
grid axis it owns. Slightly more recomputation, exactly reproducible.

What these kernels do not do
----------------------------
They do not skip zeros. BDH's activations are unstructured-sparse, and a
dense GPU pipeline cannot skip individual zero lanes inside a tile -- the
multiply happens whether the operand is zero or not. Realising BDH's FLOP
advantage as *time* needs structured (block-level) sparsity, which is an
architectural change, not a kernel change. What is won here is memory
traffic, which is real and large but is not the same claim. Anything
reported on the iso-FLOP axis should say which of the two it means; see
`docs/ARCHITECTURE.md`, "A fairness note for the scaling comparison".
"""

from __future__ import annotations

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from ..bdh_arm import _norm, neuron_count
from .cross_attention import default_interpret

DEFAULT_BLOCK_N = 64


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------

def reference_scores(xq, xk, enc):
    """(B,Lq,D), (B,Lk,D), (nh,D,N) -> (B,nh,Lq,Lk), unmasked."""
    q = jax.nn.relu(jnp.einsum("bqd,hdn->bhqn", xq, enc))
    k = jax.nn.relu(jnp.einsum("bkd,hdn->bhkn", xk, enc))
    return jnp.einsum("bhqn,bhkn->bhqk", q, k)


def reference_gate_decode(xq, yn, enc, encv, dec):
    """(B,Lq,D), (B,nh,Lq,D), (nh,D,N), (nh,D,N), (nh,N,D) -> (B,nh,Lq,D)."""
    q = jax.nn.relu(jnp.einsum("bqd,hdn->bhqn", xq, enc))
    y = jax.nn.relu(jnp.einsum("bhqd,hdn->bhqn", yn, encv))
    return jnp.einsum("bhqn,hnd->bhqd", q * y, dec)


# --------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------

def _pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _pad(x, axis, target):
    if x.shape[axis] == target:
        return x
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, target - x.shape[axis])
    return jnp.pad(x, pad)


def _block_n(n: int, requested: int | None) -> int:
    """Neuron tile width. Must divide N; N is padded to a multiple of it."""
    bn = requested or DEFAULT_BLOCK_N
    return min(_pow2(bn), _pow2(n))


# --------------------------------------------------------------------------
# Core A: scores
# --------------------------------------------------------------------------

def _scores_fwd_kernel(xq_ref, xk_ref, enc_ref, s_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        s_ref[...] = jnp.zeros_like(s_ref)

    e = enc_ref[...]                                     # (D, BN)
    q = jnp.maximum(jnp.dot(xq_ref[...], e), 0.0)        # (Lq, BN)
    k = jnp.maximum(jnp.dot(xk_ref[...], e), 0.0)        # (Lk, BN)
    s_ref[...] += jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())))


def _scores_bwd_act_kernel(xq_ref, xk_ref, enc_ref, ds_ref, dxq_ref, dxk_ref):
    """dxq, dxk for one (batch, head), reducing over the neuron tiles."""
    @pl.when(pl.program_id(2) == 0)
    def _init():
        dxq_ref[...] = jnp.zeros_like(dxq_ref)
        dxk_ref[...] = jnp.zeros_like(dxk_ref)

    e = enc_ref[...]
    xq, xk, ds = xq_ref[...], xk_ref[...], ds_ref[...]

    zq = jnp.dot(xq, e)
    zk = jnp.dot(xk, e)
    q, k = jnp.maximum(zq, 0.0), jnp.maximum(zk, 0.0)

    gq = jnp.dot(ds, k) * (zq > 0)                                        # (Lq,BN)
    gk = jax.lax.dot_general(ds, q, (((0,), (0,)), ((), ()))) * (zk > 0)  # (Lk,BN)

    dxq_ref[...] += jax.lax.dot_general(gq, e, (((1,), (1,)), ((), ())))
    dxk_ref[...] += jax.lax.dot_general(gk, e, (((1,), (1,)), ((), ())))


def _scores_bwd_enc_kernel(xq_ref, xk_ref, enc_ref, ds_ref, denc_ref):
    """d_enc for one (head, neuron tile), reducing over the batch.

    Both the query and the key path run through the same encoder, so both
    contribute here -- the reference ties Q and K to one matrix.
    """
    b = pl.program_id(2)

    @pl.when(b == 0)
    def _init():
        denc_ref[...] = jnp.zeros_like(denc_ref)

    # `None` block dims are squeezed out inside the kernel, so each ref is
    # already this grid step's single batch row / head.
    e = enc_ref[...]
    xq, xk, ds = xq_ref[...], xk_ref[...], ds_ref[...]

    zq = jnp.dot(xq, e)
    zk = jnp.dot(xk, e)
    q, k = jnp.maximum(zq, 0.0), jnp.maximum(zk, 0.0)

    gq = jnp.dot(ds, k) * (zq > 0)
    gk = jax.lax.dot_general(ds, q, (((0,), (0,)), ((), ()))) * (zk > 0)

    denc_ref[...] += jax.lax.dot_general(
        xq, gq, (((0,), (0,)), ((), ()))
    ) + jax.lax.dot_general(xk, gk, (((0,), (0,)), ((), ())))


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _scores(xq, xk, enc, block_n, interpret):
    return _scores_fwd(xq, xk, enc, block_n, interpret)[0]


def _scores_fwd(xq, xk, enc, block_n, interpret):
    b, lq, d = xq.shape
    lk = xk.shape[1]
    nh, _, n = enc.shape
    nt = n // block_n

    s = pl.pallas_call(
        _scores_fwd_kernel,
        grid=(b, nh, nt),
        in_specs=[
            pl.BlockSpec((None, lq, d), lambda i, j, t: (i, 0, 0)),
            pl.BlockSpec((None, lk, d), lambda i, j, t: (i, 0, 0)),
            pl.BlockSpec((None, d, block_n), lambda i, j, t: (j, 0, t)),
        ],
        out_specs=pl.BlockSpec((None, None, lq, lk), lambda i, j, t: (i, j, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((b, nh, lq, lk), jnp.float32),
        interpret=interpret,
    )(xq, xk, enc)
    return s, (xq, xk, enc)


def _scores_bwd(block_n, interpret, res, ds):
    xq, xk, enc = res
    b, lq, d = xq.shape
    lk = xk.shape[1]
    nh, _, n = enc.shape
    nt = n // block_n
    ds = ds.astype(jnp.float32)

    dxq_h, dxk_h = pl.pallas_call(
        _scores_bwd_act_kernel,
        grid=(b, nh, nt),
        in_specs=[
            pl.BlockSpec((None, lq, d), lambda i, j, t: (i, 0, 0)),
            pl.BlockSpec((None, lk, d), lambda i, j, t: (i, 0, 0)),
            pl.BlockSpec((None, d, block_n), lambda i, j, t: (j, 0, t)),
            pl.BlockSpec((None, None, lq, lk), lambda i, j, t: (i, j, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((None, None, lq, d), lambda i, j, t: (i, j, 0, 0)),
            pl.BlockSpec((None, None, lk, d), lambda i, j, t: (i, j, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((b, nh, lq, d), jnp.float32),
            jax.ShapeDtypeStruct((b, nh, lk, d), jnp.float32),
        ],
        interpret=interpret,
    )(xq, xk, enc, ds)

    denc = pl.pallas_call(
        _scores_bwd_enc_kernel,
        grid=(nh, nt, b),
        in_specs=[
            pl.BlockSpec((None, lq, d), lambda j, t, i: (i, 0, 0)),
            pl.BlockSpec((None, lk, d), lambda j, t, i: (i, 0, 0)),
            pl.BlockSpec((None, d, block_n), lambda j, t, i: (j, 0, t)),
            pl.BlockSpec((None, None, lq, lk), lambda j, t, i: (i, j, 0, 0)),
        ],
        out_specs=pl.BlockSpec((None, d, block_n), lambda j, t, i: (j, 0, t)),
        out_shape=jax.ShapeDtypeStruct((nh, d, n), jnp.float32),
        interpret=interpret,
    )(xq, xk, enc, ds)

    # xq and xk are shared across heads; the kernel produced one partial per
    # head, so the head axis is summed here rather than with atomics.
    return dxq_h.sum(axis=1), dxk_h.sum(axis=1), denc


_scores.defvjp(_scores_fwd, _scores_bwd)


def bdh_scores(xq, xk, enc, *, block_n=None, interpret=None):
    """Unmasked BDH interaction scores, neuron axis never in HBM.

    Padding the neuron axis with zeros is safe: `relu(0) == 0`, so padded
    neurons contribute nothing to the score and receive no gradient.
    """
    interpret = default_interpret() if interpret is None else interpret
    lq, lk, d, n = xq.shape[1], xk.shape[1], xq.shape[2], enc.shape[2]
    bn = _block_n(n, block_n)
    np_ = -(-n // bn) * bn

    s = _scores(
        _pad(_pad(xq, 1, _pow2(lq)), 2, _pow2(d)),
        _pad(_pad(xk, 1, _pow2(lk)), 2, _pow2(d)),
        _pad(_pad(enc, 1, _pow2(d)), 2, np_),
        bn,
        interpret,
    )
    return s[:, :, :lq, :lk]


# --------------------------------------------------------------------------
# Core B: gate and decode
# --------------------------------------------------------------------------

def _gate_fwd_kernel(xq_ref, yn_ref, enc_ref, encv_ref, dec_ref, o_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)

    q = jnp.maximum(jnp.dot(xq_ref[...], enc_ref[...]), 0.0)    # (Lq, BN)
    y = jnp.maximum(jnp.dot(yn_ref[...], encv_ref[...]), 0.0)   # (Lq, BN)
    o_ref[...] += jnp.dot(q * y, dec_ref[...])                  # (Lq, D)


def _gate_bwd_act_kernel(
    xq_ref, yn_ref, enc_ref, encv_ref, dec_ref, do_ref, dxq_ref, dyn_ref
):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        dxq_ref[...] = jnp.zeros_like(dxq_ref)
        dyn_ref[...] = jnp.zeros_like(dyn_ref)

    e, ev = enc_ref[...], encv_ref[...]
    zq = jnp.dot(xq_ref[...], e)
    zy = jnp.dot(yn_ref[...], ev)
    q, y = jnp.maximum(zq, 0.0), jnp.maximum(zy, 0.0)

    # dG flows back through the gate: each factor is scaled by the other.
    dg = jax.lax.dot_general(do_ref[...], dec_ref[...], (((1,), (1,)), ((), ())))
    gq = (dg * y) * (zq > 0)
    gy = (dg * q) * (zy > 0)

    dxq_ref[...] += jax.lax.dot_general(gq, e, (((1,), (1,)), ((), ())))
    dyn_ref[...] += jax.lax.dot_general(gy, ev, (((1,), (1,)), ((), ())))


def _gate_bwd_w_kernel(
    xq_ref, yn_ref, enc_ref, encv_ref, dec_ref, do_ref,
    denc_ref, dencv_ref, ddec_ref,
):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        denc_ref[...] = jnp.zeros_like(denc_ref)
        dencv_ref[...] = jnp.zeros_like(dencv_ref)
        ddec_ref[...] = jnp.zeros_like(ddec_ref)

    e, ev = enc_ref[...], encv_ref[...]
    xq, yn, do = xq_ref[...], yn_ref[...], do_ref[...]

    zq = jnp.dot(xq, e)
    zy = jnp.dot(yn, ev)
    q, y = jnp.maximum(zq, 0.0), jnp.maximum(zy, 0.0)

    dg = jax.lax.dot_general(do, dec_ref[...], (((1,), (1,)), ((), ())))
    gq = (dg * y) * (zq > 0)
    gy = (dg * q) * (zy > 0)

    denc_ref[...] += jax.lax.dot_general(xq, gq, (((0,), (0,)), ((), ())))
    dencv_ref[...] += jax.lax.dot_general(yn, gy, (((0,), (0,)), ((), ())))
    ddec_ref[...] += jax.lax.dot_general(q * y, do, (((0,), (0,)), ((), ())))


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _gate(xq, yn, enc, encv, dec, block_n, interpret):
    return _gate_fwd(xq, yn, enc, encv, dec, block_n, interpret)[0]


def _gate_fwd(xq, yn, enc, encv, dec, block_n, interpret):
    b, lq, d = xq.shape
    nh, _, n = enc.shape
    nt = n // block_n

    o = pl.pallas_call(
        _gate_fwd_kernel,
        grid=(b, nh, nt),
        in_specs=[
            pl.BlockSpec((None, lq, d), lambda i, j, t: (i, 0, 0)),
            pl.BlockSpec((None, None, lq, d), lambda i, j, t: (i, j, 0, 0)),
            pl.BlockSpec((None, d, block_n), lambda i, j, t: (j, 0, t)),
            pl.BlockSpec((None, d, block_n), lambda i, j, t: (j, 0, t)),
            pl.BlockSpec((None, block_n, d), lambda i, j, t: (j, t, 0)),
        ],
        out_specs=pl.BlockSpec((None, None, lq, d), lambda i, j, t: (i, j, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((b, nh, lq, d), jnp.float32),
        interpret=interpret,
    )(xq, yn, enc, encv, dec)
    return o, (xq, yn, enc, encv, dec)


def _gate_bwd(block_n, interpret, res, do):
    xq, yn, enc, encv, dec = res
    b, lq, d = xq.shape
    nh, _, n = enc.shape
    nt = n // block_n
    do = do.astype(jnp.float32)

    act_specs = [
        pl.BlockSpec((None, lq, d), lambda i, j, t: (i, 0, 0)),
        pl.BlockSpec((None, None, lq, d), lambda i, j, t: (i, j, 0, 0)),
        pl.BlockSpec((None, d, block_n), lambda i, j, t: (j, 0, t)),
        pl.BlockSpec((None, d, block_n), lambda i, j, t: (j, 0, t)),
        pl.BlockSpec((None, block_n, d), lambda i, j, t: (j, t, 0)),
        pl.BlockSpec((None, None, lq, d), lambda i, j, t: (i, j, 0, 0)),
    ]
    dxq_h, dyn = pl.pallas_call(
        _gate_bwd_act_kernel,
        grid=(b, nh, nt),
        in_specs=act_specs,
        out_specs=[
            pl.BlockSpec((None, None, lq, d), lambda i, j, t: (i, j, 0, 0)),
            pl.BlockSpec((None, None, lq, d), lambda i, j, t: (i, j, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((b, nh, lq, d), jnp.float32),
            jax.ShapeDtypeStruct((b, nh, lq, d), jnp.float32),
        ],
        interpret=interpret,
    )(xq, yn, enc, encv, dec, do)

    denc, dencv, ddec = pl.pallas_call(
        _gate_bwd_w_kernel,
        grid=(nh, nt, b),
        in_specs=[
            pl.BlockSpec((None, lq, d), lambda j, t, i: (i, 0, 0)),
            pl.BlockSpec((None, None, lq, d), lambda j, t, i: (i, j, 0, 0)),
            pl.BlockSpec((None, d, block_n), lambda j, t, i: (j, 0, t)),
            pl.BlockSpec((None, d, block_n), lambda j, t, i: (j, 0, t)),
            pl.BlockSpec((None, block_n, d), lambda j, t, i: (j, t, 0)),
            pl.BlockSpec((None, None, lq, d), lambda j, t, i: (i, j, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((None, d, block_n), lambda j, t, i: (j, 0, t)),
            pl.BlockSpec((None, d, block_n), lambda j, t, i: (j, 0, t)),
            pl.BlockSpec((None, block_n, d), lambda j, t, i: (j, t, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((nh, d, n), jnp.float32),
            jax.ShapeDtypeStruct((nh, d, n), jnp.float32),
            jax.ShapeDtypeStruct((nh, n, d), jnp.float32),
        ],
        interpret=interpret,
    )(xq, yn, enc, encv, dec, do)

    return dxq_h.sum(axis=1), dyn, denc, dencv, ddec


_gate.defvjp(_gate_fwd, _gate_bwd)


def bdh_gate_decode(xq, yn, enc, encv, dec, *, block_n=None, interpret=None):
    """Gate the query neurons by the value neurons and decode, per head.

    Returns `(B, nh, Lq, D)`; the caller sums over heads, which is the same
    thing the reference's `reshape(B, T, nh*N) @ decoder` does.
    """
    interpret = default_interpret() if interpret is None else interpret
    lq, d, n = xq.shape[1], xq.shape[2], enc.shape[2]
    bn = _block_n(n, block_n)
    np_ = -(-n // bn) * bn
    plq, pd = _pow2(lq), _pow2(d)

    o = _gate(
        _pad(_pad(xq, 1, plq), 2, pd),
        _pad(_pad(yn, 2, plq), 3, pd),
        _pad(_pad(enc, 1, pd), 2, np_),
        _pad(_pad(encv, 1, pd), 2, np_),
        _pad(_pad(dec, 1, np_), 2, pd),
        bn,
        interpret,
    )
    return o[:, :, :lq, :d]


# --------------------------------------------------------------------------
# Flax block
# --------------------------------------------------------------------------

class FusedBDHBlock(nn.Module):
    """`BDHBlock` with both neuron-space stages replaced by Pallas kernels.

    Parameter tree is identical to `BDHBlock`'s -- same names, same shapes,
    same affine-free norms -- so one set of parameters can be applied to
    both and compared. The pieces left in XLA are the mask, the value
    matmul and the LayerNorms, all of them over small tensors.
    """

    hidden_dim: int
    num_heads: int
    neuron_multiplier: int = 4
    block_n: int | None = None
    interpret: bool | None = None

    @nn.compact
    def __call__(
        self,
        queries: jnp.ndarray,       # (B, Lq, D)
        context: jnp.ndarray,       # (B, Lk, D)
        context_mask: jnp.ndarray,  # (B, Lk) bool
    ) -> jnp.ndarray:
        d, nh = self.hidden_dim, self.num_heads
        n = neuron_count(d, nh, self.neuron_multiplier)
        init = nn.initializers.normal(stddev=0.02)

        encoder = self.param("encoder", init, (nh, d, n))
        encoder_v = self.param("encoder_v", init, (nh, d, n))
        decoder = self.param("decoder", init, (nh * n, d))

        xq = _norm("norm_q")(queries)
        xk = _norm("norm_k")(context)

        kw = dict(block_n=self.block_n, interpret=self.interpret)

        scores = bdh_scores(xq, xk, encoder, **kw)
        scores = jnp.where(context_mask[:, None, None, :], scores, 0.0)

        y_kv = _norm("norm_kv")(jnp.einsum("bhqk,bkd->bhqd", scores, xk))

        # The reference concatenates the heads and multiplies by one
        # (nh*N, D) matrix; that is the same arithmetic as multiplying each
        # head's slice and summing, which is what the kernel returns.
        per_head = bdh_gate_decode(
            xq, y_kv, encoder, encoder_v, decoder.reshape(nh, n, d), **kw
        )
        y = _norm("norm_out")(per_head.sum(axis=1))
        return _norm("norm_residual")(queries + y)
