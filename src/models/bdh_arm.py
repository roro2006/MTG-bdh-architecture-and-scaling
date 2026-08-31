"""JAX/Flax port of BDH's sparse, Hebbian-plasticity block, dropped into the
same position in the front-end that CrossAttentionArm occupies.

Source of truth is the PyTorch reference at github.com/pathwaycom/bdh
(`bdh.py`). This module keeps the mechanism and changes the indexing, for
reasons set out below; `docs/ARCHITECTURE.md` ("Where the two architectures
diverge") is the contract it has to satisfy.

What is kept from the reference
-------------------------------
The whole of what makes BDH BDH:

  - a wide neuron space ``N`` per head, entered through a low-rank
    ``encoder`` (D -> N) and left through a low-rank ``decoder`` (nh*N -> D);
  - ReLU on the latents, so activations are non-negative and sparse, which
    is the property the architecture's efficiency claim rests on;
  - **no softmax** on the interaction scores. Scores are raw inner products
    of two non-negative sparse vectors, so they are themselves non-negative;
    this is the linear/Hebbian accumulation, not a normalised attention;
  - the multiplicative gate ``x_sparse * y_sparse``, which makes the
    decoded signal doubly sparse;
  - affine-free LayerNorm everywhere, so the block carries no norm
    parameters.

What is changed, and why
------------------------
The reference is a causal language model. Two of its pieces encode *token
order*, and this project is committed to being structurally incapable of
using order within a pack or pool (docs/ARCHITECTURE.md, "Why not a plain
causal transformer"). So:

  - **RoPE is dropped.** It is a positional phase on the query/key
    features. A pool is a set; there is no position for it to encode.
  - **The causal mask is dropped.** ``scores.tril(diagonal=-1)`` says token
    t sees tokens before t. Here the pack queries see the whole pool, and
    the pool is masked by validity rather than by order.
  - **Q and K come from different sets.** The reference asserts ``K is Q``:
    self-attention over one stream. This arm is a cross-interaction, pack
    against pool, exactly as CrossAttentionArm is -- otherwise the two arms
    would not be answering the same question and the grid would not be
    comparing them on equal terms. The same ``encoder`` produces both,
    which is the closest analogue of the reference's tied Q/K.
  - **V is the pool stream**, since the values being accumulated are the
    cards already drafted.

Dropping RoPE and causality is not a softening of BDH. It is the same
substitution the attention arm already makes: CrossAttentionArm is not a
causal transformer either. Both arms give up order for the same reason, so
the comparison stays clean.

Sizing
------
The reference's ``mlp_internal_dim_multiplier=128`` gives ``N = 32*D`` at
``nh=4``, i.e. ~25M parameters for a single layer at D=256, against ~0.8M
for one CrossAttentionBlock. That is not a knob the scaling grid can leave
alone. ``neuron_multiplier`` here is that multiplier renamed, and its
default of 4 is chosen so that one BDH layer costs ``12*D**2`` parameters
against a cross-attention block's ``12*D**2 + 15*D`` -- iso-parameter to
within a term linear in D. See ``count_params_analytic`` in pick_model.py.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


def _norm(name: str) -> nn.LayerNorm:
    """The reference's ``nn.LayerNorm(D, elementwise_affine=False)``.

    Affine-free, so a BDH layer's parameters are exactly the three
    projection matrices and nothing else.
    """
    return nn.LayerNorm(use_scale=False, use_bias=False, name=name)


def neuron_count(hidden_dim: int, num_heads: int, neuron_multiplier: int) -> int:
    """Per-head neuron-space width ``N``, the reference's
    ``mlp_internal_dim_multiplier * D // nh``."""
    total = neuron_multiplier * hidden_dim
    if total % num_heads:
        raise ValueError(
            f"neuron_multiplier*hidden_dim ({total}) must be divisible by "
            f"num_heads ({num_heads})"
        )
    return total // num_heads


class BDHBlock(nn.Module):
    """One BDH layer: encode to the sparse neuron space, accumulate over the
    pool, gate, decode back into the residual stream.

    Only the query (pack) stream is updated. The pool is context, the same
    way it is context for CrossAttentionBlock.
    """

    hidden_dim: int
    num_heads: int
    neuron_multiplier: int = 4
    collect_density: bool = False

    @nn.compact
    def __call__(
        self,
        queries: jnp.ndarray,       # (B, Lq, D) pack residual stream
        context: jnp.ndarray,       # (B, Lk, D) pool residual stream
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

        # Sparse, non-negative neuron activations. The reference ties Q and
        # K through the same encoder; the only difference here is that they
        # are drawn from two different sets.
        q_sparse = nn.relu(jnp.einsum("bqd,hdn->bhqn", xq, encoder))
        k_sparse = nn.relu(jnp.einsum("bkd,hdn->bhkn", xk, encoder))

        # No softmax: a plain inner product of non-negative vectors, zeroed
        # wherever the pool slot is padding. An empty pool therefore
        # contributes exactly zero rather than needing the null key the
        # attention arm's softmax requires -- there is no denominator here
        # to go to zero (cf. CrossAttentionArm's null_key).
        scores = jnp.einsum("bhqn,bhkn->bhqk", q_sparse, k_sparse)
        scores = jnp.where(context_mask[:, None, None, :], scores, 0.0)

        y_kv = jnp.einsum("bhqk,bkd->bhqd", scores, xk)
        y_kv = _norm("norm_kv")(y_kv)

        y_sparse = nn.relu(jnp.einsum("bhqd,hdn->bhqn", y_kv, encoder_v))

        # The multiplicative gate: a neuron contributes only if it was
        # active on the way in *and* is selected on the way out.
        xy_sparse = q_sparse * y_sparse

        # (B, nh, Lq, N) -> (B, Lq, nh*N), head-major, matching the
        # reference's transpose(1, 2).reshape(B, 1, T, N * nh).
        b, _, lq, _ = xy_sparse.shape
        flat = jnp.transpose(xy_sparse, (0, 2, 1, 3)).reshape(b, lq, nh * n)

        if self.collect_density:
            self._sow_density(q_sparse, k_sparse, xy_sparse, context_mask)

        y = _norm("norm_out")(flat @ decoder)
        return _norm("norm_residual")(queries + y)

    def _sow_density(self, q_sparse, k_sparse, xy_sparse, context_mask):
        """Record how sparse the activations actually are.

        Off by default and free when off, because these are the numbers the
        iso-FLOP axis depends on and a guessed density is worse than none.
        Rows are kept separate rather than averaged so the caller can mask
        out padded pack slots, which are not real decisions.

        `score_density` is the fraction of neurons active in *both* the
        query and the key -- the only ones that contribute to a score.
        Computing it costs as much as the score matmul itself, which is why
        this is a diagnostic pass and not something to leave on.
        """
        n = q_sparse.shape[-1]
        qi = (q_sparse > 0).astype(jnp.float32)
        ki = (k_sparse > 0).astype(jnp.float32)
        both = jnp.einsum("bhqn,bhkn->bhqk", qi, ki) / n
        valid = context_mask[:, None, None, :].astype(jnp.float32)

        # (B, Lq): averaged over heads and neurons, per pack slot.
        self.sow("density", "query_rows", qi.mean(axis=(1, 3)))
        self.sow("density", "gate_rows", (xy_sparse > 0).mean(axis=(1, 3)))
        # The numerator sums over heads as well as keys, so the denominator
        # counts (head, valid key) pairs, not valid keys alone.
        pairs = q_sparse.shape[1] * jnp.maximum(valid.sum(axis=(1, 3)), 1.0)
        self.sow("density", "score_rows", (both * valid).sum(axis=(1, 3)) / pairs)


class BDHArm(nn.Module):
    """Pool-conditioned representations for the candidates in the current
    pack, via BDH's sparse Hebbian block.

    Input and output contract is identical to CrossAttentionArm: same
    arguments in the same order, ``(B, L_pack, D)`` out, so PickModel can
    swap one for the other and change nothing else.
    """

    hidden_dim: int
    num_heads: int
    num_layers: int
    neuron_multiplier: int = 4
    fused: bool = False
    collect_density: bool = False

    @nn.compact
    def __call__(
        self,
        pack_representations: jnp.ndarray,   # (B, L_pack, D)
        pack_mask: jnp.ndarray,              # (B, L_pack)
        pool_representations: jnp.ndarray,   # (B, L_pool, D)
        pool_mask: jnp.ndarray,              # (B, L_pool)
        context: jnp.ndarray,                # (B, D) pack/pick number features
    ) -> jnp.ndarray:
        if self.fused:
            # Imported here rather than at module scope: the kernels import
            # this module for its LayerNorm and sizing helpers, so a
            # top-level import would be circular.
            from .kernels.bdh import FusedBDHBlock as Block
        else:
            Block = BDHBlock

        # Pack/pick number rides on the query side, exactly as it does in
        # the attention arm, so the two arms see the same information.
        queries = pack_representations + context[:, None, :]

        for layer in range(self.num_layers):
            kwargs = {"collect_density": True} if self.collect_density else {}
            queries = Block(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                neuron_multiplier=self.neuron_multiplier,
                name=f"block_{layer}",
                **kwargs,
            )(queries, pool_representations, pool_mask.astype(bool))

        # Zero the padded pack slots. The pointer head masks the scores
        # anyway, but a padded slot carrying a finite value into a fused
        # kernel is the kind of thing that is easier to never allow.
        return queries * pack_mask.astype(queries.dtype)[..., None]


def measure_density(
    arm: BDHArm, params, pack_repr, pack_mask, pool_repr, pool_mask, context
) -> dict[str, float]:
    """Realised activation density, averaged over real pack slots only.

    The iso-FLOP comparison in docs/ARCHITECTURE.md turns on how sparse BDH
    actually is, and that is an empirical property of a trained model, not
    something to assume. Feed this a batch and it returns the three numbers
    `src/models/flops.py::bdh_ideal_flops` needs:

      query  -- fraction of neurons firing on the query side
      gate   -- fraction surviving the multiplicative gate, which is what
                the decoder actually has to multiply
      score  -- fraction of neurons active in both a query and a key, the
                only ones that contribute to an interaction score

    Padded pack slots are excluded: they are not decisions, and including
    them would report the sparsity of arithmetic nobody cares about.
    """
    if not arm.collect_density:
        arm = arm.clone(collect_density=True)
    _, state = arm.apply(
        params,
        pack_repr,
        pack_mask,
        pool_repr,
        pool_mask,
        context,
        mutable=["density"],
    )

    weight = pack_mask.astype(jnp.float32)
    total = jnp.maximum(weight.sum(), 1.0)

    def average(name: str) -> float:
        rows = [
            v[0]
            for block in state["density"].values()
            for k, v in block.items()
            if k == name
        ]
        return float(sum((r * weight).sum() for r in rows) / (total * len(rows)))

    return {
        "query": average("query_rows"),
        "gate": average("gate_rows"),
        "score": average("score_rows"),
    }
