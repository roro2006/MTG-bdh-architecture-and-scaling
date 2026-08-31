"""The attention arm: pool-to-pack cross-attention.

This is the "Transformer" side of the scaling grid. It consumes the same
pack/pool set encodings that the BDH arm (bdh_arm.py) consumes and produces
the same shape of output for the shared pointer head to score, so the only
thing differing between the two runs is the interaction mechanism itself --
see docs/ARCHITECTURE.md, "Where the two architectures diverge".

The decision at each pick is "given what I already have, which of these
options fits", so each pack card is a query and the pool cards are the keys
and values. Pack and pick number ride along on the query side as a learned
feature, not as a causal position.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention from pack queries to pool keys, then an MLP."""

    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(
        self,
        queries: jnp.ndarray,
        context: jnp.ndarray,
        query_mask: jnp.ndarray,
        context_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        attention_mask = nn.make_attention_mask(query_mask, context_mask)

        h = nn.LayerNorm(name="norm_attn")(queries)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=0.0,
            deterministic=True,
            name="cross_attn",
        )(h, nn.LayerNorm(name="norm_context")(context), mask=attention_mask)
        queries = queries + h

        h = nn.LayerNorm(name="norm_mlp")(queries)
        h = nn.Dense(self.hidden_dim * self.mlp_ratio, name="mlp_in")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim, name="mlp_out")(h)
        return queries + h


class CrossAttentionArm(nn.Module):
    """Pool-conditioned representations for the candidates in the current pack.

    Returns one vector per pack slot, which the pointer head turns into one
    score per slot. Shape contract is shared with BDHArm.
    """

    hidden_dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: int = 4
    fused: bool = False

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
            # Local import to keep src.models.kernels off the import path of
            # anyone who only wants the reference blocks.
            from .kernels.cross_attention import FusedCrossAttentionBlock as Block
        else:
            Block = CrossAttentionBlock

        batch = pack_representations.shape[0]

        # A learned null key, always visible.
        #
        # At pack 0 pick 0 the pool is empty, so every key would be masked
        # and the attention softmax would divide by zero -- NaNs on 1 row in
        # 42, which is 140,237 rows of the corpus. Giving the attention
        # something legal to attend to when there is nothing in the pool is
        # both the numerically safe fix and the semantically right one: "I
        # have no cards yet" is a real state, not a degenerate one.
        null_key = self.param(
            "null_key", nn.initializers.normal(stddev=0.02), (1, 1, self.hidden_dim)
        )
        null_key = jnp.broadcast_to(null_key, (batch, 1, self.hidden_dim))
        pool_representations = jnp.concatenate([null_key, pool_representations], axis=1)
        pool_mask = jnp.concatenate(
            [jnp.ones((batch, 1), dtype=bool), pool_mask.astype(bool)], axis=1
        )

        queries = pack_representations + context[:, None, :]
        for layer in range(self.num_layers):
            queries = Block(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                name=f"block_{layer}",
            )(queries, pool_representations, pack_mask, pool_mask)

        return nn.LayerNorm(name="norm_out")(queries)
