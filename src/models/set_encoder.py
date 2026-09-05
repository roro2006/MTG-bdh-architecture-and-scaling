"""Permutation-invariant encoder applied separately to the pack and the pool.

Card order within either set carries no signal, so this has no positional
encoding of any kind -- the architecture is structurally incapable of using
order rather than merely trained to ignore it (docs/ARCHITECTURE.md, "Why
not a plain causal transformer").

Two modes, both order-respecting in the same sense:

  - "attention": self-attention blocks over the set, which is
    permutation-*equivariant* per element and permutation-*invariant* once
    pooled. This is the Set Transformer shape (Lee et al., 2019) minus the
    induced points, which only pay off for sets far larger than 14 or 41.
  - "mean": a shared per-card MLP then masked mean pooling. The simpler
    baseline named in the architecture doc, kept because it is the honest
    control for whether set attention is doing anything at all.

`fused=True` swaps the attention core for the Pallas kernel in
`kernels/set_encoder.py`, which is where the majority of the model's
forward FLOPs live (docs/PROJECT_PLAN.md section 5). The parameter tree is
unchanged either way, so a checkpoint reads back down either path.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


def masked_mean(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """(B, L, D) + (B, L) -> (B, D), averaging only over real elements.

    An all-padding set (the pool at the very first pick) averages to zero
    rather than dividing by zero.
    """
    weights = mask.astype(x.dtype)[..., None]
    total = (x * weights).sum(axis=-2)
    count = weights.sum(axis=-2)
    return total / jnp.maximum(count, 1.0)


class SetAttentionBlock(nn.Module):
    """Pre-norm self-attention + feed-forward, with no positional term."""

    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        attention_mask = nn.make_attention_mask(mask, mask)

        h = nn.LayerNorm(name="norm_attn")(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=0.0,
            deterministic=True,
            name="attn",
        )(h, h, mask=attention_mask)
        x = x + h

        h = nn.LayerNorm(name="norm_mlp")(x)
        h = nn.Dense(self.hidden_dim * self.mlp_ratio, name="mlp_in")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim, name="mlp_out")(h)
        return x + h


class SetEncoder(nn.Module):
    """Encodes a variable-size set of card embeddings.

    Returns both the per-element representations and a single pooled
    vector. The pack needs the per-element form (each card is a candidate
    the pointer head has to score); the pool is mostly used pooled, but its
    per-element form is what the cross-attention arm attends over.
    """

    hidden_dim: int
    num_heads: int
    num_layers: int
    mode: str = "attention"
    mlp_ratio: int = 4
    # Route the self-attention through the Pallas kernel. Parameter tree and
    # values are unchanged (tests/test_kernels.py), so this is an execution
    # choice; `ModelConfig.fused_kernels` sets it for the whole model. It has
    # no effect in "mean" mode, which has no attention to fuse.
    fused: bool = False

    @nn.compact
    def __call__(
        self, card_embeddings: jnp.ndarray, mask: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.mode not in ("attention", "mean"):
            raise ValueError(f"mode must be 'attention' or 'mean', got {self.mode!r}")

        x = card_embeddings
        if self.mode == "attention":
            if self.fused:
                # Local import to keep src.models.kernels off the import path
                # of anyone who only wants the reference blocks.
                from .kernels.set_encoder import FusedSetAttentionBlock as Block
            else:
                Block = SetAttentionBlock
            for layer in range(self.num_layers):
                x = Block(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    name=f"block_{layer}",
                )(x, mask)
        else:
            for layer in range(self.num_layers):
                h = nn.LayerNorm(name=f"norm_{layer}")(x)
                h = nn.Dense(self.hidden_dim * self.mlp_ratio, name=f"mlp_in_{layer}")(h)
                h = nn.gelu(h)
                h = nn.Dense(self.hidden_dim, name=f"mlp_out_{layer}")(h)
                x = x + h

        x = nn.LayerNorm(name="norm_out")(x)
        # Zero the padding so it cannot leak into anything downstream that
        # forgets to re-apply the mask.
        x = x * mask.astype(x.dtype)[..., None]
        return x, masked_mean(x, mask)
