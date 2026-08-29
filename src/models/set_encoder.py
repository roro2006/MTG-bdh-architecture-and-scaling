"""Permutation-invariant encoder applied separately to the pack and the pool.

Card order within either set carries no signal, so this deliberately has
no positional encoding — a Set Transformer-style induced-set-attention
block (Lee et al., 2019), or a simpler shared-MLP-then-pool baseline to
start from. See docs/ARCHITECTURE.md, "The shared front-end".
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class SetEncoder(nn.Module):
    """Encodes a variable-size set of card embeddings into one representation."""

    hidden_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, card_embeddings: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
