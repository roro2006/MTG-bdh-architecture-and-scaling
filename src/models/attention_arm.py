"""The attention arm: pool-to-pack cross-attention plus a pointer-network
output head.

This is the "Transformer" side of the scaling grid. It consumes the same
pack/pool set encodings that the BDH arm (bdh_arm.py) consumes, and differs
from it only in this file's interaction mechanism — see docs/ARCHITECTURE.md,
"Where the two architectures diverge".

The output head scores only the cards physically present in the current
pack; the softmax is taken over just those scores, so the model cannot
express a pick outside the pack.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class CrossAttentionArm(nn.Module):
    """Pool-conditioned scoring over the candidates in the current pack."""

    hidden_dim: int
    num_heads: int
    num_layers: int

    @nn.compact
    def __call__(
        self,
        pack_card_embeddings: jnp.ndarray,
        pool_representation: jnp.ndarray,
        pack_pick_number: jnp.ndarray,
    ) -> jnp.ndarray:
        """Returns one score per candidate card in the pack."""
        raise NotImplementedError
