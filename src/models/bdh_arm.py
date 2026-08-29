"""JAX port of BDH's sparse, Hebbian-plasticity block, dropped into the same
position in the front-end that CrossAttentionArm occupies.

No JAX reference implementation of BDH exists yet; the source of truth is
the bare PyTorch reference at github.com/pathwaycom/bdh (bdh.py, train.py).
This module is the porting target, and per docs/PROJECT_PLAN.md section 3a
it does not go anywhere near the scaling grid until it passes its own
acceptance test on a small toy task: stable training, no NaNs, and the
sparse/positive activation pattern actually showing up in practice.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class BDHArm(nn.Module):
    """Sparse/Hebbian interaction mechanism, same input/output contract
    as CrossAttentionArm.
    """

    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(
        self,
        pack_card_embeddings: jnp.ndarray,
        pool_representation: jnp.ndarray,
        pack_pick_number: jnp.ndarray,
    ) -> jnp.ndarray:
        raise NotImplementedError
