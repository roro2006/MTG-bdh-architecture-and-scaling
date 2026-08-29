"""Composite card embeddings: color identity, mana value, type, and a
keyword-text embedding, combined into one per-card vector.

Built from Scryfall oracle data rather than a plain id-lookup table, so
that (a) a card released after training gets a reasonable embedding from
its attributes with no retraining, and (b) whatever internal structure the
model builds beyond these attributes is, by construction, the non-trivial
residual an interpretability pass should be looking at.

See docs/ARCHITECTURE.md, "The shared front-end".
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class CardEmbedding(nn.Module):
    """Composes a card's structured attributes into a single embedding.

    Not yet implemented. Expected sub-components:
      - color identity: 5-dim multi-hot -> small dense projection
      - mana value: scalar -> small dense or bucket embedding
      - card type: categorical -> embedding lookup
      - keyword/ability text: derived from oracle text, embedding lookup
        over a small fixed keyword vocabulary (not raw subword tokens)
    """

    embed_dim: int

    @nn.compact
    def __call__(self, card_features: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
