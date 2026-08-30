"""Composite card embeddings: colour identity, mana value, type, rarity and
keyword flags, projected into one per-card vector.

Built from Scryfall oracle data (see src/data/card_features.py) rather than
a plain id-lookup table, so that (a) a card released after training gets a
reasonable embedding from its attributes with no retraining, and (b)
whatever internal structure the model builds beyond these attributes is, by
construction, the non-trivial residual an interpretability pass should be
looking at.

See docs/ARCHITECTURE.md, "The shared front-end".
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

# Card ids are padded with -1. Gathers clamp to a real row and then zero the
# result, so padding never reads uninitialised memory and never contributes.
PAD_ID = -1


def gather_card_features(feature_table: jnp.ndarray, card_ids: jnp.ndarray) -> jnp.ndarray:
    """(V, F) table + (..., L) ids -> (..., L, F), zeroed at padding.

    Padding is -1, which would wrap around under JAX's clamping gather, so
    it is clamped explicitly and then masked out. Doing both matters: the
    clamp keeps the gather in bounds, the mask keeps the value from being
    the features of card 0.
    """
    valid = card_ids >= 0
    safe_ids = jnp.where(valid, card_ids, 0)
    gathered = feature_table[safe_ids]
    return gathered * valid[..., None].astype(gathered.dtype)


class CardEmbedding(nn.Module):
    """Composes a card's structured attributes into a single embedding.

    A two-layer MLP over the feature block assembled by
    `CardFeatures.dense()`. It is deliberately small: this is a projection
    of attributes the data already carries, not the place where the model's
    capacity is meant to live. The interaction mechanism is what the
    scaling grid is varying.
    """

    embed_dim: int
    hidden_dim: int | None = None

    @nn.compact
    def __call__(self, card_features: jnp.ndarray) -> jnp.ndarray:
        hidden_dim = self.hidden_dim or self.embed_dim
        x = nn.Dense(hidden_dim, name="proj_in")(card_features)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim, name="proj_out")(x)
        return nn.LayerNorm(name="norm")(x)


class ContextFeatures(nn.Module):
    """Pack and pick number, as a small learned vector.

    Folded in on the query side of the interaction, not as a causal
    position: it says "how far into the draft am I", which genuinely
    changes what a good pick looks like, without implying an order over
    the cards in the pack or pool.
    """

    embed_dim: int
    packs_per_draft: int = 3
    picks_per_pack: int = 14

    @nn.compact
    def __call__(self, pack_number: jnp.ndarray, pick_number: jnp.ndarray) -> jnp.ndarray:
        pack_embed = nn.Embed(self.packs_per_draft, self.embed_dim, name="pack_number")
        pick_embed = nn.Embed(self.picks_per_pack, self.embed_dim, name="pick_number")
        return pack_embed(pack_number) + pick_embed(pick_number)
