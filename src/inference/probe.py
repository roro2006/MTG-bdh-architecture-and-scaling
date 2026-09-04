"""A restored checkpoint, callable on hand-built pack/pool states.

This is the shared bottom layer under everything that runs a *trained*
model rather than training one: `src/inference/drafter.py` ranks a pack
with it, and `src/analysis/synergy.py` probes a model's pool use with it.
It lived in synergy.py first, because that was the only caller; it moved
here when the inference entry point needed the same three things, and
duplicating checkpoint restore is how the two copies end up disagreeing
about padding or about which softmax the probabilities came from.

The direction of the dependency is deliberate: analysis imports inference,
not the other way round. Probing a model is a thing you do *to* a deployed
drafter, so the drafter cannot be made to depend on the probes.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..data.card_features import CardFeatures
from ..data.dataset import PAD_ID, PickData


class PickProbe:
    """A restored checkpoint, callable on hand-built pack/pool states.

    Holds the feature table and the corpus geometry so callers can hand it
    plain lists of card ids and get back a probability per pack slot.
    """

    def __init__(self, model, params, feature_table, geometry, vocab):
        self.model = model
        self.params = params
        self.table = jnp.asarray(feature_table)
        self.geometry = geometry
        self.vocab = vocab
        self._apply = jax.jit(
            lambda p, pack, pool, pn, kn: self.model.apply(
                p, self.table, pack, pool, pn, kn
            )
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path, processed_dir: str | Path
    ) -> tuple["PickProbe", PickData]:
        from ..training.checkpoint import restore

        processed_dir = Path(processed_dir)
        model, params, metadata = restore(checkpoint_dir)
        features = CardFeatures.load(processed_dir / "card_features.npz")
        table = features.dense()
        data = PickData.load(processed_dir)

        if table.shape[1] != metadata["model_config"]["card_feature_dim"]:
            raise ValueError(
                f"checkpoint was trained on a {metadata['model_config']['card_feature_dim']}"
                f"-column feature table, but {processed_dir} now holds "
                f"{table.shape[1]} columns. The feature layout changed under it; "
                "retrain, or point at the matching processed directory."
            )
        return cls(model, params, table, data.geometry, data.vocab), data

    # -- shaping ----------------------------------------------------------

    def pad_pack(self, packs: np.ndarray | list) -> np.ndarray:
        return _pad(packs, self.geometry.max_pack_size)

    def pad_pool(self, pools: np.ndarray | list) -> np.ndarray:
        return _pad(pools, self.geometry.max_pool_size)

    def logits(
        self,
        pack_ids: np.ndarray,
        pool_ids: np.ndarray,
        pack_number: np.ndarray,
        pick_number: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            self._apply(
                self.params,
                jnp.asarray(np.asarray(pack_ids, dtype=np.int32)),
                jnp.asarray(np.asarray(pool_ids, dtype=np.int32)),
                jnp.asarray(np.asarray(pack_number, dtype=np.int32)),
                jnp.asarray(np.asarray(pick_number, dtype=np.int32)),
            )
        )

    def log_probs(self, *args) -> np.ndarray:
        logits = self.logits(*args)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    def probabilities(self, *args) -> np.ndarray:
        return np.exp(self.log_probs(*args))


def _pad(rows, width: int) -> np.ndarray:
    """(n, width) int32, PAD_ID-padded, from ragged lists or a padded array."""
    if isinstance(rows, np.ndarray) and rows.ndim == 2:
        if rows.shape[1] == width:
            return rows.astype(np.int32)
        out = np.full((rows.shape[0], width), PAD_ID, dtype=np.int32)
        keep = min(width, rows.shape[1])
        out[:, :keep] = rows[:, :keep]
        return out
    out = np.full((len(rows), width), PAD_ID, dtype=np.int32)
    for i, row in enumerate(rows):
        row = [int(c) for c in row if int(c) >= 0][:width]
        out[i, : len(row)] = row
    return out
