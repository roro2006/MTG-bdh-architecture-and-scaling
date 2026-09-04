"""Assembles the front-end, an interaction arm, and the pointer head.

Everything except the arm is shared between the two sides of the scaling
grid, so `PickModel(arm="attention")` and `PickModel(arm="bdh")` differ in
exactly one component (docs/ARCHITECTURE.md, "Where the two architectures
diverge").

The output head scores only the cards physically present in the current
pack and takes a softmax over just those scores, so the model is
structurally unable to express a pick outside the pack.

Parameter counts are derived term by term in `count_params_analytic` and
checked against the realised pytree, per docs/PROJECT_PLAN.md section 3b.
An iso-parameter and an iso-FLOP comparison are different experiments here,
and neither is trustworthy if N is whatever `count_params()` happened to
return.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .attention_arm import CrossAttentionArm
from .bdh_arm import BDHArm, neuron_count
from .embeddings import CardEmbedding, ContextFeatures, gather_card_features
from .set_encoder import SetEncoder

# Large negative rather than -inf: -inf produces NaN gradients when a whole
# row is masked, and a row of an all-padding pack is possible in principle.
MASK_SCORE = -1e9


@dataclass(frozen=True)
class ModelConfig:
    """The knobs the scaling grid sweeps, plus the shapes fixed by the data."""

    hidden_dim: int
    num_heads: int = 4
    pool_encoder_layers: int = 2
    pack_encoder_layers: int = 1
    arm_layers: int = 2
    mlp_ratio: int = 4
    set_encoder_mode: str = "attention"
    # BDH's neuron width per head is neuron_multiplier * hidden_dim // heads.
    # 4 makes a BDH layer cost 12*D^2 against a cross-attention block's
    # 12*D^2 + 15*D; the reference's own default of 128 is 32x larger and
    # would make an iso-parameter grid impossible. See src/models/bdh_arm.py.
    neuron_multiplier: int = 4
    # Route the set encoders and both arms through their Pallas kernels.
    # Parameter trees and values are unchanged (tests/test_kernels.py), so
    # this is purely an execution choice and can be flipped per run.
    fused_kernels: bool = False
    # Sow BDH's realised activation density so a *trained* model can be
    # measured, not just a freshly initialised arm. Off by default: it costs
    # three extra reductions per block and is only wanted for the iso-FLOP
    # accounting and the sparsity half of PROJECT_PLAN.md's 3a acceptance
    # gate. Adds no parameters, so a checkpoint is readable either way.
    collect_density: bool = False
    embed_hidden_dim: int | None = None
    card_feature_dim: int = 65
    packs_per_draft: int = 3
    picks_per_pack: int = 14

    @property
    def embed_hidden(self) -> int:
        return self.embed_hidden_dim or self.hidden_dim

    def scaled_to(self, hidden_dim: int) -> "ModelConfig":
        """Same shape, different width -- the grid's N axis."""
        return replace(self, hidden_dim=hidden_dim)


class PointerHead(nn.Module):
    """One score per pack slot, softmaxed over the pack alone."""

    @nn.compact
    def __call__(self, candidates: jnp.ndarray, pack_mask: jnp.ndarray) -> jnp.ndarray:
        scores = nn.Dense(1, name="score")(candidates)[..., 0]
        return jnp.where(pack_mask, scores, MASK_SCORE)


class PickModel(nn.Module):
    """Pack + pool + pack/pick number -> one logit per card in the pack."""

    config: ModelConfig
    arm: str = "attention"

    @nn.compact
    def __call__(
        self,
        feature_table: jnp.ndarray,  # (V, F), constant
        pack_ids: jnp.ndarray,       # (B, L_pack) int, -1 padded
        pool_ids: jnp.ndarray,       # (B, L_pool) int, -1 padded
        pack_number: jnp.ndarray,    # (B,)
        pick_number: jnp.ndarray,    # (B,)
    ) -> jnp.ndarray:
        config = self.config
        pack_mask = pack_ids >= 0
        pool_mask = pool_ids >= 0

        embed = CardEmbedding(
            embed_dim=config.hidden_dim,
            hidden_dim=config.embed_hidden,
            name="card_embedding",
        )
        pack_embeddings = embed(gather_card_features(feature_table, pack_ids))
        pool_embeddings = embed(gather_card_features(feature_table, pool_ids))

        pack_repr, _ = SetEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.pack_encoder_layers,
            mode=config.set_encoder_mode,
            mlp_ratio=config.mlp_ratio,
            fused=config.fused_kernels,
            name="pack_encoder",
        )(pack_embeddings, pack_mask)

        pool_repr, _ = SetEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.pool_encoder_layers,
            mode=config.set_encoder_mode,
            mlp_ratio=config.mlp_ratio,
            fused=config.fused_kernels,
            name="pool_encoder",
        )(pool_embeddings, pool_mask)

        context = ContextFeatures(
            embed_dim=config.hidden_dim,
            packs_per_draft=config.packs_per_draft,
            picks_per_pack=config.picks_per_pack,
            name="context",
        )(pack_number, pick_number)

        if self.arm == "attention":
            candidates = CrossAttentionArm(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.arm_layers,
                mlp_ratio=config.mlp_ratio,
                fused=config.fused_kernels,
                name="arm",
            )(pack_repr, pack_mask, pool_repr, pool_mask, context)
        elif self.arm == "bdh":
            candidates = BDHArm(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.arm_layers,
                neuron_multiplier=config.neuron_multiplier,
                fused=config.fused_kernels,
                collect_density=config.collect_density,
                name="arm",
            )(pack_repr, pack_mask, pool_repr, pool_mask, context)
        else:
            raise ValueError(f"unknown arm {self.arm!r}")

        return PointerHead(name="pointer")(candidates, pack_mask)


# --------------------------------------------------------------------------
# Analytic parameter counts
#
# Derived term by term rather than read off the pytree, so that N in the
# scaling fit is a quantity we understand as a function of the config. The
# test suite asserts these against the realised parameter tree; if flax ever
# changes a default (a bias here, a scale there) the assertion fails rather
# than the scaling law quietly shifting underneath us.
# --------------------------------------------------------------------------

def _layer_norm(d: int) -> int:
    return 2 * d  # scale + bias


def _dense(fan_in: int, fan_out: int) -> int:
    return fan_in * fan_out + fan_out  # kernel + bias


def _attention(d: int) -> int:
    """query, key, value and output projections, each (d -> d) with a bias."""
    return 4 * (d * d + d)


def _mlp(d: int, ratio: int) -> int:
    return _dense(d, ratio * d) + _dense(ratio * d, d)


def _set_block(d: int, ratio: int) -> int:
    return _layer_norm(d) + _attention(d) + _layer_norm(d) + _mlp(d, ratio)


def _mean_block(d: int, ratio: int) -> int:
    return _layer_norm(d) + _mlp(d, ratio)


def _cross_block(d: int, ratio: int) -> int:
    return (
        _layer_norm(d)      # norm_attn
        + _layer_norm(d)    # norm_context
        + _attention(d)
        + _layer_norm(d)    # norm_mlp
        + _mlp(d, ratio)
    )


def _bdh_block(d: int, num_heads: int, n: int) -> int:
    """encoder + encoder_v + decoder. No biases, no norm parameters.

    Equals 3 * neuron_multiplier * d**2, so neuron_multiplier=4 lands
    within O(d) of _cross_block's 12*d**2 + 15*d.
    """
    return 3 * num_heads * d * n


def count_params_analytic(config: ModelConfig, arm: str = "attention") -> dict[str, int]:
    """Per-component parameter counts, and their total under "total"."""
    d = config.hidden_dim
    r = config.mlp_ratio
    block = _set_block if config.set_encoder_mode == "attention" else _mean_block

    counts = {
        "card_embedding": (
            _dense(config.card_feature_dim, config.embed_hidden)
            + _dense(config.embed_hidden, d)
            + _layer_norm(d)
        ),
        "pack_encoder": config.pack_encoder_layers * block(d, r) + _layer_norm(d),
        "pool_encoder": config.pool_encoder_layers * block(d, r) + _layer_norm(d),
        "context": (config.packs_per_draft + config.picks_per_pack) * d,
        "pointer": _dense(d, 1),
    }
    if arm == "attention":
        counts["arm"] = (
            d  # the learned null key
            + config.arm_layers * _cross_block(d, r)
            + _layer_norm(d)
        )
    elif arm == "bdh":
        # A BDH layer is exactly its three projections: encoder (D->N),
        # encoder_v (D->N) and decoder (nh*N->D), all bias-free, with
        # affine-free LayerNorms that carry no parameters at all. There is
        # no null key either -- an empty pool contributes a zero score
        # rather than needing something legal to attend to.
        n = neuron_count(d, config.num_heads, config.neuron_multiplier)
        counts["arm"] = config.arm_layers * _bdh_block(d, config.num_heads, n)
    else:
        raise NotImplementedError(f"no analytic count for arm {arm!r}")

    counts["total"] = sum(counts.values())
    return counts


def count_params_actual(variables) -> int:
    """N for the scaling law: the size of the *params* collection only.

    Takes either a full variables dict or a bare params tree. The
    distinction is not pedantic -- flax returns every collection from
    `init`, and `collect_density=True` adds a sown one whose arrays are
    activations, not weights. Summing the whole tree counted those as
    parameters, which would have moved N by the batch size of whatever
    dummy input built the tree and quietly bent the fitted alpha.
    """
    tree = variables["params"] if "params" in variables else variables
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(tree)))


def init_model(
    config: ModelConfig,
    feature_table: jnp.ndarray,
    arm: str = "attention",
    seed: int = 0,
    batch_size: int = 2,
    max_pack: int = 14,
    max_pool: int = 41,
):
    """Builds the model and its initial parameters with a dummy batch."""
    model = PickModel(config=config, arm=arm)
    dummy_pack = jnp.zeros((batch_size, max_pack), dtype=jnp.int32)
    dummy_pool = jnp.zeros((batch_size, max_pool), dtype=jnp.int32)
    dummy_scalar = jnp.zeros((batch_size,), dtype=jnp.int32)
    params = model.init(
        jax.random.key(seed),
        feature_table,
        dummy_pack,
        dummy_pool,
        dummy_scalar,
        dummy_scalar,
    )
    return model, params


def cross_entropy_loss(
    logits: jnp.ndarray, label_pos: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mean cross-entropy over the pack, and top-1 accuracy.

    The softmax runs over pack slots only, which is what makes this a closed
    decision task rather than an open-vocabulary one.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target = jnp.take_along_axis(log_probs, label_pos[:, None], axis=-1)[:, 0]
    loss = -target.mean()
    accuracy = (logits.argmax(axis=-1) == label_pos).mean()
    return loss, accuracy
