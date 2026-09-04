"""Does the model use the pool, and if so, for synergy or for colour?

A drafter that has learned nothing about card interaction can still post a
respectable loss, because most of a pick is explained by raw card quality
and by staying in two colours. Loss alone cannot tell those apart from
synergy, and neither can a wider feature table -- adding 73 mechanical
columns buys nothing if the model reads none of them. This module is how we
find out, and it is worth running *before* a scaling grid rather than after,
because "the text features bought synergy" and "the text features bought a
wider table" produce very similar learning curves.

Three probes, in increasing order of how specific a claim they support:

1. `pool_ablation` -- one scalar. Re-score real picks with the pool replaced
   by a decoy of the same size. How much worse the model gets is how much it
   was using the pool at all. A model that ignores the pool scores the same
   either way, and no amount of synergy analysis downstream will find
   anything in it.

2. `pool_sensitivity` -- hold one pack fixed, vary the pool, watch the
   candidate ranking move. This is the qualitative picture: which card the
   model switches to when the pool changes, and by how much.

3. `pairwise_synergy` -- for a candidate c and an anchor s, how much does
   seeding the pool with copies of s raise c's log-probability? That matrix
   is the closest thing to a direct read of what the interaction arm learned.

The colour control is what makes (3) meaningful. "Pool is mono-red, so take
the red card" is real behaviour and shows up as a large synergy score, but
it is not what the mechanical columns were added for. So `pairwise_synergy`
splits its result by whether candidate and anchor share a colour, and
`synergy_summary` reports the two separately. A model whose entire pool
effect lives in the colour-sharing half has learned colour-matching; one
with a substantial effect in the colour-disjoint half has learned something
else, and the mechanic columns are the natural place to look for what.

Usage:

    python -m src.analysis.synergy \
        --checkpoint runs/attn_d64_s3000 \
        --processed-dir data/processed/FIN.PremierDraft
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..data.card_features import CardFeatures
from ..data.dataset import PAD_ID, PickData, split_by_draft

# Below this many rows the ablation's difference of two noisy means is not
# worth reporting; the default sample is well above it.
MIN_ABLATION_ROWS = 64


# --------------------------------------------------------------------------
# Running the model
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Probe 1: does the model use the pool at all?
# --------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Real pools against decoy pools, over the same rows."""

    mode: str
    rows: int
    real_loss: float
    decoy_loss: float
    real_accuracy: float
    decoy_accuracy: float
    mean_total_variation: float
    top1_flip_rate: float

    @property
    def pool_utilisation(self) -> float:
        """The headline scalar: nats of loss the real pool is worth.

        Zero means the model's prediction does not depend on which pool it
        is holding, and every synergy claim about it is empty. Negative
        would mean the decoy pool helps, which is a bug, not a finding.
        """
        return self.decoy_loss - self.real_loss

    def summary(self) -> str:
        return (
            f"pool ablation ({self.mode}, {self.rows:,} rows)\n"
            f"  loss      real {self.real_loss:.4f}  decoy {self.decoy_loss:.4f}"
            f"   -> pool is worth {self.pool_utilisation:+.4f} nats\n"
            f"  accuracy  real {self.real_accuracy:.4f}  decoy {self.decoy_accuracy:.4f}"
            f"   ({self.decoy_accuracy - self.real_accuracy:+.4f})\n"
            f"  mean total-variation between the two distributions "
            f"{self.mean_total_variation:.4f}\n"
            f"  top-1 pick changes on {self.top1_flip_rate:.1%} of rows"
        )


def decoy_pools(
    data: PickData,
    rows: np.ndarray,
    mode: str = "permuted",
    seed: int = 0,
) -> np.ndarray:
    """Pools of the right size that do not belong to these picks.

    Two controls, and the difference between them matters:

      "permuted" -- take some *other* row's real pool, truncated or padded
        to this row's pool size. The decoy is a coherent draft pool: right
        colour concentration, right curve, right card quality. Only its
        pairing with this particular pack is destroyed. This is the control
        that isolates "does the model use *this* pool".

      "random" -- sample card ids uniformly from the vocabulary. Destroys
        pool coherence as well as pairing, so it measures something larger
        and easier: a model that has only learned "pools are usually two
        colours" already beats this decoy. Kept because a model that cannot
        beat *this* has learned nothing about pools whatsoever.

    "permuted" is the default because it is the harder and more informative
    of the two; a result quoted without saying which was used is ambiguous.
    """
    if mode not in ("permuted", "random"):
        raise ValueError(f"mode must be 'permuted' or 'random', got {mode!r}")

    rng = np.random.default_rng(seed)
    real = data.pools_padded(rows)
    sizes = (real >= 0).sum(axis=1)
    width = real.shape[1]
    out = np.full((rows.size, width), PAD_ID, dtype=np.int32)

    if mode == "random":
        for i, n in enumerate(sizes):
            if n:
                out[i, :n] = rng.integers(0, data.vocab.size, size=int(n))
        return out

    # "permuted": borrow another row's pool. Drawn from rows with at least
    # as large a pool where possible, so a decoy is a truncated real pool
    # rather than a real pool topped up with padding.
    donor_order = np.argsort(sizes, kind="stable")
    donors = rows[donor_order[::-1]]
    donor_pools = data.pools_padded(donors)
    shuffle = rng.permutation(donors.size)
    for i, n in enumerate(sizes):
        if not n:
            continue
        # Walk the shuffled donor list until one is big enough and is not
        # this row itself; fall back to the largest available.
        for offset in range(donors.size):
            j = shuffle[(i + offset) % donors.size]
            if donors[j] == rows[i]:
                continue
            candidate = donor_pools[j]
            candidate = candidate[candidate >= 0]
            if candidate.size >= n:
                out[i, :n] = candidate[:n]
                break
        else:  # pragma: no cover - only if every donor is too small
            pool = donor_pools[shuffle[i]]
            pool = pool[pool >= 0]
            out[i, : pool.size] = pool
    return out


def pool_ablation(
    probe: PickProbe,
    data: PickData,
    rows: np.ndarray,
    mode: str = "permuted",
    seed: int = 0,
    batch_size: int = 512,
) -> AblationResult:
    """Scores `rows` twice, with their real pools and with decoy pools.

    Rows at pick 0 of pack 0 have an empty pool and are excluded: their two
    scorings are identical by construction, and including them would dilute
    every number here toward zero for no reason.
    """
    rows = np.asarray(rows, dtype=np.int64)
    pool_sizes = (data.pools_padded(rows) >= 0).sum(axis=1)
    rows = rows[pool_sizes > 0]
    if rows.size < MIN_ABLATION_ROWS:
        raise ValueError(
            f"{rows.size} rows with a non-empty pool is too few to compare two "
            f"noisy means; want at least {MIN_ABLATION_ROWS}"
        )

    decoys = decoy_pools(data, rows, mode=mode, seed=seed)

    real_loss = decoy_loss = real_hits = decoy_hits = 0.0
    tv_total = flips = 0.0
    for start in range(0, rows.size, batch_size):
        chunk = rows[start : start + batch_size]
        batch = data.batch(chunk)
        args = (batch["pack_ids"], batch["pack_number"], batch["pick_number"])
        label = batch["label_pos"]

        lp_real = probe.log_probs(args[0], batch["pool_ids"], args[1], args[2])
        lp_decoy = probe.log_probs(args[0], decoys[start : start + batch_size], *args[1:])

        real_loss += -lp_real[np.arange(chunk.size), label].sum()
        decoy_loss += -lp_decoy[np.arange(chunk.size), label].sum()
        real_pick = lp_real.argmax(axis=1)
        decoy_pick = lp_decoy.argmax(axis=1)
        real_hits += (real_pick == label).sum()
        decoy_hits += (decoy_pick == label).sum()
        flips += (real_pick != decoy_pick).sum()
        tv_total += 0.5 * np.abs(np.exp(lp_real) - np.exp(lp_decoy)).sum()

    n = float(rows.size)
    return AblationResult(
        mode=mode,
        rows=int(rows.size),
        real_loss=float(real_loss / n),
        decoy_loss=float(decoy_loss / n),
        real_accuracy=float(real_hits / n),
        decoy_accuracy=float(decoy_hits / n),
        mean_total_variation=float(tv_total / n),
        top1_flip_rate=float(flips / n),
    )


# --------------------------------------------------------------------------
# Probe 2: one pack, many pools
# --------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """One fixed pack scored against several pools."""

    pack: tuple[str, ...]
    pool_labels: tuple[str, ...]
    probabilities: np.ndarray = field(repr=False)  # (n_pools, n_candidates)
    top_pick: tuple[str, ...] = ()
    rank_correlation: tuple[float, ...] = ()
    max_log_ratio: float = 0.0

    def summary(self, top: int = 5) -> str:
        order = np.argsort(-self.probabilities.mean(axis=0))[:top]
        header = "  pool".ljust(30) + "".join(
            f"{self.pack[i][:16]:>18}" for i in order
        )
        lines = [
            f"pack of {len(self.pack)} cards, {len(self.pool_labels)} pools",
            header,
        ]
        for r, label in enumerate(self.pool_labels):
            row = "".join(f"{self.probabilities[r, i]:>18.3f}" for i in order)
            lines.append(f"  {label[:28]:<28}" + row + f"   -> {self.top_pick[r][:20]}")
        lines.append(
            f"  rank correlation vs. first pool: "
            + ", ".join(f"{c:+.2f}" for c in self.rank_correlation)
        )
        lines.append(
            f"  largest log-probability swing on any candidate: "
            f"{self.max_log_ratio:.2f} nats"
        )
        return "\n".join(lines)


def pool_sensitivity(
    probe: PickProbe,
    pack: list[int],
    pools: list[list[int]],
    pool_labels: list[str] | None = None,
    pack_number: int = 1,
    pick_number: int = 3,
) -> SensitivityResult:
    """Holds one pack fixed and scores it against each pool in turn.

    Rank correlation is Spearman against the *first* pool, which the caller
    should make the neutral or empty one; the interesting number is how far
    the others move away from it.
    """
    from scipy.stats import spearmanr

    pack = [int(c) for c in pack]
    n_pools = len(pools)
    packs = probe.pad_pack([pack] * n_pools)
    pool_array = probe.pad_pool(pools)
    probs = probe.probabilities(
        packs,
        pool_array,
        np.full(n_pools, pack_number),
        np.full(n_pools, pick_number),
    )[:, : len(pack)]

    names = tuple(probe.vocab.id_to_card[c] for c in pack)
    labels = tuple(pool_labels or [f"pool {i}" for i in range(n_pools)])
    top_pick = tuple(names[int(row.argmax())] for row in probs)

    correlations = []
    for row in probs:
        if len(pack) < 3:
            correlations.append(float("nan"))
        else:
            correlations.append(float(spearmanr(probs[0], row).statistic))

    with np.errstate(divide="ignore"):
        logs = np.log(np.clip(probs, 1e-12, None))
    max_swing = float((logs.max(axis=0) - logs.min(axis=0)).max())

    return SensitivityResult(
        pack=names,
        pool_labels=labels,
        probabilities=probs,
        top_pick=top_pick,
        rank_correlation=tuple(correlations),
        max_log_ratio=max_swing,
    )


# --------------------------------------------------------------------------
# Probe 3: pairwise synergy, with a colour control
# --------------------------------------------------------------------------

def pairwise_synergy(
    probe: PickProbe,
    candidates: list[int],
    anchors: list[int],
    pool_copies: int = 6,
    pack_number: int = 1,
    pick_number: int = 6,
    baseline_pool: list[int] | None = None,
) -> np.ndarray:
    """(len(candidates), len(anchors)) of log-probability lift.

    Entry [c, s] is `log p(candidate c | pool of s) - log p(c | baseline)`,
    with every candidate in one pack so the softmax is over the same set of
    options in both scorings. That normalisation is the point: a raw
    probability rises whenever the *other* candidates fall, so the lift has
    to be read as "how much did c gain relative to this pack", not as an
    absolute preference.

    The pool is `pool_copies` copies of the anchor. Real drafters never hold
    six copies of one common, so this is a deliberately exaggerated state --
    it makes the anchor's contribution legible above the noise floor, at the
    cost of asking the model about a position it never saw in training.
    Read the sign and the ordering, not the magnitude.
    """
    candidates = [int(c) for c in candidates]
    anchors = [int(a) for a in anchors]
    if len(candidates) > probe.geometry.max_pack_size:
        raise ValueError(
            f"{len(candidates)} candidates will not fit in a pack of "
            f"{probe.geometry.max_pack_size}; the softmax has to be over one pack"
        )

    base = baseline_pool if baseline_pool is not None else []
    pools = [list(base)] + [[a] * pool_copies for a in anchors]
    packs = probe.pad_pack([candidates] * len(pools))
    log_probs = probe.log_probs(
        packs,
        probe.pad_pool(pools),
        np.full(len(pools), pack_number),
        np.full(len(pools), pick_number),
    )[:, : len(candidates)]

    return log_probs[1:].T - log_probs[0][:, None]


def interaction_residual(lift: np.ndarray) -> np.ndarray:
    """`lift` with its additive main effects removed.

    A lift matrix is not yet evidence of interaction. Two things inflate it
    that have nothing to do with a candidate and an anchor *pairing*: a
    candidate that gains from any non-empty pool whatsoever (a row effect,
    which is mostly "this card is good and the empty-pool baseline
    underrates it"), and an anchor that is a loud colour beacon whichever
    candidate it is shown against (a column effect). Both are one-card
    properties. A model that had learned nothing but one-card properties
    would still post a large `within_colour_spread`, because that statistic
    cannot tell a row effect from an interaction.

    Subtracting the row mean, the column mean and the grand mean leaves the
    part of the lift that depends on *which* candidate meets *which*
    anchor. That residual is the interaction term the arm exists to
    compute, and it is the only part of this matrix that PROJECT_PLAN.md
    section 4's claim -- "synergy lives in the interaction arm" -- predicts
    at all. Read `interaction_variance_share` in `synergy_summary` for how
    much of the raw lift survives the subtraction: a perfectly additive
    matrix scores 0 and has no pairwise content, however large its entries.
    """
    return (
        lift
        - lift.mean(axis=1, keepdims=True)
        - lift.mean(axis=0, keepdims=True)
        + lift.mean()
    )


def synergy_summary(
    probe: PickProbe,
    features: CardFeatures,
    lift: np.ndarray,
    candidates: list[int],
    anchors: list[int],
) -> dict:
    """Splits a lift matrix by whether candidate and anchor share a colour.

    This is the whole reason the probe exists. "The pool is red, so take the
    red card" produces a large lift and is not synergy in the sense the
    mechanical feature columns were added for. Read `colour_gap` for how
    much sharing a colour is worth on its own, and `within_colour_spread`
    for whether the model distinguishes *which* same-colour card is in the
    pool -- the latter is where card-level synergy lives, and a pure colour
    matcher has almost none of it.

    Colourless cards are excluded from both halves rather than assigned to
    one: they match every colour trivially, and counting them either way
    would move the comparison without saying anything about it.
    """
    identity = features.color_identity
    cand_colors = identity[candidates]  # (C, 5)
    anchor_colors = identity[anchors]   # (A, 5)

    shares = (cand_colors @ anchor_colors.T) > 0
    colored = (cand_colors.sum(axis=1) > 0)[:, None] & (
        anchor_colors.sum(axis=1) > 0
    )[None, :]

    same = lift[shares & colored]
    different = lift[(~shares) & colored]

    def _mean(values):
        return float(values.mean()) if values.size else float("nan")

    def _std(values):
        return float(values.std()) if values.size else float("nan")

    # Two numbers, read together, and neither is a ratio -- a ratio of
    # magnitudes hides the sign, and the sign is the whole finding. A model
    # trained on FIN puts mean_lift_cross_colour at about -0.6: a candidate
    # that shares no colour with the pool is actively *penalised*. That is
    # colour matching, not synergy, however large the effect looks.
    #
    #   colour_gap          -- nats explained purely by sharing a colour.
    #   within_colour_spread -- variation *among* colour-matched pairs. This
    #     is where card-level synergy has to show up if it exists at all: a
    #     pure colour matcher treats every same-colour anchor alike and has
    #     a spread near zero, so a large spread here is the part of the pool
    #     effect that colour cannot account for.
    #
    # `within_colour_spread` is necessary but not sufficient, and the two
    # `interaction_*` numbers are what close the gap. See
    # `interaction_residual`: a candidate that gains from *any* pool and an
    # anchor that shouts its colour at *every* candidate are both one-card
    # effects, and both inflate the spread without any pairing having been
    # learned. `interaction_variance_share` is the fraction of the lift's
    # variance that survives removing them -- the part that is genuinely
    # about this candidate meeting this anchor.
    resid = interaction_residual(lift)
    total_var = float(lift.var())

    return {
        "pairs": int(lift.size),
        "same_colour_pairs": int(same.size),
        "cross_colour_pairs": int(different.size),
        "mean_lift_same_colour": _mean(same),
        "mean_lift_cross_colour": _mean(different),
        "colour_gap": (
            _mean(same) - _mean(different) if same.size and different.size
            else float("nan")
        ),
        "within_colour_spread": _std(same),
        "cross_colour_spread": _std(different),
        # The interaction term, which is the part colour and card quality
        # together cannot account for.
        "interaction_spread": _std(resid),
        "interaction_variance_share": (
            float(resid.var() / total_var) if total_var > 0 else float("nan")
        ),
        "interaction_spread_same_colour": _std(resid[shares & colored]),
        "interaction_spread_cross_colour": _std(resid[(~shares) & colored]),
    }


def strongest_pairs(
    probe: PickProbe,
    lift: np.ndarray,
    candidates: list[int],
    anchors: list[int],
    top: int = 10,
) -> list[tuple[str, str, float]]:
    """The (candidate, anchor, lift) triples the model likes most."""
    flat = np.argsort(-lift, axis=None)[:top]
    out = []
    for f in flat:
        c, a = np.unravel_index(f, lift.shape)
        out.append(
            (
                probe.vocab.id_to_card[candidates[c]],
                probe.vocab.id_to_card[anchors[a]],
                float(lift[c, a]),
            )
        )
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _draftable_ids(data: PickData) -> np.ndarray:
    """Card ids that actually turn up in a pack.

    The vocabulary is wider than the draftable set -- basic lands above all
    -- and an anchor drawn uniformly from it can be a card no drafter ever
    chose. A pool of six Swamps is pure colour signal and no card, so it
    answers the colour question loudly and the synergy question not at all,
    which is the one confound this probe is built to avoid. Sampling from
    what packs contain keeps both axes of the matrix on the distribution
    the model was trained on.
    """
    return np.unique(data.pack[data.pack >= 0]).astype(np.int64)


def _pick_example_pack(data: PickData, seed: int, min_size: int = 6) -> int:
    """A mid-pack row with enough options left for a ranking to be visible."""
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(
        (data.pack_size >= min_size) & (data.pick_number >= 2) & (data.pack_number == 1)
    )
    if not eligible.size:
        eligible = np.flatnonzero(data.pack_size >= min_size)
    return int(rng.choice(eligible))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe a trained checkpoint for pool use and synergy."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--rows", type=int, default=4096,
                        help="validation rows for the ablation")
    parser.add_argument("--anchors", type=int, default=24,
                        help="anchor cards for the pairwise matrix")
    parser.add_argument("--candidates", type=int, default=10,
                        help="candidates per pack for the pairwise matrix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    probe, data = PickProbe.from_checkpoint(args.checkpoint, args.processed_dir)
    features = CardFeatures.load(Path(args.processed_dir) / "card_features.npz")
    splits = split_by_draft(data, seed=0)
    rng = np.random.default_rng(args.seed)

    report: dict = {"checkpoint": str(args.checkpoint)}

    # -- 1. ablation ------------------------------------------------------
    sample = rng.choice(splits.val, size=min(args.rows, splits.val.size), replace=False)
    print("=" * 70)
    for mode in ("permuted", "random"):
        result = pool_ablation(probe, data, sample, mode=mode, seed=args.seed)
        print(result.summary())
        print("-" * 70)
        report[f"ablation_{mode}"] = asdict(result)
        report[f"ablation_{mode}"]["pool_utilisation"] = result.pool_utilisation

    # -- 2. one pack, several pools ---------------------------------------
    row = _pick_example_pack(data, args.seed)
    pack = [int(c) for c in data.pack[row] if c >= 0]
    real_pool = [int(c) for c in data.pool_of(row)]
    other = _pick_example_pack(data, args.seed + 1)
    result = pool_sensitivity(
        probe,
        pack,
        pools=[[], real_pool, [int(c) for c in data.pool_of(other)]],
        pool_labels=["empty", "this drafter's real pool", "another drafter's pool"],
        pack_number=int(data.pack_number[row]),
        pick_number=int(data.pick_number[row]),
    )
    print(result.summary())
    print("-" * 70)
    report["sensitivity"] = {
        "pack": list(result.pack),
        "pool_labels": list(result.pool_labels),
        "top_pick": list(result.top_pick),
        "rank_correlation": list(result.rank_correlation),
        "max_log_ratio": result.max_log_ratio,
    }

    # -- 3. pairwise synergy ----------------------------------------------
    draftable = _draftable_ids(data)
    n_cand = min(args.candidates, probe.geometry.max_pack_size, draftable.size)
    n_anchor = min(args.anchors, draftable.size)
    candidates = rng.choice(draftable, size=n_cand, replace=False).tolist()
    anchors = rng.choice(draftable, size=n_anchor, replace=False).tolist()
    lift = pairwise_synergy(probe, candidates, anchors)
    summary = synergy_summary(probe, features, lift, candidates, anchors)
    print("pairwise synergy (log-probability lift from a pool of one card)")
    print(f"  drawn from {draftable.size:,} cards seen in a pack, "
          f"of {data.vocab.size:,} in the vocabulary")
    for key, value in summary.items():
        print(f"  {key:32s} {value}")

    # Raw lift first, then the same ranking with the additive main effects
    # removed. The two lists disagreeing is the point: a pair that only
    # tops the raw list is a good card meeting a loud anchor, and a pair
    # that survives into the residual list is the model asserting that
    # these two in particular belong together.
    print("  strongest raw lifts:")
    for cand, anchor_name, value in strongest_pairs(probe, lift, candidates, anchors):
        print(f"    {value:+.3f}  {cand[:34]:<34} <- pool of {anchor_name[:30]}")
    resid = interaction_residual(lift)
    print("  strongest interactions (main effects removed):")
    interactions = strongest_pairs(probe, resid, candidates, anchors)
    for cand, anchor_name, value in interactions:
        print(f"    {value:+.3f}  {cand[:34]:<34} <- pool of {anchor_name[:30]}")

    report["pairwise"] = summary
    report["pairwise"]["draftable_cards"] = int(draftable.size)
    report["pairwise"]["vocab_size"] = int(data.vocab.size)
    report["strongest_raw_lifts"] = strongest_pairs(
        probe, lift, candidates, anchors
    )
    report["strongest_interactions"] = interactions
    print("=" * 70)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
