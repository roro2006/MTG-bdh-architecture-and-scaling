"""Measures a *trained* BDH model's realised activation density.

Two things need this number and neither can use an initialised model.

`docs/PROJECT_PLAN.md` 3a makes "actually exhibits the sparse/positive
activation pattern the architecture is supposed to produce" an acceptance
condition on the port. At initialisation that condition is trivially met
and says nothing: the encoder is symmetric noise, so about half the
neurons fire and the gate -- a product of two such -- fires on about a
quarter, regardless of whether the architecture works. The claim is only
meaningful about a model that has been trained on the task.

`src/models/flops.py::bdh_ideal_flops` needs the same three fractions to
bound what sparsity could buy. `docs/ARCHITECTURE.md`'s fairness note puts
that ceiling near 18%, derived against an *assumed* density; this replaces
the assumption with a measurement.

    python -m src.training.density --checkpoint runs/bdh_d64_s3000 \\
        --processed-dir data/processed/FIN.PremierDraft

`src/models/bdh_arm.py::measure_density` does the same job one level down,
on a bare arm and its own inputs. This module exists because the trained
thing is a whole PickModel: the arm's inputs are outputs of the pack and
pool encoders, so they cannot be synthesised, and the density has to be
collected through a full forward pass.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..data.card_features import CardFeatures
from ..data.dataset import PickData, split_by_draft
from ..models.flops import bdh_ideal_flops
from .checkpoint import restore

# The three quantities bdh_ideal_flops consumes, and what each one gates.
DENSITY_KEYS = ("query", "gate", "score")

_SOWN = {"query": "query_rows", "gate": "gate_rows", "score": "score_rows"}


def measure_density_on_batches(
    model,
    params,
    feature_table: jnp.ndarray,
    data: PickData,
    indices: np.ndarray,
    batch_size: int = 512,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Average density over real pack slots, across `indices`.

    Padded slots are excluded rather than counted as inactive. They are not
    decisions, and letting them in would report the sparsity of arithmetic
    nobody cares about -- the same exclusion `measure_density` makes.

    The average is weighted by slot, not by batch: a batch is not a unit of
    anything here, and late picks carry far fewer real slots than early
    ones, so an unweighted mean over batches would quietly reweight the
    draft.
    """
    if not model.config.collect_density:
        raise ValueError(
            "the model was rebuilt without collect_density; nothing is sown. "
            "Set collect_density=True on the ModelConfig before applying."
        )

    # Apply against the weights alone. `sow` *appends* to whatever is
    # already in the collection, so handing back a variables dict that
    # still carries a density collection would grow it batch by batch and
    # leave the stale entry sitting in front of the fresh one.
    weights = {"params": params["params"]} if "params" in params else params

    totals = {name: 0.0 for name in DENSITY_KEYS}
    slots = 0.0
    batches = 0

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        if len(chunk) == 0:
            break
        batch = data.batch(chunk)
        pack_ids = jnp.asarray(batch["pack_ids"])
        _, state = model.apply(
            weights,
            feature_table,
            pack_ids,
            jnp.asarray(batch["pool_ids"]),
            jnp.asarray(batch["pack_number"]),
            jnp.asarray(batch["pick_number"]),
            mutable=["density"],
        )

        weight = (pack_ids >= 0).astype(jnp.float32)  # (B, L_pack)
        slots += float(weight.sum())

        for name in DENSITY_KEYS:
            rows = _sown_rows(state, _SOWN[name])
            if not rows:
                raise ValueError(f"no {name!r} density was sown; is the arm BDH?")
            # Mean over blocks, sum over slots: every block sees the same
            # slots, so the per-block denominator is shared.
            per_block = [float((r * weight).sum()) for r in rows]
            totals[name] += sum(per_block) / len(per_block)

        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    if slots == 0.0:
        raise ValueError("no real pack slots in the requested rows")
    return {name: totals[name] / slots for name in DENSITY_KEYS}


def _sown_rows(state, key: str) -> list[jnp.ndarray]:
    """Every block's sown (B, L_pack) array for one density key.

    `sow` stores a tuple per variable and appends to it, so the *last*
    entry is this call's; the arm nests one collection per block, hence the
    walk rather than a lookup.
    """
    found: list[jnp.ndarray] = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key:
                    found.append(v[-1] if isinstance(v, tuple) else v)
                else:
                    walk(v)

    walk(state.get("density", {}))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure a trained BDH model's activation density."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-batches", type=int, default=40,
        help="density is an average over slots and settles quickly; the "
             "default is ~20k pack slots, not the whole split",
    )
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="density is a property of the model, not of the split, but "
             "held-out rows keep it honest by default",
    )
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    model, params, metadata = restore(checkpoint)
    if metadata["arm"] != "bdh":
        print(f"{checkpoint} is the {metadata['arm']!r} arm; density is BDH-only.")
        return 1

    # Rebuilding with collect_density=True changes no parameters, so the
    # restored tree drops straight in.
    model = model.clone(config=replace(model.config, collect_density=True))

    processed = Path(args.processed_dir)
    table = jnp.asarray(CardFeatures.load(processed / "card_features.npz").dense())
    data = PickData.load(processed)
    splits = split_by_draft(data, seed=0)
    indices = getattr(splits, args.split)

    density = measure_density_on_batches(
        model, params, table, data, indices,
        batch_size=args.batch_size, max_batches=args.max_batches,
    )

    print(f"density for {checkpoint} on the {args.split} split")
    for name in DENSITY_KEYS:
        print(f"  {name:6s} {density[name]:.4f}")

    # Only the arm is affected -- sparsity cannot touch the encoders -- and
    # bdh_ideal_flops already carries the arm's dense cost to compare against.
    ideal = bdh_ideal_flops(
        model.config,
        score_density=density["score"],
        gate_density=density["gate"],
    )
    dense_total = ideal["dense_total"]
    saving = 1.0 - ideal["total"] / dense_total
    print(
        f"\nBDH arm forward FLOPs/example: dense {dense_total:,.0f} -> "
        f"perfectly-sparse {ideal['total']:,.0f} ({100 * saving:.1f}% skippable)"
    )

    out = checkpoint / "density.json"
    out.write_text(
        json.dumps(
            {
                "split": args.split,
                "density": density,
                "arm_dense_forward_flops": dense_total,
                "arm_ideal_forward_flops": ideal["total"],
                "arm_ideal_breakdown": ideal,
                "skippable_fraction": saving,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
