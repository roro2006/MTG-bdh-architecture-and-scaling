"""Does the optimal learning rate move with model width?

This exists because of a specific way scaling-law fits go wrong. If one
learning rate is used across every width in the grid, and the optimal rate
falls as width grows (it usually does), then the large models are
systematically under-tuned relative to the small ones. The fitted A/N^alpha
then absorbs that tuning degradation and reports it as capacity scaling --
the exponent comes out wrong, and nothing about the fit looks unhealthy.
Chinchilla tuned per scale for exactly this reason.

There are two ways to deal with it. muP (Tensor Programs V) reparameterises
so the optimum transfers across width, which is principled but real work.
The cheaper move, and the right first one, is to measure whether the
optimum moves here at all: sweep a few rates at a few widths on short runs
and look at where the minimum sits. If it barely moves over the grid's
range, a single rate is defensible and can be stated as measured rather
than assumed. If it moves a lot, that settles whether muP is worth building.

Short runs are enough for this. The question is where the optimum sits, not
what the final loss is, and the ordering of learning rates is usually
established well before convergence.

    python -m src.training.lr_sweep --processed-dir data/processed/FIN.PremierDraft \\
        --widths 32 64 128 --learning-rates 1e-4 3e-4 1e-3 3e-3 --steps 300
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..data.card_features import CardFeatures
from ..data.dataset import PickData, split_by_draft
from ..models.pick_model import ModelConfig
from .train import TrainConfig, train_model

DEFAULT_WIDTHS = (32, 64, 128)
DEFAULT_RATES = (1e-4, 3e-4, 1e-3, 3e-3)


def sweep(
    data: PickData,
    feature_table: jnp.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    widths=DEFAULT_WIDTHS,
    learning_rates=DEFAULT_RATES,
    steps: int = 300,
    batch_size: int = 512,
    seed: int = 0,
    arm: str = "attention",
    verbose: bool = True,
) -> list[dict]:
    """Trains width x learning-rate short runs. Returns one record each."""
    records: list[dict] = []
    for width in widths:
        model_config = ModelConfig(
            hidden_dim=width,
            card_feature_dim=int(feature_table.shape[1]),
            packs_per_draft=data.packs_per_draft,
            picks_per_pack=data.picks_per_pack,
        )
        for rate in learning_rates:
            train_config = TrainConfig(
                batch_size=batch_size,
                learning_rate=rate,
                total_steps=steps,
                warmup_steps=max(20, steps // 10),
                eval_every=steps,        # only the endpoint matters here
                eval_batches=20,
                seed=seed,
            )
            result = train_model(
                data, feature_table, train_indices, val_indices,
                model_config, train_config, arm=arm, verbose=False,
            )
            loss = result["best_val_loss"]
            diverged = not math.isfinite(loss)
            records.append(
                {
                    "width": width,
                    "learning_rate": rate,
                    "num_params": result["num_params"],
                    "val_loss": loss,
                    "diverged": diverged,
                    "elapsed_s": result["elapsed_s"],
                }
            )
            if verbose:
                flag = "  DIVERGED" if diverged else ""
                print(
                    f"  d={width:<4} lr={rate:<8.1e} val {loss:.4f}"
                    f"  ({result['elapsed_s']:.0f}s){flag}",
                    flush=True,
                )
    return records


def best_rate_per_width(records: list[dict]) -> dict[int, float]:
    best: dict[int, tuple[float, float]] = {}
    for record in records:
        if record["diverged"]:
            continue
        width = record["width"]
        if width not in best or record["val_loss"] < best[width][0]:
            best[width] = (record["val_loss"], record["learning_rate"])
    return {width: rate for width, (_, rate) in best.items()}


def format_table(records: list[dict]) -> str:
    widths = sorted({r["width"] for r in records})
    rates = sorted({r["learning_rate"] for r in records})
    lookup = {(r["width"], r["learning_rate"]): r for r in records}
    best = best_rate_per_width(records)

    header = f"{'lr \\\\ d':>10}" + "".join(f"{w:>10}" for w in widths)
    lines = [header, "-" * len(header)]
    for rate in rates:
        row = f"{rate:>10.1e}"
        for width in widths:
            record = lookup.get((width, rate))
            if record is None:
                row += f"{'-':>10}"
            elif record["diverged"]:
                row += f"{'div':>10}"
            else:
                marker = "*" if best.get(width) == rate else " "
                row += f"{record['val_loss']:>9.4f}{marker}"
        lines.append(row)
    lines.append("")
    lines.append("best rate per width: " + ", ".join(
        f"d={w}: {r:.1e}" for w, r in sorted(best.items())
    ))
    if len(set(best.values())) == 1:
        lines.append(
            "-> the optimum does not move over this range; a single rate across "
            "the grid is defensible, and now measured rather than assumed."
        )
    else:
        lines.append(
            "-> the optimum MOVES with width. A single rate would under-tune part "
            "of the grid and bias alpha. Either tune per width or adopt muP."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=list(DEFAULT_RATES)
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-fraction", type=float, default=0.1,
        help="fraction of training drafts to use; the sweep does not need all of it",
    )
    args = parser.parse_args(argv)

    processed = Path(args.processed_dir)
    features = CardFeatures.load(processed / "card_features.npz")
    table = jnp.asarray(features.dense())
    data = PickData.load(processed)
    splits = split_by_draft(data, seed=0)

    from .run import subsample_by_draft

    train_indices = subsample_by_draft(
        data, splits.train, args.data_fraction, seed=args.seed
    )
    print(
        f"LR sweep: widths {args.widths}, rates {args.learning_rates}, "
        f"{args.steps} steps, {train_indices.size:,} train rows"
    )
    records = sweep(
        data, table, train_indices, splits.val,
        widths=args.widths, learning_rates=args.learning_rates,
        steps=args.steps, batch_size=args.batch_size, seed=args.seed,
    )
    print()
    print(format_table(records))

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
