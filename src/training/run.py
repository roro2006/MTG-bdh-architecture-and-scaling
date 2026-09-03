"""Command-line entry point for training one cell.

    python -m src.training.run --processed-dir data/processed/FIN.PremierDraft \
        --width 64 --steps 3000 --out-dir runs/attn_d64

Writes the trained parameters and a metrics file holding the learning
curve, the exact full-split evaluation, and the per-pick breakdown. The
grid runner (grid.py) calls train_model directly rather than shelling out
to this, but every flag it sweeps is exposed here so a single cell can be
reproduced by hand.

`--max-seconds` and `--resume` exist for running a cell on a runtime that
will be taken away: a Colab session caps at 12h and is reclaimed after 90
minutes idle, so a long cell has to run as a sequence of bounded segments.
A budgeted run exits EXIT_INCOMPLETE (75) with resumable state on disk; the
identical command with --resume continues it. See scripts/ for the driver
that loops on that exit code, and src/training/README.md for why resume
state is a separate artefact from the best-val checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..data.card_features import CardFeatures
from ..data.dataset import PickData, split_by_draft
from ..models.pick_model import ModelConfig
from .checkpoint import save_checkpoint
from .evaluate import evaluate_by_pick, format_by_pick, summarise_by_pick
from .train import TrainConfig, frequency_baseline, train_model, uniform_baseline

# Returned instead of 0 when --max-seconds cut the run short. A caller
# driving a Colab session loops on this: re-invoke with --resume until it
# returns 0. Distinct from 1 so a genuine crash is never mistaken for
# "there is more to do".
EXIT_INCOMPLETE = 75


def subsample_by_draft(
    data: PickData, indices: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Takes a fraction of the *drafts* in `indices`, not of the rows.

    The D axis of the scaling grid has to respect the same discipline as
    the split: a fraction drawn over rows would put most drafts in the
    subset with a handful of their picks missing, which is a different
    (and easier) distribution than seeing fewer complete drafts.
    """
    if fraction >= 1.0:
        return indices
    drafts = np.unique(data.draft_idx[indices])
    rng = np.random.default_rng(seed)
    keep = rng.permutation(drafts)[: max(1, int(round(fraction * drafts.size)))]
    return indices[np.isin(data.draft_idx[indices], keep)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one model.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--arm", default="attention", choices=["attention", "bdh"])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--arm-layers", type=int, default=2)
    parser.add_argument("--pool-layers", type=int, default=2)
    parser.add_argument("--pack-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--neuron-multiplier",
        type=int,
        default=4,
        help=(
            "BDH neuron width per head is this times width over heads. The "
            "default makes a BDH layer iso-parameter with a cross-attention "
            "block; the BDH paper's own default of 128 is 32x larger."
        ),
    )
    parser.add_argument(
        "--fused-kernels",
        action="store_true",
        help=(
            "Run the arm through its Pallas kernel. Same parameters and same "
            "values (tests/test_kernels.py); needs a GPU or TPU to be worth "
            "anything, since Pallas falls back to a slow interpreter on CPU."
        ),
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--epochs", type=float, default=None,
        help="train for this many passes over the subsampled training set, "
             "overriding --steps. Preferred for grid cells: with fixed steps "
             "a small --data-fraction silently means many epochs, and the "
             "fitted beta then measures data repetition rather than data scale.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-fraction", type=float, default=1.0,
        help="fraction of training DRAFTS to use -- the grid's D axis",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--skip-full-eval", action="store_true",
        help="skip the exact full-split evaluation (the sampled one still runs)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="continue from the resume state in --out-dir if one is present. "
             "Restores optimiser moments, step counter and batch-stream "
             "position, so a resumed run sees the same examples in the same "
             "order as an uninterrupted one. Refuses if the saved state was "
             "written by a differently configured run.",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="stop cleanly at the next --eval-every boundary once this much "
             "wall clock has passed, leaving a resumable state behind, and "
             f"exit {EXIT_INCOMPLETE} rather than 0. For running a long cell "
             "as a sequence of bounded segments on a Colab runtime that caps "
             "at 12h and dies after 90 minutes idle.",
    )
    args = parser.parse_args(argv)

    processed = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = CardFeatures.load(processed / "card_features.npz")
    table = jnp.asarray(features.dense())
    data = PickData.load(processed)
    splits = split_by_draft(data, seed=0)

    train_indices = subsample_by_draft(
        data, splits.train, args.data_fraction, seed=args.seed
    )
    print(
        f"data: {data.size:,} rows | train {train_indices.size:,} "
        f"({args.data_fraction:g} of {splits.train.size:,}) | val {splits.val.size:,}"
    )

    model_config = ModelConfig(
        hidden_dim=args.width,
        num_heads=args.num_heads,
        pool_encoder_layers=args.pool_layers,
        pack_encoder_layers=args.pack_layers,
        arm_layers=args.arm_layers,
        neuron_multiplier=args.neuron_multiplier,
        fused_kernels=args.fused_kernels,
        card_feature_dim=table.shape[1],
        # Same pattern as card_feature_dim: shapes fixed by the corpus are
        # read off the corpus. These size the two ContextFeatures embeddings,
        # so a set with a different pack geometry would index out of range.
        packs_per_draft=data.packs_per_draft,
        picks_per_pack=data.picks_per_pack,
    )
    steps = args.steps
    if args.epochs is not None:
        steps = max(1, int(round(args.epochs * train_indices.size / args.batch_size)))
        print(f"--epochs {args.epochs:g} over {train_indices.size:,} rows -> {steps:,} steps")

    epochs = steps * args.batch_size / max(train_indices.size, 1)
    if epochs > 2.0:
        print(
            f"  NOTE: this cell makes {epochs:.1f} passes over its training set. "
            "Beyond about one pass the D axis starts measuring repetition rather "
            "than data scale; see src/training/README.md."
        )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        total_steps=steps,
        eval_every=args.eval_every,
        warmup_steps=max(50, steps // 20),
        seed=args.seed,
    )

    print(f"\nTRAIN {args.arm} arm, d={args.width}, steps={args.steps}")
    result = train_model(
        data, table, train_indices, splits.val, model_config, train_config,
        arm=args.arm, checkpoint_dir=out_dir,
        resume=args.resume, max_seconds=args.max_seconds,
    )

    if not result["completed"]:
        # Stopped on the segment budget. The resume state in out_dir is the
        # artefact that matters here; writing metrics.json now would leave a
        # file that looks like a finished cell but holds a truncated curve.
        progress = {
            "completed": False,
            "stopped_at_step": result["stopped_at_step"],
            "total_steps": steps,
            "best_val_loss": result["best_val_loss"],
            "best_step": result["best_step"],
            "elapsed_s": result["elapsed_s"],
            "history": result["history"],
        }
        (out_dir / "progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )
        print(
            f"\nincomplete: {result['stopped_at_step']:,} of {steps:,} steps. "
            f"Re-run the same command with --resume to continue.\n"
            f"wrote {out_dir / 'progress.json'} and resume state"
        )
        return EXIT_INCOMPLETE

    # The saved checkpoint tracks best-val; make the final artefact agree.
    best_params = result["best_params"]
    print(
        f"\nbest val {result['best_val_loss']:.4f} at step {result['best_step']:,} "
        f"(final {result['final_val_loss']:.4f})"
    )

    metrics = {
        "arm": args.arm,
        "neuron_multiplier": args.neuron_multiplier,
        "fused_kernels": args.fused_kernels,
        "num_params": result["num_params"],
        "param_breakdown": result["param_breakdown"],
        "train_rows": int(train_indices.size),
        "train_drafts": int(np.unique(data.draft_idx[train_indices]).size),
        "data_fraction": args.data_fraction,
        "seed": args.seed,
        "completed": True,
        "steps": steps,
        "examples_seen": steps * args.batch_size,
        "epochs": epochs,
        "history": result["history"],
        "sampled_final_val_loss": result["final_val_loss"],
        "best_val_loss": result["best_val_loss"],
        "best_step": result["best_step"],
        "elapsed_s": result["elapsed_s"],
        "model_config": result["model_config"],
        "train_config": result["train_config"],
    }

    uniform = uniform_baseline(data, splits.val)
    freq_loss, freq_accuracy = frequency_baseline(data, train_indices, splits.val)
    metrics["baselines"] = {
        "uniform": uniform,
        "pick_rate_prior": freq_loss,
        "pick_rate_prior_accuracy": freq_accuracy,
    }
    print(
        f"baselines on val: uniform {uniform:.4f}, "
        f"pick-rate prior {freq_loss:.4f} (acc {freq_accuracy:.4f})"
    )

    if not args.skip_full_eval:
        print("\nexact evaluation over the full val split, by pick number:")
        by_pick = evaluate_by_pick(
            result["model"], best_params, table, data, splits.val, args.eval_batch_size,
        )
        print(format_by_pick(by_pick))
        summary = summarise_by_pick(by_pick)
        metrics["by_pick"] = by_pick
        metrics["summary"] = summary
        print(
            f"\nall picks     : loss {summary['all_picks']['loss']:.4f} "
            f"acc {summary['all_picks']['accuracy']:.4f} "
            f"({summary['all_picks']['rows']:,} rows)"
        )
        print(
            f"picks {summary['decision_picks']['picks']}    : "
            f"loss {summary['decision_picks']['loss']:.4f} "
            f"acc {summary['decision_picks']['accuracy']:.4f} "
            f"({summary['decision_picks']['rows']:,} rows)"
        )
        print(
            f"forced rows   : {summary['forced_rows']:,} "
            f"({100 * summary['forced_fraction']:.1f}% of val, loss identically 0)"
        )

    save_checkpoint(
        out_dir, best_params, model_config=model_config, arm=args.arm,
        train_config=train_config, metrics=metrics,
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    # Leftover from an earlier segment of this same cell; keeping it would
    # leave a finished run advertising itself as incomplete.
    (out_dir / "progress.json").unlink(missing_ok=True)
    print(f"\nwrote {out_dir / 'params.msgpack'} and {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
