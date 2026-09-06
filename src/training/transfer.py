"""Zero-shot cross-set evaluation: train on N sets, draft a set never seen.

    python -m src.training.transfer --held-out FIN --arm attention \
        --width 64 --steps 92000 --out-dir runs/transfer_attention_d64

This is docs/PROJECT_PLAN.md section 10 step 6, and the only headline in the
project that the scaling grid cannot produce: every grid cell trains and
evaluates on the same set, so none of them says whether the model learned to
draft or learned to draft *Final Fantasy*.

What makes it possible without a single new parameter
-----------------------------------------------------
`CardEmbedding` is an MLP over a card's structured attributes and
`PointerHead` scores pack slots, so nothing in `PickModel` is indexed by
vocabulary id (see src/data/multiset.py for the full argument). The work
here is entirely data plumbing: one shared card index across sets, one
concatenated corpus, and a training index that provably excludes every row
of the held-out set.

The comparison this run is for
------------------------------
`--held-out FIN` by default, because FIN's same-set ceiling is already
measured at this width: runs/attention_d64_s92000 reports 1.0033 loss and
62.39% accuracy on picks 0-8 of the FIN val split, from a model trained on
FIN's own train split. This run evaluates on *those same rows* -- the split
is drawn per set before concatenation precisely so that it can -- with the
same width, the same optimiser, the same number of steps and the same batch
size. The only difference is which drafts the gradient came from, so the gap
between the two numbers is the transfer gap and nothing else.

Two things about that comparison to keep honest, both reported in
metrics.json rather than left to the reader:

  - The held-out SET is unseen; not all of its cards are. Basic lands and
    the occasional reprint appear in the training sets too. `card_overlap`
    in the metrics says how many.
  - The transfer model is trained on more data than the same-set model, and
    at 92,000 steps x 512 it makes far fewer passes over it. Matching steps
    rather than epochs is the deliberate choice: it holds the optimisation
    budget fixed, which is what "same model, different data" has to mean
    for the gap to be attributable to the data.

AFR is not among the training sets and cannot be. Its export omits every
draft's first pick, so every pool reconstructed from it is short one card;
docs/DATA.md records the exclusion. Holding out FIN therefore leaves eight
training sets, not nine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..data.dataset import decision_rows
from ..data.multiset import load_multiset, processed_dirs, set_code
from ..models.pick_model import ModelConfig
from .checkpoint import save_checkpoint
from .evaluate import evaluate_by_pick, format_by_pick, summarise_by_pick
from .run import EXIT_INCOMPLETE, subsample_by_draft
from .train import TrainConfig, frequency_baseline, train_model, uniform_baseline

# Picks 0-8: where the pack still holds six or more cards and a real
# decision exists. `summarise_by_pick` uses the same cut for the model, so
# the baselines have to as well.
DECISION_PICKS = 9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train on several sets and evaluate on a held-out one."
    )
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument(
        "--held-out", default="FIN",
        help="set code evaluated on and excluded from training (default FIN, "
             "whose same-set ceiling at d=64 is already measured)",
    )
    parser.add_argument(
        "--train-sets", nargs="*", default=None,
        help="set codes to train on. Default: every ingested set that loads, "
             "minus the held-out one.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--arm", default="attention", choices=["attention", "bdh"])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--arm-layers", type=int, default=2)
    parser.add_argument("--pool-layers", type=int, default=2)
    parser.add_argument("--pack-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--neuron-multiplier", type=int, default=4)
    parser.add_argument("--fused-kernels", action="store_true")
    parser.add_argument(
        "--steps", type=int, default=92000,
        help="default matches runs/*_d64_s92000, the same-set reference this "
             "run is compared against",
    )
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0,
                        help="must stay 0 to evaluate the same val rows as the "
                             "single-set reference run")
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--skip-full-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.processed_root)
    available = [p.name for p in processed_dirs(root)]
    held_out_name = next(
        (n for n in available if set_code(n) == set_code(args.held_out)), None
    )
    if held_out_name is None:
        raise SystemExit(
            f"--held-out {args.held_out} is not ingested under {root} "
            f"(have {sorted(set_code(n) for n in available)})"
        )
    # An explicitly named set that will not load is an error; "everything
    # ingested" is a request that has to tolerate one, because AFR is
    # ingested and unusable (docs/DATA.md).
    explicit = bool(args.train_sets)
    if explicit:
        train_names = [p.name for p in processed_dirs(root, args.train_sets)]
    else:
        train_names = [n for n in available if n != held_out_name]
    if held_out_name in train_names:
        raise SystemExit(
            f"{set_code(held_out_name)} is both held out and in --train-sets; "
            "that would make the evaluation meaningless"
        )

    print(
        f"held out : {set_code(held_out_name)}\n"
        f"train on : {', '.join(set_code(n) for n in train_names)} "
        f"({len(train_names)} sets, before any that fail to load)\n"
    )
    corpus = load_multiset(
        [root / n for n in [*train_names, held_out_name]],
        split_seed=args.split_seed,
        skip_unloadable=not explicit,
    )
    skipped = [n for n, _ in corpus.skipped]
    if held_out_name in skipped:
        raise SystemExit(
            f"the held-out set {set_code(held_out_name)} does not load; there is "
            "nothing to evaluate on"
        )
    train_names = [n for n in train_names if n not in skipped]
    if not train_names:
        raise SystemExit("no training sets left after skipping the unloadable ones")
    data = corpus.data
    print("\n" + data.describe() + "\n")

    table = jnp.asarray(corpus.feature_table)
    val_indices = corpus.splits[held_out_name].val

    train_pool = corpus.rows(train_names, "train")
    train_indices = decision_rows(data, train_pool)
    train_indices = subsample_by_draft(
        data, train_indices, args.data_fraction, seed=args.seed
    )

    # The whole experiment rests on this: not one gradient may come from a
    # row of the held-out set. Cheap to assert, catastrophic to get wrong,
    # and an off-by-one in the row offsets would not show up any other way.
    held = data.slice_of(held_out_name)
    leaked = int(
        ((train_indices >= held.row_start) & (train_indices < held.row_stop)).sum()
    )
    if leaked:
        raise AssertionError(
            f"{leaked:,} training rows fall inside the held-out set's row range "
            f"[{held.row_start:,}, {held.row_stop:,}); the split is leaking"
        )
    if not (
        (val_indices >= held.row_start) & (val_indices < held.row_stop)
    ).all():
        raise AssertionError("val rows are not all from the held-out set")

    overlap = corpus.card_index.overlap(held_out_name, train_names)
    print(
        f"\ntrained on: {', '.join(set_code(n) for n in train_names)} "
        f"({len(train_names)} sets)"
        + (f", skipped {', '.join(set_code(n) for n in skipped)}" if skipped else "")
    )
    print(
        f"train rows: {train_indices.size:,} (decision picks only, "
        f"{args.data_fraction:g} of {train_pool.size:,} split rows)\n"
        f"val rows  : {val_indices.size:,}, all {set_code(held_out_name)}\n"
        f"cards     : {corpus.card_index.size:,} in the shared index; "
        f"{overlap['cards_also_in_training_sets']} of "
        f"{set_code(held_out_name)}'s {overlap['held_out_cards']} "
        f"({overlap['fraction_seen']:.1%}) are also printed in a training set"
    )

    model_config = ModelConfig(
        hidden_dim=args.width,
        num_heads=args.num_heads,
        pool_encoder_layers=args.pool_layers,
        pack_encoder_layers=args.pack_layers,
        arm_layers=args.arm_layers,
        neuron_multiplier=args.neuron_multiplier,
        fused_kernels=args.fused_kernels,
        card_feature_dim=int(table.shape[1]),
        # The ENVELOPE geometry, not the held-out set's: these size the two
        # ContextFeatures embeddings, and a training row from a 3x15 set
        # would index out of range on anything smaller.
        packs_per_draft=data.packs_per_draft,
        picks_per_pack=data.picks_per_pack,
    )

    steps = args.steps
    if args.epochs is not None:
        steps = max(1, int(round(args.epochs * train_indices.size / args.batch_size)))
        print(f"--epochs {args.epochs:g} -> {steps:,} steps")
    epochs = steps * args.batch_size / max(train_indices.size, 1)

    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        total_steps=steps,
        eval_every=args.eval_every,
        warmup_steps=max(50, steps // 20),
        seed=args.seed,
    )

    print(
        f"\nTRANSFER {args.arm} arm, d={args.width}, steps={steps:,} "
        f"({epochs:.2f} passes over the training rows)\n"
        f"the val curve below is already the zero-shot number: it is measured "
        f"on {set_code(held_out_name)}, which contributes no gradient\n"
    )
    result = train_model(
        data, table, train_indices, val_indices, model_config, train_config,
        arm=args.arm, checkpoint_dir=out_dir,
        resume=args.resume, max_seconds=args.max_seconds,
    )

    if not result["completed"]:
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
            "Re-run the same command with --resume to continue."
        )
        return EXIT_INCOMPLETE

    best_params = result["best_params"]
    print(
        f"\nbest zero-shot val {result['best_val_loss']:.4f} at step "
        f"{result['best_step']:,} (final {result['final_val_loss']:.4f})"
    )

    metrics = {
        "experiment": "zero_shot_cross_set",
        "arm": args.arm,
        "neuron_multiplier": args.neuron_multiplier,
        "fused_kernels": args.fused_kernels,
        "num_params": result["num_params"],
        "param_breakdown": result["param_breakdown"],
        "train_rows": int(train_indices.size),
        "train_drafts": int(np.unique(data.draft_idx[train_indices]).size),
        "data_fraction": args.data_fraction,
        "seed": args.seed,
        "split_seed": args.split_seed,
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
        "transfer": {
            "held_out_set": set_code(held_out_name),
            "train_sets": [set_code(n) for n in train_names],
            "excluded_sets": [
                {"set": set_code(n), "reason": reason} for n, reason in corpus.skipped
            ],
            "shared_card_index_size": corpus.card_index.size,
            "reprints_in_index": len(corpus.card_index.reprints()),
            "card_overlap": overlap,
            "corpus_geometry": data.geometry.to_dict(),
            "per_set": [
                {
                    "set": s.code,
                    "rows": s.rows,
                    "drafts": s.n_drafts,
                    "cards": s.vocab_size,
                    "geometry": s.geometry.to_dict(),
                    "role": "held_out" if s.name == held_out_name else "train",
                }
                for s in data.sets
            ],
        },
    }

    # Both slices, because the headline is picks 0-8 and the baselines are
    # usually quoted over all picks. PROJECT_PLAN.md section 8 is explicit
    # about it: on FIN the pick-rate prior gets 45.3% over all picks and
    # 36.3% over picks 0-8, so comparing a picks-0-8 model number against an
    # all-picks baseline hands the model ~0.4 nats it did not earn.
    #
    # Both priors are counted over the UNFILTERED train splits, forced picks
    # included, because that is how run.py counts the single-set one and the
    # published FIN figures (1.5662 all picks, 1.9474 picks 0-8) come from
    # there. Keeping the definition identical makes `pick_rate_prior_same_set`
    # below reproduce those two numbers exactly, which is a end-to-end check
    # on the concatenation: it can only come out right if FIN's rows, ids and
    # splits all survived the merge unchanged.
    same_set_pool = corpus.splits[held_out_name].train
    slices = {
        "all_picks": val_indices,
        "decision_picks": val_indices[data.pick_number[val_indices] < DECISION_PICKS],
    }
    baselines: dict[str, dict] = {}
    for slice_name, rows in slices.items():
        transferred_loss, transferred_acc = frequency_baseline(
            data, train_indices, rows
        )
        same_set_loss, same_set_acc = frequency_baseline(data, same_set_pool, rows)
        baselines[slice_name] = {
            "rows": int(rows.size),
            "uniform": uniform_baseline(data, rows),
            # Card quality learned from the TRAINING sets and applied blind
            # to the held-out set. Cards it has never seen fall back to the
            # Laplace prior, which is the honest zero-shot version of this
            # baseline and the floor the transferred model has to clear.
            "pick_rate_prior_transferred": transferred_loss,
            "pick_rate_prior_transferred_accuracy": transferred_acc,
            # The same prior fitted on the held-out set's own train split.
            # Not available to the model; reported because it brackets the
            # result -- a per-set statistic a zero-shot model has no right
            # to match.
            "pick_rate_prior_same_set": same_set_loss,
            "pick_rate_prior_same_set_accuracy": same_set_acc,
        }
    metrics["baselines"] = baselines
    # Kept flat as well, so anything reading a single-set metrics.json for
    # `baselines.uniform` still finds the all-picks number where it was.
    metrics["baselines"]["uniform"] = baselines["all_picks"]["uniform"]

    print(f"\nbaselines on the {set_code(held_out_name)} val split:")
    for slice_name, row in baselines.items():
        if not isinstance(row, dict):
            continue
        print(
            f"  {slice_name:<15} ({row['rows']:>7,} rows) uniform "
            f"{row['uniform']:.4f} | transferred pick-rate prior "
            f"{row['pick_rate_prior_transferred']:.4f} "
            f"(acc {row['pick_rate_prior_transferred_accuracy']:.4f}) | "
            f"same-set pick-rate prior {row['pick_rate_prior_same_set']:.4f} "
            f"(acc {row['pick_rate_prior_same_set_accuracy']:.4f})"
        )

    if not args.skip_full_eval:
        print(
            f"\nexact evaluation over the full {set_code(held_out_name)} val "
            "split, by pick number:"
        )
        by_pick = evaluate_by_pick(
            result["model"], best_params, table, data, val_indices,
            args.eval_batch_size,
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
    (out_dir / "progress.json").unlink(missing_ok=True)
    print(f"\nwrote {out_dir / 'params.msgpack'} and {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
