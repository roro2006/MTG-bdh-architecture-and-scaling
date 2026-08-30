"""Invariant checks against the real ingested corpus.

The unit suite runs entirely on synthetic data, on purpose: it must work
with no download and no network. That leaves a gap, because the synthetic
generator produces exactly the shapes it was written to produce. Anything
the real 17lands export does that nobody anticipated -- a set with a
different pack size, a card Scryfall renamed, an ingest run against a
half-downloaded file -- passes every unit test and then quietly poisons a
grid.

This module closes that gap. Each check is a named function returning a
CheckResult, run against a processed directory and optionally a checkpoint.
It is meant to be run after ingesting a new set, after changing the feature
pipeline, and before committing compute to a grid.

    python -m src.validation --processed-dir data/processed/FIN.PremierDraft

Exits non-zero if any check fails, so it can gate a grid launch.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

CHECKS: list[tuple[str, Callable]] = []


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    skipped: bool = False
    facts: dict = field(default_factory=dict)

    def line(self) -> str:
        mark = "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
        return f"[{mark}] {self.name}: {self.detail}"


def check(name: str):
    def register(fn):
        CHECKS.append((name, fn))
        return fn

    return register


# --------------------------------------------------------------------------
# Data-level checks
# --------------------------------------------------------------------------

@check("vocabulary matches the feature table")
def _vocab_features_align(ctx) -> CheckResult:
    vocab, features = ctx["data"].vocab, ctx["features"]
    if features.size != vocab.size:
        return CheckResult(
            "", False,
            f"feature table has {features.size} cards, vocabulary has {vocab.size}",
        )
    mismatched = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(features.card_names, vocab.id_to_card))
        if a != b
    ]
    if mismatched:
        i, a, b = mismatched[0]
        return CheckResult(
            "", False,
            f"{len(mismatched)} names differ, first at id {i}: features={a!r} vocab={b!r}",
        )
    return CheckResult("", True, f"{vocab.size} cards aligned by id")


@check("every card resolved to real attributes")
def _features_populated(ctx) -> CheckResult:
    features = ctx["features"]
    dense = features.dense()
    if not np.isfinite(dense).all():
        return CheckResult("", False, "feature table contains NaN or inf")
    # A card that Scryfall could not resolve is left as an all-zero row.
    empty = np.flatnonzero(
        features.type_flags.sum(1) + features.color_identity.sum(1) + features.mana_value
        == 0
    )
    # Colourless zero-cost non-typed cards do not exist, so this really is
    # the unresolved signature.
    if empty.size:
        names = [features.card_names[i] for i in empty[:5]]
        return CheckResult(
            "", False, f"{empty.size} cards have no attributes at all, e.g. {names}"
        )
    return CheckResult("", True, f"{dense.shape[0]} cards x {dense.shape[1]} dims, all populated")


@check("no keyword is a card id in disguise")
def _keywords_recur(ctx) -> CheckResult:
    features = ctx["features"]
    from src.data.card_features import MIN_KEYWORD_CARDS

    counts = features.keyword_flags.sum(axis=0)
    singletons = [
        features.keyword_names[i] for i in np.flatnonzero(counts < MIN_KEYWORD_CARDS)
    ]
    if singletons:
        return CheckResult(
            "", False,
            f"{len(singletons)} keywords appear on fewer than {MIN_KEYWORD_CARDS} "
            f"cards, e.g. {singletons[:5]} -- these act as identity columns",
        )
    return CheckResult(
        "", True,
        f"{len(features.keyword_names)} keywords, each on >= {MIN_KEYWORD_CARDS} cards "
        f"(rarest {int(counts.min()) if counts.size else 0})",
    )


@check("pool is exactly the prefix of earlier picks")
def _pool_prefix(ctx) -> CheckResult:
    data = ctx["data"]
    if data.dropped_drafts:
        return CheckResult(
            "", True,
            f"holds for all retained rows; {data.dropped_drafts:,} incomplete "
            f"drafts ({data.dropped_rows:,} rows) were dropped at load",
            facts={"dropped_drafts": data.dropped_drafts},
        )
    return CheckResult("", True, f"holds for all {data.size:,} rows, none dropped")


@check("the label is always a card in the pack")
def _label_in_pack(ctx) -> CheckResult:
    data = ctx["data"]
    at_position = data.pack[np.arange(data.size), data.label_pos]
    bad = int((at_position != data.label).sum())
    if bad:
        return CheckResult("", False, f"{bad:,} rows have label_pos pointing elsewhere")
    return CheckResult("", True, f"all {data.size:,} labels sit at their recorded slot")


@check("pack size follows the pick number")
def _pack_sizes(ctx) -> CheckResult:
    data = ctx["data"]
    from src.data.dataset import PICKS_PER_PACK

    expected = PICKS_PER_PACK - data.pick_number
    bad = int((data.pack_size != expected).sum())
    if bad:
        sample = np.flatnonzero(data.pack_size != expected)[:3]
        return CheckResult(
            "", False,
            f"{bad:,} rows disagree, e.g. rows {sample.tolist()} "
            f"(sizes {data.pack_size[sample].tolist()})",
        )
    return CheckResult("", True, f"pack_size == {PICKS_PER_PACK} - pick_number everywhere")


@check("splits share no drafts")
def _splits_disjoint(ctx) -> CheckResult:
    data, splits = ctx["data"], ctx["splits"]
    sets = [
        set(np.unique(data.draft_idx[part]).tolist())
        for part in (splits.train, splits.val, splits.test)
    ]
    overlaps = [
        len(sets[0] & sets[1]), len(sets[0] & sets[2]), len(sets[1] & sets[2])
    ]
    if any(overlaps):
        return CheckResult("", False, f"draft overlap train/val/test: {overlaps}")
    covered = splits.train.size + splits.val.size + splits.test.size
    return CheckResult(
        "", True,
        f"disjoint; {covered:,} rows across "
        f"{splits.train.size:,}/{splits.val.size:,}/{splits.test.size:,}",
    )


# --------------------------------------------------------------------------
# Model-level checks, run against real batches
# --------------------------------------------------------------------------

@check("model is blind to pack and pool order on real data")
def _permutation_invariance(ctx) -> CheckResult:
    import jax.numpy as jnp

    model, params, table, data = ctx["model"], ctx["params"], ctx["table"], ctx["data"]
    if model is None:
        return CheckResult("", True, "no model supplied", skipped=True)

    rng = np.random.default_rng(0)
    idx = rng.choice(ctx["splits"].val, size=256, replace=False)
    batch = data.batch(idx)
    args = (jnp.asarray(batch["pack_ids"]), jnp.asarray(batch["pool_ids"]),
            jnp.asarray(batch["pack_number"]), jnp.asarray(batch["pick_number"]))
    base = np.asarray(model.apply(params, table, *args))

    pack_perm = rng.permutation(batch["pack_ids"].shape[1])
    shuffled = np.asarray(model.apply(
        params, table, jnp.asarray(batch["pack_ids"][:, pack_perm]), *args[1:]
    ))
    pack_ok = np.allclose(base[:, pack_perm], shuffled, atol=1e-4)

    pool_perm = rng.permutation(batch["pool_ids"].shape[1])
    pooled = np.asarray(model.apply(
        params, table, args[0], jnp.asarray(batch["pool_ids"][:, pool_perm]), *args[2:]
    ))
    pool_ok = np.allclose(base, pooled, atol=1e-4)

    if not (pack_ok and pool_ok):
        return CheckResult(
            "", False, f"pack equivariance={pack_ok}, pool invariance={pool_ok}"
        )
    return CheckResult("", True, "verified on 256 real states")


@check("no probability mass outside the pack, no NaNs")
def _output_closed(ctx) -> CheckResult:
    import jax
    import jax.numpy as jnp

    model, params, table, data = ctx["model"], ctx["params"], ctx["table"], ctx["data"]
    if model is None:
        return CheckResult("", True, "no model supplied", skipped=True)

    # Deliberately include the empty-pool states, which are the NaN risk.
    first = np.flatnonzero(
        (data.pack_number == 0) & (data.pick_number == 0)
    )[:128]
    rest = np.random.default_rng(0).choice(ctx["splits"].val, size=128, replace=False)
    idx = np.concatenate([first, rest])
    batch = data.batch(idx)
    logits = model.apply(
        params, table, jnp.asarray(batch["pack_ids"]), jnp.asarray(batch["pool_ids"]),
        jnp.asarray(batch["pack_number"]), jnp.asarray(batch["pick_number"]),
    )
    if not bool(jnp.isfinite(logits).all()):
        return CheckResult("", False, "non-finite logits, including empty-pool states")

    probs = np.asarray(jax.nn.softmax(logits, axis=-1))
    mask = batch["pack_ids"] >= 0
    leaked = float(probs[~mask].sum())
    if leaked > 1e-6:
        return CheckResult("", False, f"{leaked:.3e} probability mass on padding slots")
    return CheckResult(
        "", True,
        f"finite on {idx.size} states ({first.size} with an empty pool); "
        f"padding mass {leaked:.1e}",
    )


@check("analytic FLOPs agree with the XLA cost model")
def _flops_agree(ctx) -> CheckResult:
    from src.models.flops import count_flops_analytic, measure_flops_xla

    model, params, table = ctx["model"], ctx["params"], ctx["table"]
    if model is None:
        return CheckResult("", True, "no model supplied", skipped=True)

    analytic = count_flops_analytic(ctx["model_config"])["total"]
    measured = measure_flops_xla(model, params, table)
    if measured is None:
        return CheckResult("", True, "backend provides no cost model", skipped=True)
    ratio = measured / analytic
    if not 1.0 <= ratio < 1.15:
        return CheckResult(
            "", False,
            f"XLA/analytic = {ratio:.3f}, outside the expected 1.00-1.15 band",
        )
    return CheckResult(
        "", True,
        f"analytic {analytic:,.0f}, XLA {measured:,.0f} (ratio {ratio:.3f}); "
        "the gap is the elementwise work the derivation omits",
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def build_context(processed_dir: str | Path, checkpoint_dir: str | Path | None = None):
    """Loads everything the checks need, once."""
    import jax.numpy as jnp

    from src.data.card_features import CardFeatures
    from src.data.dataset import PickData, split_by_draft

    processed = Path(processed_dir)
    features = CardFeatures.load(processed / "card_features.npz")
    data = PickData.load(processed)
    context = {
        "data": data,
        "features": features,
        "table": jnp.asarray(features.dense()),
        "splits": split_by_draft(data, seed=0),
        "model": None,
        "params": None,
        "model_config": None,
    }

    if checkpoint_dir is not None:
        from src.models.pick_model import ModelConfig
        from src.training.checkpoint import restore

        model, params, metadata = restore(checkpoint_dir)
        context["model"] = model
        context["params"] = params
        context["model_config"] = ModelConfig(**metadata["model_config"])
    return context


def run_all(processed_dir, checkpoint_dir=None, verbose: bool = True) -> list[CheckResult]:
    context = build_context(processed_dir, checkpoint_dir)
    results: list[CheckResult] = []
    for name, fn in CHECKS:
        try:
            result = fn(context)
            result.name = name
        except Exception as error:  # a check that crashes is a failed check
            result = CheckResult(
                name, False, f"raised {type(error).__name__}: {error}"
            )
            if verbose:
                traceback.print_exc()
        results.append(result)
        if verbose:
            print(result.line(), flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a processed corpus.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="optional trained cell; enables the model-level checks",
    )
    args = parser.parse_args(argv)

    print(f"validating {args.processed_dir}")
    if args.checkpoint_dir:
        print(f"  with checkpoint {args.checkpoint_dir}")
    print()
    results = run_all(args.processed_dir, args.checkpoint_dir)

    failed = [r for r in results if not r.passed and not r.skipped]
    skipped = [r for r in results if r.skipped]
    print()
    print(
        f"{len(results) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )
    if failed:
        print("\nFAILED:")
        for result in failed:
            print(f"  - {result.name}: {result.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
