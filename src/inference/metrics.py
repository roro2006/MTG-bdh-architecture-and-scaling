"""Drafter metrics: top-1, top-3, and whether the probabilities mean anything.

PROJECT_PLAN section 7, "Drafter metrics, not classifier metrics": a tool
that shows a ranked list is judged on top-3 as well as top-1, and on
calibration, and both are reported. `src/training/evaluate.py` already
reports loss and top-1 for the training loop; this is the same split seen
as a ranking rather than as a classification.

Two things about this task make naive top-k and naive calibration read far
better than the model deserves, and both are the same hazard the per-pick
breakdown exists for:

  - Top-k is trivially 1.0 whenever the pack holds k cards or fewer. 7.1%
    of FIN's rows are one-card packs, where top-1 is 100% for any model at
    all, and top-3 collects every pack of three as well -- 21.4% of rows.
    A top-3 number over all picks is therefore mostly a measurement of the
    corpus's pack-size distribution.

  - Calibration is worse. A one-card pack has probability exactly 1.0 on
    the only legal answer and is always right, which is a perfectly
    calibrated bin at the far end of the confidence axis. Enough of those
    drag the expected calibration error toward zero no matter what the
    model does with the packs that need a decision.

So every number here is reported twice: over all picks, and over the
picks-0-8 slice where the pack still holds six or more cards. The headline
is picks-0-8, exactly as `summarise_by_pick` has it. `trivial_fraction`
says how much of the all-picks figure was forced, so the gap between the
two columns is never mysterious.

Calibration is measured on the top-1 confidence -- the probability the
model put on the card it would actually recommend -- because that is the
number a person reading the ranked list would act on.
"""

from __future__ import annotations

import numpy as np

# Ten equal-width bins over [0, 1] is the usual convention for ECE and is
# what makes a number here comparable to a published one. It is also about
# as fine as an 8k-row sample supports: past this the tail bins hold too
# few rows for their accuracy to mean anything.
DEFAULT_BINS = 10

# Matches evaluate.summarise_by_pick: picks 0-8 are where the pack still
# holds six or more cards and a real decision exists.
DECISION_PICKS = 9

DEFAULT_BATCH = 1024


def ranked_probabilities(
    probe, data, rows: np.ndarray, batch_size: int = DEFAULT_BATCH
) -> dict[str, np.ndarray]:
    """Per-row rank of the human's pick, and the model's top-1 confidence.

    Returns the three arrays every metric below is derived from, so the
    model runs once however many ways the result is later sliced:

      `label_rank`  1 for a top-1 agreement, 2 for the runner-up, ...
      `confidence`  probability on the model's own first choice.
      `pack_size`   cards actually on offer, which is what makes a top-k
                    number trivial or not.
    """
    rows = np.asarray(rows, dtype=np.int64)
    label_rank = np.empty(rows.size, dtype=np.int32)
    confidence = np.empty(rows.size, dtype=np.float64)
    label_prob = np.empty(rows.size, dtype=np.float64)

    for start in range(0, rows.size, batch_size):
        chunk = rows[start : start + batch_size]
        batch = data.batch(chunk)
        probs = probe.probabilities(
            batch["pack_ids"], batch["pool_ids"],
            batch["pack_number"], batch["pick_number"],
        )
        label = batch["label_pos"].astype(np.int64)
        taken = probs[np.arange(chunk.size), label]
        # Rank by strictly-greater count, so the human's pick ranks 1 when
        # nothing beats it. Counting ties as beating it would report a
        # model that splits its mass evenly as worse than it is.
        rank = (probs > taken[:, None]).sum(axis=1) + 1
        label_rank[start : start + chunk.size] = rank
        confidence[start : start + chunk.size] = probs.max(axis=1)
        label_prob[start : start + chunk.size] = taken

    return {
        "label_rank": label_rank,
        "confidence": confidence,
        "label_prob": label_prob,
        "pack_size": data.pack_size[rows].astype(np.int32),
        "pick_number": data.pick_number[rows].astype(np.int32),
    }


def calibration(
    confidence: np.ndarray, correct: np.ndarray, bins: int = DEFAULT_BINS
) -> dict:
    """Expected and maximum calibration error, plus the reliability table.

    ECE is the row-weighted mean gap between a bin's mean confidence and
    its accuracy. The signed `overconfidence` is kept alongside it because
    ECE is an absolute value and therefore cannot say which way the model
    is wrong -- and for a drafting tool "claims 80%, is right 60% of the
    time" and the reverse call for opposite responses.

    Empty bins are dropped rather than counted as zero error: a bin with no
    rows is not evidence of good calibration.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidence.size == 0:
        return {"ece": float("nan"), "mce": float("nan"),
                "overconfidence": float("nan"), "bins": []}

    edges = np.linspace(0.0, 1.0, bins + 1)
    # `right=True` with a clip keeps confidence exactly 1.0 in the last bin
    # rather than in a bin of its own past the final edge.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)

    table = []
    ece = 0.0
    mce = 0.0
    for b in range(bins):
        selected = index == b
        n = int(selected.sum())
        if not n:
            continue
        mean_conf = float(confidence[selected].mean())
        accuracy = float(correct[selected].mean())
        gap = abs(mean_conf - accuracy)
        ece += (n / confidence.size) * gap
        mce = max(mce, gap)
        table.append({
            "low": float(edges[b]),
            "high": float(edges[b + 1]),
            "rows": n,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
            "gap": float(mean_conf - accuracy),
        })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "overconfidence": float(confidence.mean() - correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "bins": table,
    }


def ranking_metrics(
    scored: dict, mask: np.ndarray | None = None,
    top_k: tuple[int, ...] = (1, 3), bins: int = DEFAULT_BINS,
) -> dict:
    """Top-k accuracy and calibration over the rows `mask` selects."""
    rank = scored["label_rank"]
    pack_size = scored["pack_size"]
    if mask is not None:
        rank, pack_size = rank[mask], pack_size[mask]
        confidence = scored["confidence"][mask]
    else:
        confidence = scored["confidence"]

    if rank.size == 0:
        return {"rows": 0}

    result: dict = {"rows": int(rank.size)}
    for k in top_k:
        result[f"top{k}"] = float((rank <= k).mean())
        # How much of that number came free from packs too small to get
        # wrong. Reported per k because it differs sharply between them.
        result[f"top{k}_trivial_fraction"] = float((pack_size <= k).mean())
    result["mean_pack_size"] = float(pack_size.mean())
    result["calibration"] = calibration(confidence, rank == 1, bins=bins)
    return result


def ranking_report(
    probe, data, rows: np.ndarray,
    decision_picks: int = DECISION_PICKS,
    top_k: tuple[int, ...] = (1, 3),
    bins: int = DEFAULT_BINS,
    batch_size: int = DEFAULT_BATCH,
) -> dict:
    """The reported thing: all-picks and picks-0-8, from one pass.

    Shaped like `evaluate.summarise_by_pick`'s output so the two can be
    read side by side; `decision_picks` means the same thing in both.
    """
    rows = np.asarray(rows, dtype=np.int64)
    scored = ranked_probabilities(probe, data, rows, batch_size=batch_size)
    decisions = scored["pick_number"] < decision_picks

    return {
        "all_picks": ranking_metrics(scored, None, top_k=top_k, bins=bins),
        "decision_picks": ranking_metrics(scored, decisions, top_k=top_k, bins=bins)
        | {"picks": f"0-{decision_picks - 1}"},
    }


def format_ranking_metrics(report: dict) -> str:
    """The two slices as a table, headline slice last and labelled as such."""
    lines = ["ranked-pick metrics"]
    for key, title in (("all_picks", "all picks"),
                       ("decision_picks", "picks 0-8  <- headline")):
        block = report.get(key)
        if not block or not block.get("rows"):
            continue
        cal = block["calibration"]
        lines.append(f"  {title}  ({block['rows']:,} rows, "
                     f"mean pack {block['mean_pack_size']:.1f})")
        for k in (1, 3):
            if f"top{k}" not in block:
                continue
            lines.append(
                f"    top-{k}   {block[f'top{k}']:.4f}"
                f"   ({block[f'top{k}_trivial_fraction']:.1%} of rows have a "
                f"pack of {k} or fewer and cannot be got wrong)"
            )
        lines.append(
            f"    calibration  ECE {cal['ece']:.4f}  MCE {cal['mce']:.4f}"
            f"   mean confidence {cal['mean_confidence']:.4f} vs accuracy "
            f"{cal['accuracy']:.4f} ({cal['overconfidence']:+.4f})"
        )
    return "\n".join(lines)


def format_reliability(calibration_result: dict) -> str:
    """The reliability table, for when a single ECE is not enough."""
    lines = [f"{'confidence':>18} {'rows':>8} {'mean conf':>10} {'accuracy':>9} {'gap':>8}"]
    for row in calibration_result["bins"]:
        lines.append(
            f"{row['low']:>8.2f}-{row['high']:<9.2f} {row['rows']:>8,} "
            f"{row['mean_confidence']:>10.4f} {row['accuracy']:>9.4f} {row['gap']:>+8.4f}"
        )
    return "\n".join(lines)
