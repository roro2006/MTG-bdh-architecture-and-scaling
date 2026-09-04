"""Take a pack and a pool, return ranked picks.

The PROJECT_PLAN section 8 deliverable. Everything below the surface
already existed -- `PickProbe` restores a checkpoint and turns pack/pool id
arrays into a distribution over pack slots -- so this module is about the
two things that stand between that and something a person can use: names
instead of ids, and a ranked list instead of an argmax.

Names are not a convenience. A caller holding card *ids* has to have got
them from this corpus's `vocab.json`, and if they got them from anywhere
else -- a different set, a re-ingest that reordered the vocabulary, a
hand-typed guess -- the model scores a pack of entirely different cards and
returns a confident, ranked, completely wrong answer. There is no shape
error to catch it, because every id in range is a legal card. So the entry
point takes names and refuses the ones it does not recognise.

Two more silent-failure guards, both of which the id-level API cannot make:

  - A pack longer than the corpus geometry is rejected rather than
    truncated. `_pad` drops the overflow, which would rank a pack the
    caller did not ask about while looking entirely normal.

  - The pool size, the pack number and the pick number are not independent.
    dataset.py's pool-as-prefix identity fixes
    `pool_size == pack_number * picks_per_pack + pick_number` for every row
    the model ever trained on. So the pack/pick number is *derived* from
    the pool by default and only has to be passed to override it; passing
    an inconsistent pair asks the model about a state that does not occur
    in a draft, which it will answer anyway.

Usage:

    python -m src.inference.drafter \
        --checkpoint runs/bdh_d64_s92000 \
        --processed-dir data/processed/FIN.PremierDraft \
        --pack "Vivi Ornitier" "Tifa, Martial Artist" \
        --pool "Cloud, Midgar Mercenary"
"""

from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.dataset import split_by_draft
from .probe import PickProbe

# How many spelling suggestions to offer for an unrecognised card name.
# Enough to catch a typo or a missing accent, few enough that the error
# stays readable when a caller passes a whole pack of names from the wrong
# set and every one of them misses.
NAME_SUGGESTIONS = 3


class UnknownCardError(ValueError):
    """A card name that is not in this corpus's vocabulary.

    Raised rather than skipped or defaulted: a pack with a card silently
    dropped from it is a different pack, and the ranking over it would look
    entirely well-formed.
    """


@dataclass(frozen=True)
class RankedPick:
    """One candidate, and where the model put it."""

    rank: int
    card: str
    card_id: int
    probability: float
    pack_position: int

    def __str__(self) -> str:
        return f"{self.rank}. {self.card}  p={self.probability:.3f}"


@dataclass
class PickRanking:
    """Every card in the pack, best first.

    The whole pack rather than the top few, because the interesting
    question about a drafting model is usually where it put the card you
    disagree with, and a truncated list cannot answer it. Print with
    `summary(top=k)` when only the head is wanted.
    """

    picks: tuple[RankedPick, ...]
    pack_number: int
    pick_number: int
    pool_size: int
    pool: tuple[str, ...] = field(default=(), repr=False)

    @property
    def best(self) -> RankedPick:
        return self.picks[0]

    def top(self, k: int = 3) -> tuple[RankedPick, ...]:
        return self.picks[:k]

    def probability_of(self, card: str) -> float:
        """Total probability on `card`, summed over duplicate copies.

        A pack can hold two copies of the same common, and they are
        separate slots with separate scores. Asking "what does the model
        think of this card" wants their sum, not whichever slot happened to
        rank higher.
        """
        matches = [p.probability for p in self.picks if p.card == card]
        if not matches:
            raise UnknownCardError(f"{card!r} is not in this pack")
        return float(sum(matches))

    def to_dict(self) -> dict:
        return {
            "pack_number": self.pack_number,
            "pick_number": self.pick_number,
            "pool_size": self.pool_size,
            "picks": [
                {
                    "rank": p.rank,
                    "card": p.card,
                    "card_id": p.card_id,
                    "probability": p.probability,
                    "pack_position": p.pack_position,
                }
                for p in self.picks
            ],
        }

    def summary(self, top: int | None = None) -> str:
        shown = self.picks if top is None else self.picks[:top]
        lines = [
            f"pack {self.pack_number + 1} pick {self.pick_number + 1}"
            f"  ({len(self.picks)} on offer, {self.pool_size} in pool)"
        ]
        for pick in shown:
            lines.append(f"   {pick.rank}. {pick.card[:40]:<40} p={pick.probability:.3f}")
        if top is not None and len(self.picks) > top:
            lines.append(f"   ... and {len(self.picks) - top} more")
        return "\n".join(lines)


class Drafter:
    """A trained checkpoint that ranks the cards in a pack, by name."""

    def __init__(self, probe: PickProbe):
        self.probe = probe
        self.vocab = probe.vocab
        self.geometry = probe.geometry
        # Case-folded index, built once. Used only where the fold is
        # unambiguous -- see `card_id`.
        self._folded: dict[str, list[str]] = {}
        for name in self.vocab.id_to_card:
            self._folded.setdefault(name.casefold(), []).append(name)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path, processed_dir: str | Path
    ) -> tuple["Drafter", "object"]:
        """Returns the drafter and the corpus it was pointed at.

        The corpus comes back because the caller almost always wants it too
        -- to draw a val split for `src.inference.metrics`, or to pull a
        real pack to rank -- and loading it twice means paying for the
        pick arrays twice.
        """
        probe, data = PickProbe.from_checkpoint(checkpoint_dir, processed_dir)
        return cls(probe), data

    # -- names --------------------------------------------------------------

    def card_id(self, name: str) -> int:
        """The vocabulary id for `name`, or a useful error.

        Exact match first. A case-only miss is then accepted, but only when
        the fold picks out exactly one card: two cards differing in case
        alone would make the fold a coin flip, which is the mis-indexing
        this method exists to prevent. Anything else raises, with the
        nearest spellings attached -- the usual cause is a missing accent
        or a dropped subtitle after the comma.
        """
        try:
            return self.vocab.card_to_id[name]
        except KeyError:
            pass

        folded = self._folded.get(name.casefold(), [])
        if len(folded) == 1:
            return self.vocab.card_to_id[folded[0]]

        suggestions = difflib.get_close_matches(
            name, self.vocab.id_to_card, n=NAME_SUGGESTIONS, cutoff=0.6
        )
        hint = f" Did you mean {', '.join(repr(s) for s in suggestions)}?" if suggestions else ""
        if len(folded) > 1:
            hint = (
                f" {len(folded)} cards differ from it only in case "
                f"({', '.join(repr(f) for f in folded[:NAME_SUGGESTIONS])}); "
                "spell it exactly."
            )
        raise UnknownCardError(
            f"{name!r} is not a card in the {self.vocab.expansion or 'this'} "
            f"vocabulary ({self.vocab.size} cards).{hint}"
        )

    def card_ids(self, names) -> list[int]:
        """`card_id` over a list, reporting *every* bad name at once.

        One name at a time would make a caller with a mistyped pack fix it
        one round trip per card.
        """
        ids: list[int] = []
        failures: list[str] = []
        for name in names:
            try:
                ids.append(self.card_id(name))
            except UnknownCardError as error:
                failures.append(str(error))
        if failures:
            raise UnknownCardError(
                f"{len(failures)} of {len(list(names))} card names were not "
                "recognised:\n  " + "\n  ".join(failures)
            )
        return ids

    # -- ranking ------------------------------------------------------------

    def state_for(self, pool_size: int) -> tuple[int, int]:
        """The (pack_number, pick_number) a pool of this size implies.

        Straight from dataset.py's pool-as-prefix identity. It holds for
        every row in the corpus, so a real draft never has to supply these
        two numbers -- the pool already says what they are.
        """
        picks_per_pack = self.geometry.picks_per_pack
        return divmod(pool_size, picks_per_pack)

    def rank(
        self,
        pack: list[str],
        pool: list[str] | None = None,
        pack_number: int | None = None,
        pick_number: int | None = None,
        strict_state: bool = True,
    ) -> PickRanking:
        """Rank the cards in `pack`, given `pool`.

        `pack_number`/`pick_number` default to whatever the pool size
        implies. Passing them explicitly is checked against that unless
        `strict_state=False`, which is for deliberately synthetic states --
        the synergy probes build several -- and not for real drafts.
        """
        pool = list(pool or [])
        if not pack:
            raise ValueError("cannot rank an empty pack")
        if len(pack) > self.geometry.max_pack_size:
            raise ValueError(
                f"pack of {len(pack)} cards does not fit this corpus's "
                f"{self.geometry.max_pack_size}-card packs. Padding would drop "
                f"the last {len(pack) - self.geometry.max_pack_size} silently and "
                "rank a pack you did not ask about."
            )
        if len(pool) > self.geometry.max_pool_size:
            raise ValueError(
                f"pool of {len(pool)} cards is larger than the "
                f"{self.geometry.max_pool_size} a draft can reach; the model has "
                "never seen a state like this and truncating would hide that."
            )

        pack_ids = self.card_ids(pack)
        pool_ids = self.card_ids(pool)

        implied_pack, implied_pick = self.state_for(len(pool_ids))
        if pack_number is None:
            pack_number = implied_pack
        if pick_number is None:
            pick_number = implied_pick
        if strict_state and (pack_number, pick_number) != (implied_pack, implied_pick):
            raise ValueError(
                f"a pool of {len(pool_ids)} cards is pack {implied_pack}, pick "
                f"{implied_pick}, not pack {pack_number}, pick {pick_number}. "
                "The pool is the prefix of the picks already made, so these are "
                "not independent; pass strict_state=False to score the "
                "impossible state anyway."
            )

        probabilities = self.probe.probabilities(
            self.probe.pad_pack([pack_ids]),
            self.probe.pad_pool([pool_ids]),
            np.array([pack_number]),
            np.array([pick_number]),
        )[0, : len(pack_ids)]

        order = np.argsort(-probabilities)
        picks = tuple(
            RankedPick(
                rank=r,
                card=self.vocab.id_to_card[pack_ids[int(pos)]],
                card_id=int(pack_ids[int(pos)]),
                probability=float(probabilities[int(pos)]),
                pack_position=int(pos),
            )
            for r, pos in enumerate(order, start=1)
        )
        return PickRanking(
            picks=picks,
            pack_number=int(pack_number),
            pick_number=int(pick_number),
            pool_size=len(pool_ids),
            pool=tuple(self.vocab.id_to_card[c] for c in pool_ids),
        )

    def rank_row(self, data, row: int) -> PickRanking:
        """Rank a real corpus row, by index. The demo path, and the one
        that can be checked against what the drafter actually took.
        """
        example = data.example(row)
        names = [self.vocab.id_to_card[c] for c in example.pack]
        pool = [self.vocab.id_to_card[c] for c in example.pool]
        return self.rank(
            names, pool, pack_number=example.pack_number,
            pick_number=example.pick_number, strict_state=False,
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _example_rows(data, splits, count: int, seed: int) -> np.ndarray:
    """Val rows with enough cards left in the pack for a ranking to mean
    something. A pack of two is a coin flip and shows nothing.
    """
    rng = np.random.default_rng(seed)
    eligible = splits.val[data.pack_size[splits.val] >= 6]
    if not eligible.size:
        eligible = splits.val
    return rng.choice(eligible, size=min(count, eligible.size), replace=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank the cards in a pack with a trained checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--pack", nargs="+", default=None,
                        help="card names on offer; quote names containing commas")
    parser.add_argument("--pool", nargs="*", default=[],
                        help="card names already taken")
    parser.add_argument("--pack-number", type=int, default=None,
                        help="defaults to what the pool size implies")
    parser.add_argument("--pick-number", type=int, default=None)
    parser.add_argument("--top", type=int, default=None,
                        help="show only the top k; default shows the whole pack")
    parser.add_argument("--examples", type=int, default=3,
                        help="real validation packs to rank when --pack is absent")
    parser.add_argument("--evaluate", action="store_true",
                        help="also report top-1, top-3 and calibration over the val split")
    parser.add_argument("--rows", type=int, default=8192,
                        help="validation rows for --evaluate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    drafter, data = Drafter.from_checkpoint(args.checkpoint, args.processed_dir)
    report: dict = {"checkpoint": str(args.checkpoint)}
    print("=" * 70)

    if args.pack:
        ranking = drafter.rank(
            args.pack, args.pool,
            pack_number=args.pack_number, pick_number=args.pick_number,
        )
        print(ranking.summary(top=args.top))
        report["ranking"] = ranking.to_dict()
    else:
        splits = split_by_draft(data, seed=0)
        rows = _example_rows(data, splits, args.examples, args.seed)
        rankings = []
        for row in rows:
            ranking = drafter.rank_row(data, int(row))
            took = data.vocab.id_to_card[int(data.label[row])]
            print(ranking.summary(top=args.top or 4))
            agreed = ranking.best.card == took
            place = next(p.rank for p in ranking.picks if p.card == took)
            print(
                f"   human took {took[:40]}: "
                + ("model agrees" if agreed else f"model ranked it #{place}")
                + f", p={ranking.probability_of(took):.3f}\n"
            )
            rankings.append(ranking.to_dict() | {"human_took": took, "human_rank": place})
        report["examples"] = rankings

    if args.evaluate:
        from .metrics import ranking_report

        splits = split_by_draft(data, seed=0)
        rng = np.random.default_rng(args.seed)
        sample = rng.choice(
            splits.val, size=min(args.rows, splits.val.size), replace=False
        )
        evaluation = ranking_report(drafter.probe, data, sample)
        print(format_report(evaluation))
        report["evaluation"] = evaluation

    print("=" * 70)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


def format_report(evaluation: dict) -> str:
    from .metrics import format_ranking_metrics

    return format_ranking_metrics(evaluation)


if __name__ == "__main__":
    raise SystemExit(main())
