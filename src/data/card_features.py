"""Builds the structured per-card feature table the composite embeddings read.

Card vectors in this project are composed from a card's actual attributes
rather than looked up in a table keyed by an arbitrary id (see
docs/ARCHITECTURE.md, "The shared front-end"). That buys two things the
project cares about: a card printed after training gets a sensible vector
from its attributes alone, and colour/cost/type/keywords are handed to the
model for free, so whatever structure it still has to learn is by
construction the part those trivial features do not explain.

Attributes come from Scryfall. Only the 363 cards in the set's vocabulary
are fetched -- via `/cards/collection`, 75 identifiers per POST -- rather
than the ~150MB oracle-cards bulk export, which would be a large download
to use a few hundred rows of.

Scryfall asks for a descriptive User-Agent and 50-100ms between requests;
both are honoured below.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vocab import Vocabulary

SCRYFALL_COLLECTION_URL = "https://api.scryfall.com/cards/collection"
USER_AGENT = "MTG-BDH-scaling-research/0.1 (github.com/rohanreddy/MTG-bdh)"
BATCH_SIZE = 75  # Scryfall's documented maximum per collection request
REQUEST_DELAY_S = 0.1

# The five colours, in the canonical WUBRG order Magic always uses.
COLORS = ("W", "U", "B", "R", "G")

# Primary card types. A card can hold several at once ("Artifact Creature"),
# so these are multi-hot rather than categorical.
CARD_TYPES = (
    "Creature",
    "Instant",
    "Sorcery",
    "Artifact",
    "Enchantment",
    "Land",
    "Planeswalker",
    "Battle",
    "Legendary",
)

RARITIES = ("common", "uncommon", "rare", "mythic", "special", "bonus")

# Mana values above this are clamped; the tail is thin and the difference
# between a 9-drop and a 12-drop is not what the model needs to resolve.
MAX_MANA_VALUE = 10.0

# Keywords carried by fewer than this many cards in the set are dropped.
#
# This threshold is load-bearing rather than cosmetic. Scryfall's `keywords`
# field includes named abilities as well as evergreen mechanics, and FIN is
# full of flavour-named ones -- "Blizzaga", "Final Heaven", "Chef's Knife" --
# each appearing on exactly one card. Measured on FIN: 84 of 118 keywords
# occur on a single card.
#
# A feature column set for exactly one card is that card's id in disguise. It
# would hand the model 84 free one-hot identity columns, which defeats the
# stated point of composing embeddings from attributes (docs/ARCHITECTURE.md:
# a new card should get a sensible vector from its attributes alone) and
# confounds the scaling measurement, since what the model can memorise is
# exactly what the N-axis is supposed to be varying.
#
# At 2 this keeps every keyword that recurs at all and drops every singleton.
# It also drops genuine evergreen mechanics that happen to appear once in
# this set (Double strike, Dash) -- an acceptable loss, since a single
# example teaches the model no more about the mechanic than about the card.
MIN_KEYWORD_CARDS = 2


@dataclass(frozen=True)
class CardFeatures:
    """Per-card attributes, indexed by vocabulary id."""

    color_identity: np.ndarray  # (V, 5) multi-hot, WUBRG
    colors: np.ndarray          # (V, 5) multi-hot of the castable colours
    mana_value: np.ndarray      # (V,) float, clamped to MAX_MANA_VALUE
    type_flags: np.ndarray      # (V, len(CARD_TYPES)) multi-hot
    keyword_flags: np.ndarray   # (V, n_keywords) multi-hot
    rarity: np.ndarray          # (V,) int, index into RARITIES
    power: np.ndarray           # (V,) float, 0 for non-creatures
    toughness: np.ndarray       # (V,) float, 0 for non-creatures
    is_creature: np.ndarray     # (V,) float, 1/0 -- makes power/toughness readable
    keyword_names: tuple[str, ...]
    card_names: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.card_names)

    def dense(self) -> np.ndarray:
        """All features as one (V, D) float32 matrix.

        The embedding module projects from this rather than re-deriving the
        layout, so the column order lives in exactly one place: here.
        """
        n_colors = len(COLORS)
        scalar_block = np.stack(
            [
                self.mana_value / MAX_MANA_VALUE,
                self.power / 10.0,
                self.toughness / 10.0,
                self.is_creature,
                self.color_identity.sum(axis=1) / n_colors,  # colour count
                (self.color_identity.sum(axis=1) > 1).astype(np.float32),  # multicolour
            ],
            axis=1,
        )
        rarity_onehot = np.zeros((self.size, len(RARITIES)), dtype=np.float32)
        rarity_onehot[np.arange(self.size), self.rarity] = 1.0
        return np.concatenate(
            [
                self.color_identity,
                self.colors,
                self.type_flags,
                self.keyword_flags,
                rarity_onehot,
                scalar_block,
            ],
            axis=1,
        ).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            color_identity=self.color_identity,
            colors=self.colors,
            mana_value=self.mana_value,
            type_flags=self.type_flags,
            keyword_flags=self.keyword_flags,
            rarity=self.rarity,
            power=self.power,
            toughness=self.toughness,
            is_creature=self.is_creature,
            keyword_names=np.array(self.keyword_names, dtype=object),
            card_names=np.array(self.card_names, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CardFeatures":
        with np.load(path, allow_pickle=True) as handle:
            return cls(
                color_identity=handle["color_identity"],
                colors=handle["colors"],
                mana_value=handle["mana_value"],
                type_flags=handle["type_flags"],
                keyword_flags=handle["keyword_flags"],
                rarity=handle["rarity"],
                power=handle["power"],
                toughness=handle["toughness"],
                is_creature=handle["is_creature"],
                keyword_names=tuple(handle["keyword_names"].tolist()),
                card_names=tuple(handle["card_names"].tolist()),
            )


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_scryfall_cards(card_names: list[str]) -> tuple[dict[str, dict], list[str]]:
    """Fetches oracle data for exactly these card names.

    Returns (name -> card object, names Scryfall could not resolve). The
    caller decides what to do about misses rather than having a silent
    default imposed here.
    """
    found: dict[str, dict] = {}
    missing: list[str] = []

    for start in range(0, len(card_names), BATCH_SIZE):
        batch = card_names[start : start + BATCH_SIZE]
        payload = {"identifiers": [{"name": name} for name in batch]}
        try:
            response = _post_json(SCRYFALL_COLLECTION_URL, payload)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"Scryfall returned {error.code} for batch starting at {start}: "
                f"{error.read().decode('utf-8', 'replace')[:400]}"
            ) from error

        for card in response.get("data", []):
            found[card["name"]] = card
        for miss in response.get("not_found", []):
            missing.append(miss.get("name", "<unknown>"))
        time.sleep(REQUEST_DELAY_S)

    # Scryfall resolves to its own canonical name, which can differ from the
    # requested one; match those back up so the caller's ids line up.
    for name in card_names:
        if name in found:
            continue
        for canonical, card in found.items():
            if canonical.split(" // ")[0] == name:
                found[name] = card
                break

    still_missing = [n for n in card_names if n not in found]
    return found, sorted(set(missing) | set(still_missing))


def _front_face(card: dict) -> dict:
    """The face carrying the castable characteristics.

    Double-faced cards put mana cost, type line, and oracle text on their
    faces rather than at top level; colour identity and mana value stay at
    top level and are read from there regardless.
    """
    if "type_line" in card and "oracle_text" in card:
        return card
    faces = card.get("card_faces")
    return faces[0] if faces else card


def _numeric(value: str | None) -> float:
    """Power/toughness, which can be '*', '1+*', or absent."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def build_features(
    vocab: Vocabulary,
    cards: dict[str, dict],
    min_keyword_cards: int = MIN_KEYWORD_CARDS,
) -> CardFeatures:
    """Turns raw Scryfall objects into arrays indexed by vocabulary id.

    `min_keyword_cards` drops keywords carried by fewer than that many cards
    in the set. This is not tidying -- see MIN_KEYWORD_CARDS for why a
    singleton keyword is a card id wearing a feature's clothes.
    """
    size = vocab.size

    keyword_counts: dict[str, int] = {}
    for name in vocab.id_to_card:
        card = cards.get(name)
        if card:
            for keyword in card.get("keywords", []):
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    keyword_names = tuple(
        sorted(kw for kw, n in keyword_counts.items() if n >= min_keyword_cards)
    )
    keyword_index = {kw: i for i, kw in enumerate(keyword_names)}

    color_identity = np.zeros((size, len(COLORS)), dtype=np.float32)
    colors = np.zeros((size, len(COLORS)), dtype=np.float32)
    mana_value = np.zeros(size, dtype=np.float32)
    type_flags = np.zeros((size, len(CARD_TYPES)), dtype=np.float32)
    keyword_flags = np.zeros((size, len(keyword_names)), dtype=np.float32)
    rarity = np.zeros(size, dtype=np.int32)
    power = np.zeros(size, dtype=np.float32)
    toughness = np.zeros(size, dtype=np.float32)
    is_creature = np.zeros(size, dtype=np.float32)

    color_index = {c: i for i, c in enumerate(COLORS)}
    rarity_index = {r: i for i, r in enumerate(RARITIES)}

    for card_id, name in enumerate(vocab.id_to_card):
        card = cards.get(name)
        if card is None:
            # Left as all-zeros: a card with no attributes gets a neutral
            # vector rather than a fabricated one.
            continue
        face = _front_face(card)

        for color in card.get("color_identity", []):
            if color in color_index:
                color_identity[card_id, color_index[color]] = 1.0
        for color in face.get("colors", card.get("colors", [])):
            if color in color_index:
                colors[card_id, color_index[color]] = 1.0

        mana_value[card_id] = min(float(card.get("cmc", 0.0)), MAX_MANA_VALUE)

        type_line = face.get("type_line", card.get("type_line", ""))
        for i, card_type in enumerate(CARD_TYPES):
            if card_type in type_line:
                type_flags[card_id, i] = 1.0
        is_creature[card_id] = 1.0 if "Creature" in type_line else 0.0

        for keyword in card.get("keywords", []):
            if keyword in keyword_index:
                keyword_flags[card_id, keyword_index[keyword]] = 1.0

        rarity[card_id] = rarity_index.get(card.get("rarity", "common"), 0)
        power[card_id] = _numeric(face.get("power", card.get("power")))
        toughness[card_id] = _numeric(face.get("toughness", card.get("toughness")))

    return CardFeatures(
        color_identity=color_identity,
        colors=colors,
        mana_value=mana_value,
        type_flags=type_flags,
        keyword_flags=keyword_flags,
        rarity=rarity,
        power=power,
        toughness=toughness,
        is_creature=is_creature,
        keyword_names=keyword_names,
        card_names=vocab.id_to_card,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Scryfall attributes for a set's vocabulary."
    )
    parser.add_argument("--processed-dir", required=True, help="dir holding vocab.json")
    parser.add_argument(
        "--raw-out",
        default=None,
        help="optional path to dump the raw Scryfall JSON alongside the features",
    )
    args = parser.parse_args(argv)

    processed = Path(args.processed_dir)
    vocab = Vocabulary.load(processed / "vocab.json")
    print(f"fetching {vocab.size} cards from Scryfall...")

    cards, missing = fetch_scryfall_cards(list(vocab.id_to_card))
    resolved = sum(1 for name in vocab.id_to_card if name in cards)
    print(f"resolved {resolved} / {vocab.size} vocabulary cards")
    if missing:
        print(f"NOT FOUND ({len(missing)}): {missing[:20]}")

    if args.raw_out:
        Path(args.raw_out).write_text(
            json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    all_keywords = {kw for c in cards.values() for kw in c.get("keywords", [])}
    features = build_features(vocab, cards)
    features.save(processed / "card_features.npz")
    dense = features.dense()
    print(
        f"features: {features.size} cards x {dense.shape[1]} dims, saved to "
        f"{processed / 'card_features.npz'}"
    )
    print(
        f"keywords: kept {len(features.keyword_names)} of {len(all_keywords)} "
        f"(dropped those on < {MIN_KEYWORD_CARDS} cards; see MIN_KEYWORD_CARDS)"
    )
    print(f"  {list(features.keyword_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
