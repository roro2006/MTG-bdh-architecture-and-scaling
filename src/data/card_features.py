"""Builds the structured per-card feature table the composite embeddings read.

Card vectors in this project are composed from a card's actual attributes
rather than looked up in a table keyed by an arbitrary id (see
docs/ARCHITECTURE.md, "The shared front-end"). That buys two things the
project cares about: a card printed after training gets a sensible vector
from its attributes alone, and colour/cost/type/keywords are handed to the
model for free, so whatever structure it still has to learn is by
construction the part those trivial features do not explain.

Attributes come from Scryfall. Only the cards in the set's vocabulary are
fetched -- via `/cards/collection`, 75 identifiers per POST -- rather than
the ~150MB oracle-cards bulk export, which would be a large download to use
a few hundred rows of.

Scryfall asks for a descriptive User-Agent and 50-100ms between requests;
both are honoured below.

Every column means the same thing in every set
----------------------------------------------

The end goal is a drafter that can draft a set it has never seen, so the
feature table has to be a *fixed* layout, not one fitted to whichever set
happens to be in hand. An earlier version derived its keyword columns from
the set itself (keeping any keyword carried by >= 2 of its cards), which
made column k a different keyword in a different set -- a table trained on
FIN meant nothing when handed to a model reading BLB. Two things replace it:

  - `GLOBAL_KEYWORDS`, a checked-in list of the evergreen keywords. Fixed,
    so column k is the same keyword everywhere, and a set that happens not
    to use one simply gets a zero column.
  - `MECHANICS`, structured predicates pattern-matched against Scryfall's
    oracle text. These are the synergy carriers: "creates tokens" and
    "sacrifice outlet" are separate columns precisely so that a bilinear
    form in the cross-attention arm can learn the *pair* is worth more than
    the parts. A single opaque text embedding could not express that.

Three things about the text pipeline are easy to get wrong and are handled
explicitly in `clean_oracle_text`:

  1. Scryfall spells a card's own name out in its oracle text ("Whenever
     Zidane, Tantalus Thief attacks..."), and for a legendary creature it
     uses the short form later on ("...Zidane deals damage"). Left in, the
     name is card identity leaking straight into the features -- exactly
     the failure the old MIN_KEYWORD_CARDS threshold existed to prevent.
     Both forms are replaced with a placeholder.
  2. Reminder text in parentheses restates what a keyword already means. It
     is redundant with the keyword flags and, worse, makes two unrelated
     cards that share one mechanic look textually similar. It is stripped.
  3. A double-faced card keeps its text on `card_faces`, with the top-level
     `oracle_text` empty. Both faces are read, or half the card vanishes.

Width is a real constraint, not tidiness. The card embedding is
`_dense(F, embed_hidden)` -- linear in the feature width, while the rest of
the model is quadratic in its hidden dimension. A wide table therefore
inflates small-N models proportionally more than large-N ones, which bends
exactly the low-N corner the Chinchilla Huber fit is most sensitive to.
`dense()` is kept under 120 columns for that reason, and
`tests/test_card_features.py` asserts it.
"""

from __future__ import annotations

import json
import re
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

# Hard ceiling on dense() width. See the module docstring: the card
# embedding is linear in this while the arms are quadratic in hidden_dim,
# so width taxes the small-N cells of the scaling grid hardest.
MAX_FEATURE_WIDTH = 120

# The evergreen keywords: the ones Magic prints in essentially every set.
#
# Fixed, checked in, and deliberately not derived from whichever set is
# being ingested. That is the whole difference from the old per-set fit --
# column k is "Flying" in FIN, in BLB, and in a set printed next year, so a
# model trained on one can read the table for another. A set that never
# prints one of these gets an all-zero column, which costs one column and
# preserves the correspondence; that trade is the point.
#
# Deciduous and set-specific mechanics are deliberately absent. They are
# covered, where they matter, by MECHANICS below -- which keys off what a
# card *does* rather than what the mechanic is called, and so survives a
# rename between sets.
GLOBAL_KEYWORDS = (
    "Deathtouch",
    "Defender",
    "Double strike",
    "First strike",
    "Flash",
    "Flying",
    "Haste",
    "Hexproof",
    "Indestructible",
    "Lifelink",
    "Menace",
    "Reach",
    "Trample",
    "Vigilance",
    "Ward",
)

# Creature types common enough to recur across sets. Used for one column --
# "does this card's text talk about a creature type at all" -- rather than
# one column each, which would be 60 columns of mostly zeros and would not
# transfer anyway, since the relevant tribe changes every set.
_CREATURE_TYPES = (
    "advisor|angel|ape|archer|artificer|assassin|barbarian|bard|bat|bear|beast|"
    "berserker|bird|boar|cat|centaur|cleric|construct|crab|demon|detective|devil|"
    "dinosaur|djinn|dog|dragon|drake|druid|dwarf|efreet|elder|elemental|elephant|"
    "elf|elk|faerie|fish|fox|frog|fungus|giant|goblin|god|golem|gorgon|griffin|"
    "hippo|horror|horse|hound|human|hydra|illusion|insect|kithkin|knight|kor|"
    "kraken|leviathan|lizard|mercenary|merfolk|minotaur|monk|monkey|mouse|mutant|"
    "naga|ninja|noble|nymph|octopus|ogre|ooze|orc|otter|ox|peasant|pegasus|"
    "phoenix|phyrexian|pilot|pirate|plant|rabbit|raccoon|rat|rebel|rhino|rogue|"
    "samurai|satyr|scientist|scout|serpent|shaman|shapeshifter|sheep|skeleton|"
    "slith|sliver|slug|snake|soldier|specter|sphinx|spider|spirit|squirrel|"
    "thopter|treefolk|troll|turtle|vampire|vedalken|viashino|wall|warlock|"
    "warrior|whale|wizard|wolf|wolverine|wraith|wurm|zombie"
)

# The card's own name, replaced out of its oracle text before anything reads
# it. "~" is Magic's own placeholder for exactly this.
NAME_PLACEHOLDER = "~"

# --------------------------------------------------------------------------
# Mechanical features
# --------------------------------------------------------------------------
#
# Each entry is (column name, regex, scope). Scope says what the pattern is
# matched against:
#   "text" -- the cleaned oracle text (name stripped, reminder text removed)
#   "type" -- the lowercased type line
#   "both" -- either one matching sets the column
#
# These are one column each on purpose. The cross-attention arm scores a
# candidate against the pool through a bilinear interaction, and a bilinear
# form can only learn "sacrifice outlet plus token maker is worth more than
# either alone" if those are two distinguishable inputs. Collapse them into
# a single "aristocrats-ish" score and the interaction has nothing left to
# find. Every column here is phrased as something a *pair* of cards might
# care about, which is what separates this table from a similarity embedding.
#
# The patterns are deliberately shallow. They are not a rules engine and
# will misfire on unusual templating; what they need to be is *consistent*,
# so that the same sentence in two different sets produces the same column.
MECHANICS: tuple[tuple[str, str, str], ...] = (
    # -- tokens ------------------------------------------------------------
    ("make_token", r"\bcreates?\b[^.]{0,120}\btoken", "text"),
    ("make_creature_token", r"\bcreates?\b[^.]{0,120}\bcreature token", "text"),
    (
        "make_multiple_tokens",
        r"\bcreates?\s+(?:two|three|four|five|six|seven|eight|ten|x|\d+)\b[^.]{0,120}\btoken",
        "text",
    ),
    ("make_treasure", r"\btreasure\b", "text"),
    ("make_utility_token", r"\b(?:food|clue|blood|map|junk|powerstone|incubator|gold)\b", "text"),
    # -- sacrifice and death -----------------------------------------------
    (
        "sacrifice_outlet",
        r"\bsacrifice (?:a|an|another|two|three|x)\b[^:.\n]{0,48}:",
        "text",
    ),
    ("sacrifice_self", r"\bsacrifice (?:~|this)\b", "text"),
    ("sacrifice_as_cost", r"\bas an additional cost\b[^.]{0,120}\bsacrifice\b", "text"),
    ("death_trigger_self", r"\bwhen ~ dies\b|\bwhen this creature dies\b", "text"),
    (
        "death_trigger_other",
        r"\bwhenever\b[^.]{0,80}\b(?:another|one or more)[^.]{0,60}\bdies?\b"
        r"|\bwhenever\b[^.]{0,60}\bcreatures? you control dies\b",
        "text",
    ),
    (
        "creatures_dying_matters",
        r"\bdies\b|\bdied\b|\bput into a graveyard from the battlefield\b",
        "text",
    ),
    # -- counters ----------------------------------------------------------
    ("make_plus1_counters", r"\+1/\+1 counters?\s+on\b|\bwith a \+1/\+1 counter", "text"),
    ("enters_with_counters", r"\benters with\b", "text"),
    (
        "plus1_counters_matter",
        r"\bnumber of \+1/\+1 counters\b|\bhas? a \+1/\+1 counter\b"
        r"|\bwhenever\b[^.]{0,80}\bcounters? (?:is|are) put\b|\bfor each \+1/\+1 counter\b",
        "text",
    ),
    ("make_minus_counters", r"-1/-1 counters?\b", "text"),
    (
        "other_counters",
        r"\b(?:charge|stun|shield|oil|lore|time|fade|level|verse|experience|energy|"
        r"poison|ki|quest|loyalty|rust|bounty|finality) counters?\b",
        "text",
    ),
    # -- graveyard ---------------------------------------------------------
    (
        "recur_to_hand",
        r"\breturn\b[^.]{0,90}\bfrom your graveyard to your hand\b"
        r"|\breturn\b[^.]{0,60}\bcard from a graveyard to (?:its owner's|your) hand\b",
        "text",
    ),
    (
        "reanimate",
        r"\breturn\b[^.]{0,90}\bfrom (?:your|a) graveyard to the battlefield\b"
        r"|\bput\b[^.]{0,90}\bfrom (?:your|a) graveyard onto the battlefield\b",
        "text",
    ),
    (
        "self_mill",
        r"\byou mill\b|\bmill (?:one|two|three|four|five|six|seven|x|\d+) cards?\b"
        r"|\bput the top\b[^.]{0,80}\bof your library into your graveyard\b|\bsurveil\b",
        "text",
    ),
    (
        "mill_opponent",
        r"\b(?:each opponent|target opponent|target player|each player|they) mills?\b",
        "text",
    ),
    (
        "graveyard_exile",
        r"\bexile\b[^.]{0,90}\b(?:from|in) (?:a|your|target player's|each) graveyard\b"
        r"|\bexile all cards from\b",
        "text",
    ),
    (
        "graveyard_size_matters",
        r"\bcards? in your graveyard\b|\bcards? in each graveyard\b"
        r"|\bnumber of\b[^.]{0,40}\bgraveyard\b|\bdelirium\b|\bthreshold\b",
        "text",
    ),
    # -- hand and discard ---------------------------------------------------
    ("discard_self", r"\bdiscard (?:a|an|one|two|three|x|\d+|your|the rest)\b", "text"),
    ("opponent_discards", r"\b(?:opponent|player)s? discards?\b", "text"),
    (
        "discard_matters",
        r"\bwhenever you discard\b|\bmadness\b|\byou(?:'ve| have)? discarded\b",
        "text",
    ),
    (
        "hand_size_matters",
        r"\bcards? in your hand\b|\bno cards in (?:your )?hand\b|\bmaximum hand size\b",
        "text",
    ),
    # -- draw and selection -------------------------------------------------
    ("draw_card", r"\bdraws? a card\b|\bdraws? (?:x|\d+|two|three|four) cards?\b", "text"),
    ("draw_multiple", r"\bdraws? (?:two|three|four|five|six|seven|x|[2-9]) cards\b", "text"),
    (
        "card_selection",
        r"\bscry\b|\bsurveil\b|\blook at the top\b|\bexplores?\b|\bconnives?\b",
        "text",
    ),
    (
        # Deliberately spans a sentence break. The templating is "Exile the
        # top card of your library. You may play it until end of turn", so a
        # within-sentence window ([^.]) never reaches the payoff clause and
        # this column reads zero on every set that prints the effect.
        "play_from_exile",
        r"\bexile the top\b[^\n]{0,140}\byou may (?:play|cast)\b"
        r"|\bexile (?:it|them|that card)\b[^\n]{0,90}\byou may (?:play|cast)\b",
        "text",
    ),
    ("tutor_library", r"\bsearch your library for (?:a|an|up to)\b", "text"),
    # -- life ---------------------------------------------------------------
    ("gain_life", r"\bgains? \d+ life\b|\bgain (?:x|that much) life\b|\blifelink\b", "text"),
    # "you lose that much life" is as much a life payment as "you lose 2
    # life"; requiring a numeral misses every damage-linked drawback.
    ("pay_life", r"\byou lose (?:\d+|that much|x) life\b|\bpay (?:\d+|x) life\b", "text"),
    (
        "drain_opponent",
        r"\b(?:each opponent|target opponent|target player|they) loses? \d+ life\b"
        r"|\bdeals \d+ damage to each opponent\b",
        "text",
    ),
    (
        "lifegain_matters",
        r"\bwhenever you gain life\b|\byou gained life\b|\blife total is greater\b"
        r"|\bif you (?:have|gained)\b[^.]{0,30}\blife\b",
        "text",
    ),
    # -- permanents that care about permanents ------------------------------
    (
        "artifacts_matter",
        r"\bartifacts? (?:you control|spell|card)\b|\banother artifact\b"
        r"|\bnumber of artifacts\b|\bmetalcraft\b|\baffinity for artifacts\b",
        "text",
    ),
    (
        "enchantments_matter",
        r"\benchantments? (?:you control|spell|card)\b|\banother enchantment\b"
        r"|\bnumber of enchantments\b|\bconstellation\b",
        "text",
    ),
    ("equipment_matters", r"\bequip\b|\bequipment\b", "both"),
    ("aura_matters", r"\baura\b|\benchant \w+|\benchanted (?:creature|permanent|land|player)\b", "both"),
    ("modifies_attached", r"\b(?:equipped|enchanted) (?:creature|permanent|land|player)\b", "text"),
    # -- tribal --------------------------------------------------------------
    ("references_creature_type", rf"\b(?:{_CREATURE_TYPES})s?\b", "text"),
    (
        "tribal_lord",
        rf"\bother (?:{_CREATURE_TYPES})s you control\b"
        rf"|\b(?:{_CREATURE_TYPES})s you control get \+",
        "text",
    ),
    ("all_creature_types", r"\bis every creature type\b|\bchangeling\b|\ball creature types\b", "text"),
    # -- combat ---------------------------------------------------------------
    ("attack_trigger_self", r"\bwhenever ~ attacks\b|\bwhenever this creature attacks\b", "text"),
    (
        "attack_trigger_other",
        r"\bwhenever\b[^.]{0,70}\bcreatures? you control attacks?\b|\bwhenever you attack\b",
        "text",
    ),
    ("block_trigger", r"\bwhenever ~ blocks\b|\bblocks or becomes blocked\b|\bwhenever\b[^.]{0,50}\bblocks\b", "text"),
    ("combat_damage_trigger", r"\bdeals combat damage to a (?:player|creature)\b", "text"),
    ("pump_temporary", r"\bgets \+\d+/\+\d+ until end of turn\b|\bgets \+x/\+x\b", "text"),
    (
        "grant_evasion",
        r"\bcan't be blocked\b|\bcan block only\b|\bmust be blocked\b"
        r"|\bgains? (?:flying|menace|trample|shadow|fear|intimidate)\b",
        "text",
    ),
    # -- triggers --------------------------------------------------------------
    ("etb_self", r"\bwhen ~ enters\b|\bwhen this (?:creature|permanent|artifact|enchantment|land) enters\b", "text"),
    ("etb_other", r"\bwhenever (?:another|a|one or more)\b[^.]{0,70}\benters\b", "text"),
    ("cast_trigger", r"\bwhenever you cast\b", "text"),
    ("phase_trigger", r"\bat the beginning of\b", "text"),
    # -- abilities and timing ---------------------------------------------------
    (
        "activated_mana_cost",
        r"\{(?:\d+|[wubrgcxs]|[wubrg]/[wubrg]|[wubrg]/p|\d+/[wubrg])\}[^:.\n]{0,40}:",
        "text",
    ),
    ("activated_tap_only", r"\{t\}[^:.\n]{0,40}:", "text"),
    ("instant_speed", r"(?:^|\n)instant\b|\bflash\b", "both"),
    ("static_anthem", r"\bcreatures you control get \+\d+/\+\d+\b|\bother creatures you control get\b", "text"),
    ("mana_ability", r"\badd \{|\badd (?:one|two|three|x) mana\b|\badd that much\b", "text"),
    # -- interaction ------------------------------------------------------------
    ("damage_target", r"\bdeals \d+ damage to (?:target|any target)\b|\bdeals x damage to\b", "text"),
    ("damage_each", r"\bdeals \d+ damage to each\b", "text"),
    ("destroy_target", r"\bdestroy target\b|\bdestroy all\b|\bdestroy that\b", "text"),
    ("exile_target", r"\bexile target\b|\bexile that (?:creature|permanent)\b|\bexile all\b", "text"),
    ("shrink_toughness", r"\bgets -\d+/-\d+\b|\bgets \+\d+/-\d+\b|\b-\d+/-\d+ until end of turn\b", "text"),
    ("counterspell", r"\bcounter target\b", "text"),
    (
        "bounce",
        r"\breturn target\b[^.]{0,80}\bto (?:its|their) owner'?s? hand\b"
        r"|\breturn\b[^.]{0,60}\bto (?:its|their) owner'?s? hand\b",
        "text",
    ),
    ("tap_down", r"\btap target\b|\bdoesn't untap\b|\btap up to\b|\bbecomes tapped\b", "text"),
    # -- mana development -------------------------------------------------------
    (
        "land_search",
        r"\bsearch your library for a (?:basic )?(?:land|plains|island|swamp|mountain|forest)\b",
        "text",
    ),
    ("any_color_mana", r"\bmana of any (?:one )?color\b|\badd one mana of any\b", "text"),
    ("cost_reduction", r"\bcosts? \{\d+\} less\b|\bspells you cast cost\b", "text"),
    ("extra_land_drop", r"\bplay an additional land\b|\badditional lands?\b", "text"),
    # -- protection and copying -------------------------------------------------
    (
        "grant_protection",
        r"\bgains? (?:hexproof|indestructible|protection)\b|\bhas hexproof\b"
        r"|\b(?:hexproof|indestructible) until end of turn\b",
        "text",
    ),
    (
        "forced_sacrifice",
        r"\b(?:each opponent|target player|target opponent|each player) sacrifices\b",
        "text",
    ),
    ("copy_effect", r"\btoken that's a copy\b|\bcopies? target\b|\bcopy of\b", "text"),
)

MECHANIC_NAMES: tuple[str, ...] = tuple(name for name, _, _ in MECHANICS)

_COMPILED_MECHANICS = tuple(
    (name, re.compile(pattern, re.IGNORECASE), scope) for name, pattern, scope in MECHANICS
)

_REMINDER_TEXT = re.compile(r"\([^()]*\)")


# --------------------------------------------------------------------------
# Oracle text cleaning
# --------------------------------------------------------------------------

def oracle_text_of(card: dict) -> str:
    """Every face's oracle text, joined.

    A double-faced card carries its rules text on `card_faces` and leaves
    the top-level `oracle_text` empty, so reading only the top level drops
    the card's text entirely -- and reading only the front face drops half
    of it.
    """
    faces = card.get("card_faces")
    if faces:
        parts = [face.get("oracle_text", "") for face in faces]
        if any(parts):
            return "\n".join(p for p in parts if p)
    return card.get("oracle_text", "") or ""


def name_variants(card: dict) -> list[str]:
    """Every spelling of its own name a card's text might use.

    Scryfall writes the full name on first reference and, for a legendary
    permanent, the pre-comma short name afterwards: "Whenever Zidane,
    Tantalus Thief attacks... Zidane deals damage". Both have to go, and
    the full name has to go first or the short form would leave a dangling
    ", Tantalus Thief" behind.

    Returned longest-first for exactly that reason.
    """
    raw = [card.get("name", "")]
    raw += [face.get("name", "") for face in card.get("card_faces", []) or []]

    variants: set[str] = set()
    for name in raw:
        for part in (name or "").split(" // "):
            part = part.strip()
            if len(part) < 3:
                continue
            variants.add(part)
            short = part.split(",")[0].strip()
            # Only a genuinely shorter form, and long enough that replacing
            # it cannot plausibly eat an ordinary word.
            if len(short) >= 3 and short != part:
                variants.add(short)
    return sorted(variants, key=len, reverse=True)


def clean_oracle_text(card: dict) -> str:
    """Oracle text with the card's own name and reminder text removed.

    Order matters: names go first, so a name appearing inside reminder text
    is neutralised whichever way the parentheses fall. Name matching is
    case-sensitive, because card names are capitalised in rules text while
    the words they collide with ("Fire" the card vs "firebreathing") are
    not; lowercasing happens afterwards, once identity is gone.
    """
    text = oracle_text_of(card)
    if not text:
        return ""

    for variant in name_variants(card):
        text = re.sub(rf"\b{re.escape(variant)}\b", NAME_PLACEHOLDER, text)

    # Reminder text nests ("... (a keyword (with an aside))"), so strip
    # innermost-outwards until nothing changes.
    while True:
        stripped = _REMINDER_TEXT.sub(" ", text)
        if stripped == text:
            break
        text = stripped

    text = text.lower()
    # Curly apostrophes appear in Scryfall text and would break \b patterns
    # written with the ASCII form.
    text = text.replace("’", "'").replace("—", " - ")
    return re.sub(r"[ \t]+", " ", text).strip()


def mechanic_flags_for(card: dict) -> np.ndarray:
    """The (len(MECHANICS),) float32 row for one Scryfall card object."""
    text = clean_oracle_text(card)
    type_line = (_front_face(card).get("type_line") or card.get("type_line") or "").lower()
    row = np.zeros(len(_COMPILED_MECHANICS), dtype=np.float32)
    for i, (_, pattern, scope) in enumerate(_COMPILED_MECHANICS):
        hit = False
        if scope in ("text", "both"):
            hit = bool(pattern.search(text))
        if not hit and scope in ("type", "both"):
            hit = bool(pattern.search(type_line))
        row[i] = 1.0 if hit else 0.0
    return row


@dataclass(frozen=True)
class CardFeatures:
    """Per-card attributes, indexed by vocabulary id."""

    color_identity: np.ndarray   # (V, 5) multi-hot, WUBRG
    colors: np.ndarray           # (V, 5) multi-hot of the castable colours
    mana_value: np.ndarray       # (V,) float, clamped to MAX_MANA_VALUE
    type_flags: np.ndarray       # (V, len(CARD_TYPES)) multi-hot
    keyword_flags: np.ndarray    # (V, len(GLOBAL_KEYWORDS)) multi-hot, FIXED layout
    mechanic_flags: np.ndarray   # (V, len(MECHANICS)) multi-hot from oracle text
    rarity: np.ndarray           # (V,) int, index into RARITIES
    power: np.ndarray            # (V,) float, 0 for non-creatures
    toughness: np.ndarray        # (V,) float, 0 for non-creatures
    is_creature: np.ndarray      # (V,) float, 1/0 -- makes power/toughness readable
    keyword_names: tuple[str, ...]
    mechanic_names: tuple[str, ...]
    card_names: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.card_names)

    def _blocks(self) -> list[tuple[np.ndarray, tuple[str, ...]]]:
        """The column layout, as (array, labels) pairs. The single source.

        `dense()` and `column_names()` both read this, so the matrix and
        its labels cannot drift apart -- which matters because the labels
        are how anyone reading a trained model knows what column 47 was.
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
        return [
            (self.color_identity, tuple(f"color_identity_{c}" for c in COLORS)),
            (self.colors, tuple(f"colors_{c}" for c in COLORS)),
            (self.type_flags, tuple(f"type_{t.lower()}" for t in CARD_TYPES)),
            (self.keyword_flags, tuple(
                "kw_" + k.lower().replace(" ", "_") for k in self.keyword_names
            )),
            (self.mechanic_flags, tuple("mech_" + m for m in self.mechanic_names)),
            (rarity_onehot, tuple(f"rarity_{r}" for r in RARITIES)),
            (scalar_block, (
                "mana_value", "power", "toughness", "is_creature",
                "color_count", "is_multicolour",
            )),
        ]

    def dense(self) -> np.ndarray:
        """All features as one (V, D) float32 matrix.

        The embedding module projects from this rather than re-deriving the
        layout, so the column order lives in exactly one place: here.
        """
        return np.concatenate([block for block, _ in self._blocks()], axis=1).astype(
            np.float32
        )

    def column_names(self) -> tuple[str, ...]:
        """One label per column of `dense()`, in the same order."""
        return tuple(label for _, labels in self._blocks() for label in labels)

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
            mechanic_flags=self.mechanic_flags,
            rarity=self.rarity,
            power=self.power,
            toughness=self.toughness,
            is_creature=self.is_creature,
            keyword_names=np.array(self.keyword_names, dtype=object),
            mechanic_names=np.array(self.mechanic_names, dtype=object),
            card_names=np.array(self.card_names, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CardFeatures":
        with np.load(path, allow_pickle=True) as handle:
            if "mechanic_flags" not in handle.files:
                raise ValueError(
                    f"{path} was written by the per-set keyword build and has no "
                    "mechanic columns. Its keyword columns are fitted to one set "
                    "and mean something different in another, so it cannot be "
                    "reused -- rerun `python -m src.data.card_features`."
                )
            return cls(
                color_identity=handle["color_identity"],
                colors=handle["colors"],
                mana_value=handle["mana_value"],
                type_flags=handle["type_flags"],
                keyword_flags=handle["keyword_flags"],
                mechanic_flags=handle["mechanic_flags"],
                rarity=handle["rarity"],
                power=handle["power"],
                toughness=handle["toughness"],
                is_creature=handle["is_creature"],
                keyword_names=tuple(handle["keyword_names"].tolist()),
                mechanic_names=tuple(handle["mechanic_names"].tolist()),
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

    # Split, transforming and Room cards: 17lands names them with the full
    # "Front // Back", and /cards/collection refuses that exact string --
    # verified against the live endpoint, which returns "Crime // Punishment"
    # in `not_found` but resolves it fine when asked for "Crime". The pass
    # above only covers the opposite case (17lands gives the front face,
    # Scryfall answers with the full name), so without this retry DSK loses
    # 23 of its 286 cards to all-zero feature rows.
    retry = [n for n in card_names if n not in found and " // " in n]
    for start in range(0, len(retry), BATCH_SIZE):
        batch = retry[start : start + BATCH_SIZE]
        payload = {"identifiers": [{"name": n.split(" // ")[0]} for n in batch]}
        try:
            response = _post_json(SCRYFALL_COLLECTION_URL, payload)
        except urllib.error.HTTPError:
            break
        by_front = {c["name"].split(" // ")[0]: c for c in response.get("data", [])}
        for name in batch:
            card = by_front.get(name.split(" // ")[0])
            if card is not None:
                found[name] = card
        time.sleep(REQUEST_DELAY_S)

    still_missing = [n for n in card_names if n not in found]
    return found, sorted(still_missing)


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


def build_features(vocab: Vocabulary, cards: dict[str, dict]) -> CardFeatures:
    """Turns raw Scryfall objects into arrays indexed by vocabulary id.

    Takes no fitting parameters, deliberately: the column layout is the same
    for every set (GLOBAL_KEYWORDS and MECHANICS are module constants), so
    there is nothing here that could be fitted to the set being built.
    """
    size = vocab.size
    keyword_index = {kw: i for i, kw in enumerate(GLOBAL_KEYWORDS)}

    color_identity = np.zeros((size, len(COLORS)), dtype=np.float32)
    colors = np.zeros((size, len(COLORS)), dtype=np.float32)
    mana_value = np.zeros(size, dtype=np.float32)
    type_flags = np.zeros((size, len(CARD_TYPES)), dtype=np.float32)
    keyword_flags = np.zeros((size, len(GLOBAL_KEYWORDS)), dtype=np.float32)
    mechanic_flags = np.zeros((size, len(MECHANICS)), dtype=np.float32)
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

        mechanic_flags[card_id] = mechanic_flags_for(card)

        rarity[card_id] = rarity_index.get(card.get("rarity", "common"), 0)
        power[card_id] = _numeric(face.get("power", card.get("power")))
        toughness[card_id] = _numeric(face.get("toughness", card.get("toughness")))

    features = CardFeatures(
        color_identity=color_identity,
        colors=colors,
        mana_value=mana_value,
        type_flags=type_flags,
        keyword_flags=keyword_flags,
        mechanic_flags=mechanic_flags,
        rarity=rarity,
        power=power,
        toughness=toughness,
        is_creature=is_creature,
        keyword_names=GLOBAL_KEYWORDS,
        mechanic_names=MECHANIC_NAMES,
        card_names=vocab.id_to_card,
    )
    width = len(features.column_names())
    if width > MAX_FEATURE_WIDTH:
        raise ValueError(
            f"feature table is {width} columns, over MAX_FEATURE_WIDTH="
            f"{MAX_FEATURE_WIDTH}. The card embedding is linear in this while "
            "the arms are quadratic in hidden_dim, so extra width taxes the "
            "small-N grid cells hardest -- drop a column rather than raise "
            "the cap without re-reading docs/PROJECT_PLAN.md."
        )
    return features


def column_report(features: CardFeatures) -> str:
    """Per-column occupancy, for eyeballing a newly built table.

    Dead columns (nothing in this set fires them) are expected and fine --
    the layout is global, so a set that does not print a mechanic gets a
    zero column and the correspondence to other sets survives. Columns that
    fire on nearly everything are the ones worth a second look, since they
    carry almost no information at their own width.
    """
    dense = features.dense()
    names = features.column_names()
    counts = (dense > 0).sum(axis=0)
    lines = [f"{dense.shape[0]} cards x {dense.shape[1]} columns"]
    dead = [n for n, c in zip(names, counts) if c == 0]
    singleton = [n for n, c in zip(names, counts) if c == 1]
    saturated = [n for n, c in zip(names, counts) if c >= 0.9 * dense.shape[0]]
    lines.append(f"  dead in this set ({len(dead)}): {dead}")
    lines.append(f"  on exactly one card ({len(singleton)}): {singleton}")
    lines.append(f"  on >=90% of cards ({len(saturated)}): {saturated}")
    return "\n".join(lines)


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
    parser.add_argument(
        "--raw-in",
        default=None,
        help="reuse a previously dumped Scryfall JSON instead of refetching",
    )
    parser.add_argument(
        "--report", action="store_true", help="print per-column occupancy"
    )
    args = parser.parse_args(argv)

    processed = Path(args.processed_dir)
    vocab = Vocabulary.load(processed / "vocab.json")

    if args.raw_in:
        cards = json.loads(Path(args.raw_in).read_text(encoding="utf-8"))
        print(f"loaded {len(cards)} cards from {args.raw_in}")
    else:
        print(f"fetching {vocab.size} cards from Scryfall...")
        cards, missing = fetch_scryfall_cards(list(vocab.id_to_card))
        if missing:
            print(f"NOT FOUND ({len(missing)}): {missing[:20]}")
        if args.raw_out:
            Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.raw_out).write_text(
                json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8"
            )

    resolved = sum(1 for name in vocab.id_to_card if name in cards)
    print(f"resolved {resolved} / {vocab.size} vocabulary cards")

    features = build_features(vocab, cards)
    features.save(processed / "card_features.npz")
    dense = features.dense()
    print(
        f"features: {features.size} cards x {dense.shape[1]} dims "
        f"({len(GLOBAL_KEYWORDS)} fixed keywords + {len(MECHANICS)} mechanics), "
        f"saved to {processed / 'card_features.npz'}"
    )
    if args.report:
        print(column_report(features))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
