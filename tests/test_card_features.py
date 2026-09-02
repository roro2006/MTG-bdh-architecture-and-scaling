"""Tests for the set-independent card feature table.

These run against hand-written Scryfall-shaped dicts rather than the live
API, so the suite needs no network. The fixtures below reproduce the exact
templating quirks that make the text pipeline hard: a legendary creature
whose oracle text refers to itself by both its full and its short name, a
double-faced card with its text on the faces, and reminder text in
parentheses that restates a keyword.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.card_features import (
    GLOBAL_KEYWORDS,
    MAX_FEATURE_WIDTH,
    MECHANIC_NAMES,
    MECHANICS,
    NAME_PLACEHOLDER,
    build_features,
    clean_oracle_text,
    mechanic_flags_for,
    name_variants,
    oracle_text_of,
)
from src.data.vocab import Vocabulary


def card(name, oracle, type_line="Creature — Human", **extra):
    base = {
        "name": name,
        "oracle_text": oracle,
        "type_line": type_line,
        "cmc": 2.0,
        "color_identity": ["R"],
        "colors": ["R"],
        "keywords": [],
        "rarity": "common",
        "power": "2",
        "toughness": "2",
    }
    base.update(extra)
    return base


# Real templating, copied from Scryfall: the full name on first reference,
# the pre-comma short name afterwards, and a possessive form.
ZIDANE = card(
    "Zidane, Tantalus Thief",
    "Whenever Zidane, Tantalus Thief attacks, draw a card.\n"
    "Zidane deals 1 damage to each opponent when Zidane's power is 3 or more.",
    type_line="Legendary Creature — Human Mutant Scout",
)

LANNERY = card(
    "Captain Lannery Storm",
    "Haste\nWhenever Captain Lannery Storm attacks, create a Treasure token. "
    "(It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")\n"
    "Whenever you sacrifice a Treasure, Captain Lannery Storm gets +1/+0 until end of turn.",
    keywords=["Haste"],
)

# A double-faced card: top-level oracle_text is empty, both faces carry text.
CECIL = {
    "name": "Cecil, Dark Knight // Cecil, Redeemed Paladin",
    "oracle_text": "",
    "cmc": 3.0,
    "color_identity": ["B", "W"],
    "keywords": ["Deathtouch", "Lifelink"],
    "rarity": "rare",
    "card_faces": [
        {
            "name": "Cecil, Dark Knight",
            "type_line": "Legendary Creature — Human Knight",
            "oracle_text": "Deathtouch\nWhenever Cecil deals damage, you lose that much life.",
            "colors": ["B"],
            "power": "3",
            "toughness": "3",
        },
        {
            "name": "Cecil, Redeemed Paladin",
            "type_line": "Legendary Creature — Human Knight",
            "oracle_text": "Lifelink\nWhenever Cecil attacks, other attacking creatures "
            "gain indestructible until end of turn.",
            "colors": ["W"],
            "power": "4",
            "toughness": "5",
        },
    ],
}


# --------------------------------------------------------------------------
# Requirement 1: the card's own name must not survive into the features
# --------------------------------------------------------------------------

def test_full_and_short_names_are_both_stripped():
    """Scryfall uses "Zidane, Tantalus Thief" once and "Zidane" thereafter.

    Leaving either in leaks card identity into the feature table, which is
    the exact failure the old per-set keyword threshold existed to prevent.
    """
    text = clean_oracle_text(ZIDANE)
    assert "zidane" not in text
    assert "tantalus" not in text
    assert NAME_PLACEHOLDER in text
    # The full name must be replaced before the short form, or a dangling
    # ", tantalus thief" would be left behind.
    assert ", tantalus thief" not in text
    # Possessives survive as text but not as identity.
    assert "~'s power" in text


def test_no_card_leaks_its_own_name_into_the_text():
    for c in (ZIDANE, LANNERY, CECIL):
        text = clean_oracle_text(c)
        for variant in name_variants(c):
            assert variant.lower() not in text, (c["name"], variant)


def test_name_stripping_is_case_sensitive_so_it_cannot_eat_common_words():
    """A card named "Fire" must lose its own name but not the word inside
    "firebreathing" -- names are capitalised in rules text, collisions are not.
    """
    fire = card("Fire Elemental", "Fire Elemental has firebreathing when it attacks.")
    text = clean_oracle_text(fire)
    assert "firebreathing" in text
    assert text.startswith(NAME_PLACEHOLDER)


def test_short_name_is_not_taken_from_a_name_without_a_comma():
    assert "Captain" not in name_variants(LANNERY)
    assert "Captain Lannery Storm" in name_variants(LANNERY)


# --------------------------------------------------------------------------
# Requirement 2: reminder text goes
# --------------------------------------------------------------------------

def test_reminder_text_is_stripped():
    text = clean_oracle_text(LANNERY)
    assert "it's an artifact" not in text
    assert "(" not in text and ")" not in text
    # The real text either side of it survives.
    assert "create a treasure token" in text
    assert "whenever you sacrifice a treasure" in text


def test_nested_reminder_text_is_stripped():
    c = card("Probe", "Draw a card. (This is a reminder (with an aside) inside.) Then win.")
    text = clean_oracle_text(c)
    assert "reminder" not in text and "aside" not in text
    assert "draw a card" in text and "then win" in text


def test_reminder_text_does_not_inflate_similarity_between_unrelated_cards():
    """Two unrelated cards sharing one keyword should not come out looking
    alike just because they carry the same parenthetical gloss.
    """
    gloss = " (Whenever this creature deals combat damage, do the thing.)"
    a = card("Alpha Beast", "Whenever Alpha Beast attacks, draw a card." + gloss)
    b = card("Beta Wurm", "Destroy target creature." + gloss, type_line="Sorcery")
    flags_a, flags_b = mechanic_flags_for(a), mechanic_flags_for(b)
    shared = float(np.dot(flags_a, flags_b))
    assert shared == 0.0, dict(
        zip(MECHANIC_NAMES, (flags_a * flags_b).tolist())
    )


# --------------------------------------------------------------------------
# Double-faced cards
# --------------------------------------------------------------------------

def test_both_faces_of_a_dfc_are_read():
    text = clean_oracle_text(CECIL)
    assert "you lose that much life" in text          # front face
    assert "gain indestructible" in text              # back face
    assert "cecil" not in text                        # both names stripped
    assert oracle_text_of(CECIL)                      # not empty despite top-level ""


def test_dfc_mechanics_cover_both_faces():
    flags = dict(zip(MECHANIC_NAMES, mechanic_flags_for(CECIL)))
    assert flags["attack_trigger_self"] == 1.0        # back face only
    assert flags["grant_protection"] == 1.0           # back face only
    assert flags["pay_life"] == 1.0                   # front face only


# --------------------------------------------------------------------------
# Requirement 3: width, and a layout that lives in exactly one place
# --------------------------------------------------------------------------

@pytest.fixture
def small_features():
    names = ("Captain Lannery Storm", "Cecil, Dark Knight // Cecil, Redeemed Paladin",
             "Zidane, Tantalus Thief")
    vocab = Vocabulary(
        card_to_id={n: i for i, n in enumerate(names)}, id_to_card=names
    )
    return build_features(vocab, {LANNERY["name"]: LANNERY, CECIL["name"]: CECIL,
                                  ZIDANE["name"]: ZIDANE})


def test_feature_width_stays_under_the_cap(small_features):
    dense = small_features.dense()
    assert dense.shape[1] <= MAX_FEATURE_WIDTH
    assert dense.dtype == np.float32
    assert np.isfinite(dense).all()


def test_column_names_and_dense_cannot_drift(small_features):
    """Both read `_blocks()`, so a column added in one place shows up in the
    other. That is the only thing keeping a trained model's column 47
    interpretable.
    """
    assert len(small_features.column_names()) == small_features.dense().shape[1]
    assert len(set(small_features.column_names())) == small_features.dense().shape[1]


def test_layout_does_not_depend_on_the_set(small_features):
    """The whole point of the rewrite: column k means the same thing in
    every set, so a table built from one vocabulary must have exactly the
    same columns as one built from a completely different vocabulary.
    """
    other = Vocabulary(card_to_id={"Nothing At All": 0}, id_to_card=("Nothing At All",))
    empty = build_features(other, {})
    assert empty.column_names() == small_features.column_names()
    assert empty.dense().shape == (1, len(small_features.column_names()))
    # An unresolved card carries no attributes: every column is zero except
    # the rarity one-hot, which has to pick something and picks "common".
    row = dict(zip(empty.column_names(), empty.dense()[0]))
    assert row["rarity_common"] == 1.0
    assert sum(v for k, v in row.items() if not k.startswith("rarity_")) == 0.0


def test_keyword_columns_are_the_fixed_global_vocabulary(small_features):
    assert small_features.keyword_names == GLOBAL_KEYWORDS
    assert small_features.keyword_flags.shape[1] == len(GLOBAL_KEYWORDS)
    # A keyword the set never prints is a zero column, not a missing one.
    hexproof = GLOBAL_KEYWORDS.index("Hexproof")
    assert small_features.keyword_flags[:, hexproof].sum() == 0.0
    # A keyword it does print is set from Scryfall's own list.
    haste = GLOBAL_KEYWORDS.index("Haste")
    assert small_features.keyword_flags[0, haste] == 1.0


def test_mechanic_names_are_unique_and_stable():
    assert len(set(MECHANIC_NAMES)) == len(MECHANIC_NAMES)
    assert small_names_sorted(MECHANIC_NAMES) == small_names_sorted(
        tuple(n for n, _, _ in MECHANICS)
    )


def small_names_sorted(names):
    return sorted(names)


# --------------------------------------------------------------------------
# The mechanics themselves
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "column, oracle",
    [
        ("make_token", "Create a 1/1 white Soldier creature token."),
        ("make_creature_token", "Create a 1/1 white Soldier creature token."),
        ("make_multiple_tokens", "Create two 1/1 white Soldier creature tokens."),
        ("make_treasure", "Create a Treasure token."),
        ("sacrifice_outlet", "{1}, Sacrifice a creature: Draw a card."),
        ("death_trigger_self", "When Probe dies, draw a card."),
        ("make_plus1_counters", "Put a +1/+1 counter on target creature."),
        ("make_minus_counters", "Put a -1/-1 counter on target creature."),
        ("reanimate", "Return target creature card from your graveyard to the battlefield."),
        ("recur_to_hand", "Return target creature card from your graveyard to your hand."),
        ("self_mill", "You mill three cards."),
        ("mill_opponent", "Each opponent mills two cards."),
        ("discard_self", "Draw a card, then discard a card."),
        ("opponent_discards", "Each opponent discards a card."),
        ("draw_card", "Draw a card."),
        ("draw_multiple", "Draw two cards."),
        ("card_selection", "Scry 2."),
        ("play_from_exile", "Exile the top card of your library. You may play it this turn."),
        ("tutor_library", "Search your library for a creature card."),
        ("gain_life", "You gain 3 life."),
        ("drain_opponent", "Each opponent loses 2 life."),
        ("lifegain_matters", "Whenever you gain life, draw a card."),
        ("artifacts_matter", "Artifacts you control get +1/+1."),
        ("enchantments_matter", "Whenever another enchantment enters, draw a card."),
        ("etb_self", "When Probe enters, draw a card."),
        ("cast_trigger", "Whenever you cast a noncreature spell, draw a card."),
        ("activated_mana_cost", "{2}{R}: Probe deals 1 damage to any target."),
        ("activated_tap_only", "{T}: Add {G}."),
        ("mana_ability", "{T}: Add {G}."),
        ("damage_target", "Probe deals 3 damage to any target."),
        ("destroy_target", "Destroy target creature."),
        ("exile_target", "Exile target creature."),
        ("shrink_toughness", "Target creature gets -3/-3 until end of turn."),
        ("counterspell", "Counter target spell."),
        ("bounce", "Return target creature to its owner's hand."),
        ("land_search", "Search your library for a basic land card."),
        ("any_color_mana", "{T}: Add one mana of any color."),
        ("attack_trigger_self", "Whenever Probe attacks, draw a card."),
        ("combat_damage_trigger", "Whenever Probe deals combat damage to a player, draw a card."),
        ("references_creature_type", "Goblins you control get +1/+1."),
        ("tribal_lord", "Other Goblins you control get +1/+1."),
        ("grant_evasion", "Target creature can't be blocked this turn."),
        ("static_anthem", "Creatures you control get +1/+1."),
        ("phase_trigger", "At the beginning of your upkeep, draw a card."),
        ("forced_sacrifice", "Each opponent sacrifices a creature."),
        ("cost_reduction", "Creature spells you cast cost {1} less to cast."),
        ("extra_land_drop", "You may play an additional land on each of your turns."),
    ],
)
def test_mechanic_fires_on_its_own_templating(column, oracle):
    flags = dict(zip(MECHANIC_NAMES, mechanic_flags_for(card("Probe", oracle))))
    assert flags[column] == 1.0, f"{column!r} did not fire on {oracle!r}"


def test_a_vanilla_creature_fires_almost_nothing():
    """A card with no rules text at all should be near the origin of the
    mechanic space; if it is not, some pattern is matching whitespace.
    """
    flags = mechanic_flags_for(card("Grizzly Bears", ""))
    assert flags.sum() == 0.0


def test_type_line_scoped_mechanics_read_the_type_line():
    equipment = card(
        "Sharp Thing", "Equipped creature gets +2/+0.", type_line="Artifact — Equipment"
    )
    flags = dict(zip(MECHANIC_NAMES, mechanic_flags_for(equipment)))
    assert flags["equipment_matters"] == 1.0
    assert flags["modifies_attached"] == 1.0

    aura = card("Clingy Thing", "Enchanted creature gets +2/+2.",
                type_line="Enchantment — Aura")
    assert dict(zip(MECHANIC_NAMES, mechanic_flags_for(aura)))["aura_matters"] == 1.0

    bolt = card("Zap", "Zap deals 3 damage to any target.", type_line="Instant")
    assert dict(zip(MECHANIC_NAMES, mechanic_flags_for(bolt)))["instant_speed"] == 1.0


def test_instant_speed_does_not_fire_on_a_card_that_merely_mentions_instants():
    """"Instant or sorcery spell" is a card that *cares* about instants, not
    one that can be cast at instant speed."""
    c = card("Slow Thing", "Whenever you cast an instant or sorcery spell, draw a card.",
             type_line="Creature — Human Wizard")
    assert dict(zip(MECHANIC_NAMES, mechanic_flags_for(c)))["instant_speed"] == 0.0


def test_pairs_of_mechanics_are_separately_addressable():
    """A sacrifice outlet and a token maker must occupy different columns.

    This is the property the cross-attention arm's bilinear interaction
    needs in order to learn that the pair is worth more than the parts --
    collapse them into one "aristocrats" score and there is nothing left
    for an interaction term to find.
    """
    outlet = card("Altar", "{1}, Sacrifice a creature: Draw a card.")
    maker = card("Factory", "At the beginning of your end step, create a 1/1 Soldier "
                            "creature token.")
    fo = dict(zip(MECHANIC_NAMES, mechanic_flags_for(outlet)))
    fm = dict(zip(MECHANIC_NAMES, mechanic_flags_for(maker)))
    assert fo["sacrifice_outlet"] == 1.0 and fo["make_creature_token"] == 0.0
    assert fm["make_creature_token"] == 1.0 and fm["sacrifice_outlet"] == 0.0


def test_save_and_load_round_trips(small_features, tmp_path):
    path = tmp_path / "card_features.npz"
    small_features.save(path)
    from src.data.card_features import CardFeatures

    reloaded = CardFeatures.load(path)
    assert reloaded.column_names() == small_features.column_names()
    assert np.array_equal(reloaded.dense(), small_features.dense())
    assert reloaded.mechanic_names == MECHANIC_NAMES


def test_loading_a_pre_rewrite_table_is_refused(tmp_path):
    """An old npz has keyword columns fitted to one set. Silently accepting
    it would train a model on columns whose meaning nobody can recover.
    """
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        color_identity=np.zeros((1, 5), dtype=np.float32),
        keyword_flags=np.zeros((1, 3), dtype=np.float32),
        keyword_names=np.array(["Job select"], dtype=object),
    )
    from src.data.card_features import CardFeatures

    with pytest.raises(ValueError, match="mechanic columns"):
        CardFeatures.load(path)
