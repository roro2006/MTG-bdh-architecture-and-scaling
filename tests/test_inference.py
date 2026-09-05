"""Tests for the inference entry point.

Like the synergy tests, these run against a randomly initialised or stubbed
model rather than a trained one: what is under test is that the entry point
ranks the right cards, refuses the states it cannot answer honestly, and
computes the reported metrics correctly -- not that any model scores well.

Where a metric needs a known answer, the model is a stub whose response is
fixed by construction and the expected number is computed independently in
numpy, so a bug in the metric cannot hide behind a bug in the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import split_by_draft
from src.inference import Drafter, PickProbe, UnknownCardError
from src.inference.metrics import (
    calibration,
    format_ranking_metrics,
    ranked_probabilities,
    ranking_metrics,
    ranking_report,
)

from . import synthetic


@pytest.fixture
def corpus(ingested):
    data, _stats, out = ingested(count=60)
    return data


def _drafter(data, hidden_dim=16, feature_dim=9, seed=0) -> Drafter:
    """A randomly initialised PickModel wrapped in a Drafter."""
    import jax
    import jax.numpy as jnp

    from src.models.pick_model import ModelConfig, PickModel

    config = ModelConfig(
        hidden_dim=hidden_dim,
        card_feature_dim=feature_dim,
        packs_per_draft=data.packs_per_draft,
        picks_per_pack=data.picks_per_pack,
    )
    model = PickModel(config=config, arm="attention")
    table = jax.random.normal(jax.random.PRNGKey(seed), (data.vocab.size, feature_dim))
    batch = data.batch(np.arange(4))
    params = model.init(
        jax.random.PRNGKey(seed),
        table,
        jnp.asarray(batch["pack_ids"]),
        jnp.asarray(batch["pool_ids"]),
        jnp.asarray(batch["pack_number"]),
        jnp.asarray(batch["pick_number"]),
    )
    return Drafter(PickProbe(model, params, table, data.geometry, data.vocab))


class _IdScoringModel:
    """Scores each pack slot by its card id. Deterministic, pool-blind.

    Lets a test compute the expected ranking straight from `data.pack`
    without running a model at all.
    """

    def apply(self, params, table, pack_ids, pool_ids, pack_number, pick_number):
        import jax.numpy as jnp

        return jnp.where(pack_ids >= 0, pack_ids.astype(jnp.float32), -1e9)


def _stub_drafter(data) -> Drafter:
    table = np.zeros((data.vocab.size, 3), dtype=np.float32)
    return Drafter(PickProbe(_IdScoringModel(), {}, table, data.geometry, data.vocab))


# --------------------------------------------------------------------------
# Names, and refusing the ones we do not know
# --------------------------------------------------------------------------

def test_ranks_by_name_and_returns_the_whole_pack(corpus):
    drafter = _drafter(corpus)
    pack = ["Card 00", "Card 01", "Zidane, Tantalus Thief"]
    ranking = drafter.rank(pack, pool=["Card 02"])

    assert [p.card for p in ranking.picks] != []
    assert sorted(p.card for p in ranking.picks) == sorted(pack)
    assert sum(p.probability for p in ranking.picks) == pytest.approx(1.0, abs=1e-5)
    assert [p.rank for p in ranking.picks] == [1, 2, 3]


def test_ranking_is_descending_and_best_is_the_head(corpus):
    drafter = _drafter(corpus)
    ranking = drafter.rank([f"Card {i:02d}" for i in range(8)], pool=[])
    probabilities = [p.probability for p in ranking.picks]
    assert probabilities == sorted(probabilities, reverse=True)
    assert ranking.best is ranking.picks[0]
    assert ranking.top(3) == ranking.picks[:3]


def test_an_unknown_name_raises_rather_than_mis_indexing(corpus):
    """The failure this entry point exists to prevent: every id in range is
    a legal card, so a bad name that resolved to *something* would produce
    a confident ranking of a pack nobody asked about.
    """
    drafter = _drafter(corpus)
    with pytest.raises(UnknownCardError, match="not a card"):
        drafter.rank(["Card 00", "Lightning Bolt"])


def test_an_unknown_name_suggests_the_nearest_spellings(corpus):
    drafter = _drafter(corpus)
    with pytest.raises(UnknownCardError, match="Did you mean"):
        drafter.card_id("Card 0")


def test_every_bad_name_is_reported_at_once(corpus):
    """One round trip per typo would be a poor way to fix a mistyped pack."""
    drafter = _drafter(corpus)
    with pytest.raises(UnknownCardError) as error:
        drafter.rank(["Nope One", "Card 00", "Nope Two"])
    message = str(error.value)
    assert "2 of 3" in message
    assert "Nope One" in message and "Nope Two" in message


def test_the_count_survives_a_one_shot_iterable(corpus):
    """The tally is taken from the names as read, not by re-reading them.

    A generator is empty the second time round, so counting it after the
    loop reported every failure as "n of 0".
    """
    drafter = _drafter(corpus)
    with pytest.raises(UnknownCardError) as error:
        drafter.card_ids(n for n in ["Nope One", "Card 00", "Nope Two"])
    assert "2 of 3" in str(error.value)


def test_case_only_misses_resolve_when_unambiguous(corpus):
    drafter = _drafter(corpus)
    assert drafter.card_id("card 00") == drafter.card_id("Card 00")
    assert drafter.card_id("ZIDANE, TANTALUS THIEF") == drafter.card_id(
        "Zidane, Tantalus Thief"
    )


def test_names_with_commas_survive(corpus):
    """Real card names carry commas and the vocabulary keeps them whole."""
    drafter = _drafter(corpus)
    ranking = drafter.rank(["Zidane, Tantalus Thief", "Card 00"])
    assert "Zidane, Tantalus Thief" in {p.card for p in ranking.picks}


def test_duplicate_copies_are_separate_slots_but_one_card(corpus):
    """A pack can hold two copies of a common. They score separately; the
    question "what does the model think of this card" wants their sum.
    """
    drafter = _drafter(corpus)
    ranking = drafter.rank(["Card 00", "Card 00", "Card 01"])
    assert len(ranking.picks) == 3
    total = ranking.probability_of("Card 00")
    slots = [p.probability for p in ranking.picks if p.card == "Card 00"]
    assert len(slots) == 2
    assert total == pytest.approx(sum(slots))


def test_asking_about_a_card_not_in_the_pack_raises(corpus):
    drafter = _drafter(corpus)
    ranking = drafter.rank(["Card 00", "Card 01"])
    with pytest.raises(UnknownCardError, match="not in this pack"):
        ranking.probability_of("Card 02")


# --------------------------------------------------------------------------
# States the model cannot honestly answer
# --------------------------------------------------------------------------

def test_an_empty_pack_is_refused(corpus):
    drafter = _drafter(corpus)
    with pytest.raises(ValueError, match="empty pack"):
        drafter.rank([])


def test_an_oversized_pack_is_refused_not_truncated(corpus):
    """Padding drops the overflow silently, which would rank a different
    pack from the one asked about and look entirely normal doing it.
    """
    drafter = _drafter(corpus)
    too_many = [f"Card {i:02d}" for i in range(corpus.geometry.max_pack_size + 1)]
    with pytest.raises(ValueError, match="does not fit"):
        drafter.rank(too_many)


def test_an_oversized_pool_is_refused(corpus):
    drafter = _drafter(corpus)
    with pytest.raises(ValueError, match="larger than"):
        drafter.rank(["Card 00"], pool=["Card 01"] * (corpus.geometry.max_pool_size + 1))


def test_the_pool_size_implies_the_pack_and_pick_number(corpus):
    """The pool-as-prefix identity, so a real draft never has to say."""
    drafter = _drafter(corpus)
    picks_per_pack = corpus.geometry.picks_per_pack

    ranking = drafter.rank(["Card 00", "Card 01"], pool=["Card 02"] * 5)
    assert (ranking.pack_number, ranking.pick_number) == (0, 5)

    ranking = drafter.rank(["Card 00", "Card 01"], pool=["Card 02"] * (picks_per_pack + 2))
    assert (ranking.pack_number, ranking.pick_number) == (1, 2)

    ranking = drafter.rank(["Card 00", "Card 01"], pool=[])
    assert (ranking.pack_number, ranking.pick_number) == (0, 0)


def test_a_state_that_contradicts_the_pool_is_refused(corpus):
    drafter = _drafter(corpus)
    with pytest.raises(ValueError, match="not independent"):
        drafter.rank(["Card 00", "Card 01"], pool=["Card 02"] * 5, pack_number=2)


def test_an_impossible_state_can_be_scored_deliberately(corpus):
    """The synergy probes build states no draft produces on purpose, so the
    guard has to be an opt-out rather than a wall.
    """
    drafter = _drafter(corpus)
    ranking = drafter.rank(
        ["Card 00", "Card 01"], pool=["Card 02"] * 5,
        pack_number=2, pick_number=9, strict_state=False,
    )
    assert (ranking.pack_number, ranking.pick_number) == (2, 9)


def test_rank_row_reproduces_the_probe_on_a_real_row(corpus):
    """The name path and the id path must agree, or one of them is wrong."""
    drafter = _drafter(corpus)
    row = 100
    ranking = drafter.rank_row(corpus, row)

    batch = corpus.batch(np.array([row]))
    direct = drafter.probe.probabilities(
        batch["pack_ids"], batch["pool_ids"],
        batch["pack_number"], batch["pick_number"],
    )[0]
    for pick in ranking.picks:
        assert pick.probability == pytest.approx(
            float(direct[pick.pack_position]), abs=1e-6
        )
    assert ranking.pool_size == len(corpus.example(row).pool)


def test_a_stub_model_ranks_exactly_as_its_scores_say(corpus):
    """End to end against a model whose ordering is known in advance."""
    drafter = _stub_drafter(corpus)
    ranking = drafter.rank(["Card 03", "Card 07", "Card 01"])
    ids = {p.card: p.card_id for p in ranking.picks}
    expected = sorted(ids, key=lambda name: -ids[name])
    assert [p.card for p in ranking.picks] == expected


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_a_perfectly_calibrated_model_has_near_zero_ece():
    """Confidence 0.7 on rows right 70% of the time, and so on."""
    rng = np.random.default_rng(0)
    confidence = rng.choice([0.15, 0.45, 0.75, 0.95], size=40000)
    correct = rng.random(40000) < confidence
    result = calibration(confidence, correct)
    assert result["ece"] < 0.01
    assert abs(result["overconfidence"]) < 0.01


def test_an_overconfident_model_is_reported_as_overconfident():
    """ECE is an absolute value and cannot say which way the model is
    wrong, so the signed gap has to survive alongside it.
    """
    confidence = np.full(1000, 0.9)
    correct = np.zeros(1000, dtype=bool)
    correct[:600] = True
    result = calibration(confidence, correct)
    assert result["ece"] == pytest.approx(0.3, abs=1e-6)
    assert result["overconfidence"] == pytest.approx(0.3, abs=1e-6)


def test_an_underconfident_model_gets_the_opposite_sign():
    confidence = np.full(1000, 0.4)
    correct = np.ones(1000, dtype=bool)
    result = calibration(confidence, correct)
    assert result["ece"] == pytest.approx(0.6, abs=1e-6)
    assert result["overconfidence"] == pytest.approx(-0.6, abs=1e-6)


def test_empty_bins_are_dropped_rather_than_counted_as_perfect():
    """A bin with no rows is not evidence of anything."""
    confidence = np.full(100, 0.95)
    correct = np.ones(100, dtype=bool)
    result = calibration(confidence, correct)
    assert len(result["bins"]) == 1
    assert result["ece"] == pytest.approx(0.05, abs=1e-6)


def test_confidence_of_exactly_one_lands_in_the_last_bin():
    result = calibration(np.ones(10), np.ones(10, dtype=bool))
    assert len(result["bins"]) == 1
    assert result["bins"][0]["high"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Ranked-pick metrics
# --------------------------------------------------------------------------

def test_top1_matches_an_independent_count(corpus):
    """Computed from the arrays directly, without the model, for a stub
    whose ordering is fixed: the human's pick is top-1 exactly when no
    other card in the pack has a higher id.
    """
    drafter = _stub_drafter(corpus)
    rows = np.arange(500)
    scored = ranked_probabilities(drafter.probe, corpus, rows)

    packs = corpus.pack[rows]
    labels = corpus.label[rows]
    beaten = ((packs > labels[:, None]) & (packs >= 0)).sum(axis=1)
    assert (scored["label_rank"] == beaten + 1).all()

    metrics = ranking_metrics(scored)
    assert metrics["top1"] == pytest.approx(float((beaten == 0).mean()))


def test_top3_is_at_least_top1(corpus):
    drafter = _drafter(corpus)
    rows = np.arange(400)
    metrics = ranking_metrics(ranked_probabilities(drafter.probe, corpus, rows))
    assert metrics["top3"] >= metrics["top1"]
    assert 0.0 <= metrics["top1"] <= 1.0


def test_a_tie_ranks_the_human_pick_first(corpus):
    """Counting ties as beating the label would report a model that splits
    its mass evenly as worse than it is.
    """
    import jax.numpy as jnp

    class _Flat:
        def apply(self, params, table, pack_ids, pool_ids, pack_number, pick_number):
            return jnp.where(pack_ids >= 0, 0.0, -1e9)

    probe = PickProbe(
        _Flat(), {}, np.zeros((corpus.vocab.size, 3), dtype=np.float32),
        corpus.geometry, corpus.vocab,
    )
    scored = ranked_probabilities(probe, corpus, np.arange(200))
    assert (scored["label_rank"] == 1).all()


def test_trivial_packs_are_counted_and_reported(corpus):
    """A pack of one card cannot be got wrong, and a top-1 number that does
    not say how many of those it contains is not interpretable.
    """
    drafter = _drafter(corpus)
    rows = np.arange(corpus.size)
    metrics = ranking_metrics(ranked_probabilities(drafter.probe, corpus, rows))
    forced = (corpus.pack_size[rows] <= 1).mean()
    assert metrics["top1_trivial_fraction"] == pytest.approx(forced)
    assert metrics["top3_trivial_fraction"] > metrics["top1_trivial_fraction"]


def test_the_report_separates_the_headline_slice(corpus):
    """picks 0-8 is the headline, per PROJECT_PLAN section 7, and it must
    hold none of the forced one-card packs the aggregate is full of.
    """
    drafter = _drafter(corpus)
    rows = np.arange(corpus.size)
    report = ranking_report(drafter.probe, corpus, rows)

    assert set(report) == {"all_picks", "decision_picks"}
    assert report["decision_picks"]["picks"] == "0-8"
    assert report["decision_picks"]["rows"] < report["all_picks"]["rows"]
    # The decision slice is where the packs are big, so nothing in it is free.
    assert report["decision_picks"]["top1_trivial_fraction"] == 0.0
    assert report["all_picks"]["top1_trivial_fraction"] > 0.0
    assert (
        report["decision_picks"]["mean_pack_size"]
        > report["all_picks"]["mean_pack_size"]
    )


def test_the_forced_picks_flatter_the_aggregate_calibration(corpus):
    """The reason calibration is reported twice: one-card packs are a
    perfectly calibrated 100%-confidence bin for any model at all.
    """
    drafter = _drafter(corpus)
    report = ranking_report(drafter.probe, corpus, np.arange(corpus.size))
    all_ece = report["all_picks"]["calibration"]["ece"]
    decision_ece = report["decision_picks"]["calibration"]["ece"]
    assert all_ece < decision_ece


def test_the_formatted_report_names_the_headline(corpus):
    drafter = _drafter(corpus)
    text = format_ranking_metrics(
        ranking_report(drafter.probe, corpus, np.arange(600))
    )
    assert "headline" in text
    assert "top-1" in text and "top-3" in text and "ECE" in text


def test_metrics_on_an_empty_selection_do_not_crash(corpus):
    drafter = _drafter(corpus)
    scored = ranked_probabilities(drafter.probe, corpus, np.arange(100))
    empty = ranking_metrics(scored, np.zeros(100, dtype=bool))
    assert empty["rows"] == 0


# --------------------------------------------------------------------------
# The CLI, end to end against a saved checkpoint
# --------------------------------------------------------------------------

@pytest.fixture
def saved_run(ingested, tmp_path):
    """A processed corpus, a feature table, and a checkpoint over it."""
    import jax
    import jax.numpy as jnp

    from src.data.card_features import build_features
    from src.models.pick_model import ModelConfig, PickModel
    from src.training.checkpoint import save_checkpoint

    data, _stats, processed = ingested(count=40)
    cards = {
        name: {
            "name": name, "oracle_text": "", "type_line": "Creature — Human",
            "cmc": 2.0, "color_identity": ["R"], "colors": ["R"],
            "keywords": [], "rarity": "common", "power": "2", "toughness": "2",
        }
        for name in data.vocab.id_to_card
    }
    features = build_features(data.vocab, cards)
    features.save(processed / "card_features.npz")

    table = jnp.asarray(features.dense())
    config = ModelConfig(
        hidden_dim=16,
        card_feature_dim=table.shape[1],
        packs_per_draft=data.packs_per_draft,
        picks_per_pack=data.picks_per_pack,
    )
    model = PickModel(config=config, arm="attention")
    batch = data.batch(np.arange(4))
    params = model.init(
        jax.random.PRNGKey(0), table,
        jnp.asarray(batch["pack_ids"]), jnp.asarray(batch["pool_ids"]),
        jnp.asarray(batch["pack_number"]), jnp.asarray(batch["pick_number"]),
    )
    checkpoint = tmp_path / "run"
    save_checkpoint(checkpoint, params, model_config=config, arm="attention")
    return checkpoint, processed, data


def test_cli_ranks_a_named_pack(saved_run, capsys):
    from src.inference.drafter import main

    checkpoint, processed, _data = saved_run
    code = main([
        "--checkpoint", str(checkpoint), "--processed-dir", str(processed),
        "--pack", "Card 00", "Card 01", "Zidane, Tantalus Thief",
        "--pool", "Card 02",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "Zidane, Tantalus Thief" in out
    assert "1 in pool" in out


def test_cli_reports_examples_and_metrics(saved_run, tmp_path, capsys):
    import json

    from src.inference.drafter import main

    checkpoint, processed, _data = saved_run
    out_path = tmp_path / "picks.json"
    code = main([
        "--checkpoint", str(checkpoint), "--processed-dir", str(processed),
        "--examples", "2", "--evaluate", "--rows", "500",
        "--json-out", str(out_path),
    ])
    assert code == 0
    text = capsys.readouterr().out
    assert "human took" in text
    assert "headline" in text

    report = json.loads(out_path.read_text())
    assert len(report["examples"]) == 2
    assert report["evaluation"]["decision_picks"]["rows"] > 0
    assert 0.0 <= report["evaluation"]["decision_picks"]["top1"] <= 1.0


def test_cli_rejects_an_unknown_card(saved_run):
    from src.inference.drafter import main

    checkpoint, processed, _data = saved_run
    with pytest.raises(UnknownCardError):
        main([
            "--checkpoint", str(checkpoint), "--processed-dir", str(processed),
            "--pack", "Card 00", "Not A Real Card",
        ])
