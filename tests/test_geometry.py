"""Pack geometry must be measured from the export, not assumed.

Every test elsewhere in the suite uses Arena's usual 3 packs x 14 picks,
which is exactly the shape a hardcoded 14 gets right. So these tests build
exports at a *different* geometry and check that the whole chain -- ingest,
the persisted stats file, PickData's pool-as-prefix identity, and the two
ContextFeatures embedding sizes -- follows the data.

The failure this guards against is silent, which is why it is worth a
dedicated module: with the wrong picks_per_pack, `pack_number * ppp +
pick_number` disagrees with every row's position, every draft is judged
invalid, and PickData's default `on_invalid="drop"` throws away the entire
corpus without raising anything at all.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.dataset import DEFAULT_GEOMETRY, PickData, matched_state_groups
from src.data.ingest import PackGeometry, ingest, load_geometry

from .synthetic import CARDS, make_drafts, write_export

# Four packs of eight, against the default's three of fourteen. Both numbers
# differ, so a test cannot pass by getting one of them right by accident.
PACKS = 4
PICKS = 8


@pytest.fixture
def odd_geometry_corpus(tmp_path):
    """An ingested export at PACKS x PICKS instead of 3 x 14."""
    rng = np.random.default_rng(0)
    drafts = make_drafts(rng, 12, picks_per_pack=PICKS, packs_per_draft=PACKS)
    csv_path = tmp_path / "draft_data_public.ODD.PremierDraft.csv.gz"
    write_export(csv_path, drafts)
    out = tmp_path / "processed"
    stats = ingest(csv_path, out, verbose=False)
    return out, stats, drafts


def test_ingest_measures_the_geometry_it_saw(odd_geometry_corpus):
    out, stats, drafts = odd_geometry_corpus

    assert stats.geometry == PackGeometry(
        packs_per_draft=PACKS, picks_per_pack=PICKS, max_pack_size=PICKS
    )
    assert stats.geometry != DEFAULT_GEOMETRY
    assert stats.rows == len(drafts) * PACKS * PICKS
    # max_pack_seen is kept as an alias so older callers still read something
    # meaningful, but it now comes from the measured geometry.
    assert stats.max_pack_seen == PICKS


def test_geometry_is_persisted_and_read_back(odd_geometry_corpus):
    out, stats, _ = odd_geometry_corpus

    payload = json.loads((out / "ingest_stats.json").read_text(encoding="utf-8"))
    assert payload["geometry"] == {
        "packs_per_draft": PACKS,
        "picks_per_pack": PICKS,
        "max_pack_size": PICKS,
    }
    assert load_geometry(out) == stats.geometry

    data = PickData.load(out)
    assert data.geometry == stats.geometry
    assert (data.packs_per_draft, data.picks_per_pack) == (PACKS, PICKS)


def test_a_non_default_corpus_survives_the_prefix_identity(odd_geometry_corpus):
    """The whole point: at the wrong geometry every draft looks invalid."""
    out, stats, drafts = odd_geometry_corpus

    data = PickData.load(out)
    assert data.dropped_drafts == 0
    assert data.dropped_rows == 0
    assert data.size == len(drafts) * PACKS * PICKS
    # Loading with on_invalid="raise" is the strongest form of the check.
    assert PickData.load(out, on_invalid="raise").size == data.size

    # ...and the counterfactual, which is what the old hardcoded constants
    # produced: force the default 3 x 14 onto this corpus and every draft
    # fails the identity. That now raises with a diagnosis rather than
    # handing back an empty corpus, which is the failure mode that cost a
    # real set (AFR) an afternoon to track down.
    with np.load(out / "picks.npz") as handle:
        arrays = {name: handle[name] for name in handle.files}
    with pytest.raises(ValueError, match="Most drafts have"):
        PickData(arrays, data.vocab, geometry=DEFAULT_GEOMETRY)

    forced = PickData(
        arrays, data.vocab, geometry=DEFAULT_GEOMETRY, max_dropped_fraction=1.0
    )
    assert forced.size == 0
    assert forced.dropped_drafts == len(drafts)


def test_pool_and_pack_widths_follow_the_geometry(odd_geometry_corpus):
    out, _, _ = odd_geometry_corpus
    data = PickData.load(out)

    # Pack ids are padded to the widest pack the file held, not to 14.
    assert data.pack.shape[1] == PICKS
    assert (data.pack_size == PICKS - data.pick_number).all()

    pools = data.pools_padded(np.arange(data.size))
    assert pools.shape == (data.size, PACKS * PICKS - 1)
    assert data.max_pool_size == PACKS * PICKS - 1

    # Every reconstructed pool is exactly the picks before it in its draft.
    for i in rng_sample(data.size, 40):
        expected = sorted(int(c) for c in data.pool_of(i))
        assert sorted(int(c) for c in pools[i] if c >= 0) == expected
        assert len(expected) == int(data.pack_number[i]) * PICKS + int(
            data.pick_number[i]
        )


def test_matched_state_bucketing_uses_the_measured_geometry(odd_geometry_corpus):
    """The bucket key is `pack_number * picks_per_pack + pick_number`. At the
    wrong picks_per_pack, states from different picks collide into one bucket
    and the Bayes-floor groups become nonsense rather than empty.
    """
    out, _, _ = odd_geometry_corpus
    data = PickData.load(out)
    rows, groups = matched_state_groups(data)
    for group in np.unique(groups):
        block = rows[groups == group]
        assert len({int(data.pick_number[i]) for i in block}) == 1
        assert len({int(data.pack_number[i]) for i in block}) == 1


def test_model_context_embeddings_are_sized_from_the_data(odd_geometry_corpus):
    """ContextFeatures holds one embedding row per pack and per pick number.

    Out-of-range indices do not raise under JAX -- they clamp and quietly
    return the wrong row -- so the only way to catch an undersized embedding
    is to check its shape.
    """
    import jax
    import jax.numpy as jnp

    from src.models.pick_model import ModelConfig, PickModel

    out, _, _ = odd_geometry_corpus
    data = PickData.load(out)

    config = ModelConfig(
        hidden_dim=16,
        card_feature_dim=7,
        packs_per_draft=data.packs_per_draft,
        picks_per_pack=data.picks_per_pack,
    )
    model = PickModel(config=config, arm="attention")
    table = jnp.zeros((len(CARDS), config.card_feature_dim), dtype=jnp.float32)
    batch = data.batch(np.arange(min(8, data.size)))
    params = model.init(
        jax.random.PRNGKey(0),
        table,
        jnp.asarray(batch["pack_ids"]),
        jnp.asarray(batch["pool_ids"]),
        jnp.asarray(batch["pack_number"]),
        jnp.asarray(batch["pick_number"]),
    )
    context = params["params"]["context"]
    assert context["pack_number"]["embedding"].shape == (PACKS, config.hidden_dim)
    assert context["pick_number"]["embedding"].shape == (PICKS, config.hidden_dim)

    # And the highest index the corpus can present is addressable.
    assert int(data.pack_number.max()) == PACKS - 1
    assert int(data.pick_number.max()) == PICKS - 1


def rng_sample(size: int, count: int) -> np.ndarray:
    return np.random.default_rng(0).choice(size, size=min(count, size), replace=False)
