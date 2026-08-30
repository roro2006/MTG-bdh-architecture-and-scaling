"""Shared fixtures.

Model initialisation and JAX compilation dominate this suite's runtime, so
anything reusable is session-scoped. The feature table in particular is
built once: every model test wants the same one, and rebuilding it per
module was buying nothing.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.models.pick_model import ModelConfig

from . import synthetic

VOCAB_SIZE = 40


@pytest.fixture(scope="session")
def feature_table():
    """A (VOCAB_SIZE, 65) stand-in for the real Scryfall-derived table."""
    return jnp.asarray(synthetic.make_feature_table(VOCAB_SIZE, seed=0))


@pytest.fixture(scope="session")
def vocab_size():
    return VOCAB_SIZE


@pytest.fixture
def model_config():
    """Factory for a small config; call with a width."""

    def build(hidden_dim: int = 32, **overrides) -> ModelConfig:
        defaults = dict(
            hidden_dim=hidden_dim,
            num_heads=4,
            pool_encoder_layers=2,
            pack_encoder_layers=1,
            arm_layers=2,
            card_feature_dim=synthetic.FEATURE_DIM,
        )
        defaults.update(overrides)
        return ModelConfig(**defaults)

    return build


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_export(tmp_path):
    """Factory writing a synthetic export; returns its path.

    Takes either a draft count or an explicit list of drafts, so a test that
    needs a specific malformed shape can build one.
    """

    def build(drafts=None, count: int = 40, name: str = "export.csv.gz", seed: int = 0):
        if drafts is None:
            drafts = synthetic.make_drafts(np.random.default_rng(seed), count)
        path = tmp_path / name
        synthetic.write_export(path, drafts)
        return path

    return build


@pytest.fixture
def ingested(synthetic_export, tmp_path):
    """Factory: writes an export, ingests it, returns (PickData, stats, path)."""

    def build(drafts=None, count: int = 40, seed: int = 0):
        from src.data.dataset import PickData
        from src.data.ingest import ingest

        csv_path = synthetic_export(drafts=drafts, count=count, seed=seed)
        out = tmp_path / "processed"
        stats = ingest(csv_path, out, verbose=False)
        return PickData.load(out), stats, out

    return build
