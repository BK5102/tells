"""Real TSFM adapters. Skipped unless the optional `tsfm` extra is installed (CI does not)."""

import numpy as np
import pytest

from tells.models.tsfm.base import run_adapter

SEQS = [
    np.zeros((3, 0), dtype=np.float32),  # an account's first day: no history
    np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.5], [1.0, 1.0, -1.0]], dtype=np.float32),
    np.random.default_rng(0).normal(size=(3, 40)).astype(np.float32),
]


@pytest.mark.parametrize("name", ["ttm", "chronos2"])
def test_adapter_shapes_empty_rows_and_determinism(name: str) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tsfm_public" if name == "ttm" else "chronos")
    from tells.models.tsfm import make_adapter

    adapter = make_adapter(name)
    out = run_adapter(adapter, SEQS)
    assert out.embeddings.shape == (3, adapter.dim)
    assert out.forecasts.shape == (3, 2)
    assert np.isnan(out.embeddings[0]).all()  # empty history stays empty
    assert np.isfinite(out.embeddings[1:]).all() and np.isfinite(out.forecasts[1:]).all()
    again = run_adapter(adapter, SEQS)
    np.testing.assert_allclose(out.embeddings[1:], again.embeddings[1:], rtol=1e-5, atol=1e-6)
    assert adapter.version
