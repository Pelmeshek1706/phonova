import numpy as np
import pytest

from tests.helpers.module_loaders import load_local_coherence_module


def test_normalize_embeddings_returns_unit_rows():
    coherence = load_local_coherence_module()

    embeddings = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = coherence._normalize_embeddings(embeddings)

    assert normalized.shape == (2, 2)
    assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.all(normalized[1] == 0.0)


def test_cosine_for_offset_and_slope_helpers_are_stable():
    coherence = load_local_coherence_module()

    normalized = coherence._normalize_embeddings(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    similarities = coherence._cosine_for_offset(normalized, 1)

    assert similarities.tolist() == [0.0, 0.0]
    assert coherence.calculate_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)
