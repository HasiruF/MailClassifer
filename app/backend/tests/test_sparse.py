import numpy as np
import pytest
from pydantic import ValidationError

from src.personalization.sparse import SparseVector, to_csr
from src.schemas.personalization import CorrectionIn


def test_to_csr_places_values_at_indices():
    vectors = [
        SparseVector(dim=4, indices=[1, 3], values=[0.5, -2.0]),
        SparseVector(dim=4, indices=[], values=[]),
    ]
    matrix = to_csr(vectors, 4)
    assert matrix.shape == (2, 4)
    np.testing.assert_array_equal(matrix.toarray(), [[0, 0.5, 0, -2.0], [0, 0, 0, 0]])


def test_to_csr_of_nothing_is_an_empty_matrix():
    assert to_csr([], 4).shape == (0, 4)


def test_to_csr_rejects_wrong_dim():
    with pytest.raises(ValueError, match="dim"):
        to_csr([SparseVector(dim=3, indices=[0], values=[1.0])], 4)


def test_sparse_vector_rejects_bad_shapes():
    with pytest.raises(ValidationError):
        SparseVector(dim=3, indices=[0, 1], values=[1.0])
    with pytest.raises(ValidationError):
        SparseVector(dim=3, indices=[3], values=[1.0])
    with pytest.raises(ValidationError):
        SparseVector(dim=3, indices=[1, 1], values=[1.0, 2.0])


def _payload(model, label):
    return {
        "model": model,
        "provider_message_id": "m1",
        "feature_vector": {"dim": 2, "indices": [0], "values": [1.0]},
        "predicted_label": "x",
        "predicted_confidence": 0.5,
        "corrected_label": label,
    }


def test_priority_and_spam_labels_are_fixed_sets():
    with pytest.raises(ValidationError):
        CorrectionIn(**_payload("priority", "urgent"))
    with pytest.raises(ValidationError):
        CorrectionIn(**_payload("spam", "junk"))
    assert CorrectionIn(**_payload("priority", "high")).corrected_label == "high"


def test_category_accepts_a_custom_label_and_trims_it():
    assert CorrectionIn(**_payload("category", "  Finance ")).corrected_label == "Finance"
