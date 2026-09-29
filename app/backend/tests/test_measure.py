import numpy as np
import pytest
import scipy.sparse as sp

from src.personalization import trainer
from src.personalization.measure import measure
from tests.fakes import make_bundle

POINT = np.array([1.0, 0, 0, 0, 0, 5.0])


def test_reports_before_and_after_on_held_out_corrections():
    bundle = make_bundle("category")
    corr_X = sp.csr_matrix(np.tile(POINT, (8, 1)))
    corr_y = trainer.correction_targets("category", ["Work"] * 8)

    rows = measure(bundle, corr_X, corr_y, weights=[1.0, 10.0])

    assert [r["weight"] for r in rows] == [1.0, 10.0]
    assert rows[1]["held_out_before"] == 0.0
    assert rows[1]["held_out_after"] == 1.0
    assert rows[1]["gate_before"] == bundle.base_gate_accuracy
    assert 0.0 <= rows[1]["gate_after"] <= 1.0


def test_needs_enough_corrections_to_split():
    bundle = make_bundle("category")
    corr_X = sp.csr_matrix(np.tile(POINT, (3, 1)))
    with pytest.raises(ValueError, match="at least 4"):
        measure(bundle, corr_X, trainer.correction_targets("category", ["Work"] * 3), weights=[10.0])
