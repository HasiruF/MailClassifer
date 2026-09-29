import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.personalization import base_data

REAL_DIR = Path(__file__).resolve().parents[1] / "training_data"


@pytest.fixture
def fresh_cache():
    base_data.load_meta.cache_clear()
    base_data.load_base_bundle.cache_clear()
    yield
    base_data.load_meta.cache_clear()
    base_data.load_base_bundle.cache_clear()


def test_loads_a_bundle_from_the_training_data_dir(tmp_path, monkeypatch, fresh_cache):
    sp.save_npz(tmp_path / "category_X.npz", sp.csr_matrix(np.eye(3)))
    sp.save_npz(tmp_path / "category_gate_X.npz", sp.csr_matrix(np.eye(3)))
    np.savez(
        tmp_path / "category_labels.npz",
        y=np.array(["Work", "Other", "Work"]),
        gate_y=np.array(["Work", "Other", "Other"]),
    )
    joblib.dump(LogisticRegression(C=3.0), tmp_path / "category_estimator.joblib")
    (tmp_path / "meta.json").write_text(
        json.dumps({"category": {"dim": 3, "classes": ["Other", "Work"], "base_gate_accuracy": 0.66}})
    )
    monkeypatch.setattr(settings, "training_data_dir", str(tmp_path))

    bundle = base_data.load_base_bundle("category")

    assert bundle.X.shape == (3, 3)
    assert list(bundle.y) == ["Work", "Other", "Work"]
    assert bundle.gate_X.shape == (3, 3)
    assert list(bundle.gate_y) == ["Work", "Other", "Other"]
    assert bundle.estimator.C == 3.0
    assert bundle.base_gate_accuracy == 0.66
    assert bundle.classes == ["Other", "Work"]
    assert base_data.feature_dim("category") == 3


@pytest.mark.skipif(not (REAL_DIR / "meta.json").exists(), reason="run scripts/export_training_matrices.py first")
@pytest.mark.parametrize("name", ["spam", "category", "priority", "priority_regressor"])
def test_real_bundles_are_internally_consistent(name, monkeypatch, fresh_cache):
    monkeypatch.setattr(settings, "training_data_dir", str(REAL_DIR))
    bundle = base_data.load_base_bundle(name)
    assert bundle.X.shape[1] == base_data.feature_dim(name)
    assert bundle.X.shape[0] == len(bundle.y)
    if bundle.gate_X is not None:
        assert bundle.gate_X.shape[1] == bundle.X.shape[1]
        assert bundle.gate_X.shape[0] == len(bundle.gate_y)
