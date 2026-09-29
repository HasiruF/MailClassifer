"""Small synthetic stand-ins for app/backend/training_data/ bundles.

Base rows only ever use features 0-4; feature 5 is always 0. A correction
that sets feature 5 is therefore learnable without contradicting base data,
the same situation as a user's emails containing words the base training
set rarely sees."""
import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge

from src.personalization.base_data import BaseBundle

DIM = 6
CLASSES = {
    "category": ["Other", "Personal", "Work"],
    "priority": ["high", "low", "medium"],
    "spam": [0, 1],
}
SEEDS = {"category": 0, "priority": 1, "spam": 2}


def _base_matrix(n: int, seed: int) -> np.ndarray:
    dense = np.zeros((n, DIM))
    dense[:, :5] = np.random.RandomState(seed).normal(size=(n, 5))
    return dense


def _labels(dense: np.ndarray, classes: list) -> np.ndarray:
    return np.asarray([classes[int(np.argmax(row[: len(classes)]))] for row in dense])


def make_bundle(name: str) -> BaseBundle:
    if name == "priority_regressor":
        X = _base_matrix(300, 3)
        y = np.clip(0.5 + 0.2 * X[:, 0], 0.1, 1.0)
        return BaseBundle(
            X=sp.csr_matrix(X), y=y, gate_X=None, gate_y=None,
            estimator=Ridge(alpha=1.0), base_gate_accuracy=None, classes=[],
        )
    classes = CLASSES[name]
    X = _base_matrix(300, SEEDS[name])
    gate_X = _base_matrix(60, 99)
    y = _labels(X, classes)
    gate_y = _labels(gate_X, classes)
    estimator = LogisticRegression(max_iter=1000)
    fitted = clone(estimator).fit(sp.csr_matrix(X), y)
    base_accuracy = float(np.mean(fitted.predict(sp.csr_matrix(gate_X)) == gate_y))
    return BaseBundle(
        X=sp.csr_matrix(X), y=y, gate_X=sp.csr_matrix(gate_X), gate_y=gate_y,
        estimator=estimator, base_gate_accuracy=base_accuracy, classes=list(classes),
    )


def patch_base_data(monkeypatch) -> None:
    from src.personalization import base_data

    monkeypatch.setattr(base_data, "load_base_bundle", make_bundle)
    monkeypatch.setattr(base_data, "feature_dim", lambda name: DIM)
