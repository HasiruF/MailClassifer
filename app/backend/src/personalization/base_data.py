import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp

from src.config import settings


@dataclass(frozen=True)
class BaseBundle:
    X: sp.csr_matrix
    y: np.ndarray
    gate_X: sp.csr_matrix | None
    gate_y: np.ndarray | None
    # Unfitted clone of the production estimator. A shared template:
    # callers clone() it and never fit it directly.
    estimator: object
    base_gate_accuracy: float | None
    classes: list


def _dir() -> Path:
    return Path(settings.training_data_dir)


@lru_cache(maxsize=None)
def load_meta() -> dict:
    return json.loads((_dir() / "meta.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_base_bundle(name: str) -> BaseBundle:
    directory = _dir()
    meta = load_meta()[name]
    labels = np.load(directory / f"{name}_labels.npz")
    gate_path = directory / f"{name}_gate_X.npz"
    return BaseBundle(
        X=sp.load_npz(directory / f"{name}_X.npz").tocsr(),
        y=labels["y"],
        gate_X=sp.load_npz(gate_path).tocsr() if gate_path.exists() else None,
        gate_y=labels["gate_y"] if "gate_y" in labels.files else None,
        estimator=joblib.load(directory / f"{name}_estimator.joblib"),
        base_gate_accuracy=meta.get("base_gate_accuracy"),
        classes=meta.get("classes", []),
    )


def feature_dim(name: str) -> int:
    return int(load_meta()[name]["dim"])
