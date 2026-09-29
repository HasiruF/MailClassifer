import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Starting value; the spec's §8 measurement picks the final one.
CORRECTION_WEIGHT = 10.0
# Regressor target for a corrected priority bucket.
BUCKET_SCORE = {"low": 0.2, "medium": 0.5, "high": 0.8}
SPAM_LABEL_TO_TARGET = {"ham": 0, "spam": 1}


def correction_targets(model: str, labels: list[str]) -> np.ndarray:
    if model == "spam":
        return np.array([SPAM_LABEL_TO_TARGET[label] for label in labels], dtype=np.int64)
    if model == "priority_regressor":
        return np.array([BUCKET_SCORE[label] for label in labels], dtype=np.float64)
    return np.array(labels, dtype=str)


def fit_personalized(base_X, base_y, estimator, corr_X, corr_y, weight: float = CORRECTION_WEIGHT):
    """Refit a clone of the production estimator on base rows plus corrections.

    Corrections are up-weighted, otherwise a handful of them against ~1,500
    base rows would barely move the model. Never fits `estimator` itself:
    it's the shared, cached template from base_data."""
    X = sp.vstack([base_X, corr_X]).tocsr()
    y = np.concatenate([np.asarray(base_y), np.asarray(corr_y)])
    sample_weight = np.concatenate([np.ones(base_X.shape[0]), np.full(corr_X.shape[0], weight)])
    fitted = clone(estimator)
    fitted.fit(X, y, sample_weight=sample_weight)
    return fitted


def export_onnx(fitted, dim: int) -> bytes:
    """Same headless export scripts/export_onnx.py uses: dense float vector
    in, classifier only, zipmap off so probabilities are a plain tensor."""
    is_classifier = hasattr(fitted, "classes_")
    options = {id(fitted): {"zipmap": False}} if is_classifier else None
    onnx_model = convert_sklearn(
        fitted,
        initial_types=[("features", FloatTensorType([None, dim]))],
        options=options,
        target_opset=17,
    )
    return onnx_model.SerializeToString()
