import numpy as np
import onnxruntime as ort
import scipy.sparse as sp

from src.personalization import trainer
from tests.fakes import DIM, make_bundle

# Base models call this point "Other" (feature 0 wins); feature 5 is a
# signal only the corrections carry.
POINT = np.array([1.0, 0, 0, 0, 0, 5.0])
BASE_ONLY_POINT = np.array([1.0, 0, 0, 0, 0, 0])


def corrections_at(point, label, n=5, model="category"):
    return sp.csr_matrix(np.tile(point, (n, 1))), trainer.correction_targets(model, [label] * n)


def test_weighted_corrections_flip_the_corrected_region_only():
    bundle = make_bundle("category")
    corr_X, corr_y = corrections_at(POINT, "Work")
    fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    assert fitted.predict(sp.csr_matrix([POINT]))[0] == "Work"
    assert fitted.predict(sp.csr_matrix([BASE_ONLY_POINT]))[0] == "Other"


def test_a_custom_label_becomes_a_new_class():
    bundle = make_bundle("category")
    corr_X, corr_y = corrections_at(POINT, "Finance")
    fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    assert "Finance" in fitted.classes_
    assert fitted.predict(sp.csr_matrix([POINT]))[0] == "Finance"


def test_the_template_estimator_is_never_fitted():
    bundle = make_bundle("category")
    corr_X, corr_y = corrections_at(POINT, "Work")
    trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    assert not hasattr(bundle.estimator, "classes_")


def test_targets_are_mapped_per_model():
    assert trainer.correction_targets("spam", ["spam", "ham"]).tolist() == [1, 0]
    assert trainer.correction_targets("priority_regressor", ["low", "medium", "high"]).tolist() == [0.2, 0.5, 0.8]
    assert trainer.correction_targets("category", ["Work"]).tolist() == ["Work"]
    assert trainer.correction_targets("priority", ["high"]).tolist() == ["high"]


def test_exported_classifier_matches_sklearn():
    bundle = make_bundle("category")
    corr_X, corr_y = corrections_at(POINT, "Finance")
    fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    session = ort.InferenceSession(trainer.export_onnx(fitted, DIM), providers=["CPUExecutionProvider"])
    rows = np.vstack([POINT, BASE_ONLY_POINT]).astype(np.float32)
    label, proba = session.run(None, {"features": rows})
    np.testing.assert_array_equal(label, fitted.predict(sp.csr_matrix(rows)))
    np.testing.assert_allclose(proba, fitted.predict_proba(sp.csr_matrix(rows)), atol=1e-5)


def test_exported_regressor_matches_sklearn():
    bundle = make_bundle("priority_regressor")
    corr_X, corr_y = corrections_at(POINT, "high", model="priority_regressor")
    fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    session = ort.InferenceSession(trainer.export_onnx(fitted, DIM), providers=["CPUExecutionProvider"])
    rows = np.vstack([POINT]).astype(np.float32)
    (out,) = session.run(None, {"features": rows})
    np.testing.assert_allclose(out.ravel(), fitted.predict(sp.csr_matrix(rows)), atol=1e-5)
