import numpy as np
import scipy.sparse as sp

from src.personalization.gate import evaluate_gate


class Scripted:
    """Returns canned predictions keyed by how many rows it's asked about."""

    def __init__(self, by_rows):
        self.by_rows = by_rows

    def predict(self, X):
        return np.asarray(self.by_rows[X.shape[0]])


BASE_CLASSES = ["Other", "Personal", "Work"]
CORR_X = sp.csr_matrix(np.zeros((5, 2)))
CORR_Y = np.array(["Work"] * 5)
GATE_X = sp.csr_matrix(np.zeros((10, 2)))
GATE_Y = np.array(["Other"] * 10)


def gate_for(own_pred, gate_pred, base_accuracy=0.9):
    fitted = Scripted({5: own_pred, 10: gate_pred})
    return evaluate_gate(fitted, CORR_X, CORR_Y, GATE_X, GATE_Y, base_accuracy, BASE_CLASSES)


def test_passes_when_corrections_took_and_nothing_regressed():
    result = gate_for(["Work"] * 5, ["Other"] * 9 + ["Work"])
    assert result.passed and result.reason is None
    assert result.own_correction_accuracy == 1.0
    assert result.gate_accuracy == 0.9


def test_rejects_when_corrections_did_not_take():
    result = gate_for(["Work"] * 3 + ["Other"] * 2, ["Other"] * 10)
    assert not result.passed
    assert "corrections" in result.reason


def test_rejects_an_accuracy_drop_over_three_points():
    result = gate_for(["Work"] * 5, ["Other"] * 8 + ["Work"] * 2, base_accuracy=0.9)
    assert not result.passed
    assert "accuracy" in result.reason


def test_custom_label_predictions_are_left_out_of_accuracy():
    result = gate_for(["Work"] * 5, ["Other"] * 9 + ["Finance"])
    assert result.passed
    assert result.custom_label_fraction == 0.1
    assert result.gate_accuracy == 1.0


def test_rejects_when_custom_labels_swallow_the_test_set():
    result = gate_for(["Work"] * 5, ["Other"] * 8 + ["Finance"] * 2)
    assert not result.passed
    assert "custom labels" in result.reason


def test_metrics_are_json_ready():
    metrics = gate_for(["Work"] * 5, ["Other"] * 10).as_metrics()
    assert metrics["passed"] is True
    assert set(metrics) == {
        "passed", "own_correction_accuracy", "gate_accuracy",
        "base_gate_accuracy", "custom_label_fraction", "reason",
    }
