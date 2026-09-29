from dataclasses import asdict, dataclass

import numpy as np

OWN_CORRECTION_MIN = 0.80
MAX_GATE_DROP = 0.03
MAX_CUSTOM_LABEL_FRACTION = 0.10


@dataclass
class GateResult:
    passed: bool
    own_correction_accuracy: float
    gate_accuracy: float
    base_gate_accuracy: float
    custom_label_fraction: float
    reason: str | None

    def as_metrics(self) -> dict:
        return asdict(self)


def evaluate_gate(fitted, corr_X, corr_y, gate_X, gate_y, base_gate_accuracy, base_classes) -> GateResult:
    own = float(np.mean(fitted.predict(corr_X) == corr_y)) if corr_X.shape[0] else 0.0

    predictions = fitted.predict(gate_X)
    known = set(base_classes)
    custom_mask = np.array([p not in known for p in predictions], dtype=bool)
    custom_fraction = float(custom_mask.mean()) if len(predictions) else 0.0
    # The gate set's ground truth only knows the base classes, so emails
    # predicted as a custom label can't be scored right or wrong.
    scored = ~custom_mask
    gate_accuracy = float(np.mean(predictions[scored] == gate_y[scored])) if scored.any() else 0.0

    reason = None
    if own < OWN_CORRECTION_MIN:
        reason = f"only {own:.0%} of your corrections are predicted as corrected (need {OWN_CORRECTION_MIN:.0%})"
    elif custom_fraction > MAX_CUSTOM_LABEL_FRACTION:
        reason = (f"{custom_fraction:.0%} of the test set was pulled into custom labels "
                  f"(max {MAX_CUSTOM_LABEL_FRACTION:.0%})")
    elif gate_accuracy < base_gate_accuracy - MAX_GATE_DROP:
        reason = (f"test-set accuracy fell from {base_gate_accuracy:.1%} to {gate_accuracy:.1%} "
                  f"(max drop {MAX_GATE_DROP:.0%})")

    return GateResult(
        passed=reason is None,
        own_correction_accuracy=own,
        gate_accuracy=gate_accuracy,
        base_gate_accuracy=float(base_gate_accuracy),
        custom_label_fraction=custom_fraction,
        reason=reason,
    )
