from sqlalchemy import select

from src.models import (
    Correction,
    CorrectionModel,
    PersonalizedModel,
    PersonalizedModelKind,
    PersonalizedModelStatus,
)
from src.personalization import jobs
from src.personalization.gate import GateResult
from tests.fakes import DIM, patch_base_data

POINT = {"dim": DIM, "indices": [0, 5], "values": [1.0, 5.0]}


def add_corrections(db, connection, model, label, n=5, start=0):
    for i in range(start, start + n):
        db.add(Correction(
            email_connection_id=connection.id,
            provider_message_id=f"msg-{i}",
            model=model,
            feature_vector=POINT,
            predicted_label="Other",
            predicted_confidence=0.7,
            corrected_label=label,
        ))
    db.commit()


def attempts(db, user_id, kind):
    db.expire_all()
    return db.scalars(
        select(PersonalizedModel)
        .where(PersonalizedModel.user_id == user_id, PersonalizedModel.model == kind)
        .order_by(PersonalizedModel.version)
    ).all()


def test_retrain_activates_a_personalized_category_model(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    add_corrections(db, connected.connection, CorrectionModel.category, "Finance")

    jobs.run_retrain(connected.user.id, CorrectionModel.category)

    (attempt,) = attempts(db, connected.user.id, PersonalizedModelKind.category)
    assert attempt.status == PersonalizedModelStatus.active
    assert attempt.version == 1 and attempt.correction_count == 5
    assert "Finance" in attempt.classes
    assert attempt.onnx and attempt.metrics["passed"] is True


def test_second_retrain_supersedes_the_first(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work")
    jobs.run_retrain(connected.user.id, CorrectionModel.category)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work", start=5)
    jobs.run_retrain(connected.user.id, CorrectionModel.category)

    first, second = attempts(db, connected.user.id, PersonalizedModelKind.category)
    assert (first.version, first.status) == (1, PersonalizedModelStatus.superseded)
    assert (second.version, second.status) == (2, PersonalizedModelStatus.active)
    assert second.correction_count == 10


def test_gate_failure_is_recorded_without_activating(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    failing = GateResult(
        passed=False, own_correction_accuracy=0.2, gate_accuracy=0.9, base_gate_accuracy=0.9,
        custom_label_fraction=0.0, reason="only 20% of your corrections are predicted as corrected (need 80%)",
    )
    monkeypatch.setattr(jobs, "evaluate_gate", lambda *args, **kwargs: failing)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work")

    jobs.run_retrain(connected.user.id, CorrectionModel.category)

    (attempt,) = attempts(db, connected.user.id, PersonalizedModelKind.category)
    assert attempt.status == PersonalizedModelStatus.rejected
    assert attempt.onnx is None
    assert attempt.metrics["reason"].startswith("only 20%")


def test_a_training_error_is_recorded_as_a_rejection(db, connected, monkeypatch):
    patch_base_data(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(jobs.trainer, "fit_personalized", boom)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work")

    jobs.run_retrain(connected.user.id, CorrectionModel.category)

    (attempt,) = attempts(db, connected.user.id, PersonalizedModelKind.category)
    assert attempt.status == PersonalizedModelStatus.rejected
    assert attempt.metrics == {"error": "RuntimeError: disk full"}


def test_priority_retrain_activates_classifier_and_regressor_together(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    add_corrections(db, connected.connection, CorrectionModel.priority, "high")

    jobs.run_retrain(connected.user.id, CorrectionModel.priority)

    (classifier,) = attempts(db, connected.user.id, PersonalizedModelKind.priority)
    (regressor,) = attempts(db, connected.user.id, PersonalizedModelKind.priority_regressor)
    assert classifier.status == regressor.status == PersonalizedModelStatus.active
    assert regressor.classes is None and regressor.onnx


def test_a_second_job_for_the_same_user_and_model_is_skipped(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work")
    assert jobs._claim(connected.user.id, CorrectionModel.category)
    try:
        jobs.run_retrain(connected.user.id, CorrectionModel.category)
    finally:
        jobs._release(connected.user.id, CorrectionModel.category)
    assert attempts(db, connected.user.id, PersonalizedModelKind.category) == []


def test_threshold_counts_only_corrections_since_the_last_attempt(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    user_id = connected.user.id
    add_corrections(db, connected.connection, CorrectionModel.category, "Work", n=3)
    assert jobs.corrections_until_retrain(db, user_id, CorrectionModel.category) == 2
    add_corrections(db, connected.connection, CorrectionModel.category, "Work", n=2, start=3)
    assert jobs.corrections_until_retrain(db, user_id, CorrectionModel.category) == 0

    jobs.run_retrain(user_id, CorrectionModel.category)

    db.expire_all()
    assert jobs.corrections_until_retrain(db, user_id, CorrectionModel.category) == 5
    assert jobs.has_changes_since_last_attempt(db, user_id, CorrectionModel.category) is False
