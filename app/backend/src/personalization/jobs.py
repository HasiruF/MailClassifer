"""Per-user retrain jobs (spec §4.2). run_retrain is synchronous: FastAPI
runs it in its background-task threadpool, one job at a time per
(user, model)."""
import threading
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from src.database import SessionLocal
from src.models import (
    Correction,
    CorrectionModel,
    EmailConnection,
    PersonalizedModel,
    PersonalizedModelKind,
    PersonalizedModelStatus,
)
from src.personalization import base_data, trainer
from src.personalization.gate import evaluate_gate
from src.personalization.sparse import SparseVector, to_csr

RETRAIN_THRESHOLD = 5

_running: set[tuple[uuid.UUID, str]] = set()
_running_lock = threading.Lock()


def is_running(user_id: uuid.UUID, model: CorrectionModel) -> bool:
    with _running_lock:
        return (user_id, model.value) in _running


def _claim(user_id: uuid.UUID, model: CorrectionModel) -> bool:
    with _running_lock:
        key = (user_id, model.value)
        if key in _running:
            return False
        _running.add(key)
        return True


def _release(user_id: uuid.UUID, model: CorrectionModel) -> None:
    with _running_lock:
        _running.discard((user_id, model.value))


def scoped_corrections(stmt, user_id: uuid.UUID, model: CorrectionModel):
    return stmt.join(EmailConnection, Correction.email_connection_id == EmailConnection.id).where(
        EmailConnection.user_id == user_id, Correction.model == model
    )


def correction_count(db: OrmSession, user_id: uuid.UUID, model: CorrectionModel) -> int:
    return db.scalar(scoped_corrections(select(func.count(Correction.id)), user_id, model)) or 0


def last_attempt(db: OrmSession, user_id: uuid.UUID, kind: PersonalizedModelKind) -> PersonalizedModel | None:
    return db.scalars(
        select(PersonalizedModel)
        .where(PersonalizedModel.user_id == user_id, PersonalizedModel.model == kind)
        .order_by(PersonalizedModel.version.desc())
        .limit(1)
    ).first()


def corrections_until_retrain(db: OrmSession, user_id: uuid.UUID, model: CorrectionModel) -> int:
    last = last_attempt(db, user_id, PersonalizedModelKind(model.value))
    new = correction_count(db, user_id, model) - (last.correction_count if last else 0)
    return max(0, RETRAIN_THRESHOLD - new)


def has_changes_since_last_attempt(db: OrmSession, user_id: uuid.UUID, model: CorrectionModel) -> bool:
    latest = db.scalar(scoped_corrections(select(func.max(Correction.created_at)), user_id, model))
    if latest is None:
        return False
    last = last_attempt(db, user_id, PersonalizedModelKind(model.value))
    return last is None or latest > last.created_at


def _next_version(db: OrmSession, user_id: uuid.UUID, kind: PersonalizedModelKind) -> int:
    current = db.scalar(
        select(func.max(PersonalizedModel.version)).where(
            PersonalizedModel.user_id == user_id, PersonalizedModel.model == kind
        )
    )
    return (current or 0) + 1


def _record_rejection(db, user_id, kind, count: int, metrics: dict) -> None:
    db.add(PersonalizedModel(
        user_id=user_id,
        model=kind,
        version=_next_version(db, user_id, kind),
        status=PersonalizedModelStatus.rejected,
        onnx=None,
        classes=None,
        correction_count=count,
        metrics=metrics,
    ))
    db.commit()


def _activate(db, user_id, kind, fitted, dim: int, count: int, metrics: dict) -> None:
    previous_active = db.scalars(
        select(PersonalizedModel).where(
            PersonalizedModel.user_id == user_id,
            PersonalizedModel.model == kind,
            PersonalizedModel.status == PersonalizedModelStatus.active,
        )
    ).all()
    for previous in previous_active:
        previous.status = PersonalizedModelStatus.superseded
    db.add(PersonalizedModel(
        user_id=user_id,
        model=kind,
        version=_next_version(db, user_id, kind),
        status=PersonalizedModelStatus.active,
        onnx=trainer.export_onnx(fitted, dim),
        classes=fitted.classes_.tolist() if hasattr(fitted, "classes_") else None,
        correction_count=count,
        metrics=metrics,
    ))


def run_retrain(user_id: uuid.UUID, model: CorrectionModel) -> None:
    if not _claim(user_id, model):
        return
    try:
        with SessionLocal() as db:
            _retrain(db, user_id, model)
    finally:
        _release(user_id, model)


def _retrain(db: OrmSession, user_id: uuid.UUID, model: CorrectionModel) -> None:
    rows = db.scalars(scoped_corrections(select(Correction), user_id, model).order_by(Correction.created_at)).all()
    if not rows:
        return
    kind = PersonalizedModelKind(model.value)
    labels = [row.corrected_label for row in rows]
    try:
        bundle = base_data.load_base_bundle(model.value)
        dim = bundle.X.shape[1]
        corr_X = to_csr([SparseVector.model_validate(row.feature_vector) for row in rows], dim)
        corr_y = trainer.correction_targets(model.value, labels)
        fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
        gate = evaluate_gate(
            fitted, corr_X, corr_y, bundle.gate_X, bundle.gate_y, bundle.base_gate_accuracy, bundle.classes
        )
        if not gate.passed:
            _record_rejection(db, user_id, kind, len(rows), gate.as_metrics())
            return

        outputs = [(kind, fitted)]
        if model == CorrectionModel.priority:
            # The score bar comes from the regressor; retraining only the
            # bucket classifier would let the two disagree. Promoted together.
            reg_bundle = base_data.load_base_bundle("priority_regressor")
            if reg_bundle.X.shape[1] != dim:
                raise ValueError(f"priority_regressor dim {reg_bundle.X.shape[1]} != priority dim {dim}")
            reg_y = trainer.correction_targets("priority_regressor", labels)
            outputs.append((
                PersonalizedModelKind.priority_regressor,
                trainer.fit_personalized(reg_bundle.X, reg_bundle.y, reg_bundle.estimator, corr_X, reg_y),
            ))
        for out_kind, out_fitted in outputs:
            _activate(db, user_id, out_kind, out_fitted, dim, len(rows), gate.as_metrics())
        db.commit()
    except Exception as exc:  # a background job has no caller to raise to; record it instead
        db.rollback()
        _record_rejection(db, user_id, kind, len(rows), {"error": f"{type(exc).__name__}: {exc}"})
