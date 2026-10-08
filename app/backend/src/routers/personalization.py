import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session as OrmSession

from src.database import get_db
from src.dependencies import get_current_session
from src.models import (
    Correction,
    CorrectionModel,
    EmailConnection,
    PersonalizedModel,
    PersonalizedModelKind,
    PersonalizedModelStatus,
    User,
)
from src.models import Session as SessionModel
from src.personalization import base_data, jobs
from src.personalization.sparse import SparseVector
from src.schemas.personalization import (
    CorrectionDetail,
    CorrectionIn,
    CorrectionOut,
    CorrectionSummary,
    LastAttempt,
    ManifestEntry,
    ModelStatus,
    RetrainOut,
    SettingsIn,
    SettingsOut,
    StatusOut,
)

router = APIRouter(prefix="/personalization", tags=["personalization"])

BASE_CATEGORIES = ("Work", "Personal", "Other")


def _current_user(
    session: SessionModel = Depends(get_current_session),
    db: OrmSession = Depends(get_db),
) -> User:
    user = db.get(User, session.user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user


def _require_enabled(user: User) -> None:
    if not user.personalization_enabled:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Personalization is not enabled")


def _current_connection(db: OrmSession, user_id: uuid.UUID) -> EmailConnection:
    connection = db.scalars(
        select(EmailConnection)
        .where(EmailConnection.user_id == user_id)
        .order_by(EmailConnection.last_used_at.desc())
        .limit(1)
    ).first()
    if connection is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No email connection for this session")
    return connection


@router.put("/settings", response_model=SettingsOut)
def update_settings(
    body: SettingsIn, user: User = Depends(_current_user), db: OrmSession = Depends(get_db)
) -> SettingsOut:
    user.personalization_enabled = body.enabled
    if not body.enabled:
        connection_ids = select(EmailConnection.id).where(EmailConnection.user_id == user.id)
        db.execute(
            delete(Correction)
            .where(Correction.email_connection_id.in_(connection_ids))
            .execution_options(synchronize_session=False)
        )
        db.execute(
            delete(PersonalizedModel)
            .where(PersonalizedModel.user_id == user.id)
            .execution_options(synchronize_session=False)
        )
    db.commit()
    return SettingsOut(enabled=user.personalization_enabled)


@router.post("/corrections", response_model=CorrectionOut)
def submit_correction(
    body: CorrectionIn,
    background: BackgroundTasks,
    user: User = Depends(_current_user),
    db: OrmSession = Depends(get_db),
) -> CorrectionOut:
    _require_enabled(user)
    expected_dim = base_data.feature_dim(body.model.value)
    if body.feature_vector.dim != expected_dim:
        raise HTTPException(422, f"feature_vector.dim must be {expected_dim} for {body.model.value}")
    connection = _current_connection(db, user.id)
    values = {
        "predicted_label": body.predicted_label,
        "predicted_confidence": body.predicted_confidence,
        "corrected_label": body.corrected_label,
        "feature_vector": body.feature_vector.model_dump(),
    }
    db.execute(
        pg_insert(Correction)
        .values(
            id=uuid.uuid4(),
            email_connection_id=connection.id,
            provider_message_id=body.provider_message_id,
            model=body.model,
            **values,
        )
        .on_conflict_do_update(
            constraint="uq_corrections_message_model",
            set_={**values, "created_at": func.now()},
        )
    )
    db.commit()
    remaining = jobs.corrections_until_retrain(db, user.id, body.model)
    scheduled = remaining == 0 and not jobs.is_running(user.id, body.model)
    if scheduled:
        background.add_task(jobs.run_retrain, user.id, body.model)
    return CorrectionOut(retrain_scheduled=scheduled, corrections_until_retrain=remaining)


def _user_corrections(user_id: uuid.UUID):
    return select(Correction).join(EmailConnection, Correction.email_connection_id == EmailConnection.id).where(
        EmailConnection.user_id == user_id
    )


# What the user chose per email, so the inbox can show corrections again
# after a reload. No feature vectors: the list stays small.
@router.get("/corrections", response_model=list[CorrectionSummary])
def list_corrections(user: User = Depends(_current_user), db: OrmSession = Depends(get_db)) -> list[CorrectionSummary]:
    rows = db.scalars(_user_corrections(user.id).order_by(Correction.created_at)).all()
    return [
        CorrectionSummary(provider_message_id=r.provider_message_id, model=r.model.value, corrected_label=r.corrected_label)
        for r in rows
    ]


# Everything stored for one email's corrections, exactly as saved: the
# receipt the inbox decodes to show what was sent.
@router.get("/corrections/{provider_message_id}", response_model=list[CorrectionDetail])
def get_message_corrections(
    provider_message_id: str, user: User = Depends(_current_user), db: OrmSession = Depends(get_db)
) -> list[CorrectionDetail]:
    rows = db.scalars(
        _user_corrections(user.id)
        .where(Correction.provider_message_id == provider_message_id)
        .order_by(Correction.model)
    ).all()
    return [
        CorrectionDetail(
            model=r.model.value,
            provider_message_id=r.provider_message_id,
            predicted_label=r.predicted_label,
            predicted_confidence=r.predicted_confidence,
            corrected_label=r.corrected_label,
            feature_vector=SparseVector(**r.feature_vector),
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.post("/retrain", response_model=RetrainOut)
def retrain_now(
    background: BackgroundTasks, user: User = Depends(_current_user), db: OrmSession = Depends(get_db)
) -> RetrainOut:
    _require_enabled(user)
    scheduled = []
    for model in CorrectionModel:
        if jobs.has_changes_since_last_attempt(db, user.id, model) and not jobs.is_running(user.id, model):
            background.add_task(jobs.run_retrain, user.id, model)
            scheduled.append(model.value)
    return RetrainOut(scheduled=scheduled)


@router.get("/status", response_model=StatusOut)
def get_status(user: User = Depends(_current_user), db: OrmSession = Depends(get_db)) -> StatusOut:
    models = []
    for model in CorrectionModel:
        last = jobs.last_attempt(db, user.id, PersonalizedModelKind(model.value))
        models.append(ModelStatus(
            model=model.value,
            correction_count=jobs.correction_count(db, user.id, model),
            corrections_until_retrain=jobs.corrections_until_retrain(db, user.id, model),
            running=jobs.is_running(user.id, model),
            last_attempt=LastAttempt(
                version=last.version,
                status=last.status.value,
                correction_count=last.correction_count,
                metrics=last.metrics,
                created_at=last.created_at,
            ) if last else None,
        ))
    custom_labels = db.scalars(
        jobs.scoped_corrections(select(Correction.corrected_label).distinct(), user.id, CorrectionModel.category)
        .where(Correction.corrected_label.not_in(BASE_CATEGORIES))
    ).all()
    return StatusOut(enabled=user.personalization_enabled, models=models, custom_labels=sorted(custom_labels))


@router.get("/models", response_model=list[ManifestEntry])
def get_manifest(user: User = Depends(_current_user), db: OrmSession = Depends(get_db)) -> list[ManifestEntry]:
    active = db.scalars(
        select(PersonalizedModel)
        .where(PersonalizedModel.user_id == user.id, PersonalizedModel.status == PersonalizedModelStatus.active)
        .order_by(PersonalizedModel.model)
    ).all()
    return [ManifestEntry(model=m.model.value, version=m.version, classes=m.classes or []) for m in active]


@router.get("/models/{model}/{version}.onnx")
def get_model_bytes(
    model: PersonalizedModelKind,
    version: int,
    user: User = Depends(_current_user),
    db: OrmSession = Depends(get_db),
) -> Response:
    row = db.scalars(
        select(PersonalizedModel).where(
            PersonalizedModel.user_id == user.id,
            PersonalizedModel.model == model,
            PersonalizedModel.version == version,
            PersonalizedModel.onnx.is_not(None),
        )
    ).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    return Response(
        content=row.onnx,
        media_type="application/octet-stream",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )
