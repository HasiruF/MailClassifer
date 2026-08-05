import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user_id
from app.db.session import get_db
from app.models.model_version import ModelVersion
from app.services.personalization import build_model_version_artifact

router = APIRouter()


@router.post("/models/{model_type}/retrain")
def trigger_retrain(
    model_type: Literal["category", "priority"],
    db: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
):
    """Synchronous for now (small per-user data). Move to a background job
    once retrained (not just centroid) models are implemented — see
    app/services/personalization.py."""
    artifact = build_model_version_artifact(db, user_id, model_type)
    version = ModelVersion(
        user_id=user_id,
        model_type=model_type,
        strategy="mixed",
        training_example_count=sum(a["example_count"] for a in artifact),
        artifact={"labels": artifact},
    )
    db.add(version)
    db.commit()
    db.refresh(version)
    return {"id": version.id, "created_at": version.created_at, "artifact": version.artifact}


@router.get("/models/{user_id}/{model_type}/latest")
def get_latest_model(
    user_id: uuid.UUID,
    model_type: Literal["category", "priority"],
    db: Session = Depends(get_db),
):
    version = (
        db.query(ModelVersion)
        .filter(ModelVersion.user_id == user_id, ModelVersion.model_type == model_type)
        .order_by(ModelVersion.created_at.desc())
        .first()
    )
    if version is None:
        raise HTTPException(status_code=404, detail="No personalized model yet for this user/type")
    return {"id": version.id, "created_at": version.created_at, "artifact": version.artifact}
