import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user_id
from app.db.session import get_db
from app.models.category import CustomCategory
from app.models.correction import Correction
from app.schemas.correction import CorrectionCreate, CorrectionOut

router = APIRouter()


@router.post("/corrections", response_model=CorrectionOut)
def create_correction(
    payload: CorrectionCreate,
    db: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
):
    custom_category_id = None
    if payload.custom_category_name:
        existing = (
            db.query(CustomCategory)
            .filter(CustomCategory.user_id == user_id, CustomCategory.name == payload.custom_category_name)
            .first()
        )
        if existing is None:
            existing = CustomCategory(user_id=user_id, name=payload.custom_category_name)
            db.add(existing)
            db.flush()
        custom_category_id = existing.id

    correction = Correction(
        user_id=user_id,
        label_type=payload.label_type,
        original_label=payload.original_label,
        corrected_label=payload.corrected_label,
        custom_category_id=custom_category_id,
        feature_vector=payload.feature_vector.model_dump(),
    )
    db.add(correction)
    db.commit()
    db.refresh(correction)
    return correction
