import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user_id
from app.db.session import get_db
from app.models.category import CustomCategory
from app.schemas.category import CustomCategoryCreate, CustomCategoryOut

router = APIRouter()


@router.get("/categories", response_model=list[CustomCategoryOut])
def list_categories(db: Session = Depends(get_db), user_id: uuid.UUID = Depends(get_current_user_id)):
    return db.query(CustomCategory).filter(CustomCategory.user_id == user_id).all()


@router.post("/categories", response_model=CustomCategoryOut)
def create_category(
    payload: CustomCategoryCreate,
    db: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
):
    category = CustomCategory(user_id=user_id, name=payload.name)
    db.add(category)
    db.commit()
    db.refresh(category)
    return category
