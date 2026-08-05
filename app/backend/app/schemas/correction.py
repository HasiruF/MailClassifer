import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class FeatureVectorIn(BaseModel):
    text_features: dict[str, float]
    numeric_features: list[float]
    numeric_feature_names: list[str]


class CorrectionCreate(BaseModel):
    label_type: Literal["category", "priority"]
    original_label: str
    corrected_label: str
    custom_category_name: str | None = None
    feature_vector: FeatureVectorIn


class CorrectionOut(BaseModel):
    id: uuid.UUID
    label_type: str
    original_label: str
    corrected_label: str
    created_at: datetime

    class Config:
        from_attributes = True
