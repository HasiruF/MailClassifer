from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from src.models import CorrectionModel
from src.personalization.sparse import SparseVector

FIXED_LABELS = {
    CorrectionModel.priority: {"low", "medium", "high"},
    CorrectionModel.spam: {"spam", "ham"},
}


class CorrectionIn(BaseModel):
    model: CorrectionModel
    provider_message_id: str = Field(min_length=1)
    feature_vector: SparseVector
    predicted_label: str
    predicted_confidence: float = Field(ge=0.0, le=1.0)
    corrected_label: str = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def _label_allowed(self) -> "CorrectionIn":
        self.corrected_label = self.corrected_label.strip()
        if not self.corrected_label:
            raise ValueError("corrected_label is blank")
        allowed = FIXED_LABELS.get(self.model)
        if allowed is not None and self.corrected_label not in allowed:
            raise ValueError(f"{self.model.value} corrections must be one of {sorted(allowed)}")
        return self


class CorrectionOut(BaseModel):
    retrain_scheduled: bool
    corrections_until_retrain: int


class SettingsIn(BaseModel):
    enabled: bool


class SettingsOut(BaseModel):
    enabled: bool


class LastAttempt(BaseModel):
    version: int
    status: str
    metrics: dict
    created_at: datetime


class ModelStatus(BaseModel):
    model: str
    correction_count: int
    corrections_until_retrain: int
    running: bool
    last_attempt: LastAttempt | None


class StatusOut(BaseModel):
    enabled: bool
    models: list[ModelStatus]
    custom_labels: list[str]


class ManifestEntry(BaseModel):
    model: str
    version: int
    classes: list[str | int]


class RetrainOut(BaseModel):
    scheduled: list[str]
