import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, LargeBinary, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class PersonalizedModelKind(str, enum.Enum):
    spam = "spam"
    category = "category"
    priority = "priority"
    priority_regressor = "priority_regressor"


class PersonalizedModelStatus(str, enum.Enum):
    active = "active"
    rejected = "rejected"
    superseded = "superseded"


class PersonalizedModel(Base):
    __tablename__ = "personalized_models"
    __table_args__ = (UniqueConstraint("user_id", "model", "version", name="uq_personalized_models_version"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    model: Mapped[PersonalizedModelKind] = mapped_column(
        Enum(PersonalizedModelKind, name="personalized_model_kind"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[PersonalizedModelStatus] = mapped_column(
        Enum(PersonalizedModelStatus, name="personalized_model_status"), nullable=False
    )
    # Null for rejected attempts: the gate runs before export.
    onnx: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    classes: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    correction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
