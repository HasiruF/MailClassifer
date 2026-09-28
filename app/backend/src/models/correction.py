import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class CorrectionModel(str, enum.Enum):
    category = "category"
    priority = "priority"
    spam = "spam"


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("email_connections.id"), nullable=False
    )
    model: Mapped[CorrectionModel] = mapped_column(Enum(CorrectionModel, name="correction_model"), nullable=False)
    # Dense feature vector the client already builds for ONNX inference — never raw
    # subject/body text, per the app's privacy boundary.
    feature_vector: Mapped[list[float]] = mapped_column(JSONB, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String, nullable=False)
    predicted_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    corrected_label: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
