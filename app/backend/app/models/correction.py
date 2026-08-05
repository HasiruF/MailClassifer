import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, ForeignKey, func, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Correction(Base):
    """
    A user's correction to a predicted label. Stores only the extracted
    feature vector the client computed (sparse TF-IDF terms + numeric
    header/style/keyword features) — never raw subject/body text. See
    src/inference/features.ts on the frontend for what produces this shape,
    and PROJECT_STATUS.md for why this boundary matters.
    """

    __tablename__ = "corrections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)

    label_type: Mapped[str] = mapped_column(String)  # 'category' | 'priority'
    original_label: Mapped[str] = mapped_column(String)
    corrected_label: Mapped[str] = mapped_column(String)
    custom_category_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("custom_categories.id"), nullable=True
    )

    # {"text_features": {term: weight, ...}, "numeric_features": [...], "numeric_feature_names": [...]}
    feature_vector: Mapped[dict] = mapped_column(JSONB)

    used_in_training: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
