import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, ForeignKey, func, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ModelVersion(Base):
    """
    A personalized model snapshot for one user. Below
    retrain_threshold_per_category, `artifact` holds centroid/kNN vectors
    (see app/services/personalization.py); above it, a proper retrained
    classifier's exported weights. Either way `artifact` is what the
    frontend fetches and blends with the base model at inference time.
    """

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    model_type: Mapped[str] = mapped_column(String)  # 'category' | 'priority'
    strategy: Mapped[str] = mapped_column(String)  # 'centroid' | 'knn' | 'retrained'
    training_example_count: Mapped[int] = mapped_column(Integer)
    artifact: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
