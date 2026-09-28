import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class ClassificationEvent(Base):
    __tablename__ = "classification_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("email_connections.id"), nullable=False
    )
    # Gmail/Outlook's own message id — not content.
    provider_message_id: Mapped[str] = mapped_column(String, nullable=False)
    category_label: Mapped[str] = mapped_column(String, nullable=False)
    priority_bucket: Mapped[str] = mapped_column(String, nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)
    spam_label: Mapped[str] = mapped_column(String, nullable=False)
    sentiment_compound: Mapped[float] = mapped_column(Float, nullable=False)
    classified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
