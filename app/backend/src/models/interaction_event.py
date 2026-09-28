import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class InteractionAction(str, enum.Enum):
    opened = "opened"
    archived = "archived"
    replied = "replied"
    flagged = "flagged"


class InteractionEvent(Base):
    __tablename__ = "interaction_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("email_connections.id"), nullable=False
    )
    provider_message_id: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[InteractionAction] = mapped_column(Enum(InteractionAction, name="interaction_action"), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
