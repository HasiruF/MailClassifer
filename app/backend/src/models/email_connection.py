import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class EmailProvider(str, enum.Enum):
    gmail = "gmail"
    outlook = "outlook"


class EmailConnection(Base):
    __tablename__ = "email_connections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    provider: Mapped[EmailProvider] = mapped_column(Enum(EmailProvider, name="email_provider"), nullable=False)
    provider_account_id: Mapped[str] = mapped_column(String, nullable=False)
    provider_email: Mapped[str] = mapped_column(String, nullable=False)
    # Null when remember_me was false on the session that created this connection —
    # kept only in an in-memory dict (keyed by this row's id) in that case, never here.
    refresh_token: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
