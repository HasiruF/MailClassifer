import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, ForeignKey, func, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CustomCategory(Base):
    """A category a user added beyond the built-in Work/Personal/Other."""

    __tablename__ = "custom_categories"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_category_name"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
