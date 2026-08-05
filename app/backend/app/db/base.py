from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import all models here so Alembic's autogenerate can discover them via
# Base.metadata — each model module registers itself on import.
from app.models import user, category, correction, model_version  # noqa: E402,F401
