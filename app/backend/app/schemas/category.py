import uuid
from datetime import datetime

from pydantic import BaseModel


class CustomCategoryCreate(BaseModel):
    name: str


class CustomCategoryOut(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime

    class Config:
        from_attributes = True
