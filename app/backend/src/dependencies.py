import uuid

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session as OrmSession

from src.database import get_db
from src.models import Session as SessionModel

SESSION_COOKIE_NAME = "dmp_session"


def get_current_session(
    dmp_session: str | None = Cookie(default=None),
    db: OrmSession = Depends(get_db),
) -> SessionModel:
    if dmp_session is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not connected")
    try:
        session_id = uuid.UUID(dmp_session)
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid session")

    session = db.get(SessionModel, session_id)
    if session is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session not found")
    return session


def get_optional_session(
    dmp_session: str | None = Cookie(default=None),
    db: OrmSession = Depends(get_db),
) -> SessionModel | None:
    if dmp_session is None:
        return None
    try:
        session_id = uuid.UUID(dmp_session)
    except ValueError:
        return None
    return db.get(SessionModel, session_id)
