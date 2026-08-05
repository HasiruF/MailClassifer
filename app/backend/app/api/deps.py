"""
TODO: replace with real auth (verify the Gmail OAuth token the frontend
already has, or issue our own session). For now, callers pass a user id
directly so the rest of the API/DB layer can be built and tested without
blocking on the auth flow.
"""

import uuid

from fastapi import Header, HTTPException


def get_current_user_id(x_user_id: str = Header(...)) -> uuid.UUID:
    try:
        return uuid.UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or missing X-User-Id header")
