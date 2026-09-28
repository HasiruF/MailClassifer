import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session as OrmSession

from src.config import settings
from src.database import get_db
from src.dependencies import SESSION_COOKIE_NAME, get_current_session, get_optional_session
from src.models import EmailConnection, EmailProvider
from src.models import Session as SessionModel
from src.models import User
from src.schemas.auth import GmailTokenResponse
from src.services import google_oauth, memory_store

router = APIRouter(prefix="/auth/gmail", tags=["auth"])

SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days, only applied when remember_me


@router.get("/start")
def start(remember_me: bool = False) -> RedirectResponse:
    state = secrets.token_urlsafe(24)
    memory_store.put_pending_state(state, remember_me)
    return RedirectResponse(google_oauth.build_authorization_url(state))


@router.get("/callback")
async def callback(
    code: str,
    state: str,
    db: OrmSession = Depends(get_db),
    existing_session: SessionModel | None = Depends(get_optional_session),
) -> RedirectResponse:
    remember_me = memory_store.pop_pending_state(state)
    if remember_me is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired OAuth state")

    try:
        tokens = await google_oauth.exchange_code_for_tokens(code)
    except google_oauth.TokenExchangeError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Google token exchange failed: {exc}")

    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token")
    userinfo = await google_oauth.fetch_userinfo(access_token)
    google_sub = userinfo["sub"]
    google_email = userinfo["email"]

    if existing_session is not None:
        user_id = existing_session.user_id
        session = existing_session
    else:
        user = User()
        db.add(user)
        db.flush()
        user_id = user.id
        session = SessionModel(user_id=user_id, remember_me=remember_me)
        db.add(session)

    connection = (
        db.query(EmailConnection)
        .filter_by(user_id=user_id, provider=EmailProvider.gmail, provider_account_id=google_sub)
        .first()
    )
    if connection is None:
        connection = EmailConnection(
            user_id=user_id,
            provider=EmailProvider.gmail,
            provider_account_id=google_sub,
            provider_email=google_email,
        )
        db.add(connection)
    else:
        connection.provider_email = google_email
    db.flush()

    if remember_me:
        if refresh_token:
            connection.refresh_token = refresh_token
    else:
        connection.refresh_token = None
        if refresh_token:
            memory_store.put_session_refresh_token(connection.id, refresh_token)

    db.commit()

    response = RedirectResponse(f"{settings.frontend_url}/inbox?connected=1")
    cookie_kwargs: dict = {
        "key": SESSION_COOKIE_NAME,
        "value": str(session.id),
        "httponly": True,
        "samesite": "lax",
        "secure": False,  # local http dev; set True once served over https
    }
    if session.remember_me:
        cookie_kwargs["max_age"] = SESSION_MAX_AGE_SECONDS
    response.set_cookie(**cookie_kwargs)
    return response


@router.post("/token", response_model=GmailTokenResponse)
async def issue_access_token(
    session: SessionModel = Depends(get_current_session),
    db: OrmSession = Depends(get_db),
) -> GmailTokenResponse:
    connection = (
        db.query(EmailConnection)
        .filter_by(user_id=session.user_id, provider=EmailProvider.gmail)
        .order_by(EmailConnection.last_used_at.desc())
        .first()
    )
    if connection is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No Gmail connection for this session")

    refresh_token = connection.refresh_token or memory_store.get_session_refresh_token(connection.id)
    if refresh_token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Gmail connection expired, please reconnect")

    try:
        tokens = await google_oauth.refresh_access_token(refresh_token)
    except google_oauth.TokenExchangeError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Google token refresh failed: {exc}")

    connection.last_used_at = datetime.now(timezone.utc)
    session.last_used_at = datetime.now(timezone.utc)
    db.commit()

    return GmailTokenResponse(access_token=tokens["access_token"], expires_in=tokens.get("expires_in", 3600))
