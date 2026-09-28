from urllib.parse import urlencode

import httpx

from src.config import settings

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"
GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def build_authorization_url(state: str) -> str:
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": f"openid email {GMAIL_READONLY_SCOPE}",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


class TokenExchangeError(Exception):
    pass


async def exchange_code_for_tokens(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    if resp.status_code != 200:
        raise TokenExchangeError(resp.text)
    return resp.json()


async def refresh_access_token(refresh_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TOKEN_ENDPOINT,
            data={
                "refresh_token": refresh_token,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "grant_type": "refresh_token",
            },
        )
    if resp.status_code != 200:
        raise TokenExchangeError(resp.text)
    return resp.json()


async def fetch_userinfo(access_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    resp.raise_for_status()
    return resp.json()
