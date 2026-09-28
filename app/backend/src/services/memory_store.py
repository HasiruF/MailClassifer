import time
import uuid

# Process-local only, by design:
# - pending_states: short-lived CSRF state -> remember_me, while the browser is
#   off at Google's consent screen. Fine to lose on a restart, the user just
#   re-starts the connect flow.
# - session_refresh_tokens: refresh tokens for connections created with
#   remember_me=False. Deliberately never written to Postgres — losing these
#   on a server restart is the whole point of unchecking "remember me".

_STATE_TTL_SECONDS = 300

_pending_states: dict[str, tuple[bool, float]] = {}
_session_refresh_tokens: dict[uuid.UUID, str] = {}


def put_pending_state(state: str, remember_me: bool) -> None:
    _pending_states[state] = (remember_me, time.monotonic() + _STATE_TTL_SECONDS)


def pop_pending_state(state: str) -> bool | None:
    entry = _pending_states.pop(state, None)
    if entry is None:
        return None
    remember_me, expires_at = entry
    if time.monotonic() > expires_at:
        return None
    return remember_me


def put_session_refresh_token(email_connection_id: uuid.UUID, refresh_token: str) -> None:
    _session_refresh_tokens[email_connection_id] = refresh_token


def get_session_refresh_token(email_connection_id: uuid.UUID) -> str | None:
    return _session_refresh_tokens.get(email_connection_id)
