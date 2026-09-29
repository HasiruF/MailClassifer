// Gmail auth via the backend token-holder (see app/backend/src/routers/auth.py)
// instead of Google Identity Services' popup token-client flow. GIS's flow
// structurally never issues a refresh token, so it hit a hard ~1hr re-auth
// wall for every user with no way around it. The backend now holds the
// refresh token (in Postgres if "remember me" was checked, otherwise only in
// an in-process dict — see memory_store.py) behind an httpOnly session
// cookie, and mints a fresh access token on request. This module never sees
// the refresh token itself, only the short-lived access token it gets back.

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:3011'

export interface GmailToken {
  accessToken: string
  expiresAt: number
}

// Navigates the whole page to the backend, which redirects to Google's
// consent screen and (after the user approves) back to /inbox. Not a fetch —
// the code-for-tokens exchange needs a client secret, so it can't happen via
// an XHR/fetch call from browser JS, only a real top-level navigation through
// Google's redirect chain.
export function startGmailConnect(rememberMe: boolean): void {
  // eslint-disable-next-line @next/next/no-location-assign-relative-destination -- destination is the backend's own origin, not an internal Next.js route
  window.location.href = `${BACKEND_URL}/auth/gmail/start?remember_me=${rememberMe}`
}

export class NotConnectedError extends Error {}

// Exchanges the httpOnly session cookie (sent automatically via
// credentials: 'include') for a fresh ~1hr Gmail access token. Throws
// NotConnectedError if there's no session cookie or it's not linked to a
// Gmail connection yet — the caller falls back to the sample-email view.
export async function fetchGmailAccessToken(): Promise<GmailToken> {
  const res = await fetch(`${BACKEND_URL}/auth/gmail/token`, {
    method: 'POST',
    credentials: 'include',
  })
  if (res.status === 401 || res.status === 404) {
    throw new NotConnectedError('Not connected to Gmail')
  }
  if (!res.ok) {
    throw new Error(`Failed to fetch Gmail access token (${res.status})`)
  }
  const data = (await res.json()) as { access_token: string; expires_in: number }
  return {
    accessToken: data.access_token,
    expiresAt: Date.now() + data.expires_in * 1000,
  }
}

// Cached across a page reload so every classify/fetch call doesn't have to
// round-trip the backend — sessionStorage (not localStorage) so it clears
// when the tab closes. Still just the short-lived access token, never
// anything the backend gave us that we shouldn't persist client-side.
const STORAGE_KEY = 'gmail_token_v1'

export function loadStoredGmailToken(): GmailToken | null {
  const raw = sessionStorage.getItem(STORAGE_KEY)
  if (!raw) return null
  try {
    const token = JSON.parse(raw) as GmailToken
    if (typeof token.accessToken !== 'string' || typeof token.expiresAt !== 'number') {
      sessionStorage.removeItem(STORAGE_KEY)
      return null
    }
    if (token.expiresAt <= Date.now()) {
      sessionStorage.removeItem(STORAGE_KEY)
      return null
    }
    return token
  } catch {
    sessionStorage.removeItem(STORAGE_KEY)
    return null
  }
}

export function saveGmailToken(token: GmailToken): void {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(token))
}

export function clearStoredGmailToken(): void {
  sessionStorage.removeItem(STORAGE_KEY)
}
