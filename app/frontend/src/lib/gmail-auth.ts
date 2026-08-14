// Client-side Google OAuth via Google Identity Services (GIS) token client.
// No backend and no client secret involved — GIS's token-client flow hands
// an access token straight to browser JS via a popup, which is the
// supported no-server pattern for a static-export app. The script itself
// must load from Google's own domain (can't be bundled/self-hosted), same
// reasoning as fetching onnxruntime-web's WASM from a CDN in engine.ts.
// The access token itself is short-lived (~1hr) and read-only-scoped, so
// the caller (inbox-context.tsx) caches it in sessionStorage to survive a
// page reload — deliberately NOT a refresh token, which would need a
// backend to hold safely and isn't part of this app.

const GIS_SRC = 'https://accounts.google.com/gsi/client'
const GMAIL_READONLY_SCOPE = 'https://www.googleapis.com/auth/gmail.readonly'
const CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID

interface TokenResponse {
  access_token?: string
  expires_in?: number
  error?: string
}

export interface GmailToken {
  accessToken: string
  expiresAt: number
}

interface TokenClient {
  requestAccessToken: () => void
}

declare global {
  interface Window {
    google?: {
      accounts: {
        oauth2: {
          initTokenClient(config: {
            client_id: string
            scope: string
            callback: (resp: TokenResponse) => void
          }): TokenClient
        }
      }
    }
  }
}

let scriptPromise: Promise<void> | null = null

function loadGisScript(): Promise<void> {
  if (scriptPromise) return scriptPromise
  scriptPromise = new Promise((resolve, reject) => {
    if (window.google?.accounts?.oauth2) {
      resolve()
      return
    }
    const script = document.createElement('script')
    script.src = GIS_SRC
    script.async = true
    script.defer = true
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('Failed to load Google Identity Services script'))
    document.head.appendChild(script)
  })
  return scriptPromise
}

// Opens the Google account picker/consent popup and resolves with a
// short-lived Gmail-readonly access token + its expiry. Rejects if the user
// closes the popup, denies consent, or (in Testing publish status) isn't on
// the project's test-user list.
export async function requestGmailAccessToken(): Promise<GmailToken> {
  if (!CLIENT_ID) {
    throw new Error(
      'NEXT_PUBLIC_GOOGLE_CLIENT_ID is not set — add it to app/frontend/.env.local',
    )
  }
  await loadGisScript()
  return new Promise((resolve, reject) => {
    const client = window.google!.accounts.oauth2.initTokenClient({
      client_id: CLIENT_ID,
      scope: GMAIL_READONLY_SCOPE,
      callback: (resp) => {
        if (resp.error || !resp.access_token) {
          reject(new Error(resp.error ?? 'Gmail authorization failed or was cancelled'))
          return
        }
        resolve({
          accessToken: resp.access_token,
          expiresAt: Date.now() + (resp.expires_in ?? 3600) * 1000,
        })
      },
    })
    client.requestAccessToken()
  })
}

// Cached across a page reload so the user doesn't have to re-click through
// the consent popup every time — sessionStorage (not localStorage) so it
// clears when the tab closes, and it's still just the short-lived
// read-only access token, not a refresh token.
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
