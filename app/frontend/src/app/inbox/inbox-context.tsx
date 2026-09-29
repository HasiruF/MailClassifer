'use client'

import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { loadModels, modelsLoaded, classify } from '@/inference/engine'
import {
  startGmailConnect,
  fetchGmailAccessToken,
  loadStoredGmailToken,
  saveGmailToken,
  clearStoredGmailToken,
} from '@/lib/gmail-auth'
import { fetchRecentInboxEmails } from '@/lib/gmail-fetch'
import type { ClassificationResult, CategoryLabel, InboxEmail } from '@/types'
import { SAMPLE_EMAILS } from '@/data/sample-emails'

export type Row = InboxEmail & { result: ClassificationResult | null }
export type GmailStatus = 'disconnected' | 'connecting' | 'fetching' | 'connected' | 'error'

interface InboxState {
  rows: Row[]
  status: 'loading' | 'ready' | 'error'
  error: string | null
  filter: CategoryLabel | 'All'
  setFilter: (f: CategoryLabel | 'All') => void
  onlyHigh: boolean
  setOnlyHigh: (v: boolean) => void
  showArchived: boolean
  setShowArchived: (v: boolean) => void
  query: string
  setQuery: (q: string) => void
  archivedIds: Set<string>
  archiveEmail: (id: string) => void
  unarchiveEmail: (id: string) => void
  toggleRead: (id: string) => void
  selectedId: string | null
  selectEmail: (id: string | null) => void
  gmailStatus: GmailStatus
  gmailError: string | null
  connectGmail: (rememberMe: boolean) => void
  refreshInbox: () => Promise<void>
}

const InboxContext = createContext<InboxState | null>(null)

// Module-level guard against React Strict Mode's dev-only double-invoke of
// the mount effect below: without it, two concurrent invocations each
// independently mint a Gmail access token and fetch the inbox, doubling
// concurrent Gmail API requests past its per-user rate limit (confirmed via
// a headless-browser repro — both runs failed with 403, and the resume
// path's catch swallowed it silently, freezing the UI at "FETCHING
// INBOX..." forever). Same pattern as engine.ts's loadModels() memoized
// promise, and same single-instance assumption (one InboxProvider per app).
let fetchAndClassifyPromise: Promise<void> | null = null

export function InboxProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [rows, setRows] = useState<Row[]>(SAMPLE_EMAILS.map((e) => ({ ...e, result: null })))
  const [filter, setFilter] = useState<CategoryLabel | 'All'>('All')
  const [onlyHigh, setOnlyHigh] = useState(false)
  const [showArchived, setShowArchived] = useState(false)
  const [query, setQuery] = useState('')
  // Archive/read state is client-only (this app has no server to persist
  // to) — real, working actions for the current session, not wired to
  // Gmail itself (archiving here never touches the real inbox).
  const [archivedIds, setArchivedIds] = useState<Set<string>>(new Set())

  function archiveEmail(id: string) {
    setArchivedIds((prev) => new Set(prev).add(id))
  }
  function unarchiveEmail(id: string) {
    setArchivedIds((prev) => {
      const next = new Set(prev)
      next.delete(id)
      return next
    })
  }
  function toggleRead(id: string) {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, unread: !r.unread } : r)))
  }
  // Detail view is client-side state, not a /inbox/[id] route — Gmail
  // message IDs only exist at runtime (after fetch), so a dynamic
  // filesystem route could never satisfy generateStaticParams() under
  // output: 'export', which pre-renders every route at build time.
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [gmailStatus, setGmailStatus] = useState<GmailStatus>('disconnected')
  const [gmailError, setGmailError] = useState<string | null>(null)

  function fetchAndClassify(token: string): Promise<void> {
    if (fetchAndClassifyPromise) return fetchAndClassifyPromise
    fetchAndClassifyPromise = (async () => {
      const emails = await fetchRecentInboxEmails(token)

      await loadModels()
      if (!modelsLoaded()) throw new Error('models did not finish loading')

      setRows(emails.map((e) => ({ ...e, result: null })))
      // Same WASM single-flight constraint as the sample-email loop below —
      // classify() calls must go one at a time, not Promise.all.
      for (const email of emails) {
        const result = await classify(email)
        setRows((prev) => prev.map((r) => (r.id === email.id ? { ...r, result } : r)))
      }
    })()
    // Clear once settled (success or failure) so the next real call —
    // refreshInbox(), or a later reconnect — starts a fresh fetch instead of
    // replaying this one's stale result.
    fetchAndClassifyPromise.finally(() => {
      fetchAndClassifyPromise = null
    })
    return fetchAndClassifyPromise
  }

  // Navigates away to the backend's OAuth start route — nothing after this
  // runs in this tab. The page that loads on return (/inbox) picks the
  // connection up silently via the mount effect below, since the session
  // cookie is already set by the time Google redirects back.
  function connectGmail(rememberMe: boolean) {
    setGmailStatus('connecting')
    setGmailError(null)
    startGmailConnect(rememberMe)
  }

  // Always asks the backend for a fresh token rather than reusing the
  // cached one — that's what makes this silent instead of erroring once the
  // cached access token expires. Only fails if the session cookie itself is
  // gone or was never connected.
  async function refreshInbox() {
    setGmailStatus('fetching')
    setGmailError(null)
    try {
      const token = await fetchGmailAccessToken()
      saveGmailToken(token)
      await fetchAndClassify(token.accessToken)
      setGmailStatus('connected')
    } catch (err) {
      clearStoredGmailToken()
      setGmailError(err instanceof Error ? err.message : String(err))
      setGmailStatus('error')
    }
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      // Cosmetic only — the session cookie from /auth/gmail/callback is
      // already set by the time this page loads, the query param doesn't
      // gate anything below.
      if (window.location.search.includes('connected=1')) {
        window.history.replaceState({}, '', window.location.pathname)
      }

      // Resume silently if a still-valid token survived a page reload —
      // skips the sample-email classification entirely rather than
      // flashing sample data before replacing it with the real inbox.
      const stored = loadStoredGmailToken()
      if (stored) {
        setGmailStatus('fetching')
        try {
          await fetchAndClassify(stored.accessToken)
          if (!cancelled) {
            setGmailStatus('connected')
            setStatus('ready')
          }
          return
        } catch (err) {
          clearStoredGmailToken()
          if (!cancelled) {
            setGmailError(err instanceof Error ? err.message : String(err))
            setGmailStatus('error')
          }
          // fall through to the sample-email view below
        }
      } else {
        // No cached access token — try the backend session cookie before
        // giving up. Succeeds silently (no popup, no redirect) whenever a
        // prior connection is still alive: same tab after the cached token
        // expired, a fresh tab in the same browser, or a returning
        // remember-me session days later.
        try {
          const token = await fetchGmailAccessToken()
          saveGmailToken(token)
          setGmailStatus('fetching')
          await fetchAndClassify(token.accessToken)
          if (!cancelled) {
            setGmailStatus('connected')
            setStatus('ready')
          }
          return
        } catch {
          // Not connected yet — fall through to the sample-email view below,
          // stay in 'disconnected' status (not 'error'; this is the normal
          // first-visit state).
        }
      }

      try {
        await loadModels()
        if (!modelsLoaded()) throw new Error('models did not finish loading')
        // classify() can't run concurrently — onnxruntime-web's WASM
        // backend throws "Session already started" if two .run() calls
        // overlap — so these go one at a time, not Promise.all.
        for (const email of SAMPLE_EMAILS) {
          if (cancelled) return
          const result = await classify(email)
          setRows((prev) => prev.map((r) => (r.id === email.id ? { ...r, result } : r)))
        }
        if (!cancelled) setStatus('ready')
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
          setStatus('error')
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <InboxContext.Provider
      value={{
        rows,
        status,
        error,
        filter,
        setFilter,
        onlyHigh,
        setOnlyHigh,
        showArchived,
        setShowArchived,
        query,
        setQuery,
        archivedIds,
        archiveEmail,
        unarchiveEmail,
        toggleRead,
        selectedId,
        selectEmail: setSelectedId,
        gmailStatus,
        gmailError,
        connectGmail,
        refreshInbox,
      }}
    >
      {children}
    </InboxContext.Provider>
  )
}

export function useInbox() {
  const ctx = useContext(InboxContext)
  if (!ctx) throw new Error('useInbox must be used within InboxProvider')
  return ctx
}
