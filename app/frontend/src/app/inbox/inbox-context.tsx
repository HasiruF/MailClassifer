'use client'

import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { loadModels, modelsLoaded, classify } from '@/inference/engine'
import {
  requestGmailAccessToken,
  loadStoredGmailToken,
  saveGmailToken,
  clearStoredGmailToken,
  type GmailToken,
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
  selectedId: string | null
  selectEmail: (id: string | null) => void
  gmailStatus: GmailStatus
  gmailError: string | null
  connectGmail: () => Promise<void>
  refreshInbox: () => Promise<void>
}

const InboxContext = createContext<InboxState | null>(null)

export function InboxProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [rows, setRows] = useState<Row[]>(SAMPLE_EMAILS.map((e) => ({ ...e, result: null })))
  const [filter, setFilter] = useState<CategoryLabel | 'All'>('All')
  const [onlyHigh, setOnlyHigh] = useState(false)
  // Detail view is client-side state, not a /inbox/[id] route — Gmail
  // message IDs only exist at runtime (after fetch), so a dynamic
  // filesystem route could never satisfy generateStaticParams() under
  // output: 'export', which pre-renders every route at build time.
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [gmailStatus, setGmailStatus] = useState<GmailStatus>('disconnected')
  const [gmailError, setGmailError] = useState<string | null>(null)
  // Kept so refreshInbox() can re-fetch without popping the OAuth consent
  // screen again, and so a page reload can silently resume (see the mount
  // effect below) instead of dropping back to the sample-email view. GIS
  // access tokens are short-lived (~1hr); once expired, the Gmail API call
  // fails and surfaces as the normal error state, recoverable via the main
  // button (fresh connectGmail call, new token).
  const [gmailToken, setGmailToken] = useState<GmailToken | null>(null)

  async function fetchAndClassify(token: string) {
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
  }

  async function connectGmail() {
    setGmailStatus('connecting')
    setGmailError(null)
    try {
      const token = await requestGmailAccessToken()
      setGmailToken(token)
      saveGmailToken(token)
      setGmailStatus('fetching')
      await fetchAndClassify(token.accessToken)
      setGmailStatus('connected')
    } catch (err) {
      setGmailError(err instanceof Error ? err.message : String(err))
      setGmailStatus('error')
    }
  }

  async function refreshInbox() {
    if (!gmailToken) return
    setGmailStatus('fetching')
    setGmailError(null)
    try {
      await fetchAndClassify(gmailToken.accessToken)
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
      // Resume silently if a still-valid token survived a page reload —
      // skips the sample-email classification entirely rather than
      // flashing sample data before replacing it with the real inbox.
      const stored = loadStoredGmailToken()
      if (stored) {
        setGmailToken(stored)
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
