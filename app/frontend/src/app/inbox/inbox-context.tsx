'use client'

import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { applyPersonalizedModels, buildCorrectionVector, classify, loadModels, modelsLoaded } from '@/inference/engine'
import {
  startGmailConnect,
  fetchGmailAccessToken,
  loadStoredGmailToken,
  saveGmailToken,
  clearStoredGmailToken,
} from '@/lib/gmail-auth'
import { fetchRecentInboxEmails } from '@/lib/gmail-fetch'
import { applyCorrections, predictedFor } from '@/lib/corrections'
import {
  fetchActiveModels,
  getStatus,
  retrainNow,
  setEnabled,
  submitCorrection,
  type CorrectionResponse,
  type PersonalizationStatus,
} from '@/lib/personalization-api'
import { toSparse } from '@/lib/sparse'
import type { ClassificationResult, InboxEmail, PersonalizableModel, PersonalizedModelArtifact } from '@/types'
import { SAMPLE_EMAILS } from '@/data/sample-emails'

export type Row = InboxEmail & {
  // What the UI renders: the model's output with this user's corrections
  // applied on top (a correction always wins for the email it was made on).
  result: ClassificationResult | null
  // Raw model output, never overwritten. Corrections are built from this.
  modelResult: ClassificationResult | null
  corrected: Partial<Record<PersonalizableModel, string>>
}
export type GmailStatus = 'disconnected' | 'connecting' | 'fetching' | 'connected' | 'error'

interface InboxState {
  rows: Row[]
  status: 'loading' | 'ready' | 'error'
  error: string | null
  filter: string
  setFilter: (f: string) => void
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
  // Phone-width only: whether the sidebar is open as a slide-over menu.
  navOpen: boolean
  setNavOpen: (open: boolean) => void
  gmailStatus: GmailStatus
  gmailError: string | null
  connectGmail: (rememberMe: boolean) => void
  refreshInbox: () => Promise<void>
  personalization: PersonalizationStatus | null
  personalizationError: string | null
  retraining: boolean
  correctEmail: (id: string, model: PersonalizableModel, label: string) => Promise<void>
  enablePersonalization: () => Promise<boolean>
  disablePersonalization: () => Promise<void>
  retrainPersonalization: () => Promise<void>
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

const RETRAIN_POLL_MS = 2000
const RETRAIN_POLL_LIMIT = 60

function emptyRow(email: InboxEmail): Row {
  return { ...email, result: null, modelResult: null, corrected: {} }
}

function withResult(row: Row, result: ClassificationResult): Row {
  return { ...row, modelResult: result, result: applyCorrections(result, row.corrected) }
}

function versionKey(artifacts: PersonalizedModelArtifact[]): string {
  return artifacts
    .map((a) => `${a.model}:${a.version}`)
    .sort()
    .join(',')
}

// Changes whenever any model records a new retrain attempt (active or
// rejected), which is how the watcher knows a job finished.
function attemptKey(status: PersonalizationStatus | null): string {
  return (status?.models ?? []).map((m) => `${m.model}:${m.last_attempt?.version ?? 0}`).join(',')
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

export function InboxProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [rows, setRows] = useState<Row[]>(SAMPLE_EMAILS.map(emptyRow))
  const [filter, setFilter] = useState<string>('All')
  const [onlyHigh, setOnlyHigh] = useState(false)
  const [showArchived, setShowArchived] = useState(false)
  const [query, setQuery] = useState('')
  // Archive/read state is client-only — real, working actions for the
  // current session, not wired to Gmail itself (archiving here never
  // touches the real inbox).
  const [archivedIds, setArchivedIds] = useState<Set<string>>(new Set())
  // Detail view is client-side state, not a /inbox/[id] route — Gmail
  // message IDs only exist at runtime (after fetch), so a dynamic
  // filesystem route could never satisfy generateStaticParams() under
  // output: 'export', which pre-renders every route at build time.
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [navOpen, setNavOpen] = useState(false)
  const [gmailStatus, setGmailStatus] = useState<GmailStatus>('disconnected')
  const [gmailError, setGmailError] = useState<string | null>(null)
  const [personalization, setPersonalization] = useState<PersonalizationStatus | null>(null)
  const [personalizationError, setPersonalizationError] = useState<string | null>(null)
  const [retraining, setRetraining] = useState(false)

  // Async flows below outlive the render they started in; refs let them
  // read current state instead of a stale closure.
  const rowsRef = useRef<Row[]>(rows)
  const personalizationRef = useRef<PersonalizationStatus | null>(null)
  const appliedVersions = useRef('')

  useEffect(() => {
    rowsRef.current = rows
  }, [rows])

  function updatePersonalization(next: PersonalizationStatus) {
    personalizationRef.current = next
    setPersonalization(next)
  }

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

  // Loads this user's active personalized models into the engine, or
  // reverts to base models when there are none or personalization is off.
  // Returns whether the loaded set changed. Never throws: personalization
  // is strictly additive, so a failure leaves the current models in place.
  async function syncPersonalization(): Promise<boolean> {
    try {
      const next = await getStatus()
      updatePersonalization(next)
      const artifacts = next.enabled ? await fetchActiveModels() : []
      const key = versionKey(artifacts)
      if (key === appliedVersions.current) return false
      await applyPersonalizedModels(artifacts)
      appliedVersions.current = key
      return true
    } catch (err) {
      setPersonalizationError(`Personalization unavailable, using base models: ${message(err)}`)
      return false
    }
  }

  async function reclassify(): Promise<void> {
    for (const row of rowsRef.current) {
      if (!row.modelResult) continue
      const result = await classify(row)
      setRows((prev) => prev.map((r) => (r.id === row.id ? withResult(r, result) : r)))
    }
  }

  function fetchAndClassify(token: string): Promise<void> {
    if (fetchAndClassifyPromise) return fetchAndClassifyPromise
    fetchAndClassifyPromise = (async () => {
      const emails = await fetchRecentInboxEmails(token)

      await loadModels()
      if (!modelsLoaded()) throw new Error('models did not finish loading')
      await syncPersonalization()

      setRows(emails.map(emptyRow))
      // Same WASM single-flight constraint as the sample-email loop below —
      // classify() calls go one at a time, not Promise.all.
      for (const email of emails) {
        const result = await classify(email)
        setRows((prev) => prev.map((r) => (r.id === email.id ? withResult(r, result) : r)))
      }
    })()
    // Clear once settled (success or failure) so the next real call starts a
    // fresh fetch. Handled on both paths: a bare .finally() would re-reject
    // into an unhandled rejection whenever the fetch fails.
    const clear = () => {
      fetchAndClassifyPromise = null
    }
    fetchAndClassifyPromise.then(clear, clear)
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
  // cached access token expires.
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
      setGmailError(message(err))
      setGmailStatus('error')
    }
  }

  function setCorrected(id: string, model: PersonalizableModel, label: string | undefined) {
    setRows((prev) =>
      prev.map((r) => {
        if (r.id !== id) return r
        const corrected = { ...r.corrected }
        if (label === undefined) delete corrected[model]
        else corrected[model] = label
        return { ...r, corrected, result: r.modelResult ? applyCorrections(r.modelResult, corrected) : r.result }
      }),
    )
  }

  // `before` is the attempt snapshot taken before the request that kicked
  // off the retrain, so a job that finishes before the first poll still
  // registers as a change.
  async function watchRetrain(before: string): Promise<void> {
    setRetraining(true)
    try {
      for (let i = 0; i < RETRAIN_POLL_LIMIT; i++) {
        await new Promise((resolve) => setTimeout(resolve, RETRAIN_POLL_MS))
        const next = await getStatus()
        updatePersonalization(next)
        if (!next.models.some((m) => m.running) && attemptKey(next) !== before) break
      }
      if (await syncPersonalization()) await reclassify()
    } catch (err) {
      setPersonalizationError(message(err))
    } finally {
      setRetraining(false)
    }
  }

  async function correctEmail(id: string, model: PersonalizableModel, label: string): Promise<void> {
    const row = rowsRef.current.find((r) => r.id === id)
    if (!row?.modelResult || row.source !== 'gmail') return
    const before = attemptKey(personalizationRef.current)
    const previous = row.corrected[model]
    const predicted = predictedFor(model, row.modelResult)
    setCorrected(id, model, label)
    setPersonalizationError(null)

    let response: CorrectionResponse
    try {
      response = await submitCorrection({
        model,
        provider_message_id: id,
        feature_vector: toSparse(buildCorrectionVector(row, model, row.modelResult)),
        predicted_label: predicted.label,
        predicted_confidence: Math.min(1, Math.max(0, predicted.confidence)),
        corrected_label: label,
      })
    } catch (err) {
      setCorrected(id, model, previous)
      setPersonalizationError(`Correction not saved: ${message(err)}`)
      return
    }
    try {
      updatePersonalization(await getStatus())
    } catch (err) {
      setPersonalizationError(message(err))
    }
    if (response.retrain_scheduled) void watchRetrain(before)
  }

  async function enablePersonalization(): Promise<boolean> {
    setPersonalizationError(null)
    try {
      await setEnabled(true)
      updatePersonalization(await getStatus())
      return true
    } catch (err) {
      setPersonalizationError(`Could not turn on personalization: ${message(err)}`)
      return false
    }
  }

  async function disablePersonalization(): Promise<void> {
    setPersonalizationError(null)
    try {
      await setEnabled(false)
    } catch (err) {
      setPersonalizationError(`Could not turn off personalization: ${message(err)}`)
      return
    }
    setRows((prev) => prev.map((r) => ({ ...r, corrected: {}, result: r.modelResult })))
    if (await syncPersonalization()) await reclassify()
  }

  async function retrainPersonalization(): Promise<void> {
    const before = attemptKey(personalizationRef.current)
    setPersonalizationError(null)
    try {
      const { scheduled } = await retrainNow()
      if (scheduled.length > 0) await watchRetrain(before)
    } catch (err) {
      setPersonalizationError(`Retrain did not start: ${message(err)}`)
    }
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      // Cosmetic only — the session cookie from /auth/gmail/callback is
      // already set by the time this page loads.
      if (window.location.search.includes('connected=1')) {
        window.history.replaceState({}, '', window.location.pathname)
      }

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
            setGmailError(message(err))
            setGmailStatus('error')
          }
          // fall through to the sample-email view below
        }
      } else {
        // No cached access token — try the backend session cookie before
        // giving up. Succeeds silently whenever a prior connection is still
        // alive.
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
          // Not connected yet — stay 'disconnected' (the normal first-visit
          // state) and fall through to the sample-email view below.
        }
      }

      try {
        await loadModels()
        if (!modelsLoaded()) throw new Error('models did not finish loading')
        for (const email of SAMPLE_EMAILS) {
          if (cancelled) return
          const result = await classify(email)
          setRows((prev) => prev.map((r) => (r.id === email.id ? withResult(r, result) : r)))
        }
        if (!cancelled) setStatus('ready')
      } catch (err) {
        if (!cancelled) {
          setError(message(err))
          setStatus('error')
        }
      }
    })()
    return () => {
      cancelled = true
    }
    // Mount-only on purpose: fetchAndClassify dedupes through the
    // module-level memo above, and its helpers read current state via refs,
    // so re-running this on every render would only re-fetch the inbox.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        navOpen,
        setNavOpen,
        gmailStatus,
        gmailError,
        connectGmail,
        refreshInbox,
        personalization,
        personalizationError,
        retraining,
        correctEmail,
        enablePersonalization,
        disablePersonalization,
        retrainPersonalization,
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
