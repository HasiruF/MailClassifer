'use client'

import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { loadModels, modelsLoaded, classify } from '@/inference/engine'
import type { ClassificationResult, CategoryLabel } from '@/types'
import { SAMPLE_EMAILS, type SampleEmail } from '@/data/sample-emails'

export type Row = SampleEmail & { result: ClassificationResult | null }

interface InboxState {
  rows: Row[]
  status: 'loading' | 'ready' | 'error'
  error: string | null
  filter: CategoryLabel | 'All'
  setFilter: (f: CategoryLabel | 'All') => void
  onlyHigh: boolean
  setOnlyHigh: (v: boolean) => void
}

const InboxContext = createContext<InboxState | null>(null)

export function InboxProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [rows, setRows] = useState<Row[]>(SAMPLE_EMAILS.map((e) => ({ ...e, result: null })))
  const [filter, setFilter] = useState<CategoryLabel | 'All'>('All')
  const [onlyHigh, setOnlyHigh] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
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
    <InboxContext.Provider value={{ rows, status, error, filter, setFilter, onlyHigh, setOnlyHigh }}>
      {children}
    </InboxContext.Provider>
  )
}

export function useInbox() {
  const ctx = useContext(InboxContext)
  if (!ctx) throw new Error('useInbox must be used within InboxProvider')
  return ctx
}
