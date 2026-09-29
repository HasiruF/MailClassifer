'use client'

import { useState } from 'react'
import { useInbox } from './inbox-context'
import { BORDER, FAINT, HIGH, INK, MUTED, SIDEBAR_BG } from './tokens'

const CATEGORIES = ['Work', 'Personal', 'Other']

function NavRow({
  active,
  label,
  count,
  onClick,
  accent,
}: {
  active: boolean
  label: string
  count: string
  onClick: () => void
  accent?: string
}) {
  return (
    <button
      onClick={onClick}
      className="flex w-full items-center justify-between rounded-sm px-3 py-1.5 font-mono text-[11px] tracking-wide focus-visible:outline-none focus-visible:ring-1"
      style={{
        background: active ? BORDER : 'transparent',
        color: active ? INK : MUTED,
      }}
    >
      <span className="flex items-center gap-2">
        {accent && <span className="size-1.5 rounded-full" style={{ background: accent }} aria-hidden />}
        {label.toUpperCase()}
      </span>
      <span style={{ color: FAINT }}>{count}</span>
    </button>
  )
}

export function Sidebar() {
  const {
    rows,
    status,
    filter,
    setFilter,
    onlyHigh,
    setOnlyHigh,
    showArchived,
    setShowArchived,
    query,
    setQuery,
    archivedIds,
    selectedId,
    selectEmail,
    gmailStatus,
    gmailError,
    connectGmail,
    refreshInbox,
    personalization,
    personalizationError,
    retraining,
    retrainPersonalization,
    disablePersonalization,
  } = useInbox()
  const categories = [...CATEGORIES, ...(personalization?.custom_labels ?? [])]

  const [rememberMe, setRememberMe] = useState(true)

  const models = personalization?.models ?? []
  const totalCorrections = models.reduce((n, m) => n + m.correction_count, 0)
  const nextRetrainIn = models.length ? Math.min(...models.map((m) => m.corrections_until_retrain)) : 5
  const lastAttempt = models
    .filter((m) => m.last_attempt)
    .sort((a, b) => (a.last_attempt!.created_at < b.last_attempt!.created_at ? 1 : -1))[0]
  const lastAttemptText = lastAttempt?.last_attempt
    ? lastAttempt.last_attempt.status === 'rejected'
      ? `LAST RETRAIN REJECTED: ${lastAttempt.last_attempt.metrics.reason ?? lastAttempt.last_attempt.metrics.error ?? 'unknown reason'}`
      : `LAST RETRAIN: ${lastAttempt.model.toUpperCase()} V${lastAttempt.last_attempt.version}`
    : null

  function turnOffPersonalization() {
    if (window.confirm('Turning off personalization deletes your corrections and personalized models. Continue?')) {
      void disablePersonalization()
    }
  }

  const live = (r: (typeof rows)[number]) => !archivedIds.has(r.id)
  const countFor = (cat: string) => {
    if (status === 'loading') return '·'
    if (cat === 'All') return String(rows.filter(live).length)
    return String(rows.filter(live).filter((r) => r.result?.category.label === cat).length)
  }
  const highCount =
    status === 'loading' ? '·' : String(rows.filter(live).filter((r) => r.result?.priority.bucket === 'high').length)
  const inList = selectedId === null

  function goToAll() {
    setFilter('All')
    setOnlyHigh(false)
    setShowArchived(false)
    selectEmail(null)
  }
  function goToCategory(c: string) {
    setFilter(c)
    setOnlyHigh(false)
    setShowArchived(false)
    selectEmail(null)
  }
  function goToHighPriority() {
    setOnlyHigh(true)
    setShowArchived(false)
    selectEmail(null)
  }
  function goToArchived() {
    setShowArchived(true)
    selectEmail(null)
  }

  return (
    <aside
      className="flex w-56 shrink-0 flex-col gap-4 px-3 py-5"
      style={{ background: SIDEBAR_BG, borderRight: `1px solid ${BORDER}` }}
    >
      <button
        onClick={() => selectEmail(null)}
        className="px-1 text-left font-mono text-xs tracking-[0.25em]"
        style={{ color: INK }}
      >
        ◆ INBOX
      </button>

      <label
        className="flex items-center gap-1.5 rounded-sm px-2 py-1.5 focus-within:ring-1"
        style={{ border: `1px solid ${BORDER}` }}
      >
        <span className="font-mono text-[11px]" style={{ color: FAINT }} aria-hidden>
          {'>'}
        </span>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="search mail"
          aria-label="Search mail"
          className="w-full bg-transparent font-mono text-[11px] tracking-wide outline-none placeholder:opacity-50"
          style={{ color: INK }}
        />
      </label>

      <nav className="flex flex-col gap-0.5">
        <NavRow
          active={inList && !showArchived && filter === 'All' && !onlyHigh}
          label="All Mail"
          count={countFor('All')}
          onClick={goToAll}
        />
        {categories.map((c) => (
          <NavRow
            key={c}
            active={inList && !showArchived && filter === c && !onlyHigh}
            label={c}
            count={countFor(c)}
            onClick={() => goToCategory(c)}
          />
        ))}
      </nav>

      <div className="h-px" style={{ background: BORDER }} />

      <nav className="flex flex-col gap-0.5">
        <NavRow
          active={inList && !showArchived && onlyHigh}
          label="High Priority"
          count={highCount}
          accent={HIGH}
          onClick={goToHighPriority}
        />
        <NavRow
          active={inList && showArchived}
          label="Archived"
          count={String(archivedIds.size)}
          onClick={goToArchived}
        />
      </nav>

      <div className="h-px" style={{ background: BORDER }} />

      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between gap-1">
          <button
            onClick={() => connectGmail(rememberMe)}
            disabled={gmailStatus === 'connecting' || gmailStatus === 'fetching' || gmailStatus === 'connected'}
            className="flex-1 px-1 text-left font-mono text-[10px] tracking-wide disabled:cursor-default"
            style={{ color: gmailStatus === 'error' ? HIGH : gmailStatus === 'connected' ? INK : MUTED }}
            title={gmailStatus === 'error' ? (gmailError ?? undefined) : undefined}
          >
            {gmailStatus === 'disconnected' && '○ CONNECT GMAIL'}
            {gmailStatus === 'connecting' && '○ CONNECTING…'}
            {gmailStatus === 'fetching' && '○ FETCHING INBOX…'}
            {gmailStatus === 'connected' && '● GMAIL CONNECTED'}
            {gmailStatus === 'error' && '○ GMAIL ERROR — RETRY'}
          </button>
          {gmailStatus === 'connected' && (
            <button
              onClick={refreshInbox}
              className="px-1 font-mono text-[11px]"
              style={{ color: MUTED }}
              title="Refresh inbox"
              aria-label="Refresh inbox"
            >
              ↻
            </button>
          )}
        </div>
        {(gmailStatus === 'disconnected' || gmailStatus === 'error') && (
          <label className="flex items-center gap-1.5 px-1 font-mono text-[10px] tracking-wide" style={{ color: FAINT }}>
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              className="size-3"
            />
            remember me
          </label>
        )}
      </div>

      {gmailStatus === 'connected' && personalization?.enabled && (
        <div
          className="flex flex-col gap-1 px-1 font-mono text-[10px] tracking-wide"
          style={{ color: FAINT }}
          data-testid="personalization-status"
        >
          <span>
            {totalCorrections} CORRECTIONS · {retraining ? 'RETRAINING…' : `RETRAIN IN ${nextRetrainIn}`}
          </span>
          {lastAttemptText && (
            <span style={{ color: lastAttempt?.last_attempt?.status === 'rejected' ? HIGH : MUTED }}>{lastAttemptText}</span>
          )}
          {personalizationError && <span style={{ color: HIGH }}>{personalizationError}</span>}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => void retrainPersonalization()}
              disabled={retraining || totalCorrections === 0}
              data-testid="retrain-now"
              className="disabled:opacity-40"
              style={{ color: MUTED }}
            >
              ↻ RETRAIN NOW
            </button>
            <button type="button" onClick={turnOffPersonalization} style={{ color: FAINT }}>
              TURN OFF
            </button>
          </div>
        </div>
      )}

      <div className="mt-auto px-1 font-mono text-[10px] tracking-wide" style={{ color: FAINT }}>
        {status === 'loading' && 'MODEL LOADING…'}
        {status === 'ready' && 'MODEL READY · ON-DEVICE'}
        {status === 'error' && <span style={{ color: HIGH }}>MODEL ERROR</span>}
      </div>
    </aside>
  )
}
