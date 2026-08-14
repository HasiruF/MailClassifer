'use client'

import { useInbox } from './inbox-context'
import { BORDER, FAINT, HIGH, INK, MUTED, SIDEBAR_BG } from './tokens'
import type { CategoryLabel } from '@/types'

const CATEGORIES: CategoryLabel[] = ['Work', 'Personal', 'Other']

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
    selectedId,
    selectEmail,
    gmailStatus,
    gmailError,
    connectGmail,
    refreshInbox,
  } = useInbox()

  const countFor = (cat: CategoryLabel | 'All') => {
    if (status === 'loading') return '·'
    if (cat === 'All') return String(rows.length)
    return String(rows.filter((r) => r.result?.category.label === cat).length)
  }
  const highCount = status === 'loading' ? '·' : String(rows.filter((r) => r.result?.priority.bucket === 'high').length)
  const inList = selectedId === null

  return (
    <aside
      className="flex w-52 shrink-0 flex-col gap-4 px-3 py-5"
      style={{ background: SIDEBAR_BG, borderRight: `1px solid ${BORDER}` }}
    >
      <button
        onClick={() => selectEmail(null)}
        className="px-1 text-left font-mono text-xs tracking-[0.25em]"
        style={{ color: INK }}
      >
        ◆ INBOX
      </button>

      <nav className="flex flex-col gap-0.5">
        <NavRow
          active={inList && filter === 'All' && !onlyHigh}
          label="All Mail"
          count={countFor('All')}
          onClick={() => {
            setFilter('All')
            setOnlyHigh(false)
            selectEmail(null)
          }}
        />
        {CATEGORIES.map((c) => (
          <NavRow
            key={c}
            active={inList && filter === c && !onlyHigh}
            label={c}
            count={countFor(c)}
            onClick={() => {
              setFilter(c)
              setOnlyHigh(false)
              selectEmail(null)
            }}
          />
        ))}
      </nav>

      <div className="h-px" style={{ background: BORDER }} />

      <nav className="flex flex-col gap-0.5">
        <NavRow
          active={inList && onlyHigh}
          label="High Priority"
          count={highCount}
          accent={HIGH}
          onClick={() => {
            setOnlyHigh(true)
            selectEmail(null)
          }}
        />
      </nav>

      <div className="h-px" style={{ background: BORDER }} />

      <div className="flex items-center justify-between gap-1">
        <button
          onClick={connectGmail}
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

      <div className="mt-auto px-1 font-mono text-[10px] tracking-wide" style={{ color: FAINT }}>
        {status === 'loading' && 'MODEL LOADING…'}
        {status === 'ready' && 'MODEL READY · ON-DEVICE'}
        {status === 'error' && <span style={{ color: HIGH }}>MODEL ERROR</span>}
      </div>
    </aside>
  )
}
