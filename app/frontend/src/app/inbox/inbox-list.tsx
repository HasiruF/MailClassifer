'use client'

import { Archive, ArchiveRestore, Mail, MailOpen } from 'lucide-react'
import type { PriorityBucket } from '@/types'
import { useInbox, type Row } from './inbox-context'
import { Avatar } from './avatar'
import { BORDER, CATEGORY_COLOR, ERROR, FAINT, HIGH, INK, LOW_OPACITY, MEDIUM, MUTED, ROW_HOVER, SPAM } from './tokens'

function PriorityDot({ bucket }: { bucket: PriorityBucket | null }) {
  if (bucket === null) {
    return (
      <span
        className="mt-2 size-2 shrink-0 animate-pulse rounded-full"
        style={{ border: `1px solid ${FAINT}` }}
        aria-hidden
      />
    )
  }
  if (bucket === 'high') {
    return <span className="mt-2 size-2.5 shrink-0 rounded-full" style={{ background: HIGH }} aria-hidden />
  }
  if (bucket === 'medium') {
    return <span className="mt-2 size-2 shrink-0 rounded-full" style={{ background: MEDIUM }} aria-hidden />
  }
  return <span className="mt-2 size-1.5 shrink-0 rounded-full" style={{ border: `1px solid ${FAINT}` }} aria-hidden />
}

function Pill({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="rounded-sm px-1.5 py-px font-mono text-[10px] tracking-wide uppercase"
      style={{ color, background: `${color}22`, border: `1px solid ${color}55` }}
    >
      {label}
    </span>
  )
}

function matchesQuery(row: Row, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (!q) return true
  return (
    row.subject.toLowerCase().includes(q) ||
    row.fromName.toLowerCase().includes(q) ||
    row.body.toLowerCase().includes(q)
  )
}

function bucketFor(ms: number): 'Today' | 'Yesterday' | 'Earlier' {
  const now = new Date()
  const d = new Date(ms)
  if (d.toDateString() === now.toDateString()) return 'Today'
  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday'
  return 'Earlier'
}

function EmailRow({ row }: { row: Row }) {
  const { selectEmail, showArchived, toggleRead, archiveEmail, unarchiveEmail } = useInbox()

  return (
    <li
      data-testid="email-row"
      data-email-id={row.id}
      data-category={row.result?.category.label ?? ''}
      style={{
        borderTop: `1px solid ${BORDER}`,
        opacity: !showArchived && row.result?.priority.bucket === 'low' ? LOW_OPACITY : 1,
      }}
    >
      <div
        className="group flex w-full items-start gap-3 px-3 py-3 transition-colors"
        style={{ background: 'transparent' }}
        onMouseEnter={(e) => (e.currentTarget.style.background = ROW_HOVER)}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
      >
        <button onClick={() => selectEmail(row.id)} className="flex min-w-0 flex-1 items-start gap-3 text-left">
          <span
            className="mt-2.5 size-1.5 shrink-0 rounded-full"
            style={{ background: row.unread ? INK : 'transparent' }}
            aria-hidden
          />
          <PriorityDot bucket={row.result?.priority.bucket ?? null} />
          <Avatar name={row.fromName} />
          <span className="min-w-0 flex-1">
            <span className="flex items-baseline gap-2">
              <span className="w-32 shrink-0 truncate text-sm sm:w-36" style={{ color: INK }}>
                {row.fromName}
              </span>
              <span className="min-w-0 flex-1 truncate text-sm">
                <span style={{ color: INK, fontWeight: row.unread ? 500 : 400 }}>{row.subject}</span>
                <span className="hidden sm:inline" style={{ color: FAINT }}>
                  {' '}
                  — {row.body}
                </span>
              </span>
            </span>
            <span className="mt-1 flex items-center gap-1.5">
              {row.result ? (
                <>
                  <Pill label={row.result.category.label} color={CATEGORY_COLOR[row.result.category.label as 'Work' | 'Personal' | 'Other'] ?? MUTED} />
                  <Pill
                    label={row.result.priority.bucket}
                    color={row.result.priority.bucket === 'high' ? HIGH : row.result.priority.bucket === 'medium' ? MEDIUM : MUTED}
                  />
                  {row.result.spam.label === 'spam' && <Pill label="spam" color={SPAM} />}
                </>
              ) : (
                <span className="font-mono text-[10px] tracking-wide uppercase" style={{ color: FAINT }}>
                  classifying…
                </span>
              )}
            </span>
          </span>
        </button>

        <div className="flex shrink-0 items-center gap-2 self-start pt-0.5">
          <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
            <button
              onClick={(e) => {
                e.stopPropagation()
                toggleRead(row.id)
              }}
              className="rounded-sm p-1 focus-visible:outline-none focus-visible:ring-1"
              style={{ color: MUTED }}
              title={row.unread ? 'Mark as read' : 'Mark as unread'}
              aria-label={row.unread ? 'Mark as read' : 'Mark as unread'}
            >
              {row.unread ? <MailOpen size={14} /> : <Mail size={14} />}
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation()
                if (showArchived) unarchiveEmail(row.id)
                else archiveEmail(row.id)
              }}
              className="rounded-sm p-1 focus-visible:outline-none focus-visible:ring-1"
              style={{ color: MUTED }}
              title={showArchived ? 'Move to inbox' : 'Archive'}
              aria-label={showArchived ? 'Move to inbox' : 'Archive'}
            >
              {showArchived ? <ArchiveRestore size={14} /> : <Archive size={14} />}
            </button>
          </div>
          <span className="hidden font-mono text-[11px] sm:block" style={{ color: FAINT }}>
            {row.receivedAt}
          </span>
        </div>
      </div>
    </li>
  )
}

function DateSection({ label, rows }: { label: string; rows: Row[] }) {
  if (rows.length === 0) return null
  return (
    <>
      <li className="px-3 py-1.5 font-mono text-[10px] tracking-[0.2em]" style={{ color: FAINT, background: 'rgba(255,255,255,0.02)' }}>
        {label.toUpperCase()}
      </li>
      {rows.map((row) => (
        <EmailRow key={row.id} row={row} />
      ))}
    </>
  )
}

export function InboxList() {
  const { rows, status, error, filter, onlyHigh, showArchived, query, archivedIds } = useInbox()

  const scoped = rows.filter((r) => archivedIds.has(r.id) === showArchived)
  const visibleRows = scoped
    .filter((r) => showArchived || filter === 'All' || r.result?.category.label === filter)
    .filter((r) => showArchived || !onlyHigh || r.result?.priority.bucket === 'high')
    .filter((r) => matchesQuery(r, query))
    .sort((a, b) => b.receivedAtMs - a.receivedAtMs)

  const unreadCount = visibleRows.filter((r) => r.unread).length
  const today = visibleRows.filter((r) => bucketFor(r.receivedAtMs) === 'Today')
  const yesterday = visibleRows.filter((r) => bucketFor(r.receivedAtMs) === 'Yesterday')
  const earlier = visibleRows.filter((r) => bucketFor(r.receivedAtMs) === 'Earlier')

  const title = showArchived ? 'Archived' : onlyHigh ? 'High Priority' : filter === 'All' ? 'All Mail' : filter

  return (
    <main className="p-6">
      <header className="mb-3 flex items-baseline justify-between">
        <h1 className="font-mono text-xs tracking-[0.25em]" style={{ color: INK }}>
          {title.toUpperCase()}
        </h1>
        <span className="font-mono text-[11px] tracking-wide" style={{ color: MUTED }}>
          {unreadCount} UNREAD · {visibleRows.length} TOTAL
        </span>
      </header>
      <div className="mb-4 h-px" style={{ background: BORDER }} />

      {status === 'error' && (
        <p className="mb-4 font-mono text-xs" style={{ color: ERROR }}>
          MODELS FAILED TO LOAD: {error ?? 'unknown error'} — check public/models/*.onnx and *.vocab.json exist (see
          scripts/export_onnx.py).
        </p>
      )}

      <ul className="overflow-hidden rounded-sm" style={{ border: `1px solid ${BORDER}` }}>
        <DateSection label="Today" rows={today} />
        <DateSection label="Yesterday" rows={yesterday} />
        <DateSection label="Earlier" rows={earlier} />
        {visibleRows.length === 0 && (
          <li className="px-3 py-6 text-center font-mono text-xs" style={{ color: FAINT }}>
            {query.trim()
              ? `NO MAIL MATCHING "${query.trim().toUpperCase()}".`
              : `NOTHING IN ${title.toUpperCase()}.`}
          </li>
        )}
      </ul>
    </main>
  )
}
