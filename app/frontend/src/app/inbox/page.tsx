'use client'

import Link from 'next/link'
import type { PriorityBucket } from '@/types'
import { useInbox } from './inbox-context'
import { BORDER, ERROR, FAINT, HIGH, INK, LOW_OPACITY, MEDIUM, MUTED, ROW_HOVER, SPAM } from './tokens'

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

export default function Inbox() {
  const { rows, status, error, filter, onlyHigh } = useInbox()

  const visibleRows = rows
    .filter((r) => filter === 'All' || r.result?.category.label === filter)
    .filter((r) => !onlyHigh || r.result?.priority.bucket === 'high')
  const unreadCount = visibleRows.filter((r) => r.unread).length

  const title = onlyHigh ? 'High Priority' : filter === 'All' ? 'All Mail' : filter

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

      <ul className="rounded-sm" style={{ border: `1px solid ${BORDER}` }}>
        {visibleRows.map((row, i) => (
          <li
            key={row.id}
            style={{
              borderTop: i === 0 ? 'none' : `1px solid ${BORDER}`,
              opacity: row.result?.priority.bucket === 'low' ? LOW_OPACITY : 1,
            }}
          >
            <Link
              href={`/inbox/${row.id}`}
              className="flex w-full items-start gap-3 px-3 py-3 text-left transition-colors"
              style={{ background: 'transparent' }}
              onMouseEnter={(e) => (e.currentTarget.style.background = ROW_HOVER)}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            >
              <span
                className="mt-2.5 size-1.5 shrink-0 rounded-full"
                style={{ background: row.unread ? INK : 'transparent' }}
                aria-hidden
              />
              <PriorityDot bucket={row.result?.priority.bucket ?? null} />
              <span className="w-36 shrink-0 truncate text-sm sm:w-40" style={{ color: INK }}>
                {row.fromName}
              </span>
              <span className="min-w-0 flex-1 truncate text-sm">
                <span style={{ color: INK, fontWeight: row.unread ? 500 : 400 }}>{row.subject}</span>
                <span className="hidden sm:inline" style={{ color: FAINT }}>
                  {' '}
                  — {row.body}
                </span>
              </span>
              <span className="hidden shrink-0 font-mono text-[11px] sm:block" style={{ color: FAINT }}>
                {row.receivedAt}
              </span>
            </Link>

            <div
              className="flex items-center gap-2 px-3 pb-2.5 font-mono text-[11px] tracking-wide uppercase"
              style={{ marginLeft: '1.75rem', color: MUTED }}
            >
              {row.result ? (
                <>
                  <span>{row.result.category.label}</span>
                  <span style={{ color: FAINT }}>·</span>
                  <span
                    style={{
                      color:
                        row.result.priority.bucket === 'high'
                          ? HIGH
                          : row.result.priority.bucket === 'medium'
                            ? MEDIUM
                            : MUTED,
                    }}
                  >
                    {row.result.priority.bucket}
                  </span>
                  <span style={{ color: FAINT }}>·</span>
                  <span>{row.result.priority.score.toFixed(2)}</span>
                  {row.result.spam.label === 'spam' && (
                    <>
                      <span style={{ color: FAINT }}>·</span>
                      <span style={{ color: SPAM }}>spam</span>
                    </>
                  )}
                </>
              ) : (
                <span>classifying…</span>
              )}
            </div>
          </li>
        ))}
        {visibleRows.length === 0 && (
          <li className="px-3 py-6 text-center font-mono text-xs" style={{ color: FAINT }}>
            NOTHING IN {title.toUpperCase()}.
          </li>
        )}
      </ul>
    </main>
  )
}
