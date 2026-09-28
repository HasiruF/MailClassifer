'use client'

import type { ReactNode } from 'react'
import { Archive, ArchiveRestore, ExternalLink, Mail, MailOpen } from 'lucide-react'
import type { CategoryLabel, PriorityBucket, SpamLabel } from '@/types'
import { useInbox } from './inbox-context'
import { Avatar } from './avatar'
import { CATEGORY_COLOR, DETAIL_BG, DETAIL_BORDER, DETAIL_FAINT, DETAIL_MUTED, HIGH, INK, MEDIUM } from './tokens'

function ActionButton({
  icon,
  label,
  onClick,
  href,
}: {
  icon: ReactNode
  label: string
  onClick?: () => void
  href?: string
}) {
  const className = 'flex items-center gap-1.5 font-mono text-[11px] tracking-wide focus-visible:outline-none focus-visible:ring-1'
  if (href) {
    return (
      <a href={href} target="_blank" rel="noreferrer" className={className} style={{ color: DETAIL_MUTED }}>
        {icon}
        {label}
      </a>
    )
  }
  return (
    <button onClick={onClick} className={className} style={{ color: DETAIL_MUTED }}>
      {icon}
      {label}
    </button>
  )
}

// Priority is a single 0-1 score across three named buckets — a semicircle
// gauge reads that at a glance the way a stacked bar-per-metric never
// could, and it's the one number worth the most visual weight (it's the
// model's headline call: does this need attention now).
function PriorityGauge({ score, bucket }: { score: number; bucket: PriorityBucket }) {
  const size = 128
  const r = 48
  const strokeWidth = 11
  const cx = size / 2
  const cy = size / 2
  const circumference = 2 * Math.PI * r
  const half = circumference / 2
  const fraction = Math.min(1, Math.max(0, score))
  const color = bucket === 'high' ? HIGH : bucket === 'medium' ? MEDIUM : DETAIL_FAINT

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size / 2 + strokeWidth} viewBox={`0 0 ${size} ${size / 2 + strokeWidth}`}>
        <g transform={`translate(0 ${strokeWidth / 2})`}>
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={DETAIL_BORDER}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${half} ${circumference}`}
            transform={`rotate(180 ${cx} ${cy})`}
          />
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${half * fraction} ${circumference}`}
            transform={`rotate(180 ${cx} ${cy})`}
          />
        </g>
      </svg>
      <div className="-mt-3 flex flex-col items-center">
        <span className="font-mono text-2xl font-semibold" style={{ color }}>
          {Math.round(score * 100)}%
        </span>
        <span className="font-mono text-[10px] tracking-[0.2em] uppercase" style={{ color: DETAIL_MUTED }}>
          {bucket}
        </span>
      </div>
    </div>
  )
}

// Category is a 3-way vote, not a single score — one segmented strip shows
// the split at a glance; the legend underneath gives the exact numbers,
// winner emphasized.
function CategoryStrip({ confidences, winner }: { confidences: Record<string, number>; winner: string }) {
  const order: CategoryLabel[] = ['Work', 'Personal', 'Other']
  return (
    <div>
      <div className="flex h-3 w-full overflow-hidden rounded-full" style={{ background: DETAIL_BORDER }}>
        {order.map((cat) => {
          const pct = (confidences[cat] ?? 0) * 100
          if (pct <= 0) return null
          return (
            <div
              key={cat}
              style={{ width: `${pct}%`, background: CATEGORY_COLOR[cat] }}
              title={`${cat} ${pct.toFixed(0)}%`}
            />
          )
        })}
      </div>
      <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1">
        {order.map((cat) => (
          <span
            key={cat}
            className="flex items-center gap-1.5 font-mono text-[11px] tracking-wide uppercase"
            style={{ color: cat === winner ? CATEGORY_COLOR[cat] : DETAIL_MUTED, fontWeight: cat === winner ? 600 : 400 }}
          >
            <span className="size-1.5 rounded-full" style={{ background: CATEGORY_COLOR[cat] }} aria-hidden />
            {cat} {Math.round((confidences[cat] ?? 0) * 100)}%
          </span>
        ))}
      </div>
    </div>
  )
}

// Spam is binary — a full bar chart is more weight than a yes/no deserves.
function SpamBadge({ label, confidence }: { label: SpamLabel; confidence: number }) {
  const color = label === 'spam' ? HIGH : DETAIL_FAINT
  return (
    <div className="flex flex-col items-start gap-2">
      <span
        className="rounded-sm px-2.5 py-1 font-mono text-xs font-semibold tracking-wide uppercase"
        style={{ color, background: `${color}22`, border: `1px solid ${color}66` }}
      >
        {label}
      </span>
      <span className="font-mono text-[11px]" style={{ color: DETAIL_MUTED }}>
        {(confidence * 100).toFixed(0)}% confidence
      </span>
    </div>
  )
}

export function EmailDetail({ id }: { id: string }) {
  const { rows, selectEmail, archivedIds, archiveEmail, unarchiveEmail, toggleRead } = useInbox()
  const row = rows.find((r) => r.id === id)

  if (!row) {
    return (
      <main className="p-6" style={{ color: INK }}>
        <button onClick={() => selectEmail(null)} className="font-mono text-xs tracking-wide" style={{ color: DETAIL_MUTED }}>
          ← INBOX
        </button>
        <p className="mt-4 font-mono text-sm">Unknown message.</p>
      </main>
    )
  }

  const archived = archivedIds.has(row.id)

  return (
    <main className="min-h-screen" style={{ background: DETAIL_BG, color: INK }}>
      <div className="mx-auto max-w-2xl px-6 py-6">
        <button onClick={() => selectEmail(null)} className="font-mono text-xs tracking-wide" style={{ color: DETAIL_MUTED }}>
          ← INBOX
        </button>

        <div className="mt-4 flex items-start gap-3">
          <Avatar name={row.fromName} size="md" />
          <div className="min-w-0 flex-1">
            <h1 className="font-serif text-2xl font-medium" style={{ color: INK }}>
              {row.subject}
            </h1>
            <p className="mt-1 font-mono text-[11px]" style={{ color: DETAIL_MUTED }}>
              FROM {row.fromName} &lt;{row.fromAddr}&gt; → {row.to} · {row.receivedAt}
            </p>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
          <ActionButton
            icon={row.unread ? <MailOpen size={13} /> : <Mail size={13} />}
            label={row.unread ? 'MARK READ' : 'MARK UNREAD'}
            onClick={() => toggleRead(row.id)}
          />
          <ActionButton
            icon={archived ? <ArchiveRestore size={13} /> : <Archive size={13} />}
            label={archived ? 'MOVE TO INBOX' : 'ARCHIVE'}
            onClick={() => (archived ? unarchiveEmail(row.id) : archiveEmail(row.id))}
          />
          {row.source === 'gmail' && (
            <ActionButton
              icon={<ExternalLink size={13} />}
              label="OPEN IN GMAIL"
              href={`https://mail.google.com/mail/u/0/#all/${row.id}`}
            />
          )}
        </div>

        <div className="my-5 h-px" style={{ background: DETAIL_BORDER }} />

        <p className="font-serif text-base leading-relaxed whitespace-pre-wrap" style={{ color: INK }}>
          {row.body}
        </p>

        <div className="my-6 h-px" style={{ background: DETAIL_BORDER }} />

        <h2 className="mb-3 font-mono text-[11px] tracking-[0.2em]" style={{ color: DETAIL_MUTED }}>
          MODEL OUTPUT
        </h2>

        {!row.result ? (
          <p className="font-mono text-xs" style={{ color: DETAIL_MUTED }}>
            classifying…
          </p>
        ) : (
          <div className="flex flex-col gap-5">
            <div>
              <p className="mb-2 font-mono text-[10px] tracking-widest" style={{ color: DETAIL_FAINT }}>
                CATEGORY
              </p>
              <CategoryStrip confidences={row.result.category.confidences} winner={row.result.category.label} />
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="mb-2 font-mono text-[10px] tracking-widest" style={{ color: DETAIL_FAINT }}>
                  PRIORITY
                </p>
                <PriorityGauge score={row.result.priority.score} bucket={row.result.priority.bucket} />
                {row.result.priority.note && (
                  <p className="mt-1 text-center font-mono text-[10px]" style={{ color: DETAIL_MUTED }}>
                    {row.result.priority.note}
                  </p>
                )}
              </div>

              <div>
                <p className="mb-2 font-mono text-[10px] tracking-widest" style={{ color: DETAIL_FAINT }}>
                  SPAM CHECK
                </p>
                <SpamBadge label={row.result.spam.label} confidence={row.result.spam.confidence} />
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
