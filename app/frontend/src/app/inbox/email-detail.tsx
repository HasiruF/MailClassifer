'use client'

import type { ReactNode } from 'react'
import { Archive, ArchiveRestore, ExternalLink, Mail, MailOpen } from 'lucide-react'
import { useInbox } from './inbox-context'
import { Avatar } from './avatar'
import { DETAIL_BG, DETAIL_BORDER, DETAIL_FAINT, DETAIL_MUTED, HIGH, INK, MEDIUM } from './tokens'

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

function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-20 shrink-0 font-mono text-[11px] tracking-wide uppercase" style={{ color: DETAIL_MUTED }}>
        {label}
      </span>
      <div className="h-1.5 flex-1 rounded-full" style={{ background: DETAIL_BORDER }}>
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.round(value * 100)}%`, background: color }}
        />
      </div>
      <span className="w-10 shrink-0 text-right font-mono text-[11px]" style={{ color: DETAIL_MUTED }}>
        {(value * 100).toFixed(0)}%
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

  const priorityColor =
    row.result?.priority.bucket === 'high' ? HIGH : row.result?.priority.bucket === 'medium' ? MEDIUM : DETAIL_FAINT
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
          <div className="flex flex-col gap-4">
            <div>
              <p className="mb-1.5 font-mono text-[10px] tracking-widest" style={{ color: DETAIL_FAINT }}>
                CATEGORY — {row.result.category.label.toUpperCase()}
              </p>
              <div className="flex flex-col gap-1.5">
                {Object.entries(row.result.category.confidences)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, conf]) => (
                    <Bar key={cls} label={cls} value={conf} color={INK} />
                  ))}
              </div>
            </div>

            <div>
              <p className="mb-1.5 font-mono text-[10px] tracking-widest" style={{ color: DETAIL_FAINT }}>
                PRIORITY — {row.result.priority.bucket.toUpperCase()}
              </p>
              <Bar label="score" value={row.result.priority.score} color={priorityColor} />
              {row.result.priority.note && (
                <p className="mt-1.5 font-mono text-[11px]" style={{ color: DETAIL_MUTED }}>
                  {row.result.priority.note}
                </p>
              )}
            </div>

            <div>
              <p className="mb-1.5 font-mono text-[10px] tracking-widest" style={{ color: DETAIL_FAINT }}>
                SPAM CHECK — {row.result.spam.label.toUpperCase()}
              </p>
              <Bar
                label="spam"
                value={row.result.spam.confidence}
                color={row.result.spam.label === 'spam' ? HIGH : DETAIL_FAINT}
              />
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
