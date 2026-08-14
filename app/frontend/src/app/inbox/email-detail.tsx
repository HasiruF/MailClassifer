'use client'

import { useInbox } from './inbox-context'
import { DETAIL_BG, DETAIL_BORDER, DETAIL_FAINT, DETAIL_MUTED, HIGH, INK, MEDIUM } from './tokens'

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
  const { rows, selectEmail } = useInbox()
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

  return (
    <main className="min-h-screen" style={{ background: DETAIL_BG, color: INK }}>
      <div className="mx-auto max-w-2xl px-6 py-6">
        <button onClick={() => selectEmail(null)} className="font-mono text-xs tracking-wide" style={{ color: DETAIL_MUTED }}>
          ← INBOX
        </button>

        <h1 className="mt-4 text-xl font-medium" style={{ color: INK }}>
          {row.subject}
        </h1>
        <p className="mt-2 font-mono text-[11px]" style={{ color: DETAIL_MUTED }}>
          FROM {row.fromName} &lt;{row.fromAddr}&gt; → {row.to} · {row.receivedAt}
        </p>

        <div className="my-5 h-px" style={{ background: DETAIL_BORDER }} />

        <p className="text-sm whitespace-pre-wrap" style={{ color: INK }}>
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
