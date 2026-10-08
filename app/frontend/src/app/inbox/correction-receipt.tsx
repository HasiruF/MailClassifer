'use client'

import { useEffect, useState, type ReactNode } from 'react'
import { Check, X } from 'lucide-react'
import { getVocabs } from '@/inference/engine'
import { getCorrectionDetail, type CorrectionDetail } from '@/lib/personalization-api'
import { decodeCorrection, type Receipt, type WeightedTerm } from '@/lib/receipt'
import type { PersonalizableModel } from '@/types'
import { CERULEAN_TEXT, CORAL_TEXT, FOCUS_RING, INK, LINE, LINE_SOFT, MUTED, SURFACE, TEAL_TEXT, TEAL_TINT } from './tokens'

const TOP_WORDS = 8
const TOP_PIECES = 12
const NEVER_SENT = ['The subject line', 'The email as written', 'Who sent it', 'Your other emails']

type State =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; detail: CorrectionDetail; receipt: Receipt }

function LeaderRow({ name, value, mono = false }: { name: string; value: string; mono?: boolean }) {
  return (
    <li className="flex items-baseline gap-1.5">
      <span className={`min-w-0 break-words ${mono ? 'font-mono' : ''}`}>{name}</span>
      <span className="min-w-4 flex-1 border-b border-dotted" style={{ borderColor: '#B9C3C0' }} aria-hidden />
      <span className="shrink-0 font-mono">{value}</span>
    </li>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2 border-t border-dashed px-5 py-3.5" style={{ borderColor: '#CFD6D2' }}>
      <h5 className="text-[13px] font-semibold">{title}</h5>
      {children}
    </div>
  )
}

function Words({ words }: { words: WeightedTerm[] }) {
  const [all, setAll] = useState(false)
  if (words.length === 0) {
    return (
      <p className="text-[13px]" style={{ color: MUTED }}>
        None of this email&apos;s words are in the model&apos;s word list.
      </p>
    )
  }
  const shown = all ? words : words.slice(0, TOP_WORDS)
  return (
    <>
      <ul className="flex flex-col gap-0.5 text-[13px]">
        {shown.map((w) => (
          <LeaderRow key={w.term} name={w.term} value={w.weight.toFixed(3)} mono />
        ))}
      </ul>
      {words.length > TOP_WORDS && (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className={`self-start rounded-sm text-[13px] font-medium hover:underline ${FOCUS_RING}`}
          style={{ color: CERULEAN_TEXT }}
        >
          {all ? 'Show fewer' : `Show all ${words.length} words`}
        </button>
      )}
    </>
  )
}

// What the server holds for one correction, decoded on this device. The
// server sends back exactly what it stored; nothing here is recomputed from
// the email.
export function CorrectionReceipt({
  messageId,
  model,
  title,
  labelFor,
  onClose,
}: {
  messageId: string
  model: PersonalizableModel
  title: string
  labelFor: (value: string) => string
  onClose: () => void
}) {
  const [state, setState] = useState<State>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    getCorrectionDetail(messageId).then(
      (rows) => {
        if (cancelled) return
        const detail = rows.find((r) => r.model === model)
        const vocabs = getVocabs()
        if (!detail) setState({ status: 'error', message: 'The server has no saved correction for this email.' })
        else if (!vocabs) setState({ status: 'error', message: 'The models are still loading. Try again in a moment.' })
        else setState({ status: 'ready', detail, receipt: decodeCorrection(model, detail.feature_vector, vocabs) })
      },
      (err: unknown) => {
        if (cancelled) return
        setState({
          status: 'error',
          message: `Couldn't load what was sent: ${err instanceof Error ? err.message : String(err)}`,
        })
      },
    )
    return () => {
      cancelled = true
    }
  }, [messageId, model])

  return (
    <section
      aria-labelledby="receipt-title"
      data-testid="correction-receipt"
      className="flex flex-col rounded-xl border"
      style={{ borderColor: LINE, background: SURFACE }}
    >
      <div className="flex items-start gap-3 px-5 pt-4 pb-3">
        <div className="flex flex-1 flex-col gap-0.5">
          <span className="text-xs font-semibold" style={{ color: MUTED }}>
            {title} correction
            {state.status === 'ready' &&
              ` · saved ${new Date(state.detail.created_at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}`}
          </span>
          <h4 id="receipt-title" className="font-display text-lg font-bold">
            What was sent to the server
          </h4>
        </div>
        <button
          type="button"
          aria-label="Close"
          onClick={onClose}
          className={`flex size-8 items-center justify-center rounded-md hover:bg-[#F6F7F2] ${FOCUS_RING}`}
          style={{ color: MUTED }}
        >
          <X size={16} strokeWidth={2} aria-hidden />
        </button>
      </div>

      {state.status === 'loading' && (
        <p className="px-5 pb-4 text-sm motion-safe:animate-pulse" style={{ color: MUTED }}>
          Loading what was sent…
        </p>
      )}
      {state.status === 'error' && (
        <p className="px-5 pb-4 text-[13px]" style={{ color: CORAL_TEXT }} role="alert">
          {state.message}
        </p>
      )}
      {state.status === 'ready' && (
        <>
          <Section title="Your correction">
            <ul className="flex flex-col gap-0.5 text-[13px]">
              <LeaderRow
                name="The model's guess"
                value={`${labelFor(state.detail.predicted_label)} · ${Math.round(state.detail.predicted_confidence * 100)}%`}
              />
              <LeaderRow name="Your answer" value={labelFor(state.detail.corrected_label)} />
              <LeaderRow name="Message ID" value={state.detail.provider_message_id} />
            </ul>
          </Section>
          <Section title="Words it counted">
            <Words words={state.receipt.words} />
          </Section>
          {state.receipt.pieces.length > 0 && (
            <Section title="Letter groups it counted">
              <div className="flex flex-wrap gap-1.5 font-mono text-xs">
                {state.receipt.pieces.slice(0, TOP_PIECES).map((p) => (
                  <span key={p.term} className="rounded px-1.5 py-0.5 whitespace-pre" style={{ background: LINE_SOFT }}>
                    {p.term}
                  </span>
                ))}
                {state.receipt.pieces.length > TOP_PIECES && (
                  <span className="px-1.5 py-0.5" style={{ color: MUTED }}>
                    + {state.receipt.pieces.length - TOP_PIECES} more
                  </span>
                )}
              </div>
              <p className="text-xs leading-relaxed" style={{ color: MUTED }}>
                Short pieces of words the model knows. They can hint at words that aren&apos;t in its word list.
              </p>
            </Section>
          )}
          {state.receipt.signals.length > 0 && (
            <Section title="Signals about the email">
              <ul className="flex flex-col gap-0.5 text-[13px]">
                {state.receipt.signals.map((s) => (
                  <LeaderRow key={s.label} name={s.label} value={s.value} />
                ))}
              </ul>
            </Section>
          )}
          <div className="m-3 mt-1 flex flex-col gap-2 rounded-lg px-4 py-3" style={{ background: TEAL_TINT }}>
            <h5 className="text-[13px] font-semibold" style={{ color: TEAL_TEXT }}>
              Never sent
            </h5>
            <ul className="grid grid-cols-1 gap-x-3 gap-y-1 text-[13px] sm:grid-cols-2" style={{ color: INK }}>
              {NEVER_SENT.map((item) => (
                <li key={item} className="flex items-center gap-1.5">
                  <Check size={14} strokeWidth={2.25} style={{ color: TEAL_TEXT }} aria-hidden />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </section>
  )
}
