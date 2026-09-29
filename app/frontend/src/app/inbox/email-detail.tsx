'use client'

import { useState, type ReactNode } from 'react'
import {
  Archive,
  ArchiveRestore,
  ArrowLeft,
  ChevronDown,
  Cpu,
  ExternalLink,
  ImageOff,
  Mail,
  MailOpen,
  MousePointerClick,
} from 'lucide-react'
import type { PersonalizableModel, PriorityBucket, SpamLabel } from '@/types'
import { predictedFor } from '@/lib/corrections'
import { splitLinks, tidyPlainText } from '@/lib/links'
import { useInbox } from './inbox-context'
import { Avatar } from './avatar'
import { CategoryChip, PriorityChip, SpamChip } from './inbox-list'
import { EmailHtmlBody } from './email-html-body'
import { CorrectionPicker, OptInPrompt, type CorrectionOption } from './correction-picker'
import {
  APRICOT_SWATCH,
  APRICOT_TEXT,
  BASE_CATEGORIES,
  CATEGORY_SWATCH,
  CERULEAN_TEXT,
  CORAL,
  CORAL_TEXT,
  FOCUS_RING,
  INK,
  LINE,
  MUTED,
  PANEL_BG,
  READING_BG,
  SURFACE,
  TEAL_TEXT,
  TEAL_TINT,
  isBaseCategory,
} from './tokens'

// Bar color for a user's own label (it has no brand swatch).
const CUSTOM_BAR = '#8FA3A8'

// Whether "How this email was sorted" is expanded. Remembered per browser
// (a viewing preference, not data), so collapsing it once keeps it
// collapsed across emails and reloads.
const SORTED_OPEN_KEY = 'inbox.sortedPanelOpen'

function readSortedOpen(): boolean {
  try {
    return localStorage.getItem(SORTED_OPEN_KEY) !== '0'
  } catch {
    return true
  }
}

function writeSortedOpen(open: boolean) {
  try {
    localStorage.setItem(SORTED_OPEN_KEY, open ? '1' : '0')
  } catch {
    // Storage blocked (private window etc.): the toggle still works for this session.
  }
}

function ToolbarButton({
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
  const className = `inline-flex h-[34px] items-center gap-1.5 rounded-[7px] border px-3 text-[13px] font-medium transition-colors hover:bg-[#F6F7F2] ${FOCUS_RING}`
  const style = { borderColor: '#D9DDD3', background: SURFACE, color: INK }
  if (href) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className={className}
        style={style}
        aria-label={label}
        title={label}
      >
        {icon}
        <span className="hidden sm:inline">{label}</span>
      </a>
    )
  }
  return (
    <button type="button" onClick={onClick} className={className} style={style} aria-label={label} title={label}>
      {icon}
      <span className="hidden sm:inline">{label}</span>
    </button>
  )
}

function VerdictCell({
  title,
  corrected,
  onCorrect,
  correcting,
  testId,
  children,
}: {
  title: string
  corrected: boolean
  onCorrect?: () => void
  correcting: boolean
  testId: string
  children: ReactNode
}) {
  return (
    <div className="flex min-w-0 flex-col gap-3 p-[18px]" style={{ background: SURFACE }}>
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold" style={{ color: MUTED }}>
          {title}
        </span>
        {corrected && (
          <span
            className="rounded-full px-2 py-px text-[11px] font-semibold"
            style={{ background: TEAL_TINT, color: TEAL_TEXT }}
          >
            Corrected
          </span>
        )}
      </div>
      {children}
      {onCorrect && (
        <button
          type="button"
          onClick={onCorrect}
          aria-expanded={correcting}
          data-testid={testId}
          className={`mt-auto h-8 self-start rounded-[7px] border px-3 text-[13px] font-medium transition-colors ${FOCUS_RING} ${
            correcting ? '' : 'hover:bg-[#F6F7F2]'
          }`}
          style={
            correcting
              ? {
                  borderColor: '#00789C',
                  background: '#EEF7F9',
                  color: '#005873',
                }
              : { borderColor: '#CFD6D2', background: SURFACE, color: INK }
          }
        >
          Correct
        </button>
      )}
    </div>
  )
}

function CategoryVerdict({ confidences, winner }: { confidences: Record<string, number>; winner: string }) {
  const classes = [
    ...BASE_CATEGORIES.filter((c) => c in confidences),
    ...Object.keys(confidences)
      .filter((c) => !isBaseCategory(c))
      .sort(),
  ]
  const winnerPct = confidences[winner]
  return (
    <>
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="truncate font-display text-2xl font-bold">{winner}</span>
        {winnerPct !== undefined && (
          <span className="font-mono text-[13px]" style={{ color: MUTED }}>
            {Math.round(winnerPct * 100)}%
          </span>
        )}
      </div>
      <div className="flex h-2 gap-0.5 overflow-hidden rounded" aria-hidden>
        {classes.map((c) =>
          confidences[c] > 0 ? (
            <span
              key={c}
              style={{
                width: `${confidences[c] * 100}%`,
                background: CATEGORY_SWATCH[c] ?? CUSTOM_BAR,
              }}
            />
          ) : null,
        )}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px]" style={{ color: MUTED }}>
        {classes.map((c) => (
          <span key={c} style={c === winner ? { color: INK, fontWeight: 500 } : undefined}>
            {c} {Math.round(confidences[c] * 100)}
          </span>
        ))}
      </div>
    </>
  )
}

const PRIORITY_LEVEL: Record<PriorityBucket, number> = {
  low: 1,
  medium: 2,
  high: 3,
}
const PRIORITY_STYLE: Record<PriorityBucket, { word: string; color: string; fill: string }> = {
  low: { word: 'Low', color: INK, fill: '#9FB3B8' },
  medium: { word: 'Medium', color: APRICOT_TEXT, fill: APRICOT_SWATCH },
  high: { word: 'High', color: CORAL_TEXT, fill: CORAL },
}

function PriorityVerdict({ score, bucket, note }: { score: number; bucket: PriorityBucket; note?: string }) {
  const style = PRIORITY_STYLE[bucket]
  const level = PRIORITY_LEVEL[bucket]
  return (
    <>
      <div className="flex items-baseline gap-2">
        <span className="font-display text-2xl font-bold" style={{ color: style.color }}>
          {style.word}
        </span>
        <span className="font-mono text-[13px]" style={{ color: MUTED }}>
          score {score.toFixed(2)}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-[3px]" aria-hidden>
        {[1, 2, 3].map((i) => (
          <span key={i} className="h-2 rounded-[2px]" style={{ background: i <= level ? style.fill : '#EEF0EA' }} />
        ))}
      </div>
      <div className="grid grid-cols-3 font-mono text-[11px]" style={{ color: MUTED }}>
        {(['low', 'medium', 'high'] as const).map((b) => (
          <span key={b} style={b === bucket ? { color: INK, fontWeight: 500 } : undefined}>
            {PRIORITY_STYLE[b].word}
          </span>
        ))}
      </div>
      {note && (
        <span className="text-xs leading-snug" style={{ color: MUTED }}>
          Lowered because this looks like spam.
        </span>
      )}
    </>
  )
}

function SpamVerdict({ label, confidence }: { label: SpamLabel; confidence: number }) {
  const pct = Math.round(confidence * 100)
  return (
    <>
      <span className="font-display text-2xl font-bold" style={{ color: label === 'spam' ? CORAL_TEXT : INK }}>
        {label === 'spam' ? 'Spam' : 'Not spam'}
      </span>
      <div className="relative h-2 rounded" style={{ background: '#EEF0EA' }} aria-hidden>
        <span
          className="absolute inset-y-0 left-0 rounded"
          style={{ width: `${Math.max(pct, 1)}%`, background: CORAL }}
        />
        <span className="absolute -inset-y-[3px] left-1/2 w-0.5" style={{ background: '#9AA8AB' }} />
      </div>
      <div className="flex justify-between font-mono text-[11px]" style={{ color: MUTED }}>
        <span>{pct}% spam</span>
        <span>flags at 50%</span>
      </div>
    </>
  )
}

const HEADINGS: Record<PersonalizableModel, string> = {
  category: 'What should this email be?',
  priority: 'How urgent is this email?',
  spam: 'Is this spam?',
}
const MODEL_TITLES: Record<PersonalizableModel, string> = {
  category: 'Category',
  priority: 'Priority',
  spam: 'Spam check',
}
const PRIORITY_OPTIONS: CorrectionOption[] = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]
const SPAM_OPTIONS: CorrectionOption[] = [
  { value: 'spam', label: 'Spam' },
  { value: 'ham', label: 'Not spam' },
]

function optionLabel(model: PersonalizableModel, value: string): string {
  if (model === 'priority') return PRIORITY_STYLE[value as PriorityBucket]?.word ?? value
  if (model === 'spam') return value === 'spam' ? 'Spam' : 'Not spam'
  return value
}

export function NoEmailSelected() {
  return (
    <div
      className="flex flex-1 flex-col items-center justify-center gap-3 px-8 text-center"
      style={{ background: READING_BG }}
    >
      <span
        className="flex size-12 items-center justify-center rounded-full border"
        style={{ borderColor: LINE, background: SURFACE, color: MUTED }}
      >
        <MousePointerClick size={20} strokeWidth={1.75} aria-hidden />
      </span>
      <p className="font-display text-lg font-semibold">Select an email to read it</p>
      <p className="max-w-xs text-sm leading-relaxed" style={{ color: MUTED }}>
        Each email shows how it was sorted. If a label is wrong, you can correct it.
      </p>
    </div>
  )
}

export function EmailDetail({ id }: { id: string }) {
  const {
    rows,
    selectEmail,
    archivedIds,
    archiveEmail,
    unarchiveEmail,
    toggleRead,
    personalization,
    personalizationError,
    correctEmail,
    enablePersonalization,
  } = useInbox()
  const [picking, setPicking] = useState<PersonalizableModel | null>(null)
  const [optInFor, setOptInFor] = useState<PersonalizableModel | null>(null)
  const [enabling, setEnabling] = useState(false)
  const [sortedOpen, setSortedOpen] = useState(readSortedOpen)
  const [showImagesFor, setShowImagesFor] = useState<string | null>(null)
  const row = rows.find((r) => r.id === id)

  const backButton = (
    <button
      type="button"
      onClick={() => selectEmail(null)}
      className={`inline-flex h-[34px] items-center gap-1.5 rounded-[7px] px-2 text-[13px] font-medium lg:hidden ${FOCUS_RING}`}
      style={{ color: INK }}
    >
      <ArrowLeft size={16} strokeWidth={1.75} aria-hidden />
      Back
    </button>
  )

  if (!row) {
    return (
      <div className="flex flex-1 flex-col gap-4 p-6">
        {backButton}
        <p className="text-sm" style={{ color: MUTED }}>
          This email is no longer in the list.
        </p>
      </div>
    )
  }

  const archived = archivedIds.has(row.id)
  // Spam uses remote images to confirm an address is live, so they stay
  // blocked until the user asks (as Gmail does). Until the email is
  // classified we can't tell, so they wait for the verdict too.
  const isSpam = row.result?.spam.label === 'spam'
  const imagesAllowed = row.result !== null && (!isSpam || showImagesFor === row.id)
  // Corrections need a real message id and a backend connection, so the
  // hand-authored sample emails can't be corrected.
  const canCorrect = row.source === 'gmail' && row.modelResult !== null
  const active = optInFor ?? picking

  const startCorrecting = (model: PersonalizableModel) => {
    if (active === model) {
      setPicking(null)
      setOptInFor(null)
    } else if (personalization?.enabled) {
      setOptInFor(null)
      setPicking(model)
    } else {
      setPicking(null)
      setOptInFor(model)
    }
  }

  const confirmOptIn = async () => {
    if (!optInFor) return
    setEnabling(true)
    const ok = await enablePersonalization()
    setEnabling(false)
    if (ok) {
      setPicking(optInFor)
      setOptInFor(null)
    }
  }

  const toggleSorted = () => {
    const next = !sortedOpen
    setSortedOpen(next)
    writeSortedOpen(next)
    if (!next) {
      setPicking(null)
      setOptInFor(null)
    }
  }

  const pick = (model: PersonalizableModel, label: string) => {
    setPicking(null)
    void correctEmail(row.id, model, label)
  }

  const optionsFor = (model: PersonalizableModel): CorrectionOption[] => {
    if (model === 'priority') return PRIORITY_OPTIONS
    if (model === 'spam') return SPAM_OPTIONS
    return [
      ...BASE_CATEGORIES.map((c) => ({
        value: c,
        label: c,
        swatch: CATEGORY_SWATCH[c],
      })),
      ...(personalization?.custom_labels ?? []).map((c) => ({
        value: c,
        label: c,
        swatch: null,
        note: 'Your label',
      })),
    ]
  }

  const currentFor = (model: PersonalizableModel): string => {
    if (!row.result) return ''
    if (model === 'category') return row.result.category.label
    if (model === 'priority') return row.result.priority.bucket
    return row.result.spam.label
  }

  const correctionPanel = () => {
    if (optInFor) {
      return <OptInPrompt busy={enabling} onEnable={() => void confirmOptIn()} onCancel={() => setOptInFor(null)} />
    }
    if (picking && row.modelResult) {
      const predicted = predictedFor(picking, row.modelResult)
      return (
        <CorrectionPicker
          key={picking}
          heading={HEADINGS[picking]}
          context={`${MODEL_TITLES[picking]} · model said ${optionLabel(picking, predicted.label)}, ${Math.round(predicted.confidence * 100)}%`}
          options={optionsFor(picking)}
          current={currentFor(picking)}
          predicted={predicted.label}
          allowNew={picking === 'category'}
          onPick={(label) => pick(picking, label)}
          onCancel={() => setPicking(null)}
        />
      )
    }
    return null
  }

  return (
    <article className="@container flex min-h-0 flex-1 flex-col overflow-y-auto">
      <div className="mx-auto flex w-full max-w-3xl flex-col px-5 pt-4 pb-10 md:px-10 md:pt-5">
        <div className="flex flex-wrap items-center gap-2">
          {backButton}
          <ToolbarButton
            icon={archived ? <ArchiveRestore size={15} strokeWidth={1.75} /> : <Archive size={15} strokeWidth={1.75} />}
            label={archived ? 'Move to inbox' : 'Archive'}
            onClick={() => (archived ? unarchiveEmail(row.id) : archiveEmail(row.id))}
          />
          <ToolbarButton
            icon={row.unread ? <MailOpen size={15} strokeWidth={1.75} /> : <Mail size={15} strokeWidth={1.75} />}
            label={row.unread ? 'Mark as read' : 'Mark as unread'}
            onClick={() => toggleRead(row.id)}
          />
          {row.source === 'gmail' && (
            <ToolbarButton
              icon={<ExternalLink size={15} strokeWidth={1.75} />}
              label="Open in Gmail"
              href={`https://mail.google.com/mail/u/0/#all/${row.id}`}
            />
          )}
          <span className="ml-auto hidden font-mono text-xs @xl:block" style={{ color: MUTED }}>
            {row.receivedAt}
          </span>
        </div>

        <h2 className="mt-7 font-display text-[26px] leading-tight font-bold tracking-tight break-words md:text-3xl">
          {row.subject}
        </h2>
        <div className="mt-4 flex min-w-0 items-center gap-3">
          <Avatar name={row.fromName} size="md" />
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="truncate text-[15px] font-semibold">
              {row.fromName}{' '}
              <span className="font-normal" style={{ color: MUTED }}>
                &lt;{row.fromAddr}&gt;
              </span>
            </span>
            <span className="truncate text-[13px]" style={{ color: MUTED }}>
              To {row.to}
            </span>
          </div>
        </div>

        <section aria-labelledby="sorted-heading" className="mt-6 flex flex-col gap-2.5">
          <div className="flex items-center justify-between gap-3">
            <h3 id="sorted-heading" className="font-display text-[15px] font-semibold">
              <button
                type="button"
                onClick={toggleSorted}
                aria-expanded={sortedOpen}
                aria-controls="sorted-panels"
                className={`-mx-1 inline-flex items-center gap-1.5 rounded-md px-1 py-0.5 transition-colors hover:bg-[#F6F7F2] ${FOCUS_RING}`}
              >
                How this email was sorted
                <ChevronDown
                  size={16}
                  strokeWidth={2}
                  className={`transition-transform motion-reduce:transition-none ${sortedOpen ? '' : '-rotate-90'}`}
                  style={{ color: MUTED }}
                  aria-hidden
                />
              </button>
            </h3>
            <span className="flex items-center gap-1.5 text-xs" style={{ color: MUTED }}>
              <Cpu size={14} strokeWidth={1.75} aria-hidden />
              Sorted on this device
            </span>
          </div>

          {!sortedOpen && row.result && (
            <div className="flex flex-wrap gap-1.5">
              {row.result.spam.label === 'spam' && <SpamChip />}
              <CategoryChip label={row.result.category.label} />
              {row.result.spam.label !== 'spam' && <PriorityChip bucket={row.result.priority.bucket} />}
            </div>
          )}

          <div id="sorted-panels" className="flex flex-col gap-2.5" hidden={!sortedOpen}>
            {!row.result ? (
              <p
                className="rounded-xl border px-[18px] py-4 text-sm motion-safe:animate-pulse"
                style={{ borderColor: LINE, color: MUTED }}
              >
                Sorting this email…
              </p>
            ) : (
              <div
                className="grid grid-cols-1 gap-px overflow-hidden rounded-xl border @xl:grid-cols-3"
                style={{ borderColor: LINE, background: LINE }}
              >
                <VerdictCell
                  title="Category"
                  corrected={row.corrected.category !== undefined}
                  onCorrect={canCorrect ? () => startCorrecting('category') : undefined}
                  correcting={active === 'category'}
                  testId="correct-category"
                >
                  <CategoryVerdict confidences={row.result.category.confidences} winner={row.result.category.label} />
                </VerdictCell>
                <VerdictCell
                  title="Priority"
                  corrected={row.corrected.priority !== undefined}
                  onCorrect={canCorrect ? () => startCorrecting('priority') : undefined}
                  correcting={active === 'priority'}
                  testId="correct-priority"
                >
                  <PriorityVerdict
                    score={row.result.priority.score}
                    bucket={row.result.priority.bucket}
                    note={row.result.priority.note}
                  />
                </VerdictCell>
                <VerdictCell
                  title="Spam check"
                  corrected={row.corrected.spam !== undefined}
                  onCorrect={canCorrect ? () => startCorrecting('spam') : undefined}
                  correcting={active === 'spam'}
                  testId="correct-spam"
                >
                  <SpamVerdict label={row.result.spam.label} confidence={row.result.spam.confidence} />
                </VerdictCell>
              </div>
            )}

            {correctionPanel()}
          </div>

          {personalizationError && (
            <p className="text-[13px]" style={{ color: CORAL_TEXT }} role="alert">
              {personalizationError}
            </p>
          )}
          {!canCorrect && row.result && row.source === 'sample' && (
            <p className="text-xs" style={{ color: MUTED }}>
              Sample emails can&apos;t be corrected. Connect Gmail to correct your own mail.
            </p>
          )}
        </section>

        <div className="mt-6 overflow-hidden rounded-xl border p-4 md:p-6" style={{ borderColor: LINE, background: SURFACE }}>
          {row.bodyHtml ? (
            <>
              {isSpam && !imagesAllowed && (
                <div
                  className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border px-3.5 py-2.5 text-[13px]"
                  style={{ borderColor: LINE, background: PANEL_BG }}
                >
                  <ImageOff size={16} strokeWidth={1.75} style={{ color: MUTED }} aria-hidden />
                  <span className="flex-1">Images are hidden because this email looks like spam.</span>
                  <button
                    type="button"
                    onClick={() => setShowImagesFor(row.id)}
                    className={`h-8 rounded-[7px] border px-3 font-medium transition-colors hover:bg-[#F6F7F2] ${FOCUS_RING}`}
                    style={{ borderColor: '#CFD6D2', background: SURFACE, color: INK }}
                  >
                    Show images
                  </button>
                </div>
              )}
              <EmailHtmlBody key={row.id} html={row.bodyHtml} blockImages={!imagesAllowed} />
            </>
          ) : (
            <p className="max-w-[65ch] text-base leading-7 break-words whitespace-pre-wrap">
              {splitLinks(tidyPlainText(row.body)).map((part, i) =>
                part.type === 'link' ? (
                  <a
                    key={i}
                    href={part.href}
                    title={part.href}
                    target="_blank"
                    rel="noopener noreferrer nofollow"
                    className={`rounded-sm underline decoration-[#00789C]/40 underline-offset-[3px] hover:decoration-[#00789C] ${FOCUS_RING}`}
                    style={{ color: CERULEAN_TEXT }}
                  >
                    {part.label}
                  </a>
                ) : (
                  part.text
                ),
              )}
            </p>
          )}
        </div>
      </div>
    </article>
  )
}
