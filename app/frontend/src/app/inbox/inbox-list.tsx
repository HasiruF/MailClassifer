'use client'

import { useState } from 'react'
import { Archive, ArchiveRestore, Mail, MailOpen, Menu } from 'lucide-react'
import type { PriorityBucket } from '@/types'
import { withShortLinks } from '@/lib/links'
import { useInbox, type Row } from './inbox-context'
import { Avatar } from './avatar'
import {
  APRICOT_TEXT,
  APRICOT_TINT,
  BASE_CATEGORIES,
  CATEGORY_SWATCH,
  CERULEAN,
  CERULEAN_TEXT,
  CORAL,
  CORAL_TEXT,
  CORAL_TINT,
  FOCUS_RING,
  INK,
  LINE,
  LINE_SOFT,
  MUTED,
  SELECTED_BG,
  SIDEBAR_BG,
  SIDEBAR_FIELD_LINE,
  SIDEBAR_LINE,
  SURFACE,
} from './tokens'

// sm: the compact tags on list rows; md: everywhere else.
const CHIP_SIZE = {
  sm: 'inline-flex h-5 items-center gap-1 rounded px-1.5 text-[11px]',
  md: 'inline-flex h-[22px] items-center gap-1.5 rounded-md px-2 text-xs',
}
type ChipSize = keyof typeof CHIP_SIZE

export function CategoryChip({ label, size = 'md' }: { label: string; size?: ChipSize }) {
  const swatch = CATEGORY_SWATCH[label]
  return (
    <span className={`${CHIP_SIZE[size]} shrink-0 border font-medium`} style={{ borderColor: LINE, background: SURFACE, color: INK }}>
      {swatch ? (
        <span className="size-2 rounded-[2px]" style={{ background: swatch }} aria-hidden />
      ) : (
        <span className="size-2 rounded-[2px] border-[1.5px]" style={{ borderColor: INK }} aria-hidden />
      )}
      {label}
    </span>
  )
}

// Low priority gets no tag: most mail is low, and tagging it would bury the
// two that matter.
export function PriorityChip({ bucket, size = 'md' }: { bucket: PriorityBucket; size?: ChipSize }) {
  if (bucket === 'high') {
    return (
      <span className={`${CHIP_SIZE[size]} shrink-0 font-semibold`} style={{ background: CORAL_TINT, color: CORAL_TEXT }}>
        {size === 'sm' ? 'High' : 'High priority'}
      </span>
    )
  }
  if (bucket === 'medium') {
    return (
      <span className={`${CHIP_SIZE[size]} shrink-0 font-semibold`} style={{ background: APRICOT_TINT, color: APRICOT_TEXT }}>
        Medium
      </span>
    )
  }
  return null
}

export function SpamChip({ size = 'md' }: { size?: ChipSize }) {
  return (
    <span className={`${CHIP_SIZE[size]} shrink-0 border font-semibold`} style={{ borderColor: CORAL, background: SURFACE, color: CORAL_TEXT }}>
      Spam
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
  const { selectedId, selectEmail, showArchived, toggleRead, archiveEmail, unarchiveEmail } = useInbox()
  const selected = row.id === selectedId
  const isSpam = row.result?.spam.label === 'spam'
  // Spam recedes: the model already set it aside, so it shouldn't compete
  // with real mail for attention.
  const textColor = isSpam ? MUTED : INK

  return (
    <li
      data-testid="email-row"
      data-email-id={row.id}
      data-category={row.result?.category.label ?? ''}
      className="group relative border-b"
      style={{ borderColor: LINE_SOFT }}
    >
      <button
        type="button"
        onClick={() => selectEmail(row.id)}
        aria-current={selected ? 'true' : undefined}
        className={`flex w-full items-start gap-3 px-4 py-2.5 text-left transition-colors md:px-5 ${FOCUS_RING} focus-visible:-outline-offset-2 ${
          selected ? '' : 'hover:bg-[#F8F8F2]'
        }`}
        style={{ background: selected ? SELECTED_BG : undefined, color: textColor }}
      >
        <span className="mt-0.5">
          <Avatar name={row.fromName} size="xs" />
        </span>
        {/* Two lines, Gmail-style: who + tags + when, then subject — preview. */}
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="flex min-w-0 items-center gap-1.5">
            {row.unread && <span className="size-[7px] shrink-0 rounded-full" style={{ background: CERULEAN }} aria-label="Unread" />}
            <span className={`min-w-0 flex-1 truncate text-sm ${row.unread ? 'font-semibold' : 'font-medium'}`}>{row.fromName}</span>
            {row.result ? (
              <>
                {isSpam && <SpamChip size="sm" />}
                <CategoryChip label={row.result.category.label} size="sm" />
                {!isSpam && <PriorityChip bucket={row.result.priority.bucket} size="sm" />}
              </>
            ) : (
              <span className={`${CHIP_SIZE.sm} shrink-0 border motion-safe:animate-pulse`} style={{ borderColor: LINE, color: MUTED }}>
                Sorting…
              </span>
            )}
            <span className="ml-1 shrink-0 font-mono text-[11px]" style={{ color: MUTED }}>
              {row.receivedAt}
            </span>
          </span>
          <span className="truncate text-[13px]">
            <span className={row.unread ? 'font-semibold' : ''}>{row.subject}</span>
            <span style={{ color: MUTED }}> — {withShortLinks(row.body)}</span>
          </span>
        </span>
      </button>

      <div
        className="absolute top-1/2 right-3 hidden -translate-y-1/2 items-center gap-0.5 rounded-md border p-0.5 group-focus-within:flex group-hover:flex"
        style={{ borderColor: LINE, background: SURFACE }}
      >
        <button
          type="button"
          onClick={() => toggleRead(row.id)}
          className={`flex size-7 items-center justify-center rounded hover:bg-[#F1F3EC] ${FOCUS_RING}`}
          style={{ color: MUTED }}
          title={row.unread ? 'Mark as read' : 'Mark as unread'}
          aria-label={row.unread ? 'Mark as read' : 'Mark as unread'}
        >
          {row.unread ? <MailOpen size={15} strokeWidth={1.75} /> : <Mail size={15} strokeWidth={1.75} />}
        </button>
        <button
          type="button"
          onClick={() => (showArchived ? unarchiveEmail(row.id) : archiveEmail(row.id))}
          className={`flex size-7 items-center justify-center rounded hover:bg-[#F1F3EC] ${FOCUS_RING}`}
          style={{ color: MUTED }}
          title={showArchived ? 'Move to inbox' : 'Archive'}
          aria-label={showArchived ? 'Move to inbox' : 'Archive'}
        >
          {showArchived ? <ArchiveRestore size={15} strokeWidth={1.75} /> : <Archive size={15} strokeWidth={1.75} />}
        </button>
      </div>
    </li>
  )
}

function DateSection({ label, rows }: { label: string; rows: Row[] }) {
  if (rows.length === 0) return null
  return (
    <>
      <li
        className="sticky top-0 z-10 px-4 pt-3 pb-1 text-xs font-semibold md:px-5"
        style={{ color: MUTED, background: SURFACE }}
      >
        {label}
      </li>
      {rows.map((row) => (
        <EmailRow key={row.id} row={row} />
      ))}
    </>
  )
}

// Phone-width replacement for the sidebar's category list: the sidebar is a
// slide-over menu there, so the everyday filters stay one tap away.
function FilterChips() {
  const { filter, setFilter, onlyHigh, setOnlyHigh, showArchived, setShowArchived, personalization } = useInbox()
  const options = ['All', ...BASE_CATEGORIES, ...(personalization?.custom_labels ?? [])]
  return (
    <nav
      aria-label="Filters"
      className="flex gap-2 overflow-x-auto px-4 pt-1 pb-3 md:hidden"
      style={{ background: SIDEBAR_BG, borderBottom: `1px solid ${SIDEBAR_LINE}` }}
    >
      {options.map((option) => {
        const active = !showArchived && !onlyHigh && filter === option
        return (
          <button
            key={option}
            type="button"
            aria-pressed={active}
            onClick={() => {
              setFilter(option)
              setOnlyHigh(false)
              setShowArchived(false)
            }}
            className={`flex h-9 shrink-0 items-center gap-1.5 rounded-full border px-3.5 text-sm ${FOCUS_RING} ${
              active ? 'font-semibold text-white' : 'font-medium'
            }`}
            style={
              active
                ? { background: CERULEAN_TEXT, borderColor: CERULEAN_TEXT }
                : { background: SURFACE, borderColor: SIDEBAR_FIELD_LINE, color: INK }
            }
          >
            {option !== 'All' &&
              (CATEGORY_SWATCH[option] ? (
                <span className="size-2 rounded-[2px]" style={{ background: CATEGORY_SWATCH[option] }} aria-hidden />
              ) : (
                <span className="size-2 rounded-[2px] border-[1.5px]" style={{ borderColor: active ? '#FFFFFF' : INK }} aria-hidden />
              ))}
            {option}
          </button>
        )
      })}
    </nav>
  )
}

export function InboxList() {
  const { rows, status, error, filter, onlyHigh, showArchived, query, archivedIds, setNavOpen } = useInbox()
  const [unreadOnly, setUnreadOnly] = useState(false)

  const scoped = rows.filter((r) => archivedIds.has(r.id) === showArchived)
  const visibleRows = scoped
    .filter((r) => showArchived || filter === 'All' || r.result?.category.label === filter)
    .filter((r) => showArchived || !onlyHigh || r.result?.priority.bucket === 'high')
    .filter((r) => matchesQuery(r, query))
    .sort((a, b) => b.receivedAtMs - a.receivedAtMs)
  const shownRows = unreadOnly ? visibleRows.filter((r) => r.unread) : visibleRows

  const unreadCount = visibleRows.filter((r) => r.unread).length
  const today = shownRows.filter((r) => bucketFor(r.receivedAtMs) === 'Today')
  const yesterday = shownRows.filter((r) => bucketFor(r.receivedAtMs) === 'Yesterday')
  const earlier = shownRows.filter((r) => bucketFor(r.receivedAtMs) === 'Earlier')

  const title = showArchived ? 'Archived' : onlyHigh ? 'High priority' : filter === 'All' ? 'All mail' : filter
  const trimmedQuery = query.trim()
  const emptyMessage = trimmedQuery
    ? `No mail matches “${trimmedQuery}”.`
    : unreadOnly
      ? 'No unread mail here.'
      : `Nothing in ${title}.`

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header
        className="flex items-center gap-1 bg-[#FDFCDC] px-2 pt-2.5 pb-1.5 md:items-end md:bg-transparent md:gap-3 md:border-b md:px-5 md:pt-[22px] md:pb-3.5"
        style={{ borderColor: LINE }}
      >
        <button
          type="button"
          onClick={() => setNavOpen(true)}
          aria-label="Open mailboxes"
          className={`flex size-11 shrink-0 items-center justify-center rounded-md md:hidden ${FOCUS_RING}`}
        >
          <Menu size={20} strokeWidth={1.75} aria-hidden />
        </button>
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <h1 className="truncate font-display text-xl font-bold tracking-tight md:text-[22px]">{title}</h1>
          <span className="text-xs md:text-[13px]" style={{ color: MUTED }}>
            {status === 'loading' && rows.every((r) => !r.result) ? 'Sorting your mail…' : `${unreadCount} unread of ${visibleRows.length}`}
          </span>
        </div>
        <button
          type="button"
          aria-pressed={unreadOnly}
          onClick={() => setUnreadOnly((v) => !v)}
          className={`mr-2 h-8 shrink-0 rounded-[7px] border px-3 text-[13px] font-medium transition-colors md:mr-0 ${FOCUS_RING} ${
            unreadOnly ? '' : 'hover:bg-[#F6F7F2]'
          }`}
          style={
            unreadOnly
              ? { background: '#DFF0F3', borderColor: CERULEAN_TEXT, color: '#005873' }
              : { background: SURFACE, borderColor: '#D9DDD3', color: INK }
          }
        >
          Unread only
        </button>
      </header>
      <FilterChips />

      {status === 'error' && (
        <p className="mx-4 mt-3 rounded-lg px-3 py-2.5 text-[13px] leading-relaxed md:mx-5" style={{ background: CORAL_TINT, color: CORAL_TEXT }} role="alert">
          Models failed to load: {error ?? 'unknown error'}. Check that public/models has the .onnx and .vocab.json
          files (scripts/export_onnx.py writes them).
        </p>
      )}

      <ul className="min-h-0 flex-1 overflow-y-auto">
        <DateSection label="Today" rows={today} />
        <DateSection label="Yesterday" rows={yesterday} />
        <DateSection label="Earlier" rows={earlier} />
        {shownRows.length === 0 && (
          <li className="px-5 py-12 text-center text-sm" style={{ color: MUTED }}>
            {emptyMessage}
          </li>
        )}
      </ul>
    </div>
  )
}
