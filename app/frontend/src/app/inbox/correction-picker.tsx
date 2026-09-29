'use client'

import { useState } from 'react'
import { DETAIL_BORDER, DETAIL_FAINT, DETAIL_MUTED, INK } from './tokens'

export function CorrectionPicker({
  options,
  allowNew,
  current,
  onPick,
  onCancel,
}: {
  options: string[]
  allowNew: boolean
  current: string
  onPick: (label: string) => void
  onCancel: () => void
}) {
  const [draft, setDraft] = useState('')
  const trimmed = draft.trim()

  return (
    <div
      className="mt-2 flex flex-col gap-2 rounded-sm p-2"
      style={{ border: `1px solid ${DETAIL_BORDER}` }}
      data-testid="correction-picker"
    >
      <div className="flex flex-wrap gap-1.5">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onPick(option)}
            data-testid={`correction-option-${option}`}
            className="rounded-sm px-2 py-1 font-mono text-[11px] tracking-wide uppercase"
            style={{
              color: option === current ? INK : DETAIL_MUTED,
              border: `1px solid ${option === current ? DETAIL_MUTED : DETAIL_BORDER}`,
            }}
          >
            {option}
          </button>
        ))}
      </div>
      {allowNew && (
        <form
          className="flex gap-1.5"
          onSubmit={(e) => {
            e.preventDefault()
            if (trimmed) onPick(trimmed)
          }}
        >
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            maxLength={40}
            placeholder="new label…"
            aria-label="New label"
            className="min-w-0 flex-1 bg-transparent px-2 py-1 font-mono text-[11px] outline-none"
            style={{ color: INK, border: `1px solid ${DETAIL_BORDER}` }}
          />
          <button
            type="submit"
            disabled={!trimmed}
            className="px-2 font-mono text-[11px] uppercase disabled:opacity-40"
            style={{ color: DETAIL_MUTED }}
          >
            add
          </button>
        </form>
      )}
      <button
        type="button"
        onClick={onCancel}
        className="self-start font-mono text-[10px] tracking-wide uppercase"
        style={{ color: DETAIL_FAINT }}
      >
        cancel
      </button>
    </div>
  )
}

// Wording follows the spec's §10: say plainly what leaves the browser.
export function OptInPrompt({ busy, onEnable, onCancel }: { busy: boolean; onEnable: () => void; onCancel: () => void }) {
  return (
    <div
      className="mt-2 flex flex-col gap-2 rounded-sm p-3"
      style={{ border: `1px solid ${DETAIL_BORDER}` }}
      data-testid="opt-in-prompt"
    >
      <p className="font-serif text-sm leading-relaxed" style={{ color: INK }}>
        Corrections train a copy of the model that only you use. To do that, this app sends its server a word-level
        summary of each email you correct: which words from its fixed vocabulary appear and how strongly, plus a few
        numbers like the recipient count. It does not send the email text or word order. You can turn this off at any
        time, which deletes everything it stored.
      </p>
      <div className="flex gap-3">
        <button
          type="button"
          onClick={onEnable}
          disabled={busy}
          data-testid="opt-in-enable"
          className="font-mono text-[11px] tracking-wide uppercase disabled:opacity-40"
          style={{ color: INK }}
        >
          {busy ? 'turning on…' : 'turn on personalization'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="font-mono text-[11px] tracking-wide uppercase"
          style={{ color: DETAIL_FAINT }}
        >
          not now
        </button>
      </div>
    </div>
  )
}
