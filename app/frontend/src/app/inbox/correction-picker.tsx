'use client'

import { useState } from 'react'
import {
  APRICOT_TEXT,
  CERULEAN_TEXT,
  FOCUS_RING,
  INK,
  LINE,
  MUTED,
  SURFACE,
  TEAL_TEXT,
} from './tokens'

export interface CorrectionOption {
  value: string
  label: string
  // Category swatch color; null draws the outlined square used for the
  // user's own labels; undefined draws none (priority, spam).
  swatch?: string | null
  note?: string
}

function OptionSwatch({ swatch }: { swatch: string | null | undefined }) {
  if (swatch === undefined) return null
  if (swatch === null) return <span className="size-2.5 rounded-[3px] border-2" style={{ borderColor: INK }} aria-hidden />
  return <span className="size-2.5 rounded-[3px]" style={{ background: swatch }} aria-hidden />
}

// Pick-then-save, rather than saving on the first click: a correction
// retrains the user's model, so it should be a deliberate choice.
export function CorrectionPicker({
  heading,
  context,
  options,
  current,
  predicted,
  allowNew,
  onPick,
  onCancel,
}: {
  heading: string
  context: string
  options: CorrectionOption[]
  current: string
  predicted: string
  allowNew: boolean
  onPick: (value: string) => void
  onCancel: () => void
}) {
  const [choice, setChoice] = useState(current)
  const [added, setAdded] = useState<CorrectionOption[]>([])
  const [draft, setDraft] = useState('')
  const trimmed = draft.trim()
  const all = [...options, ...added.filter((a) => !options.some((o) => o.value === a.value))]

  function addLabel() {
    if (!trimmed) return
    if (!all.some((o) => o.value === trimmed)) {
      setAdded((prev) => [...prev, { value: trimmed, label: trimmed, swatch: null, note: 'New label' }])
    }
    setChoice(trimmed)
    setDraft('')
  }

  return (
    <form
      className="flex flex-col gap-4 rounded-xl border p-5"
      style={{ borderColor: LINE, background: SURFACE }}
      data-testid="correction-picker"
      onSubmit={(e) => {
        e.preventDefault()
        if (choice !== current) onPick(choice)
      }}
    >
      <div className="flex flex-col gap-1">
        <span className="text-xs font-semibold" style={{ color: MUTED }}>
          {context}
        </span>
        <h4 className="font-display text-lg font-bold">{heading}</h4>
      </div>

      <fieldset className="flex flex-col gap-1.5">
        <legend className="sr-only">{heading}</legend>
        {all.map((option) => {
          const checked = choice === option.value
          const note = option.value === predicted ? "Model's pick" : option.note
          return (
            <label
              key={option.value}
              data-testid={`correction-option-${option.value}`}
              className={`flex h-11 cursor-pointer items-center gap-2.5 rounded-lg px-3 text-sm transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-[#00789C] ${
                checked ? 'font-semibold' : 'font-medium hover:bg-[#F8F8F2]'
              }`}
              style={{
                border: checked ? `2px solid ${CERULEAN_TEXT}` : `1px solid ${LINE}`,
                background: checked ? '#EEF7F9' : undefined,
              }}
            >
              <input
                type="radio"
                name="correction"
                value={option.value}
                checked={checked}
                onChange={() => setChoice(option.value)}
                className="size-4 accent-[#00789C] outline-none"
              />
              <OptionSwatch swatch={option.swatch} />
              <span className="flex-1">{option.label}</span>
              {note && (
                <span className="text-xs font-normal" style={{ color: MUTED }}>
                  {note}
                </span>
              )}
            </label>
          )
        })}
      </fieldset>

      {allowNew && (
        <div className="flex flex-col gap-1.5">
          <label htmlFor="correction-new-label" className="text-[13px] font-medium">
            Or create a label
          </label>
          <div className="flex gap-2">
            <input
              id="correction-new-label"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  addLabel()
                }
              }}
              maxLength={40}
              placeholder="e.g. Travel"
              className={`h-10 min-w-0 flex-1 rounded-lg border px-3 text-sm placeholder:text-[#56696F] ${FOCUS_RING}`}
              style={{ borderColor: '#CFD6D2', color: INK }}
            />
            <button
              type="button"
              onClick={addLabel}
              disabled={!trimmed}
              className={`h-10 rounded-lg border px-3.5 text-sm font-medium transition-colors hover:bg-[#F6F7F2] disabled:cursor-not-allowed disabled:opacity-50 ${FOCUS_RING}`}
              style={{ borderColor: '#CFD6D2', color: INK }}
            >
              Add
            </button>
          </div>
        </div>
      )}

      <div className="flex items-center gap-2.5 pt-1">
        <button
          type="submit"
          disabled={choice === current}
          data-testid="correction-save"
          className={`h-10 rounded-lg px-4 text-sm font-semibold text-white transition-colors hover:bg-[#006A8A] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-[#00789C] ${FOCUS_RING}`}
          style={{ background: CERULEAN_TEXT }}
        >
          Save correction
        </button>
        <button
          type="button"
          onClick={onCancel}
          className={`h-10 rounded-lg px-3 text-sm hover:underline ${FOCUS_RING}`}
          style={{ color: MUTED }}
        >
          Cancel
        </button>
      </div>
    </form>
  )
}

// Says plainly what leaves the browser (spec §10): a bag of vocabulary
// words and a few numbers, never the text or its word order.
export function OptInPrompt({ busy, onEnable, onCancel }: { busy: boolean; onEnable: () => void; onCancel: () => void }) {
  return (
    <section
      aria-labelledby="opt-in-heading"
      className="flex flex-col gap-3.5 rounded-xl border p-5"
      style={{ borderColor: LINE, background: SURFACE }}
      data-testid="opt-in-prompt"
    >
      <h4 id="opt-in-heading" className="font-display text-lg font-bold">
        Turn on personalization?
      </h4>
      <p className="text-sm leading-relaxed">
        Your corrections train a copy of the models that only you use. Turning this off later deletes everything it
        stored.
      </p>
      <div className="grid grid-cols-1 gap-px overflow-hidden rounded-[10px] border sm:grid-cols-2" style={{ borderColor: LINE, background: LINE }}>
        <div className="flex flex-col gap-2 px-3.5 py-3" style={{ background: '#FFF7EF' }}>
          <span className="text-xs font-semibold" style={{ color: APRICOT_TEXT }}>
            Sent to the server
          </span>
          <span className="text-[13px] leading-snug">Which words from a fixed vocabulary appear, and how strongly</span>
          <span className="text-[13px] leading-snug">A few numbers, like the recipient count</span>
        </div>
        <div className="flex flex-col gap-2 px-3.5 py-3" style={{ background: '#F1FAFB' }}>
          <span className="text-xs font-semibold" style={{ color: TEAL_TEXT }}>
            Stays on this device
          </span>
          <span className="text-[13px] leading-snug">The email&apos;s text</span>
          <span className="text-[13px] leading-snug">Word order</span>
        </div>
      </div>
      <div className="flex items-center gap-2.5">
        <button
          type="button"
          onClick={onEnable}
          disabled={busy}
          data-testid="opt-in-enable"
          className={`h-10 rounded-lg px-4 text-sm font-semibold text-white transition-colors hover:bg-[#006A8A] disabled:cursor-wait disabled:opacity-60 ${FOCUS_RING}`}
          style={{ background: CERULEAN_TEXT }}
        >
          {busy ? 'Turning on…' : 'Turn on and correct'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className={`h-10 rounded-lg px-3 text-sm hover:underline ${FOCUS_RING}`}
          style={{ color: MUTED }}
        >
          Not now
        </button>
      </div>
    </section>
  )
}
