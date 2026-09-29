// Plain-text email bodies (what gmail-fetch.ts prefers) spell out every
// link in full, and marketing mail's links are mostly long tracking URLs.
// These helpers are display-only: they shorten what the reader sees, while
// classification keeps using the original text.

export type TextPart = { type: 'text'; text: string } | { type: 'link'; href: string; label: string }

const URL_RE = /https?:\/\/[^\s<>"'`]+/g
// Sentence punctuation that follows a URL in prose rather than being part of it.
const TRAILING_PUNCTUATION = /[.,;:!?)\]}'"]+$/
const MAX_LABEL = 36

export function shortLink(href: string): string {
  let label: string
  let droppedSomething = false
  try {
    const url = new URL(href)
    label = url.hostname.replace(/^www\./, '') + url.pathname.replace(/\/+$/, '')
    droppedSomething = url.search !== '' || url.hash !== ''
  } catch {
    label = href
  }
  if (label.length > MAX_LABEL) return `${label.slice(0, MAX_LABEL - 1)}…`
  return droppedSomething ? `${label}…` : label
}

export function splitLinks(text: string): TextPart[] {
  const parts: TextPart[] = []
  let last = 0
  for (const match of text.matchAll(URL_RE)) {
    const start = match.index
    const href = match[0].replace(TRAILING_PUNCTUATION, '')
    if (start > last) parts.push({ type: 'text', text: text.slice(last, start) })
    parts.push({ type: 'link', href, label: shortLink(href) })
    last = start + href.length
  }
  if (last < text.length) parts.push({ type: 'text', text: text.slice(last) })
  return parts
}

export function withShortLinks(text: string): string {
  return splitLinks(text)
    .map((part) => (part.type === 'link' ? part.label : part.text))
    .join('')
}

// Plain-text mail often pads sections with long runs of blank lines. For
// display only: line endings are normalized (Gmail's plain text uses CRLF),
// trailing spaces go, and 2+ blank lines become one.
export function tidyPlainText(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+$/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}
