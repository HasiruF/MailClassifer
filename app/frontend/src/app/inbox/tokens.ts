// Flat, light inbox palette built on five brand colors: cerulean #0081A7,
// tropical teal #00AFB9, light yellow #FDFCDC, soft apricot #FED9B7 and
// vibrant coral #F07167. No shadows or gradients; structure comes from 1px
// lines and flat fills.
//
// Each color has one job so it keeps a stable meaning across the app:
// cerulean is actions + Work, teal is Personal + "on/connected", apricot is
// Other + selection + medium priority, coral is only ever "look at this"
// (high priority, spam, a rejected retrain). The *_TEXT variants are darker
// shades of the same hue, because the brand values themselves fall short of
// 4.5:1 contrast as text or under white text.
//
// Type (see app/layout.tsx): Schibsted Grotesk for headings (font-display),
// Instrument Sans for everything read (font-sans), IBM Plex Mono only for
// numbers — times, counts and model confidences (font-mono).

export const INK = '#12303A'
export const MUTED = '#56696F'
export const FAINT = '#8A979A'
export const LINE = '#E3E5DA'
export const LINE_SOFT = '#F0EFE4'
export const SURFACE = '#FFFFFF'
export const PANEL_BG = '#F7F7F0'
// The reading pane: a cool, cerulean-leaning tint, so the open email reads
// as its own surface next to the white list and the yellow sidebar. The
// message itself sits on a white card on top of it.
export const READING_BG = '#F1F6F7'

// Sidebar ground: the palette's light yellow, with lines tuned to sit on it.
export const SIDEBAR_BG = '#FDFCDC'
export const SIDEBAR_LINE = '#EBE7C3'
export const SIDEBAR_FIELD_LINE = '#E3DFB8'

export const CERULEAN = '#0081A7'
export const CERULEAN_TEXT = '#00789C'
export const CERULEAN_DEEP = '#005873'
export const CERULEAN_TINT = '#DFF0F3'

export const TEAL = '#00AFB9'
export const TEAL_TEXT = '#005F66'
export const TEAL_TINT = '#D6F1F2'

export const APRICOT = '#FED9B7'
export const APRICOT_SWATCH = '#F5B98A'
export const APRICOT_TEXT = '#8F4F1B'
export const APRICOT_TINT = '#FEEAD6'
export const SELECTED_BG = '#FFF1E3'

export const CORAL = '#F07167'
export const CORAL_TEXT = '#B3372E'
export const CORAL_TINT = '#FDE4E1'

export const BASE_CATEGORIES = ['Work', 'Personal', 'Other'] as const

// Category swatches. A user's own labels have no swatch color: they render
// as an outlined square, so the model's categories and yours stay distinct.
export const CATEGORY_SWATCH: Record<string, string> = {
  Work: CERULEAN,
  Personal: TEAL,
  Other: APRICOT_SWATCH,
}

export function isBaseCategory(label: string): boolean {
  return (BASE_CATEGORIES as readonly string[]).includes(label)
}

// Keyboard focus, shared by every interactive element in the inbox.
export const FOCUS_RING = 'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#00789C]'

// Sender avatars: [background, text] pairs, picked deterministically per
// sender (avatarColorsFor) so a person keeps the same color across sessions.
const AVATAR_COLORS: [string, string][] = [
  ['#D8EDF3', CERULEAN_DEEP],
  ['#D4F1F2', TEAL_TEXT],
  ['#FDE6D2', APRICOT_TEXT],
  [CORAL_TINT, CORAL_TEXT],
  ['#F1EFCC', '#5F5A1C'],
]

export function avatarColorsFor(name: string): [string, string] {
  let hash = 0
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) >>> 0
  }
  return AVATAR_COLORS[hash % AVATAR_COLORS.length]
}

export function initialsFor(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean)
  if (words.length === 0) return '?'
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase()
  return (words[0][0] + words[1][0]).toUpperCase()
}
