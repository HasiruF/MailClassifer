// "Signal Console" shared tokens. Cool navy shell (nav/list/chrome) vs a
// warm dark-wood interior for an opened message — closed mail is cool,
// reading one warms up, like unsealing a letter. Color is functional: sky
// blue means "medium," burnt orange means "high," nothing else gets a
// signal color. Type follows the same split (see layout.tsx): IBM Plex
// Mono for all console chrome (nav, badges, metadata), Newsreader serif
// only inside an opened message's subject/body — the one place text is
// read rather than scanned.
export const INK = '#F5E9D8'
export const INK_DIM = '#D9CBB0'
export const MUTED = '#8B93A6'
export const FAINT = '#565C70'
export const BORDER = '#242B3D'
export const BG = '#0D111C'
export const SIDEBAR_BG = '#0A0D16'
export const ROW_HOVER = '#161B29'
export const MEDIUM = '#2FA4D7'
export const HIGH = '#E76F2E'
export const SPAM = MUTED
export const ERROR = '#E76F2E'

// The opened-message page: warm dark wood instead of cool navy.
export const DETAIL_BG = '#3E2C23'
export const DETAIL_BORDER = '#5A4534'
export const DETAIL_MUTED = '#B8A688'
export const DETAIL_FAINT = '#7A6A50'

// Low-priority rows recede — the model already said "don't worry about
// this one," so the row itself goes gray instead of just its label.
export const LOW_OPACITY = 0.5

// Category pills — a distinct hue family from priority's sky-blue/orange so
// the two signals never visually collide (a Work-category chip must never
// read like a medium-priority one). Kept as desaturated as MEDIUM/HIGH to
// stay in the same console register rather than turning bright/app-like.
export const CATEGORY_COLOR: Record<'Work' | 'Personal' | 'Other', string> = {
  Work: '#6B8CAE', // steel blue
  Personal: '#7FA66B', // sage
  Other: '#9B7EBD', // plum
}

// Sender-avatar badge colors — deterministic per sender (see avatarColorFor
// below) so the same person always gets the same badge across sessions,
// the way a real client's contact colors would.
const AVATAR_HUES = ['#6B8CAE', '#7FA66B', '#9B7EBD', '#C08552', '#4F8F8B', '#B08BC9']

export function avatarColorFor(name: string): string {
  let hash = 0
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) >>> 0
  }
  return AVATAR_HUES[hash % AVATAR_HUES.length]
}

export function initialsFor(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean)
  if (words.length === 0) return '?'
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase()
  return (words[0][0] + words[1][0]).toUpperCase()
}
