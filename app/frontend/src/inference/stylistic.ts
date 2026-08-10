// Client-side port of scripts/shared/stylistic_features.py — must stay
// numerically identical (same regexes, same clamping, same slice windows)
// since the category/priority models were fit on this exact feature set.

const INFORMAL_GREETING = /\b(hey|hi\s+\w+|hiya|yo|sup)\b/i
const FORMAL_GREETING = /\b(dear\s+\w+|to\s+whom\s+it\s+may\s+concern|greetings)\b/i
const FAMILY_WORDS = /\b(mom|dad|mother|father|honey|sweetie|love\s+you|xoxo|dearest)\b/i
const FORMAL_CLOSING = /\b(regards|sincerely|best\s+regards|respectfully)\b/i
const INFORMAL_CLOSING = /\b(cheers|talk\s+soon|luv|later|ttyl)\b/i
const CAPS_WORD = /\b[A-Z]{3,}\b/g

export const STYLE_COLS = [
  'text_len_log',
  'excl_density',
  'caps_word_count',
  'informal_greeting',
  'formal_greeting',
  'family_words',
  'formal_closing',
  'informal_closing',
  'question_count',
] as const

export interface StylisticFeatures {
  text_len_log: number
  excl_density: number
  caps_word_count: number
  informal_greeting: number
  formal_greeting: number
  family_words: number
  formal_closing: number
  informal_closing: number
  question_count: number
}

export function extractStylisticFeatures(subject: string, body: string): StylisticFeatures {
  subject = subject || ''
  body = body || ''
  const full = `${subject} ${body}`
  const textLen = full.length
  const exclDensity = ((full.match(/!/g)?.length ?? 0) / Math.max(textLen, 1)) * 1000
  const capsWords = full.match(CAPS_WORD)?.length ?? 0
  const bodyHead = body.slice(0, 100)
  const bodyTail = body.slice(-200)
  const questionCount = full.match(/\?/g)?.length ?? 0

  return {
    text_len_log: Math.log1p(textLen),
    excl_density: exclDensity,
    caps_word_count: Math.min(capsWords, 10),
    informal_greeting: INFORMAL_GREETING.test(bodyHead) ? 1 : 0,
    formal_greeting: FORMAL_GREETING.test(bodyHead) ? 1 : 0,
    family_words: FAMILY_WORDS.test(full) ? 1 : 0,
    formal_closing: FORMAL_CLOSING.test(bodyTail) ? 1 : 0,
    informal_closing: INFORMAL_CLOSING.test(bodyTail) ? 1 : 0,
    question_count: Math.min(questionCount, 5),
  }
}
