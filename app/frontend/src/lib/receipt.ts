import type { CategoryVocab, Vocabs } from '../inference/features'
import type { VocabData } from '../inference/tfidf'
import type { PersonalizableModel } from '../types'
import { toDense, type SparseVector } from './sparse'

// Turns a stored correction vector back into what a person can read: the
// words and letter groups it counted, and the signals it measured. The
// layouts mirror features.ts's build*Vector() concatenation order:
//   spam:     word
//   category: word | char_wb | numeric
//   priority: word | char_wb | category one-hot | numeric

export interface WeightedTerm {
  term: string
  weight: number
}

export interface Signal {
  label: string
  value: string
}

export interface Receipt {
  words: WeightedTerm[]
  pieces: WeightedTerm[]
  signals: Signal[]
}

const termLists = new WeakMap<VocabData, string[]>()

function termsOf(vocab: VocabData): string[] {
  let terms = termLists.get(vocab)
  if (!terms) {
    terms = []
    for (const [term, index] of Object.entries(vocab.vocabulary)) terms[index] = term
    termLists.set(vocab, terms)
  }
  return terms
}

function weightedTerms(vector: SparseVector, vocab: VocabData, offset: number): WeightedTerm[] {
  const terms = termsOf(vocab)
  const size = vocab.idf.length
  const out: WeightedTerm[] = []
  vector.indices.forEach((index, k) => {
    const local = index - offset
    if (local >= 0 && local < size) out.push({ term: terms[local], weight: vector.values[k] })
  })
  return out.sort((a, b) => b.weight - a.weight)
}

const yesNo = (raw: number) => (raw >= 0.5 ? 'yes' : 'no')
const count = (raw: number) => String(Math.max(0, Math.round(raw)))

// Plain-language names for the numeric columns (header_features.py,
// stylistic_features.py, priority_keyword_features.py), and how to show a
// raw value. n_recipients and text_len_log are stored as log1p.
const SIGNALS: Record<string, { label: string; show: (raw: number) => string }> = {
  n_recipients: { label: 'Recipients', show: (raw) => count(Math.expm1(raw)) },
  is_reply_or_forward: { label: 'Reply or forward', show: yesNo },
  sender_automated: { label: 'Sent from an automated address', show: yesNo },
  has_list_unsubscribe: { label: 'Has an unsubscribe link', show: yesNo },
  has_precedence_bulk: { label: 'Marked as bulk mail', show: yesNo },
  text_len_log: { label: 'Length', show: (raw) => `${count(Math.expm1(raw))} characters` },
  excl_density: { label: 'Exclamation marks', show: (raw) => `${Math.max(0, raw).toFixed(1)} per 1,000 characters` },
  caps_word_count: { label: 'Words in capitals', show: count },
  informal_greeting: { label: 'Informal greeting', show: yesNo },
  formal_greeting: { label: 'Formal greeting', show: yesNo },
  family_words: { label: 'Family words', show: yesNo },
  formal_closing: { label: 'Formal sign-off', show: yesNo },
  informal_closing: { label: 'Informal sign-off', show: yesNo },
  question_count: { label: 'Question marks', show: count },
  spam_conf: { label: 'Spam score', show: (raw) => `${Math.round(Math.min(1, Math.max(0, raw)) * 100)}%` },
  vader_compound: { label: 'Tone (−1 negative, 1 positive)', show: (raw) => raw.toFixed(2) },
  kw_high: { label: 'Urgent words', show: count },
  kw_med: { label: 'Request words', show: count },
  kw_low: { label: 'For-your-information words', show: count },
  excl_subj: { label: 'Exclamation marks in subject', show: count },
}

// A column absent from the sparse vector was stored as 0, which unscales to
// the column's mean.
function numericSignals(dense: Float32Array, start: number, vocab: CategoryVocab): Signal[] {
  return vocab.numeric_cols.map((col, i) => {
    const raw = dense[start + i] * vocab.numeric_scale[i] + vocab.numeric_mean[i]
    const signal = SIGNALS[col]
    return signal ? { label: signal.label, value: signal.show(raw) } : { label: col, value: raw.toFixed(2) }
  })
}

export function decodeCorrection(model: PersonalizableModel, vector: SparseVector, vocabs: Vocabs): Receipt {
  if (model === 'spam') {
    return { words: weightedTerms(vector, vocabs.spam.word, 0), pieces: [], signals: [] }
  }
  const vocab = model === 'category' ? vocabs.category : vocabs.priority
  const wordSize = vocab.word.idf.length
  const charSize = vocab.char.idf.length
  const dense = toDense(vector)
  const signals: Signal[] = []
  let numericStart = wordSize + charSize
  if (model === 'priority') {
    const categories = vocabs.priority.categories
    const given = categories.findIndex((_, i) => dense[numericStart + i] >= 0.5)
    if (given >= 0) signals.push({ label: 'Category it was given', value: categories[given] })
    numericStart += categories.length
  }
  signals.push(...numericSignals(dense, numericStart, vocab))
  return {
    words: weightedTerms(vector, vocab.word, 0),
    pieces: weightedTerms(vector, vocab.char, wordSize),
    signals,
  }
}
