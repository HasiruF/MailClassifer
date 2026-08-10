import type { EmailInput } from '../types'
import { wordTfidfVector, charWbTfidfVector, type VocabData } from './tfidf'
import { extractStylisticFeatures } from './stylistic'

// Client-side re-implementation of scripts/shared/header_features.py and
// priority_keyword_features.py. Must stay numerically identical to the
// Python versions — the exported ONNX models were fit on features computed
// by those exact functions, so drift here silently produces wrong
// predictions with no error.

const AUTOMATED_LOCAL_PART =
  /^(no[-_]?reply|do[-_]?not[-_]?reply|mailer[-_]?daemon|support|notifications?|alerts?|updates?|digest|newsletter|bounces?|postmaster)/i
const REPLY_FWD_PREFIX = /^\s*(re|fw|fwd)\s*:/i

const HIGH_KW =
  /\b(urgent|asap|emergency|critical|escalat\w*|breach|outage|deadline|overdue|end of day|eod|cob|due by|expires?|right away|time sensitive|crisis|disaster|lawsuit|legal action|margin call|default|liquidat\w*)\b/gi
const MED_KW =
  /\b(action required|please respond|please advise|need your (approval|input|sign[\s-]?off)|response needed|follow up|reminder)\b/gi
const LOW_KW =
  /\b(unsubscribe|newsletter|fyi|for your information|no action needed|digest|weekly update|monthly report|do not reply|automatic notification|out of office|calendar invite|meeting reminder|rescheduled)\b/gi

function countAddresses(field: string): number {
  return (field.match(/[\w.+-]+@[\w.-]+/g) ?? []).length
}

function senderLocalPart(fromAddr: string): string {
  const m = fromAddr.match(/([\w.+-]+)@[\w.-]+/)
  return m ? m[1] : ''
}

function countMatches(re: RegExp, text: string): number {
  return (text.match(re) ?? []).length
}

export function extractHeaderFeatures(email: EmailInput) {
  const nRecipients = countAddresses(email.to) + countAddresses(email.cc ?? '')
  const isReplyOrForward = REPLY_FWD_PREFIX.test(email.subject) ? 1 : 0
  const senderAutomated = AUTOMATED_LOCAL_PART.test(senderLocalPart(email.fromAddr)) ? 1 : 0
  const hasListUnsubscribe = email.listUnsubscribe ? 1 : 0
  const hasPrecedenceBulk = ['bulk', 'list'].includes((email.precedence ?? '').toLowerCase()) ? 1 : 0
  return { nRecipients, isReplyOrForward, senderAutomated, hasListUnsubscribe, hasPrecedenceBulk }
}

export function extractKeywordFeatures(subject: string, body: string) {
  const text = `${subject} ${body}`
  return {
    kwHigh: countMatches(HIGH_KW, text),
    kwMed: countMatches(MED_KW, text),
    kwLow: countMatches(LOW_KW, text),
    exclSubj: (subject.match(/!/g) ?? []).length,
  }
}

export function cleanText(text: string): string {
  return text
    .toLowerCase()
    .replace(/https?:\/\/\S+|www\.\S+/g, ' url ')
    .replace(/\S+@\S+/g, ' email ')
    .replace(/[^a-z\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

// ── Dense vectors for the headless spam/category/priority ONNX models ──
//
// None of the exported ONNX graphs take raw text — see tfidf.ts's module
// docstring for why (skl2onnx can't embed char_wb TF-IDF at all, and its
// word-TF-IDF conversion isn't bit-exact either). So the *entire* feature
// vector the original sklearn Pipeline/ColumnTransformer built has to be
// reconstructed here and concatenated in the exact order scripts/
// export_onnx.py used when it exported each classifier, or the model's
// fixed-position coefficients line up with the wrong features.

export interface SpamVocab {
  word: VocabData
  classes: number[]
}

export interface CategoryVocab {
  word: VocabData
  char: VocabData
  numeric_cols: string[]
  numeric_mean: number[]
  numeric_scale: number[]
  classes: string[]
}

export interface PriorityVocab extends CategoryVocab {
  categories: string[]
}

// n_recipients is always numeric_cols[0] — the fitted StandardScaler's
// mean_/scale_ were computed on log1p(n_recipients), matching
// header_features.py's log1p_recipients (see export_onnx.py's docstring).
function buildScaledNumericVector(
  cols: string[],
  values: Record<string, number>,
  mean: number[],
  scale: number[],
): Float32Array {
  const out = new Float32Array(cols.length)
  for (let i = 0; i < cols.length; i++) {
    const raw = i === 0 ? Math.log1p(values[cols[i]]) : values[cols[i]]
    out[i] = (raw - mean[i]) / scale[i]
  }
  return out
}

function concat(...parts: Float32Array[]): Float32Array {
  const total = parts.reduce((n, p) => n + p.length, 0)
  const out = new Float32Array(total)
  let offset = 0
  for (const p of parts) {
    out.set(p, offset)
    offset += p.length
  }
  return out
}

export function buildSpamVector(cleanedText: string, vocab: SpamVocab): Float32Array {
  return wordTfidfVector(cleanedText, vocab.word)
}

export function buildCategoryVector(email: EmailInput, cleanedText: string, vocab: CategoryVocab): Float32Array {
  const header = extractHeaderFeatures(email)
  const style = extractStylisticFeatures(email.subject, email.body)
  const values: Record<string, number> = {
    n_recipients: header.nRecipients,
    is_reply_or_forward: header.isReplyOrForward,
    sender_automated: header.senderAutomated,
    has_list_unsubscribe: header.hasListUnsubscribe,
    has_precedence_bulk: header.hasPrecedenceBulk,
    ...style,
  }
  const wordVec = wordTfidfVector(cleanedText, vocab.word)
  const charVec = charWbTfidfVector(cleanedText, vocab.char)
  const numericVec = buildScaledNumericVector(vocab.numeric_cols, values, vocab.numeric_mean, vocab.numeric_scale)
  return concat(wordVec, charVec, numericVec)
}

export function buildPriorityVector(
  email: EmailInput,
  cleanedText: string,
  categoryLabel: string,
  spamConf: number,
  vaderCompound: number,
  vocab: PriorityVocab,
): Float32Array {
  const header = extractHeaderFeatures(email)
  const style = extractStylisticFeatures(email.subject, email.body)
  const keywords = extractKeywordFeatures(email.subject, email.body)
  const values: Record<string, number> = {
    n_recipients: header.nRecipients,
    is_reply_or_forward: header.isReplyOrForward,
    sender_automated: header.senderAutomated,
    spam_conf: spamConf,
    vader_compound: vaderCompound,
    ...style,
    kw_high: keywords.kwHigh,
    kw_med: keywords.kwMed,
    kw_low: keywords.kwLow,
    excl_subj: keywords.exclSubj,
  }
  const wordVec = wordTfidfVector(cleanedText, vocab.word)
  const charVec = charWbTfidfVector(cleanedText, vocab.char)
  const catVec = new Float32Array(vocab.categories.map((c) => (c === categoryLabel ? 1 : 0)))
  const numericVec = buildScaledNumericVector(vocab.numeric_cols, values, vocab.numeric_mean, vocab.numeric_scale)
  return concat(wordVec, charVec, catVec, numericVec)
}
