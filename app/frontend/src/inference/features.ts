import type { EmailInput } from '../types'

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
