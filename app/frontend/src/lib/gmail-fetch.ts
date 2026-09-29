// Fetches the signed-in user's most recent inbox messages via the Gmail
// REST API (https://gmail.googleapis.com/gmail/v1/...), called directly
// from the browser with the access token from gmail-auth.ts — no backend
// in between, consistent with the rest of this app's client-side-only
// architecture. Only ever hits gmail.readonly-scoped endpoints.

import type { InboxEmail } from '@/types'

const API_BASE = 'https://gmail.googleapis.com/gmail/v1/users/me'
const MAX_RESULTS = 40
// Gmail enforces a per-user cap on simultaneously in-flight requests,
// separate from (and undocumented, unlike) the 250-quota-units/sec limit —
// firing all MAX_RESULTS gets via Promise.all threw a 429
// "Too many concurrent requests for user." Throttle conservatively rather
// than guess the real ceiling.
const FETCH_CONCURRENCY = 8

interface GmailHeader {
  name: string
  value: string
}

interface GmailMessagePart {
  mimeType: string
  body?: { data?: string }
  parts?: GmailMessagePart[]
}

interface GmailMessage {
  id: string
  labelIds?: string[]
  payload: { headers: GmailHeader[] } & GmailMessagePart
}

interface GmailListResponse {
  messages?: { id: string }[]
}

async function gmailFetch<T>(path: string, token: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) {
    throw new Error(`Gmail API ${path} failed: ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

function header(headers: GmailHeader[], name: string): string {
  return headers.find((h) => h.name.toLowerCase() === name.toLowerCase())?.value ?? ''
}

function decodeBase64Url(data: string): string {
  const base64 = data.replace(/-/g, '+').replace(/_/g, '/')
  const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4)
  const binary = atob(padded)
  const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0))
  return new TextDecoder('utf-8').decode(bytes)
}

function stripHtml(html: string): string {
  return html
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function collectParts(part: GmailMessagePart, acc: GmailMessagePart[] = []): GmailMessagePart[] {
  acc.push(part)
  for (const child of part.parts ?? []) collectParts(child, acc)
  return acc
}

// Gmail's API hands back the raw MIME tree, not "the" body. Two views of it:
// `text` feeds classification (the first text/plain part, falling back to
// text/html with tags stripped, since some senders, marketing mail
// especially, omit a plain-text alternative); `html` is the text/html part
// as-is, for display only.
export function extractBodies(payload: GmailMessagePart): { text: string; html: string | undefined } {
  const parts = collectParts(payload)
  const plain = parts.find((p) => p.mimeType === 'text/plain' && p.body?.data)
  const htmlPart = parts.find((p) => p.mimeType === 'text/html' && p.body?.data)
  const html = htmlPart?.body?.data ? decodeBase64Url(htmlPart.body.data) : undefined
  const text = plain?.body?.data ? decodeBase64Url(plain.body.data) : html ? stripHtml(html) : ''
  return { text, html }
}

function parseFrom(fromHeader: string): { fromName: string; fromAddr: string } {
  const match = fromHeader.match(/^(.*?)\s*<([^>]+)>$/)
  if (match) {
    const name = match[1].replace(/^"|"$/g, '').trim()
    return { fromName: name || match[2], fromAddr: match[2] }
  }
  return { fromName: fromHeader, fromAddr: fromHeader }
}

function formatReceivedAt(dateHeader: string): string {
  const date = new Date(dateHeader)
  if (isNaN(date.getTime())) return ''
  const now = new Date()
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  }
  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)
  if (date.toDateString() === yesterday.toDateString()) return 'Yesterday'
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

function messageToInboxEmail(msg: GmailMessage): InboxEmail {
  const headers = msg.payload.headers
  const { fromName, fromAddr } = parseFrom(header(headers, 'From'))
  const dateHeader = header(headers, 'Date')
  const parsedMs = Date.parse(dateHeader)
  const { text, html } = extractBodies(msg.payload)
  return {
    id: msg.id,
    subject: header(headers, 'Subject'),
    body: text,
    bodyHtml: html,
    to: header(headers, 'To'),
    cc: header(headers, 'Cc') || undefined,
    fromAddr,
    fromName,
    listUnsubscribe: header(headers, 'List-Unsubscribe') !== '',
    precedence: header(headers, 'Precedence') || undefined,
    receivedAt: formatReceivedAt(dateHeader),
    receivedAtMs: Number.isNaN(parsedMs) ? Date.now() : parsedMs,
    unread: (msg.labelIds ?? []).includes('UNREAD'),
    source: 'gmail',
  }
}

// Runs `fn` over `items` with at most `limit` calls in flight at once —
// a fixed pool of workers, each pulling the next index as it finishes,
// rather than chunking into sequential batches (keeps the concurrency
// slots always full instead of bottlenecking on the slowest item per chunk).
async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let next = 0
  async function worker() {
    while (next < items.length) {
      const i = next++
      results[i] = await fn(items[i])
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))
  return results
}

export async function fetchRecentInboxEmails(token: string): Promise<InboxEmail[]> {
  const list = await gmailFetch<GmailListResponse>(
    `/messages?maxResults=${MAX_RESULTS}&labelIds=INBOX`,
    token,
  )
  const ids = list.messages ?? []
  const messages = await mapWithConcurrency(ids, FETCH_CONCURRENCY, (m) =>
    gmailFetch<GmailMessage>(`/messages/${m.id}?format=full`, token),
  )
  return messages.map(messageToInboxEmail)
}
