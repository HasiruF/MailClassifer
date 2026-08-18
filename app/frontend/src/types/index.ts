export type CategoryLabel = 'Work' | 'Personal' | 'Other'
export type PriorityBucket = 'low' | 'medium' | 'high'
export type SpamLabel = 'spam' | 'ham'

export interface EmailInput {
  id: string
  subject: string
  body: string
  to: string
  cc?: string
  fromAddr: string
  listUnsubscribe?: boolean
  precedence?: string
}

export interface ClassificationResult {
  spam: { label: SpamLabel; confidence: number }
  category: { label: CategoryLabel | string; confidences: Record<string, number> }
  priority: { score: number; bucket: PriorityBucket; note?: string }
}

// Display shape the inbox UI renders — shared by the hand-authored sample
// data (src/data/sample-emails.ts) and real messages fetched from Gmail
// (src/lib/gmail-fetch.ts), so the inbox list/detail views don't care which
// source produced a given row.
export interface InboxEmail extends EmailInput {
  fromName: string
  receivedAt: string
  receivedAtMs: number
  unread: boolean
  // 'gmail' rows can deep-link back to the real message (id is a real
  // Gmail message id); 'sample' rows can't — see EmailDetail's
  // "Open in Gmail" link.
  source: 'sample' | 'gmail'
}
