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

export type PersonalizableModel = 'spam' | 'category' | 'priority'
export type BackendModelName = PersonalizableModel | 'priority_regressor'

// A user's personalized ONNX graph, as downloaded from the backend.
export interface PersonalizedModelArtifact {
  model: BackendModelName
  version: number
  classes: (string | number)[]
  bytes: ArrayBuffer
}

// What the models actually saw and said, before rounding, spam suppression,
// or any user correction. Corrections are built from these values, so the
// backend trains on exactly what production computed.
export interface ClassificationInputs {
  spamConf: number
  vaderCompound: number
  categoryLabel: string
  priorityBucket: PriorityBucket
  priorityConfidences: Record<string, number>
}

export interface ClassificationResult {
  spam: { label: SpamLabel; confidence: number }
  category: { label: CategoryLabel | string; confidences: Record<string, number> }
  priority: { score: number; bucket: PriorityBucket; note?: string }
  inputs: ClassificationInputs
}

// Display shape the inbox UI renders — shared by the hand-authored sample
// data (src/data/sample-emails.ts) and real messages fetched from Gmail
// (src/lib/gmail-fetch.ts), so the inbox list/detail views don't care which
// source produced a given row.
export interface InboxEmail extends EmailInput {
  fromName: string
  // The message's text/html part, for display only (classification always
  // uses `body`). Absent for plain-text-only mail and the sample emails.
  bodyHtml?: string
  receivedAt: string
  receivedAtMs: number
  unread: boolean
  // 'gmail' rows can deep-link back to the real message (id is a real
  // Gmail message id); 'sample' rows can't — see EmailDetail's
  // "Open in Gmail" link.
  source: 'sample' | 'gmail'
}
