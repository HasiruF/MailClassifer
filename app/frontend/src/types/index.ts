// Mirrors the label shapes produced by scripts/classify_email.py's
// EmailClassifier.classify() — keep in sync if the Python output shape changes.

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
  sentAt?: string
}

export interface ClassificationResult {
  spam: { label: SpamLabel; confidence: number }
  category: { label: CategoryLabel | string; confidences: Record<string, number> }
  priority: { score: number; bucket: PriorityBucket; note?: string }
}

// A correction the user makes to a prediction. Stored locally always; only
// leaves the device if the user opts into personalization sync, and only as
// a feature vector (see inference/features.ts) — never raw subject/body.
export interface Correction {
  id: string
  emailId: string
  labelType: 'category' | 'priority'
  originalLabel: string
  correctedLabel: string
  customCategoryName?: string // set when correctedLabel is a user-defined category
  createdAt: string
  synced: boolean
}

export interface CustomCategory {
  id: string
  name: string
  createdAt: string
}

// The numeric feature vector extracted client-side for a given email —
// this, not raw text, is what gets uploaded when personalization sync is on.
export interface FeatureVector {
  textFeatures: Record<string, number> // sparse TF-IDF: term -> weight
  numericFeatures: number[] // header/style/keyword features, fixed order
  numericFeatureNames: string[]
}
