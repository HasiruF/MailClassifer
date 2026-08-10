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
