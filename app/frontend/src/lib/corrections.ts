import type { ClassificationResult, PersonalizableModel, PriorityBucket, SpamLabel } from '../types'

// Representative score for a corrected bucket. Same values the backend
// trains the priority regressor on (trainer.BUCKET_SCORE).
export const BUCKET_SCORE: Record<PriorityBucket, number> = { low: 0.2, medium: 0.5, high: 0.8 }

export function predictedFor(
  model: PersonalizableModel,
  result: ClassificationResult,
): { label: string; confidence: number } {
  const { inputs } = result
  if (model === 'category') {
    return { label: inputs.categoryLabel, confidence: result.category.confidences[inputs.categoryLabel] ?? 0 }
  }
  if (model === 'priority') {
    return { label: inputs.priorityBucket, confidence: inputs.priorityConfidences[inputs.priorityBucket] ?? 0 }
  }
  const label: SpamLabel = inputs.spamConf >= 0.5 ? 'spam' : 'ham'
  return { label, confidence: label === 'spam' ? inputs.spamConf : 1 - inputs.spamConf }
}

// Display-only: a user's correction always wins for the email it was made
// on, whatever the model says.
export function applyCorrection(
  result: ClassificationResult,
  model: PersonalizableModel,
  label: string,
): ClassificationResult {
  if (model === 'category') return { ...result, category: { ...result.category, label } }
  if (model === 'priority') {
    const bucket = label as PriorityBucket
    return { ...result, priority: { score: BUCKET_SCORE[bucket], bucket } }
  }
  return { ...result, spam: { ...result.spam, label: label as SpamLabel } }
}

export function applyCorrections(
  result: ClassificationResult,
  corrected: Partial<Record<PersonalizableModel, string>>,
): ClassificationResult {
  return (Object.entries(corrected) as [PersonalizableModel, string][]).reduce(
    (shown, [model, label]) => applyCorrection(shown, model, label),
    result,
  )
}
