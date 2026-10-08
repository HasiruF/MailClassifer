import type { ClassificationResult } from '../types'

// Emails whose model category changed between two classifications (e.g.
// before and after a new personal model loaded), mapped to the category they
// had before. Corrected emails are left out: the user's correction, not the
// model, decides what they see, so nothing visibly moved.
export function resortedCategories(
  before: Map<string, ClassificationResult>,
  after: Map<string, ClassificationResult>,
  correctedIds: Set<string>,
): Record<string, string> {
  const from: Record<string, string> = {}
  for (const [id, next] of after) {
    const previous = before.get(id)
    if (!previous || correctedIds.has(id)) continue
    if (previous.category.label !== next.category.label) from[id] = previous.category.label
  }
  return from
}
