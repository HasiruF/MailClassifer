import type { Correction, FeatureVector } from '../types'
import { getUnsyncedCorrections, markSynced } from '../storage/db'
import { extractFeatureVector } from '../inference/features'

const API_BASE = '/api'

// Sends only the extracted feature vector + label — never raw subject/body.
// Backend trains (kNN/centroid for new/small categories, periodic retrain
// once a category has enough volume) purely on this numeric representation.
async function postCorrection(correction: Correction, featureVector: FeatureVector) {
  const res = await fetch(`${API_BASE}/corrections`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      label_type: correction.labelType,
      original_label: correction.originalLabel,
      corrected_label: correction.correctedLabel,
      custom_category_name: correction.customCategoryName ?? null,
      feature_vector: featureVector,
    }),
  })
  if (!res.ok) throw new Error(`Failed to sync correction: ${res.status}`)
}

// Call after the user opts into personalization sync, and periodically
// after that (e.g. on app load, or after each new correction) to flush
// anything saved locally while offline or before sync was enabled.
export async function syncPendingCorrections(getEmailById: (id: string) => Promise<{ subject: string; body: string; to: string; fromAddr: string } | undefined>) {
  const pending = await getUnsyncedCorrections()
  for (const correction of pending) {
    const email = await getEmailById(correction.emailId)
    if (!email) continue
    const featureVector = extractFeatureVector({ id: correction.emailId, ...email })
    await postCorrection(correction, featureVector)
    await markSynced(correction.id)
  }
}

export async function fetchPersonalizedModel(userId: string, modelType: 'category' | 'priority') {
  const res = await fetch(`${API_BASE}/models/${userId}/${modelType}/latest`)
  if (!res.ok) return null
  return res.json()
}
