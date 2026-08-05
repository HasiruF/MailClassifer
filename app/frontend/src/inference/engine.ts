import * as ort from 'onnxruntime-web'
import type { EmailInput, ClassificationResult } from '../types'
import { extractFeatureVector } from './features'

// Loads the three ONNX models exported from scripts/*/train_*.py via
// skl2onnx (see PROJECT_STATUS.md "JS/browser export" — not done yet).
// Until public/models/*.onnx exist, classify() throws rather than silently
// returning wrong predictions.

const MODEL_PATHS = {
  spam: '/models/spam_classifier.onnx',
  category: '/models/category_classifier.onnx',
  priority: '/models/priority_classifier.onnx',
} as const

let sessions: Partial<Record<keyof typeof MODEL_PATHS, ort.InferenceSession>> = {}

export async function loadModels() {
  const entries = await Promise.all(
    (Object.keys(MODEL_PATHS) as (keyof typeof MODEL_PATHS)[]).map(async (key) => {
      const session = await ort.InferenceSession.create(MODEL_PATHS[key])
      return [key, session] as const
    }),
  )
  sessions = Object.fromEntries(entries)
}

export function modelsLoaded(): boolean {
  return Object.keys(sessions).length === Object.keys(MODEL_PATHS).length
}

export async function classify(email: EmailInput): Promise<ClassificationResult> {
  if (!modelsLoaded()) {
    throw new Error(
      'Models not loaded — call loadModels() first, and make sure ' +
        'public/models/*.onnx have been exported (see scripts/*/export_model.py).',
    )
  }
  const _features = extractFeatureVector(email)
  // TODO: build ort.Tensor inputs matching each pipeline's expected input
  // names/shapes (run `session.inputNames` to inspect once the exported
  // ONNX graphs exist), run session.run(), and map outputs back to
  // ClassificationResult in the same shape classify_email.py returns.
  throw new Error('classify() inference not yet wired up — pending ONNX export step.')
}
