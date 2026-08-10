import * as ort from 'onnxruntime-web'
import type { EmailInput, ClassificationResult, CategoryLabel, PriorityBucket } from '../types'
import {
  cleanText,
  buildSpamVector,
  buildCategoryVector,
  buildPriorityVector,
  type SpamVocab,
  type CategoryVocab,
  type PriorityVocab,
} from './features'
import { vaderCompound } from './sentiment'

// Mirrors scripts/classify_email.py's EmailClassifier.classify(). All four
// ONNX graphs (see scripts/export_onnx.py) are headless — dense float
// vector in, built by features.ts's build*Vector() functions, not raw
// text. Spam and category are computed independently from the cleaned text
// + header/style features; priority is the only model that depends on
// another model's output (spam's confidence, category's label, and VADER
// sentiment all feed into it).
//
// Fetches the WASM runtime from jsdelivr rather than bundling it — Next.js/
// webpack's WASM asset handling for onnxruntime-web is a known source of
// build headaches, and this app never sends user data anywhere regardless,
// so fetching the *runtime engine* (not email content) from a CDN doesn't
// weaken the privacy boundary. Version must match the installed
// onnxruntime-web version exactly (package.json pins it unranged, with
// this exact number, for the same reason) — a mismatch between the WASM
// binary and the JS wrapper is an ABI break, not a graceful fallback.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/'

const MODEL_PATHS = {
  spam: '/models/spam_classifier.onnx',
  category: '/models/category_classifier.onnx',
  priority: '/models/priority_classifier.onnx',
  priorityRegressor: '/models/priority_regressor.onnx',
} as const

type ModelKey = keyof typeof MODEL_PATHS

let sessions: Partial<Record<ModelKey, ort.InferenceSession>> = {}
let spamVocab: SpamVocab | null = null
let categoryVocab: CategoryVocab | null = null
let priorityVocab: PriorityVocab | null = null
// Memoizes the in-flight/completed load so repeated calls (e.g. React 19
// Strict Mode's dev-only double-invoke of useEffect on mount) share one
// result instead of re-running everything.
let loadPromise: Promise<void> | null = null

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`)
  return res.json() as Promise<T>
}

export function loadModels(): Promise<void> {
  if (loadPromise) return loadPromise
  loadPromise = (async () => {
    // Vocab JSON is plain HTTP — safe to fetch concurrently with everything
    // else. ONNX sessions are NOT: onnxruntime-web's WASM backend only
    // tolerates one InferenceSession.create() in flight at a time — racing
    // multiple concurrently (e.g. via Promise.all/map, which is what this
    // looked like before) throws "Session already started". So these are
    // created one at a time, awaited in sequence, not in parallel.
    const vocabPromise = Promise.all([
      fetchJson<SpamVocab>('/models/spam_classifier.vocab.json').then((v) => {
        spamVocab = v
      }),
      fetchJson<CategoryVocab>('/models/category_classifier.vocab.json').then((v) => {
        categoryVocab = v
      }),
      fetchJson<PriorityVocab>('/models/priority_classifier.vocab.json').then((v) => {
        priorityVocab = v
      }),
    ])

    const newSessions: Partial<Record<ModelKey, ort.InferenceSession>> = {}
    for (const key of Object.keys(MODEL_PATHS) as ModelKey[]) {
      newSessions[key] = await ort.InferenceSession.create(MODEL_PATHS[key])
    }
    sessions = newSessions

    await vocabPromise
  })()
  return loadPromise
}

export function modelsLoaded(): boolean {
  return (
    Object.keys(sessions).length === Object.keys(MODEL_PATHS).length &&
    spamVocab !== null &&
    categoryVocab !== null &&
    priorityVocab !== null
  )
}

function requireSession(key: ModelKey): ort.InferenceSession {
  const s = sessions[key]
  if (!s) throw new Error(`Model "${key}" not loaded — call loadModels() first.`)
  return s
}

export async function classify(email: EmailInput): Promise<ClassificationResult> {
  if (!modelsLoaded() || !spamVocab || !categoryVocab || !priorityVocab) {
    throw new Error('Models not loaded — call loadModels() first.')
  }

  const cleanedText = cleanText(`${email.subject} ${email.body}`)

  // ── spam ──
  const spamVec = buildSpamVector(cleanedText, spamVocab)
  const spamFeeds = { features: new ort.Tensor('float32', spamVec, [1, spamVec.length]) }
  const spamOut = await requireSession('spam').run(spamFeeds)
  const spamProba = spamOut.probabilities.data as Float32Array
  const spamIdx = spamVocab.classes.indexOf(1)
  const spamConf = spamProba[spamIdx]
  const spamLabel = spamConf >= 0.5 ? 'spam' : 'ham'

  // ── category ──
  const categoryVec = buildCategoryVector(email, cleanedText, categoryVocab)
  const categoryFeeds = { features: new ort.Tensor('float32', categoryVec, [1, categoryVec.length]) }
  const categoryOut = await requireSession('category').run(categoryFeeds)
  const categoryProba = categoryOut.probabilities.data as Float32Array
  const categoryLabel = (categoryOut.label.data as string[])[0] as CategoryLabel
  const categoryConfidences: Record<string, number> = {}
  categoryVocab.classes.forEach((cls, i) => {
    categoryConfidences[cls] = categoryProba[i]
  })

  // ── priority ──
  const compound = vaderCompound(cleanedText)
  const priorityVec = buildPriorityVector(email, cleanedText, categoryLabel, spamConf, compound, priorityVocab)
  const priorityFeeds = { features: new ort.Tensor('float32', priorityVec, [1, priorityVec.length]) }
  // Same constraint as loadModels(): onnxruntime-web's WASM backend only
  // tolerates one session operation in flight at a time, whether that's
  // create() or run() — these two .run() calls used to race via
  // Promise.all, which is what actually threw "Session already started"
  // (confirmed via a real headless-browser repro, not guessed).
  const priorityClfOut = await requireSession('priority').run(priorityFeeds)
  const priorityRegOut = await requireSession('priorityRegressor').run(priorityFeeds)
  const priorityBucket = (priorityClfOut.label.data as string[])[0] as PriorityBucket
  // skl2onnx names a bare regressor's sole output "variable" — confirmed
  // against the actual exported graph during the earlier attempt.
  const rawScore = (priorityRegOut.variable.data as Float32Array)[0]
  const priorityScore = Math.min(1.0, Math.max(0.1, rawScore))

  let priority: ClassificationResult['priority'] = {
    score: Math.round(priorityScore * 100) / 100,
    bucket: priorityBucket,
  }

  // Spam suppresses priority — spam/phishing routinely fakes urgency to
  // game attention, so a message already flagged as spam shouldn't rank as
  // high-priority regardless of what language it uses. Category is
  // deliberately not wired in here — see classify_email.py for why.
  if (spamLabel === 'spam') {
    priority = {
      score: 0.1,
      bucket: 'low',
      note: `classified as spam (conf=${spamConf.toFixed(2)}) — priority suppressed`,
    }
  }

  return {
    spam: { label: spamLabel, confidence: Math.round(spamConf * 10000) / 10000 },
    category: { label: categoryLabel, confidences: categoryConfidences },
    priority,
  }
}
