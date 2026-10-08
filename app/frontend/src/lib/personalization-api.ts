import type { BackendModelName, PersonalizableModel, PersonalizedModelArtifact } from '../types'
import type { SparseVector } from './sparse'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:3011'

export interface LastAttempt {
  version: number
  status: 'active' | 'rejected' | 'superseded'
  correction_count: number
  metrics: { reason?: string | null; error?: string; passed?: boolean; [key: string]: unknown }
  created_at: string
}

export interface ModelStatus {
  model: PersonalizableModel
  correction_count: number
  corrections_until_retrain: number
  running: boolean
  last_attempt: LastAttempt | null
}

export interface PersonalizationStatus {
  enabled: boolean
  models: ModelStatus[]
  custom_labels: string[]
}

export interface CorrectionPayload {
  model: PersonalizableModel
  provider_message_id: string
  feature_vector: SparseVector
  predicted_label: string
  predicted_confidence: number
  corrected_label: string
}

export interface CorrectionResponse {
  retrain_scheduled: boolean
  corrections_until_retrain: number
}

interface ManifestEntry {
  model: BackendModelName
  version: number
  classes: (string | number)[]
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const headers: Record<string, string> = init.body ? { 'Content-Type': 'application/json' } : {}
  const res = await fetch(`${BACKEND_URL}${path}`, { ...init, headers, credentials: 'include' })
  if (!res.ok) throw new Error(`${init.method ?? 'GET'} ${path} failed (${res.status})`)
  return res
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  return (await request(path, init)).json() as Promise<T>
}

export function getStatus(): Promise<PersonalizationStatus> {
  return json('/personalization/status')
}

export function setEnabled(enabled: boolean): Promise<{ enabled: boolean }> {
  return json('/personalization/settings', { method: 'PUT', body: JSON.stringify({ enabled }) })
}

export function submitCorrection(payload: CorrectionPayload): Promise<CorrectionResponse> {
  return json('/personalization/corrections', { method: 'POST', body: JSON.stringify(payload) })
}

export interface CorrectionSummary {
  provider_message_id: string
  model: PersonalizableModel
  corrected_label: string
}

export interface CorrectionDetail {
  model: PersonalizableModel
  provider_message_id: string
  predicted_label: string
  predicted_confidence: number
  corrected_label: string
  feature_vector: SparseVector
  created_at: string
}

export function listCorrections(): Promise<CorrectionSummary[]> {
  return json('/personalization/corrections')
}

export function getCorrectionDetail(messageId: string): Promise<CorrectionDetail[]> {
  return json(`/personalization/corrections/${encodeURIComponent(messageId)}`)
}

export function retrainNow(): Promise<{ scheduled: string[] }> {
  return json('/personalization/retrain', { method: 'POST' })
}

export async function fetchActiveModels(): Promise<PersonalizedModelArtifact[]> {
  const manifest = await json<ManifestEntry[]>('/personalization/models')
  const artifacts: PersonalizedModelArtifact[] = []
  for (const entry of manifest) {
    const res = await request(`/personalization/models/${entry.model}/${entry.version}.onnx`)
    artifacts.push({ ...entry, bytes: await res.arrayBuffer() })
  }
  return artifacts
}
