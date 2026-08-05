import { openDB, type DBSchema, type IDBPDatabase } from 'idb'
import type { Correction, CustomCategory, EmailInput, ClassificationResult } from '../types'

// All email content and corrections live here by default. Nothing leaves
// the device unless the user explicitly enables personalization sync
// (see api/client.ts syncCorrection) — and even then, only an extracted
// feature vector goes out, never this raw store.

interface AppDB extends DBSchema {
  emails: {
    key: string
    value: EmailInput & { classification?: ClassificationResult }
  }
  corrections: {
    key: string
    value: Correction
    indexes: { 'by-synced': number } // 0/1, IndexedDB can't index booleans directly
  }
  customCategories: {
    key: string
    value: CustomCategory
  }
}

let dbPromise: Promise<IDBPDatabase<AppDB>> | null = null

export function getDB() {
  if (!dbPromise) {
    dbPromise = openDB<AppDB>('email-classifier', 1, {
      upgrade(db) {
        db.createObjectStore('emails', { keyPath: 'id' })
        const corrections = db.createObjectStore('corrections', { keyPath: 'id' })
        corrections.createIndex('by-synced', 'synced')
        db.createObjectStore('customCategories', { keyPath: 'id' })
      },
    })
  }
  return dbPromise
}

export async function saveCorrection(correction: Correction) {
  const db = await getDB()
  await db.put('corrections', correction)
}

export async function getUnsyncedCorrections(): Promise<Correction[]> {
  const db = await getDB()
  const all = await db.getAll('corrections')
  return all.filter((c) => !c.synced)
}

export async function markSynced(correctionId: string) {
  const db = await getDB()
  const correction = await db.get('corrections', correctionId)
  if (correction) {
    correction.synced = true
    await db.put('corrections', correction)
  }
}

export async function addCustomCategory(category: CustomCategory) {
  const db = await getDB()
  await db.put('customCategories', category)
}

export async function getCustomCategories(): Promise<CustomCategory[]> {
  const db = await getDB()
  return db.getAll('customCategories')
}
