import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { wordTfidfVector, charWbTfidfVector, type VocabData } from './tfidf'

const here = dirname(fileURLToPath(import.meta.url))
const vocabJson = JSON.parse(
  readFileSync(join(here, '../../public/models/category_classifier.vocab.json'), 'utf-8'),
)
const wordVocab: VocabData = vocabJson.word
const charVocab: VocabData = vocabJson.char

interface Fixture {
  text: string
  word_nonzero: Record<string, number>
  char_nonzero: Record<string, number>
}
const fixtures: Fixture[] = JSON.parse(
  readFileSync(join(here, '__fixtures__/tfidf_fixtures.json'), 'utf-8'),
)

describe('TF-IDF parity against the real fitted sklearn TfidfVectorizer', () => {
  for (const fx of fixtures) {
    it(`matches sklearn for "${fx.text.slice(0, 40)}"`, () => {
      const wordVec = wordTfidfVector(fx.text, wordVocab)
      const charVec = charWbTfidfVector(fx.text, charVocab)

      for (const [idxStr, expected] of Object.entries(fx.word_nonzero)) {
        expect(wordVec[Number(idxStr)]).toBeCloseTo(expected, 6)
      }
      for (const [idxStr, expected] of Object.entries(fx.char_nonzero)) {
        expect(charVec[Number(idxStr)]).toBeCloseTo(expected, 6)
      }

      // every OTHER entry must be exactly zero — otherwise a term is
      // leaking into the vector that sklearn's vectorizer didn't produce
      const wordNonzeroIdx = new Set(Object.keys(fx.word_nonzero).map(Number))
      wordVec.forEach((v, i) => {
        if (!wordNonzeroIdx.has(i)) expect(v).toBe(0)
      })
      const charNonzeroIdx = new Set(Object.keys(fx.char_nonzero).map(Number))
      charVec.forEach((v, i) => {
        if (!charNonzeroIdx.has(i)) expect(v).toBe(0)
      })
    })
  }
})
