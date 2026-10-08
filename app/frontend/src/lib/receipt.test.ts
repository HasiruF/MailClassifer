import { describe, expect, it } from 'vitest'
import type { VocabData } from '../inference/tfidf'
import type { Vocabs } from '../inference/features'
import { decodeCorrection } from './receipt'

function vocab(terms: string[], analyzer: 'word' | 'char_wb' = 'word'): VocabData {
  return {
    vocabulary: Object.fromEntries(terms.map((t, i) => [t, i])),
    idf: terms.map(() => 1),
    ngram_range: [1, 1],
    analyzer,
    stop_words: [],
  }
}

const vocabs: Vocabs = {
  spam: { word: vocab(['free', 'prize']), classes: [0, 1] },
  category: {
    word: vocab(['linkedin', 'profile', 'week']),
    char: vocab([' li', 'nked'], 'char_wb'),
    numeric_cols: ['n_recipients', 'sender_automated', 'text_len_log'],
    numeric_mean: [0.5, 0.2, 5],
    numeric_scale: [0.5, 0.4, 1],
    classes: ['Other', 'Personal', 'Work'],
  },
  priority: {
    word: vocab(['urgent']),
    char: vocab(['urg'], 'char_wb'),
    categories: ['Other', 'Personal', 'Work'],
    numeric_cols: ['spam_conf'],
    numeric_mean: [0.2],
    numeric_scale: [0.1],
    classes: ['high', 'low', 'medium'],
  },
}

describe('decodeCorrection', () => {
  it('reads a spam vector as weighted words, heaviest first', () => {
    const receipt = decodeCorrection('spam', { dim: 2, indices: [0, 1], values: [0.2, 0.7] }, vocabs)
    expect(receipt.words).toEqual([
      { term: 'prize', weight: 0.7 },
      { term: 'free', weight: 0.2 },
    ])
    expect(receipt.pieces).toEqual([])
    expect(receipt.signals).toEqual([])
  })

  it('splits a category vector into words, letter groups and unscaled signals', () => {
    // word 0..2 | char 3..4 | numeric 5..7
    const vector = {
      dim: 8,
      indices: [0, 1, 4, 5, 6, 7],
      values: [0.4, 0.3, 0.5, (Math.log1p(1) - 0.5) / 0.5, 2, Math.log1p(99) - 5],
    }
    const receipt = decodeCorrection('category', vector, vocabs)
    expect(receipt.words.map((w) => w.term)).toEqual(['linkedin', 'profile'])
    expect(receipt.pieces).toEqual([{ term: 'nked', weight: 0.5 }])
    expect(receipt.signals).toEqual([
      { label: 'Recipients', value: '1' },
      { label: 'Sent from an automated address', value: 'yes' },
      { label: 'Length', value: '99 characters' },
    ])
  })

  it('treats a numeric column missing from the sparse vector as its mean', () => {
    // Only sender_automated stored; the others sit exactly at their means.
    const receipt = decodeCorrection('category', { dim: 8, indices: [6], values: [-0.5] }, vocabs)
    expect(receipt.signals).toEqual([
      { label: 'Recipients', value: '1' },
      { label: 'Sent from an automated address', value: 'no' },
      { label: 'Length', value: '147 characters' },
    ])
  })

  it('reads the category a priority vector was given, then its signals', () => {
    // word 0 | char 1 | categories 2..4 | numeric 5
    const vector = { dim: 6, indices: [0, 4, 5], values: [1, 1, 4] }
    const receipt = decodeCorrection('priority', vector, vocabs)
    expect(receipt.words).toEqual([{ term: 'urgent', weight: 1 }])
    expect(receipt.signals).toEqual([
      { label: 'Category it was given', value: 'Work' },
      { label: 'Spam score', value: '60%' },
    ])
  })
})
