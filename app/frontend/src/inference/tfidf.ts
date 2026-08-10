// Client-side re-implementation of sklearn's TfidfVectorizer.transform(),
// exact enough to reproduce it bit-for-bit (verified in tfidf.test.ts
// against fixtures dumped straight from the real fitted vectorizer via
// scripts/dump_tfidf_fixtures.py).
//
// Only 'word' and 'char_wb' analyzers are implemented — the only two this
// project's models use. Both operate on text that has already been through
// features.ts's cleanText() (lowercase, alpha + single spaces only), which
// is what the Python side fits/transforms on too.

export interface VocabData {
  vocabulary: Record<string, number>
  idf: number[]
  ngram_range: [number, number]
  analyzer: 'word' | 'char_wb'
  stop_words: string[]
}

function wordNgrams(text: string, stopWords: Set<string>, minN: number, maxN: number): string[] {
  const tokens = text.split(' ').filter((t) => t.length >= 2 && !stopWords.has(t))
  if (maxN === 1) return tokens

  const out: string[] = minN === 1 ? [...tokens] : []
  const start = minN === 1 ? 2 : minN
  for (let n = start; n <= maxN; n++) {
    for (let i = 0; i + n <= tokens.length; i++) {
      out.push(tokens.slice(i, i + n).join(' '))
    }
  }
  return out
}

function charWbNgrams(text: string, minN: number, maxN: number): string[] {
  const out: string[] = []
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const w = ` ${word} `
    const wLen = w.length
    for (let n = minN; n <= Math.min(maxN, wLen); n++) {
      let offset = 0
      out.push(w.slice(offset, offset + n))
      while (offset + n < wLen) {
        offset += 1
        out.push(w.slice(offset, offset + n))
      }
      if (offset === 0) break
    }
  }
  return out
}

// tf (sublinear, always on for these models) * idf, then L2-normalized —
// terms absent from the fitted vocabulary are simply skipped, same as
// sklearn (CountVectorizer never counts out-of-vocabulary terms).
function tfidfFromTerms(terms: string[], vocab: VocabData): Float32Array {
  const dim = vocab.idf.length
  const vec = new Float32Array(dim)
  const counts = new Map<string, number>()
  for (const term of terms) {
    counts.set(term, (counts.get(term) ?? 0) + 1)
  }
  for (const [term, count] of counts) {
    const idx = vocab.vocabulary[term]
    if (idx === undefined) continue
    const tf = 1 + Math.log(count)
    vec[idx] = tf * vocab.idf[idx]
  }
  let normSq = 0
  for (let i = 0; i < dim; i++) normSq += vec[i] * vec[i]
  const norm = Math.sqrt(normSq)
  if (norm > 0) {
    for (let i = 0; i < dim; i++) vec[i] /= norm
  }
  return vec
}

export function wordTfidfVector(text: string, vocab: VocabData): Float32Array {
  const stopWords = new Set(vocab.stop_words)
  const terms = wordNgrams(text, stopWords, vocab.ngram_range[0], vocab.ngram_range[1])
  return tfidfFromTerms(terms, vocab)
}

export function charWbTfidfVector(text: string, vocab: VocabData): Float32Array {
  const terms = charWbNgrams(text, vocab.ngram_range[0], vocab.ngram_range[1])
  return tfidfFromTerms(terms, vocab)
}
