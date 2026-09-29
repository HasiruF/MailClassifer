// Dev-only check (spec §12): the vectors the client sends as corrections
// must match, index for index, the sklearn vectors in the backend's base
// training matrices. Regenerate the fixture with
// scripts/dump_sparse_parity_fixture.py (repo root) if the models retrain.
//
// Usage (from app/frontend): npx tsx scripts/sparse_parity_check.mjs
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { buildSpamVector, buildCategoryVector, buildPriorityVector, cleanText } from '../src/inference/features.ts'
import { toSparse } from '../src/lib/sparse.ts'

const here = dirname(fileURLToPath(import.meta.url))
const vocab = (name) => JSON.parse(readFileSync(join(here, '../public/models', `${name}.vocab.json`), 'utf-8'))
const spamVocab = vocab('spam_classifier')
const categoryVocab = vocab('category_classifier')
const priorityVocab = vocab('priority_classifier')
const fixture = JSON.parse(readFileSync(join(here, 'fixtures/sparse_parity.json'), 'utf-8'))

const TOLERANCE = 1e-5
let failures = 0

function compare(label, got, want) {
  const sameIndices =
    got.indices.length === want.indices.length && got.indices.every((index, k) => index === want.indices[k])
  const maxDiff = sameIndices ? Math.max(0, ...got.values.map((v, k) => Math.abs(v - want.values[k]))) : Infinity
  const ok = sameIndices && maxDiff <= TOLERANCE
  if (!ok) failures++
  console.log(`${ok ? 'OK  ' : 'FAIL'} ${label}: ${got.indices.length} non-zeros, max value diff ${maxDiff.toExponential(2)}`)
}

for (const { email, spam, category, priority } of fixture) {
  const input = { id: email.id, subject: email.subject, body: email.body, to: email.to, fromAddr: email.fromAddr }
  const text = cleanText(`${email.subject} ${email.body}`)
  compare(`${email.id} spam`, toSparse(buildSpamVector(text, spamVocab)), spam)
  compare(`${email.id} category`, toSparse(buildCategoryVector(input, text, categoryVocab)), category)
  compare(
    `${email.id} priority`,
    toSparse(buildPriorityVector(input, text, priority.categoryLabel, priority.spamConf, priority.vaderCompound, priorityVocab)),
    priority,
  )
}

if (failures > 0) {
  console.error(`\n${failures} mismatch(es)`)
  process.exit(1)
}
console.log('\nAll vectors match.')
