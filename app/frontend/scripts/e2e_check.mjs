// End-to-end cross-check: runs the same 4 test emails from
// scripts/classify_email.py's __main__ block through the exported ONNX
// pipeline (via onnxruntime-node, standing in for onnxruntime-web — same
// underlying ONNX Runtime, no browser needed for this check) and diffs the
// results against classify_email.py's own printed output (run
// `PYTHONIOENCODING=utf-8 python classify_email.py --models ../models`
// from scripts/ to regenerate that reference output if the models ever
// retrain).
//
// This is a dev-only verification script, not part of the app build.
import * as ort from 'onnxruntime-node'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { wordTfidfVector, charWbTfidfVector } from '../src/inference/tfidf.ts'
import { extractStylisticFeatures } from '../src/inference/stylistic.ts'
import { extractHeaderFeatures, extractKeywordFeatures, cleanText } from '../src/inference/features.ts'
import { vaderCompound } from '../src/inference/sentiment.ts'

const base = join(dirname(fileURLToPath(import.meta.url)), '../public/models')

function loadJson(name) {
  return JSON.parse(readFileSync(`${base}/${name}`, 'utf-8'))
}

const spamVocab = loadJson('spam_classifier.vocab.json')
const categoryVocab = loadJson('category_classifier.vocab.json')
const priorityVocab = loadJson('priority_classifier.vocab.json')

const spamSess = await ort.InferenceSession.create(`${base}/spam_classifier.onnx`)
const categorySess = await ort.InferenceSession.create(`${base}/category_classifier.onnx`)
const prioritySess = await ort.InferenceSession.create(`${base}/priority_classifier.onnx`)
const priorityRegSess = await ort.InferenceSession.create(`${base}/priority_regressor.onnx`)

function buildScaledNumericVector(cols, values, mean, scale) {
  const out = new Float32Array(cols.length)
  for (let i = 0; i < cols.length; i++) {
    const raw = i === 0 ? Math.log1p(values[cols[i]]) : values[cols[i]]
    out[i] = (raw - mean[i]) / scale[i]
  }
  return out
}
function concat(...parts) {
  const total = parts.reduce((n, p) => n + p.length, 0)
  const out = new Float32Array(total)
  let offset = 0
  for (const p of parts) {
    out.set(p, offset)
    offset += p.length
  }
  return out
}

async function classify(email) {
  const cleanedText = cleanText(`${email.subject} ${email.body}`)

  const spamVec = wordTfidfVector(cleanedText, spamVocab.word)
  const spamOut = await spamSess.run({ features: new ort.Tensor('float32', spamVec, [1, spamVec.length]) })
  const spamProba = spamOut.probabilities.data
  const spamConf = spamProba[spamVocab.classes.indexOf(1)]
  const spamLabel = spamConf >= 0.5 ? 'spam' : 'ham'

  const header = extractHeaderFeatures(email)
  const style = extractStylisticFeatures(email.subject, email.body)
  const catValues = {
    n_recipients: header.nRecipients,
    is_reply_or_forward: header.isReplyOrForward,
    sender_automated: header.senderAutomated,
    has_list_unsubscribe: header.hasListUnsubscribe,
    has_precedence_bulk: header.hasPrecedenceBulk,
    ...style,
  }
  const catVec = concat(
    wordTfidfVector(cleanedText, categoryVocab.word),
    charWbTfidfVector(cleanedText, categoryVocab.char),
    buildScaledNumericVector(categoryVocab.numeric_cols, catValues, categoryVocab.numeric_mean, categoryVocab.numeric_scale),
  )
  const categoryOut = await categorySess.run({ features: new ort.Tensor('float32', catVec, [1, catVec.length]) })
  const categoryLabel = categoryOut.label.data[0]
  const categoryProba = categoryOut.probabilities.data
  const categoryConf = {}
  categoryVocab.classes.forEach((c, i) => (categoryConf[c] = categoryProba[i]))

  const compound = vaderCompound(cleanedText)
  const keywords = extractKeywordFeatures(email.subject, email.body)
  const prioValues = {
    n_recipients: header.nRecipients,
    is_reply_or_forward: header.isReplyOrForward,
    sender_automated: header.senderAutomated,
    spam_conf: spamConf,
    vader_compound: compound,
    ...style,
    kw_high: keywords.kwHigh,
    kw_med: keywords.kwMed,
    kw_low: keywords.kwLow,
    excl_subj: keywords.exclSubj,
  }
  const catOneHot = new Float32Array(priorityVocab.categories.map((c) => (c === categoryLabel ? 1 : 0)))
  const prioVec = concat(
    wordTfidfVector(cleanedText, priorityVocab.word),
    charWbTfidfVector(cleanedText, priorityVocab.char),
    catOneHot,
    buildScaledNumericVector(priorityVocab.numeric_cols, prioValues, priorityVocab.numeric_mean, priorityVocab.numeric_scale),
  )
  const [prioClfOut, prioRegOut] = await Promise.all([
    prioritySess.run({ features: new ort.Tensor('float32', prioVec, [1, prioVec.length]) }),
    priorityRegSess.run({ features: new ort.Tensor('float32', prioVec, [1, prioVec.length]) }),
  ])
  const bucket = prioClfOut.label.data[0]
  let score = Math.min(1.0, Math.max(0.1, prioRegOut.variable.data[0]))
  let note
  if (spamLabel === 'spam') {
    score = 0.1
    note = `classified as spam (conf=${spamConf.toFixed(2)}) — priority suppressed`
  }

  return {
    spam: { label: spamLabel, confidence: Number(spamConf.toFixed(2)) },
    category: {
      label: categoryLabel,
      ...Object.fromEntries(Object.entries(categoryConf).map(([k, v]) => [`conf_${k.toLowerCase()}`, Number(v.toFixed(2))])),
    },
    priority: { score: Number(score.toFixed(2)), bucket, ...(note ? { note } : {}) },
  }
}

// Matches classify_email.py's __main__ test_emails list exactly, in order.
const testEmails = [
  { subject: 'URGENT: Approval needed by EOD', body: 'Hi, please sign off on the attached contract before end of day. Legal is waiting.', to: 'ceo@enron.com', fromAddr: '' },
  { subject: 'Weekly newsletter - Energy market digest', body: 'This is your weekly roundup. Unsubscribe at any time. No action needed.', to: 'all@enron.com', fromAddr: 'digest@energymarketnews.com', listUnsubscribe: true, precedence: 'bulk' },
  { subject: 'Lunch on Friday?', body: 'Hey, are you free for lunch this Friday? Let me know.', to: 'friend@gmail.com', fromAddr: '' },
  { subject: 'You have won a prize!!!', body: 'Congratulations! Click here to claim your $1000 reward. Limited time offer.', to: 'victim@email.com', fromAddr: '' },
]

for (const [i, email] of testEmails.entries()) {
  const result = await classify(email)
  console.log(`\nEmail ${i + 1}: "${email.subject}"`)
  console.log(JSON.stringify(result, null, 2))
}
