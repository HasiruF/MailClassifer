# Show the Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make three things the app already does visible: which emails a new personal model re-sorted, the retrain gate's three checks, and exactly what a correction sent to the server.

**Architecture:** Three small pure modules in `src/lib/` (`resort.ts`, `retrain-report.ts`, `receipt.ts`) hold all the logic and are unit-tested. The inbox context gains re-sort state, active model versions, and restores saved corrections on load. The backend adds one field to the status response and two read-only GET endpoints for corrections. UI changes are in the list, sidebar, email detail, and one new receipt component.

**Tech Stack:** FastAPI + SQLAlchemy 2 + pytest (backend, `dmp_test` DB), Next.js 16 + React 19 + Tailwind v4 + vitest + lucide-react (frontend).

**Spec:** `docs/superpowers/specs/2026-10-08-show-the-work-design.md`

## Global Constraints

- Frontend style: no semicolons, single quotes, print width 120, trailing commas.
- Next.js 16: read `node_modules/next/dist/docs/` before using any Next API (this plan uses none beyond existing patterns).
- Colors come from `src/app/inbox/tokens.ts`; teal = "your model / on", coral = failure, cerulean = actions.
- Gate thresholds mirror `app/backend/src/personalization/gate.py`: OWN_CORRECTION_MIN 0.80, MAX_GATE_DROP 0.03, MAX_CUSTOM_LABEL_FRACTION 0.10.
- Never send email text anywhere; the new endpoints only read what is already stored.
- Commits carry no AI-attribution trailer.
- Backend tests: `cd app/backend && .venv/Scripts/python.exe -m pytest -q`. Frontend: `cd app/frontend && npm test`, `npx tsc --noEmit`, `npm run lint`.

---

### Task 1: Backend — attempt size and correction read endpoints

**Files:**
- Modify: `app/backend/src/schemas/personalization.py`
- Modify: `app/backend/src/routers/personalization.py`
- Test: `app/backend/tests/test_personalization_routes.py`

**Interfaces:**
- Produces: `GET /personalization/status` → `last_attempt.correction_count: int`; `GET /personalization/corrections` → `[{provider_message_id, model, corrected_label}]`; `GET /personalization/corrections/{provider_message_id}` → `[{model, provider_message_id, predicted_label, predicted_confidence, corrected_label, feature_vector: {dim, indices, values}, created_at}]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_personalization_routes.py`)

```python
from src.models import Correction, CorrectionModel, EmailConnection, EmailProvider, User


def test_status_reports_how_many_corrections_an_attempt_used(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    for i in range(5):
        client.post("/personalization/corrections", json=correction(f"m{i}", "Finance"))
    attempt = client.get("/personalization/status").json()["models"][0]["last_attempt"]
    assert attempt["correction_count"] == 5


def test_reading_corrections_requires_a_session(client):
    assert client.get("/personalization/corrections").status_code == 401
    assert client.get("/personalization/corrections/m1").status_code == 401


def test_listed_corrections_say_what_the_user_chose(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    client.post("/personalization/corrections", json=correction("m1", "Work"))
    client.post("/personalization/corrections", json=correction("m2", "Finance"))
    listed = client.get("/personalization/corrections").json()
    assert sorted(listed, key=lambda c: c["provider_message_id"]) == [
        {"provider_message_id": "m1", "model": "category", "corrected_label": "Work"},
        {"provider_message_id": "m2", "model": "category", "corrected_label": "Finance"},
    ]


def test_a_messages_correction_detail_is_exactly_what_was_stored(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    client.post("/personalization/corrections", json=correction("m1", "Work"))
    [detail] = client.get("/personalization/corrections/m1").json()
    assert detail["model"] == "category"
    assert detail["provider_message_id"] == "m1"
    assert detail["predicted_label"] == "Other"
    assert detail["predicted_confidence"] == 0.7
    assert detail["corrected_label"] == "Work"
    assert detail["feature_vector"] == VECTOR
    assert "created_at" in detail
    assert client.get("/personalization/corrections/unknown").json() == []


def test_corrections_are_scoped_to_their_owner(client, connected, db):
    other = User()
    db.add(other)
    db.flush()
    their_connection = EmailConnection(
        user_id=other.id, provider=EmailProvider.gmail, provider_account_id="google-sub-2",
        provider_email="them@example.com",
    )
    db.add(their_connection)
    db.flush()
    db.add(Correction(
        email_connection_id=their_connection.id, provider_message_id="theirs", model=CorrectionModel.category,
        feature_vector=VECTOR, predicted_label="Other", predicted_confidence=0.5, corrected_label="Work",
    ))
    db.commit()
    assert client.get("/personalization/corrections").json() == []
    assert client.get("/personalization/corrections/theirs").json() == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_personalization_routes.py`
Expected: the five new tests FAIL (KeyError `correction_count`, 405 for the GETs).

- [ ] **Step 3: Implement**

`schemas/personalization.py` — add `correction_count` to `LastAttempt` and two response models:

```python
class LastAttempt(BaseModel):
    version: int
    status: str
    correction_count: int
    metrics: dict
    created_at: datetime


class CorrectionSummary(BaseModel):
    provider_message_id: str
    model: str
    corrected_label: str


class CorrectionDetail(BaseModel):
    model: str
    provider_message_id: str
    predicted_label: str
    predicted_confidence: float
    corrected_label: str
    feature_vector: SparseVector
    created_at: datetime
```

`routers/personalization.py` — import the two schemas and `SparseVector`, pass `correction_count=last.correction_count` in `get_status`, and add:

```python
def _user_corrections(user_id: uuid.UUID):
    return select(Correction).join(EmailConnection, Correction.email_connection_id == EmailConnection.id).where(
        EmailConnection.user_id == user_id
    )


# What the user chose per email, so the inbox can show corrections again
# after a reload. No feature vectors: the list stays small.
@router.get("/corrections", response_model=list[CorrectionSummary])
def list_corrections(user: User = Depends(_current_user), db: OrmSession = Depends(get_db)) -> list[CorrectionSummary]:
    rows = db.scalars(_user_corrections(user.id).order_by(Correction.created_at)).all()
    return [
        CorrectionSummary(provider_message_id=r.provider_message_id, model=r.model.value, corrected_label=r.corrected_label)
        for r in rows
    ]


# Everything stored for one email's corrections, exactly as saved: the
# receipt the inbox decodes to show what was sent.
@router.get("/corrections/{provider_message_id}", response_model=list[CorrectionDetail])
def get_message_corrections(
    provider_message_id: str, user: User = Depends(_current_user), db: OrmSession = Depends(get_db)
) -> list[CorrectionDetail]:
    rows = db.scalars(
        _user_corrections(user.id)
        .where(Correction.provider_message_id == provider_message_id)
        .order_by(Correction.model)
    ).all()
    return [
        CorrectionDetail(
            model=r.model.value,
            provider_message_id=r.provider_message_id,
            predicted_label=r.predicted_label,
            predicted_confidence=r.predicted_confidence,
            corrected_label=r.corrected_label,
            feature_vector=SparseVector(**r.feature_vector),
            created_at=r.created_at,
        )
        for r in rows
    ]
```

- [ ] **Step 4: Run the full backend suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass (44 existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add app/backend/src/schemas/personalization.py app/backend/src/routers/personalization.py app/backend/tests/test_personalization_routes.py
git commit -m "Report retrain attempt size and expose saved corrections for reading"
```

---

### Task 2: Retrain report checks (`retrain-report.ts`)

**Files:**
- Modify: `app/frontend/src/lib/personalization-api.ts` (add `correction_count` to `LastAttempt`)
- Create: `app/frontend/src/lib/retrain-report.ts`
- Test: `app/frontend/src/lib/retrain-report.test.ts`

**Interfaces:**
- Consumes: `LastAttempt` from `personalization-api.ts` (now with `correction_count: number`).
- Produces: `interface ReportCheck { label: string; value: string; caption: string; passed: boolean; bar?: { fill: number; mark: number } }`; `reportChecks(model: PersonalizableModel, attempt: LastAttempt): ReportCheck[] | null` (null when metrics lack gate numbers).

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from 'vitest'
import type { LastAttempt } from './personalization-api'
import { reportChecks } from './retrain-report'

function attempt(metrics: LastAttempt['metrics'], overrides: Partial<LastAttempt> = {}): LastAttempt {
  return { version: 2, status: 'active', correction_count: 10, metrics, created_at: '2026-10-08T10:00:00Z', ...overrides }
}

const passed = {
  passed: true,
  own_correction_accuracy: 0.9,
  gate_accuracy: 0.776,
  base_gate_accuracy: 0.781,
  custom_label_fraction: 0.04,
  reason: null,
}

describe('reportChecks', () => {
  it('turns a passed attempt into three passing checks for the category model', () => {
    const checks = reportChecks('category', attempt(passed))!
    expect(checks.map((c) => [c.label, c.value, c.passed])).toEqual([
      ['Learned your corrections', '9 of 10', true],
      ['Still sorts other mail well', '77.6%', true],
      ['Your labels stay specific', '4%', true],
    ])
    expect(checks[0].caption).toBe('Needs 8 of 10')
    expect(checks[0].bar).toEqual({ fill: 0.9, mark: 0.8 })
    expect(checks[1].caption).toBe('78.1% before · may drop 3 points at most')
  })

  it('marks the check that failed', () => {
    const checks = reportChecks('category', attempt({ ...passed, own_correction_accuracy: 0.6 }, { status: 'rejected' }))!
    expect(checks[0]).toMatchObject({ value: '6 of 10', passed: false })
    expect(checks[1].passed).toBe(true)
  })

  it('fails the accuracy check only past a 3 point drop', () => {
    expect(reportChecks('category', attempt({ ...passed, gate_accuracy: 0.752 }))![1].passed).toBe(true)
    expect(reportChecks('category', attempt({ ...passed, gate_accuracy: 0.74 }))![1].passed).toBe(false)
  })

  it('leaves out the labels check for models without custom labels', () => {
    expect(reportChecks('priority', attempt(passed))!.map((c) => c.label)).toEqual([
      'Learned your corrections',
      'Still sorts other mail well',
    ])
  })

  it('returns null when the attempt errored before the gate ran', () => {
    expect(reportChecks('category', attempt({ error: 'boom' }, { status: 'rejected' }))).toBeNull()
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/lib/retrain-report.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

In `personalization-api.ts`, `LastAttempt` gains `correction_count: number` after `status`.

`src/lib/retrain-report.ts`:

```ts
import type { PersonalizableModel } from '../types'
import type { LastAttempt } from './personalization-api'

// Mirrors app/backend/src/personalization/gate.py. The backend decides; these
// only explain the decision it already stored.
export const OWN_CORRECTION_MIN = 0.8
export const MAX_GATE_DROP = 0.03
export const MAX_CUSTOM_LABEL_FRACTION = 0.1

export interface ReportCheck {
  label: string
  value: string
  caption: string
  passed: boolean
  // Fill and pass mark (both 0–1) for checks drawn as a bar.
  bar?: { fill: number; mark: number }
}

interface GateMetrics {
  own_correction_accuracy: number
  gate_accuracy: number
  base_gate_accuracy: number
  custom_label_fraction: number
}

function gateMetrics(metrics: LastAttempt['metrics']): GateMetrics | null {
  const keys = ['own_correction_accuracy', 'gate_accuracy', 'base_gate_accuracy', 'custom_label_fraction'] as const
  return keys.every((k) => typeof metrics[k] === 'number') ? (metrics as unknown as GateMetrics) : null
}

const percent = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`

// The gate's checks for one attempt, or null when the job failed before the
// gate ran (its metrics then only hold an error).
export function reportChecks(model: PersonalizableModel, attempt: LastAttempt): ReportCheck[] | null {
  const m = gateMetrics(attempt.metrics)
  if (!m) return null
  const total = attempt.correction_count
  const checks: ReportCheck[] = [
    {
      label: 'Learned your corrections',
      value: `${Math.round(m.own_correction_accuracy * total)} of ${total}`,
      caption: `Needs ${Math.ceil(OWN_CORRECTION_MIN * total - 1e-9)} of ${total}`,
      passed: m.own_correction_accuracy >= OWN_CORRECTION_MIN,
      bar: { fill: m.own_correction_accuracy, mark: OWN_CORRECTION_MIN },
    },
    {
      label: 'Still sorts other mail well',
      value: percent(m.gate_accuracy),
      caption: `${percent(m.base_gate_accuracy)} before · may drop 3 points at most`,
      passed: m.gate_accuracy >= m.base_gate_accuracy - MAX_GATE_DROP,
    },
  ]
  // Only the category model can learn labels of the user's own.
  if (model === 'category') {
    checks.push({
      label: 'Your labels stay specific',
      value: percent(m.custom_label_fraction, 0),
      caption: 'of test mail went to your labels · 10% at most',
      passed: m.custom_label_fraction <= MAX_CUSTOM_LABEL_FRACTION,
    })
  }
  return checks
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npx vitest run src/lib/retrain-report.test.ts`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/src/lib/personalization-api.ts app/frontend/src/lib/retrain-report.ts app/frontend/src/lib/retrain-report.test.ts
git commit -m "Explain a retrain attempt as the gate's three checks"
```

---

### Task 3: Re-sort diffing (`resort.ts`)

**Files:**
- Create: `app/frontend/src/lib/resort.ts`
- Test: `app/frontend/src/lib/resort.test.ts`

**Interfaces:**
- Produces: `resortedCategories(before: Map<string, ClassificationResult>, after: Map<string, ClassificationResult>, correctedIds: Set<string>): Record<string, string>` — email id → category it had before.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from 'vitest'
import type { ClassificationResult } from '../types'
import { resortedCategories } from './resort'

function result(category: string): ClassificationResult {
  return {
    spam: { label: 'ham', confidence: 0.1 },
    category: { label: category, confidences: { [category]: 0.9 } },
    priority: { score: 0.3, bucket: 'low' },
    inputs: { spamConf: 0.1, vaderCompound: 0, categoryLabel: category, priorityBucket: 'low', priorityConfidences: {} },
  }
}

describe('resortedCategories', () => {
  const before = new Map([
    ['a', result('Other')],
    ['b', result('Work')],
    ['c', result('Other')],
  ])

  it('maps each email whose category changed to the category it had', () => {
    const after = new Map([
      ['a', result('LINKEDIN')],
      ['b', result('Work')],
    ])
    expect(resortedCategories(before, after, new Set())).toEqual({ a: 'Other' })
  })

  it('leaves out emails the user corrected', () => {
    const after = new Map([
      ['a', result('LINKEDIN')],
      ['c', result('Personal')],
    ])
    expect(resortedCategories(before, after, new Set(['c']))).toEqual({ a: 'Other' })
  })

  it('leaves out emails with no earlier result', () => {
    expect(resortedCategories(before, new Map([['new', result('Work')]]), new Set())).toEqual({})
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/lib/resort.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/lib/resort.ts`

```ts
import type { ClassificationResult } from '../types'

// Emails whose model category changed between two classifications (e.g.
// before and after a new personal model loaded), mapped to the category they
// had before. Corrected emails are left out: the user's correction, not the
// model, decides what they see, so nothing visibly moved.
export function resortedCategories(
  before: Map<string, ClassificationResult>,
  after: Map<string, ClassificationResult>,
  correctedIds: Set<string>,
): Record<string, string> {
  const from: Record<string, string> = {}
  for (const [id, next] of after) {
    const previous = before.get(id)
    if (!previous || correctedIds.has(id)) continue
    if (previous.category.label !== next.category.label) from[id] = previous.category.label
  }
  return from
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npx vitest run src/lib/resort.test.ts`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/src/lib/resort.ts app/frontend/src/lib/resort.test.ts
git commit -m "Work out which emails a new model re-sorted"
```

---

### Task 4: Decoding a stored correction (`receipt.ts`)

**Files:**
- Modify: `app/frontend/src/inference/features.ts` (export `Vocabs`)
- Modify: `app/frontend/src/inference/engine.ts` (add `getVocabs()`)
- Create: `app/frontend/src/lib/receipt.ts`
- Test: `app/frontend/src/lib/receipt.test.ts`

**Interfaces:**
- Produces: `interface Vocabs { spam: SpamVocab; category: CategoryVocab; priority: PriorityVocab }` (features.ts); `getVocabs(): Vocabs | null` (engine.ts); `interface WeightedTerm { term: string; weight: number }`, `interface Signal { label: string; value: string }`, `interface Receipt { words: WeightedTerm[]; pieces: WeightedTerm[]; signals: Signal[] }`, `decodeCorrection(model: PersonalizableModel, vector: SparseVector, vocabs: Vocabs): Receipt` (receipt.ts).

- [ ] **Step 1: Write the failing test**

```ts
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

const numericCols = ['n_recipients', 'sender_automated', 'text_len_log']
const vocabs: Vocabs = {
  spam: { word: vocab(['free', 'prize']), classes: [0, 1] },
  category: {
    word: vocab(['linkedin', 'profile', 'week']),
    char: vocab([' li', 'nked'], 'char_wb'),
    numeric_cols: numericCols,
    numeric_mean: [0.5, 0.2, 5],
    numeric_scale: [0.5, 0.4, 1],
    classes: ['Other', 'Personal', 'Work'],
  },
  priority: {
    word: vocab(['urgent']),
    char: vocab(['urg']),
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
    // n_recipients: log1p(1)=0.693 → scaled (0.693-0.5)/0.5; sender_automated: (1-0.2)/0.4 = 2;
    // text_len_log: log1p(99)=4.605 → scaled -0.395
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/lib/receipt.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`features.ts`, after `PriorityVocab`:

```ts
export interface Vocabs {
  spam: SpamVocab
  category: CategoryVocab
  priority: PriorityVocab
}
```

`engine.ts`, import `type Vocabs` from `./features` and add after `modelsLoaded()`:

```ts
// The fitted vocabularies, for turning a stored feature vector back into
// words (src/lib/receipt.ts). Null until loadModels() has finished.
export function getVocabs(): Vocabs | null {
  if (!spamVocab || !categoryVocab || !priorityVocab) return null
  return { spam: spamVocab, category: categoryVocab, priority: priorityVocab }
}
```

`src/lib/receipt.ts`:

```ts
import type { CategoryVocab, Vocabs } from '../inference/features'
import type { VocabData } from '../inference/tfidf'
import type { PersonalizableModel } from '../types'
import { toDense, type SparseVector } from './sparse'

// Turns a stored correction vector back into what a person can read: the
// words and letter groups it counted, and the signals it measured. The
// layouts mirror features.ts's build*Vector() concatenation order:
//   spam:     word
//   category: word | char_wb | numeric
//   priority: word | char_wb | category one-hot | numeric

export interface WeightedTerm {
  term: string
  weight: number
}

export interface Signal {
  label: string
  value: string
}

export interface Receipt {
  words: WeightedTerm[]
  pieces: WeightedTerm[]
  signals: Signal[]
}

const termLists = new WeakMap<VocabData, string[]>()

function termsOf(vocab: VocabData): string[] {
  let terms = termLists.get(vocab)
  if (!terms) {
    terms = []
    for (const [term, index] of Object.entries(vocab.vocabulary)) terms[index] = term
    termLists.set(vocab, terms)
  }
  return terms
}

function weightedTerms(vector: SparseVector, vocab: VocabData, offset: number): WeightedTerm[] {
  const terms = termsOf(vocab)
  const size = vocab.idf.length
  const out: WeightedTerm[] = []
  vector.indices.forEach((index, k) => {
    const local = index - offset
    if (local >= 0 && local < size) out.push({ term: terms[local], weight: vector.values[k] })
  })
  return out.sort((a, b) => b.weight - a.weight)
}

const yesNo = (raw: number) => (raw >= 0.5 ? 'yes' : 'no')
const count = (raw: number) => String(Math.max(0, Math.round(raw)))

// Plain-language names for the numeric columns (header_features.py,
// stylistic_features.py, priority_keyword_features.py), and how to show a
// raw value. n_recipients and text_len_log are stored as log1p.
const SIGNALS: Record<string, { label: string; show: (raw: number) => string }> = {
  n_recipients: { label: 'Recipients', show: (raw) => count(Math.expm1(raw)) },
  is_reply_or_forward: { label: 'Reply or forward', show: yesNo },
  sender_automated: { label: 'Sent from an automated address', show: yesNo },
  has_list_unsubscribe: { label: 'Has an unsubscribe link', show: yesNo },
  has_precedence_bulk: { label: 'Marked as bulk mail', show: yesNo },
  text_len_log: { label: 'Length', show: (raw) => `${count(Math.expm1(raw))} characters` },
  excl_density: { label: 'Exclamation marks', show: (raw) => `${Math.max(0, raw).toFixed(1)} per 1,000 characters` },
  caps_word_count: { label: 'Words in capitals', show: count },
  informal_greeting: { label: 'Informal greeting', show: yesNo },
  formal_greeting: { label: 'Formal greeting', show: yesNo },
  family_words: { label: 'Family words', show: yesNo },
  formal_closing: { label: 'Formal sign-off', show: yesNo },
  informal_closing: { label: 'Informal sign-off', show: yesNo },
  question_count: { label: 'Question marks', show: count },
  spam_conf: { label: 'Spam score', show: (raw) => `${Math.round(Math.min(1, Math.max(0, raw)) * 100)}%` },
  vader_compound: { label: 'Tone (−1 negative, 1 positive)', show: (raw) => raw.toFixed(2) },
  kw_high: { label: 'Urgent words', show: count },
  kw_med: { label: 'Request words', show: count },
  kw_low: { label: 'For-your-information words', show: count },
  excl_subj: { label: 'Exclamation marks in subject', show: count },
}

// A column absent from the sparse vector was stored as 0, which unscales to
// the column's mean.
function numericSignals(dense: Float32Array, start: number, vocab: CategoryVocab): Signal[] {
  return vocab.numeric_cols.map((col, i) => {
    const raw = dense[start + i] * vocab.numeric_scale[i] + vocab.numeric_mean[i]
    const signal = SIGNALS[col]
    return signal ? { label: signal.label, value: signal.show(raw) } : { label: col, value: raw.toFixed(2) }
  })
}

export function decodeCorrection(model: PersonalizableModel, vector: SparseVector, vocabs: Vocabs): Receipt {
  if (model === 'spam') {
    return { words: weightedTerms(vector, vocabs.spam.word, 0), pieces: [], signals: [] }
  }
  const vocab = model === 'category' ? vocabs.category : vocabs.priority
  const wordSize = vocab.word.idf.length
  const charSize = vocab.char.idf.length
  const dense = toDense(vector)
  const signals: Signal[] = []
  let numericStart = wordSize + charSize
  if (model === 'priority') {
    const categories = vocabs.priority.categories
    const given = categories.findIndex((_, i) => dense[numericStart + i] >= 0.5)
    if (given >= 0) signals.push({ label: 'Category it was given', value: categories[given] })
    numericStart += categories.length
  }
  signals.push(...numericSignals(dense, numericStart, vocab))
  return {
    words: weightedTerms(vector, vocab.word, 0),
    pieces: weightedTerms(vector, vocab.char, wordSize),
    signals,
  }
}
```

Check the test's "missing column" expectation by hand before running: n_recipients raw = mean 0.5 → expm1(0.5) = 0.649 → "1"; sender_automated raw = -0.5×0.4+0.2 = 0 → "no"; text_len_log raw = 5 → expm1(5) = 147.4 → "147 characters". Priority: spam_conf raw = 4×0.1+0.2 = 0.6 → "60%".

- [ ] **Step 4: Run it to verify it passes**

Run: `npx vitest run src/lib/receipt.test.ts`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/src/inference/features.ts app/frontend/src/inference/engine.ts app/frontend/src/lib/receipt.ts app/frontend/src/lib/receipt.test.ts
git commit -m "Decode a stored correction vector back into words, letter groups and signals"
```

---

### Task 5: API client and inbox context state

**Files:**
- Modify: `app/frontend/src/lib/personalization-api.ts`
- Modify: `app/frontend/src/lib/personalization-api.test.ts`
- Modify: `app/frontend/src/app/inbox/inbox-context.tsx`

**Interfaces:**
- Consumes: `resortedCategories` (Task 3).
- Produces (personalization-api.ts): `interface CorrectionSummary { provider_message_id: string; model: PersonalizableModel; corrected_label: string }`, `interface CorrectionDetail { model: PersonalizableModel; provider_message_id: string; predicted_label: string; predicted_confidence: number; corrected_label: string; feature_vector: SparseVector; created_at: string }`, `listCorrections(): Promise<CorrectionSummary[]>`, `getCorrectionDetail(messageId: string): Promise<CorrectionDetail[]>`.
- Produces (context): `type Resorted = { version: number | null; from: Record<string, string> }`; on `InboxState`: `activeVersions: Partial<Record<BackendModelName, number>>`, `resorted: Resorted | null`, `showResortedOnly: boolean`, `setShowResortedOnly: (v: boolean) => void`, `dismissResorted: () => void`.

- [ ] **Step 1: Write the failing API test** (append to `personalization-api.test.ts`; extend the import to `getCorrectionDetail, submitCorrection`)

```ts
describe('getCorrectionDetail', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('asks for one message, encoded, with credentials', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('[]', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await expect(getCorrectionDetail('a/b')).resolves.toEqual([])
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:3011/personalization/corrections/a%2Fb')
    expect(init.credentials).toBe('include')
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/lib/personalization-api.test.ts`
Expected: FAIL (`getCorrectionDetail` is not a function).

- [ ] **Step 3: Implement the API functions** (personalization-api.ts, after `submitCorrection`)

```ts
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npx vitest run src/lib/personalization-api.test.ts`
Expected: 3 passed.

- [ ] **Step 5: Add the context state** (inbox-context.tsx)

Imports: add `listCorrections` to the personalization-api import, `import { resortedCategories } from '@/lib/resort'`, and `BackendModelName` to the types import.

Below `GmailStatus`:

```ts
// Emails a newly loaded personal model moved to another category, mapped to
// the category they had before. `version` is the new category model version.
export type Resorted = { version: number | null; from: Record<string, string> }
```

`InboxState` gains:

```ts
  activeVersions: Partial<Record<BackendModelName, number>>
  resorted: Resorted | null
  showResortedOnly: boolean
  setShowResortedOnly: (v: boolean) => void
  dismissResorted: () => void
```

In `InboxProvider`, after the `retraining` state:

```ts
  const [activeVersions, setActiveVersions] = useState<Partial<Record<BackendModelName, number>>>({})
  const [resorted, setResorted] = useState<Resorted | null>(null)
  const [showResortedOnly, setShowResortedOnly] = useState(false)
```

and after `appliedVersions`: `const activeVersionsRef = useRef<Partial<Record<BackendModelName, number>>>({})`.

In `syncPersonalization`, after `appliedVersions.current = key`:

```ts
      const versions = Object.fromEntries(artifacts.map((a) => [a.model, a.version]))
      activeVersionsRef.current = versions
      setActiveVersions(versions)
```

Replace `reclassify` so it returns the new results:

```ts
  async function reclassify(): Promise<Map<string, ClassificationResult>> {
    const results = new Map<string, ClassificationResult>()
    for (const row of rowsRef.current) {
      if (!row.modelResult) continue
      const result = await classify(row)
      results.set(row.id, result)
      setRows((prev) => prev.map((r) => (r.id === row.id ? withResult(r, result) : r)))
    }
    return results
  }

  function dismissResorted() {
    setResorted(null)
    setShowResortedOnly(false)
  }

  // Saved corrections by email, so they still show after a reload. Never
  // throws: without them the inbox just shows the model's labels.
  async function savedCorrections(): Promise<Record<string, Row['corrected']>> {
    if (!personalizationRef.current?.enabled) return {}
    try {
      const byEmail: Record<string, Row['corrected']> = {}
      for (const c of await listCorrections()) {
        byEmail[c.provider_message_id] = { ...byEmail[c.provider_message_id], [c.model]: c.corrected_label }
      }
      return byEmail
    } catch {
      return {}
    }
  }
```

In `fetchAndClassify`, replace `setRows(emails.map(emptyRow))` with:

```ts
      const saved = await savedCorrections()
      setRows(emails.map((email) => ({ ...emptyRow(email), corrected: saved[email.id] ?? {} })))
```

In `watchRetrain`, replace `if (await syncPersonalization()) await reclassify()` with:

```ts
      const before = new Map(rowsRef.current.flatMap((r) => (r.modelResult ? [[r.id, r.modelResult] as const] : [])))
      const corrected = new Set(rowsRef.current.filter((r) => r.corrected.category !== undefined).map((r) => r.id))
      if (await syncPersonalization()) {
        const from = resortedCategories(before, await reclassify(), corrected)
        setShowResortedOnly(false)
        setResorted(
          Object.keys(from).length > 0 ? { version: activeVersionsRef.current.category ?? null, from } : null,
        )
      }
```

In `disablePersonalization`, after the `setRows(...)` that clears corrections, add `dismissResorted()`.

Add to the provider value: `activeVersions, resorted, showResortedOnly, setShowResortedOnly, dismissResorted,`.

- [ ] **Step 6: Type check and test**

Run: `npx tsc --noEmit && npm test`
Expected: no type errors; all tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/src/lib/personalization-api.ts app/frontend/src/lib/personalization-api.test.ts app/frontend/src/app/inbox/inbox-context.tsx
git commit -m "Track re-sorted emails and active model versions, and restore saved corrections on load"
```

---

### Task 6: Re-sorted banner, tags and filter in the list

**Files:**
- Modify: `app/frontend/src/app/inbox/inbox-list.tsx`
- Modify: `app/frontend/src/app/inbox/sidebar.tsx` (navigation clears the re-sorted filter)

**Interfaces:**
- Consumes: `resorted`, `showResortedOnly`, `setShowResortedOnly`, `dismissResorted` (Task 5).

- [ ] **Step 1: Add the tag and banner** (inbox-list.tsx)

Imports: add `RotateCcw, X` to the lucide import and `TEAL_TEXT, TEAL_TINT` to the tokens import.

After `SpamChip`:

```tsx
function ResortedTag({ from }: { from: string }) {
  return (
    <span
      className={`${CHIP_SIZE.sm} shrink-0 font-semibold`}
      style={{ background: TEAL_TINT, color: TEAL_TEXT }}
      title={`Your model moved this from ${from}`}
    >
      <RotateCcw size={11} strokeWidth={2.5} aria-hidden />
      was {from}
    </span>
  )
}

function ResortedBanner() {
  const { resorted, showResortedOnly, setShowResortedOnly, dismissResorted } = useInbox()
  if (!resorted) return null
  const count = Object.keys(resorted.from).length
  const who = resorted.version ? `Your model (version ${resorted.version})` : 'Your model'
  return (
    <div
      role="status"
      data-testid="resorted-banner"
      className="mx-4 mt-3 flex items-center gap-2.5 rounded-lg py-1.5 pr-1.5 pl-3 md:mx-5"
      style={{ background: TEAL_TINT, color: TEAL_TEXT }}
    >
      <RotateCcw size={16} strokeWidth={2} className="shrink-0" aria-hidden />
      <span className="min-w-0 flex-1 text-[13px] font-medium" style={{ color: INK }}>
        {who} re-sorted {count} {count === 1 ? 'email' : 'emails'}.
      </span>
      <button
        type="button"
        aria-pressed={showResortedOnly}
        onClick={() => setShowResortedOnly(!showResortedOnly)}
        className={`h-8 shrink-0 rounded-md px-2 text-[13px] font-semibold hover:bg-[#C4EAEC] ${FOCUS_RING}`}
      >
        {showResortedOnly ? 'Show all' : 'Show only these'}
      </button>
      <button
        type="button"
        aria-label="Dismiss"
        onClick={dismissResorted}
        className={`flex size-8 shrink-0 items-center justify-center rounded-md hover:bg-[#C4EAEC] ${FOCUS_RING}`}
      >
        <X size={14} strokeWidth={2} aria-hidden />
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Show the tag on rows**

In `EmailRow`, read `resorted` from `useInbox()` and set `const wasCategory = resorted?.from[row.id]`. After `<CategoryChip label={row.result.category.label} size="sm" />` add `{wasCategory && <ResortedTag from={wasCategory} />}`.

- [ ] **Step 3: Filter and title**

In `InboxList`, read `resorted, showResortedOnly` from `useInbox()` and change the filters:

```tsx
  const scoped = rows.filter((r) => archivedIds.has(r.id) === showArchived)
  const visibleRows = scoped
    .filter((r) => !showResortedOnly || resorted?.from[r.id] !== undefined)
    .filter((r) => showArchived || showResortedOnly || filter === 'All' || r.result?.category.label === filter)
    .filter((r) => showArchived || showResortedOnly || !onlyHigh || r.result?.priority.bucket === 'high')
    .filter((r) => matchesQuery(r, query))
    .sort((a, b) => b.receivedAtMs - a.receivedAtMs)
```

```tsx
  const title = showArchived
    ? 'Archived'
    : showResortedOnly
      ? 'Re-sorted by your model'
      : onlyHigh
        ? 'High priority'
        : filter === 'All'
          ? 'All mail'
          : filter
```

Render `<ResortedBanner />` right after `<FilterChips />`. In `FilterChips`'s chip `onClick`, also call `setShowResortedOnly(false)` (read it from `useInbox()`).

- [ ] **Step 4: Sidebar navigation leaves the re-sorted view**

In `sidebar.tsx`, read `setShowResortedOnly` from `useInbox()` and call `setShowResortedOnly(false)` inside `go()`.

- [ ] **Step 5: Type check, lint, test**

Run: `npx tsc --noEmit && npm run lint && npm test`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/src/app/inbox/inbox-list.tsx app/frontend/src/app/inbox/sidebar.tsx
git commit -m "Show which emails a new model re-sorted in the list"
```

---

### Task 7: Retrain report card in the sidebar

**Files:**
- Modify: `app/frontend/src/app/inbox/sidebar.tsx`

**Interfaces:**
- Consumes: `reportChecks`, `ReportCheck` (Task 2); `activeVersions`, `resorted`, `setShowResortedOnly` (Task 5); `LastAttempt` type.

- [ ] **Step 1: Add the components** (above `export function Sidebar()`)

Imports: `import { reportChecks, type ReportCheck } from '@/lib/retrain-report'`, `import type { LastAttempt } from '@/lib/personalization-api'`, `import type { PersonalizableModel } from '@/types'`.

```tsx
function ReportCheckRow({ check }: { check: ReportCheck }) {
  const failed = !check.passed
  return (
    <li className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={`flex items-center gap-1.5 text-[13px] ${failed ? 'font-semibold' : 'font-medium'}`}
          style={{ color: failed ? CORAL_TEXT : INK }}
        >
          {failed ? (
            <X size={13} strokeWidth={2.5} aria-hidden />
          ) : (
            <Check size={13} strokeWidth={2.5} style={{ color: TEAL_TEXT }} aria-hidden />
          )}
          <span className="sr-only">{failed ? 'Failed:' : 'Passed:'}</span>
          {check.label}
        </span>
        <span className="font-mono text-xs font-medium" style={{ color: failed ? CORAL_TEXT : INK }}>
          {check.value}
        </span>
      </div>
      {check.bar && (
        <span className="relative h-1.5 rounded-full" style={{ background: '#EBE8CC' }} aria-hidden>
          <span
            className="absolute inset-y-0 left-0 rounded-full"
            style={{ width: `${Math.min(1, check.bar.fill) * 100}%`, background: failed ? CORAL : CERULEAN }}
          />
          <span className="absolute -inset-y-[3px] w-0.5" style={{ left: `${check.bar.mark * 100}%`, background: INK }} />
        </span>
      )}
      <span className="text-[11px] leading-snug" style={{ color: MUTED }}>
        {check.caption}
      </span>
    </li>
  )
}

function RetrainReportCard({
  model,
  attempt,
  activeVersion,
  resortedCount,
  onShowResorted,
}: {
  model: PersonalizableModel
  attempt: LastAttempt
  activeVersion: number | undefined
  resortedCount: number
  onShowResorted: () => void
}) {
  const checks = reportChecks(model, attempt)
  const rejected = attempt.status === 'rejected'
  const name = model[0].toUpperCase() + model.slice(1)
  return (
    <div className="flex flex-col gap-2.5 border-t pt-2.5" style={{ borderColor: '#EFEDD6' }}>
      {rejected ? (
        <div className="flex flex-col gap-1">
          <span className="text-[13px] font-semibold" style={{ color: CORAL_TEXT }}>
            {name} version {attempt.version} not applied
          </span>
          <span className="text-xs leading-relaxed" style={{ color: '#3D5359' }}>
            {checks ? '' : `${attempt.metrics.reason ?? attempt.metrics.error ?? 'Reason not recorded.'} `}
            {activeVersion
              ? `Version ${activeVersion} is still sorting your inbox.`
              : 'The standard model is still sorting your inbox.'}
          </span>
        </div>
      ) : (
        <div className="flex flex-col gap-1">
          <span className="flex items-center gap-1.5 text-[13px] font-semibold" style={{ color: TEAL_TEXT }}>
            <Check size={15} strokeWidth={2.25} aria-hidden />
            {name} model updated
          </span>
          <span className="text-xs leading-relaxed" style={{ color: MUTED }}>
            {checks
              ? `Version ${attempt.version} passed all ${checks.length} checks and is sorting your inbox.`
              : `Version ${attempt.version} is now sorting your inbox.`}
          </span>
        </div>
      )}
      {checks && (
        <ul className="flex flex-col gap-2.5" data-testid="retrain-checks">
          {checks.map((check) => (
            <ReportCheckRow key={check.label} check={check} />
          ))}
        </ul>
      )}
      {!rejected && resortedCount > 0 && (
        <button
          type="button"
          onClick={onShowResorted}
          className={`self-start rounded-sm text-[13px] font-semibold hover:underline ${FOCUS_RING}`}
          style={{ color: CERULEAN_TEXT }}
        >
          See the {resortedCount} {resortedCount === 1 ? 'email' : 'emails'} it re-sorted
        </button>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Use it in the personalization panel**

Read `activeVersions, resorted, setShowResortedOnly` from `useInbox()`. Replace the whole `{!retraining && lastAttempt?.last_attempt && ( ... )}` block with:

```tsx
              {!retraining && lastAttempt?.last_attempt && (
                <RetrainReportCard
                  model={lastAttempt.model}
                  attempt={lastAttempt.last_attempt}
                  activeVersion={activeVersions[lastAttempt.model]}
                  resortedCount={resorted ? Object.keys(resorted.from).length : 0}
                  onShowResorted={() => {
                    go({})
                    setShowResortedOnly(true)
                  }}
                />
              )}
```

(`go({})` resets to All mail and clears the re-sorted view first, so the `setShowResortedOnly(true)` after it wins.)

- [ ] **Step 3: Type check, lint, test**

Run: `npx tsc --noEmit && npm run lint && npm test`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add app/frontend/src/app/inbox/sidebar.tsx
git commit -m "Show the retrain gate's checks in the personalization panel"
```

---

### Task 8: Receipt and model version in the email view

**Files:**
- Create: `app/frontend/src/app/inbox/correction-receipt.tsx`
- Modify: `app/frontend/src/app/inbox/email-detail.tsx`

**Interfaces:**
- Consumes: `getCorrectionDetail`, `CorrectionDetail` (Task 5); `getVocabs` (Task 4); `decodeCorrection`, `Receipt` (Task 4); `activeVersions`, `resorted` (Task 5).
- Produces: `CorrectionReceipt({ messageId: string; model: PersonalizableModel; title: string; labelFor: (value: string) => string; onClose: () => void })`.

- [ ] **Step 1: Write the receipt component** (`correction-receipt.tsx`)

```tsx
'use client'

import { useEffect, useState } from 'react'
import { Check, X } from 'lucide-react'
import { getVocabs } from '@/inference/engine'
import { getCorrectionDetail, type CorrectionDetail } from '@/lib/personalization-api'
import { decodeCorrection, type Receipt, type WeightedTerm } from '@/lib/receipt'
import type { PersonalizableModel } from '@/types'
import { CERULEAN_TEXT, CORAL_TEXT, FOCUS_RING, INK, LINE, LINE_SOFT, MUTED, SURFACE, TEAL_TEXT, TEAL_TINT } from './tokens'

const TOP_WORDS = 8
const TOP_PIECES = 12

type State =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; detail: CorrectionDetail; receipt: Receipt }

function LeaderRow({ name, value, mono = false }: { name: string; value: string; mono?: boolean }) {
  return (
    <li className="flex items-baseline gap-1.5">
      <span className={mono ? 'font-mono' : ''}>{name}</span>
      <span className="min-w-4 flex-1 border-b border-dotted" style={{ borderColor: '#B9C3C0' }} aria-hidden />
      <span className="font-mono">{value}</span>
    </li>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 border-t border-dashed px-5 py-3.5" style={{ borderColor: '#CFD6D2' }}>
      <h5 className="text-[13px] font-semibold">{title}</h5>
      {children}
    </div>
  )
}

function Words({ words }: { words: WeightedTerm[] }) {
  const [all, setAll] = useState(false)
  const shown = all ? words : words.slice(0, TOP_WORDS)
  return (
    <>
      <ul className="flex flex-col gap-0.5 text-[13px]">
        {shown.map((w) => (
          <LeaderRow key={w.term} name={w.term} value={w.weight.toFixed(3)} mono />
        ))}
      </ul>
      {words.length > TOP_WORDS && (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className={`self-start rounded-sm text-[13px] font-medium hover:underline ${FOCUS_RING}`}
          style={{ color: CERULEAN_TEXT }}
        >
          {all ? 'Show fewer' : `Show all ${words.length} words`}
        </button>
      )}
    </>
  )
}

// What the server holds for one correction, decoded on this device. The
// server sends back exactly what it stored; nothing here is recomputed from
// the email.
export function CorrectionReceipt({
  messageId,
  model,
  title,
  labelFor,
  onClose,
}: {
  messageId: string
  model: PersonalizableModel
  title: string
  labelFor: (value: string) => string
  onClose: () => void
}) {
  const [state, setState] = useState<State>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    getCorrectionDetail(messageId).then(
      (rows) => {
        if (cancelled) return
        const detail = rows.find((r) => r.model === model)
        const vocabs = getVocabs()
        if (!detail) setState({ status: 'error', message: 'The server has no saved correction for this email.' })
        else if (!vocabs) setState({ status: 'error', message: 'The models are still loading. Try again in a moment.' })
        else setState({ status: 'ready', detail, receipt: decodeCorrection(model, detail.feature_vector, vocabs) })
      },
      (err: unknown) => {
        if (!cancelled) {
          setState({ status: 'error', message: `Couldn't load what was sent: ${err instanceof Error ? err.message : err}` })
        }
      },
    )
    return () => {
      cancelled = true
    }
  }, [messageId, model])

  return (
    <section
      aria-labelledby="receipt-title"
      data-testid="correction-receipt"
      className="flex flex-col rounded-xl border"
      style={{ borderColor: LINE, background: SURFACE }}
    >
      <div className="flex items-start gap-3 px-5 pt-4 pb-3">
        <div className="flex flex-1 flex-col gap-0.5">
          <span className="text-xs font-semibold" style={{ color: MUTED }}>
            {title} correction
            {state.status === 'ready' &&
              ` · saved ${new Date(state.detail.created_at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}`}
          </span>
          <h4 id="receipt-title" className="font-display text-lg font-bold">
            What was sent to the server
          </h4>
        </div>
        <button
          type="button"
          aria-label="Close"
          onClick={onClose}
          className={`flex size-8 items-center justify-center rounded-md hover:bg-[#F6F7F2] ${FOCUS_RING}`}
          style={{ color: MUTED }}
        >
          <X size={16} strokeWidth={2} aria-hidden />
        </button>
      </div>

      {state.status === 'loading' && (
        <p className="px-5 pb-4 text-sm motion-safe:animate-pulse" style={{ color: MUTED }}>
          Loading what was sent…
        </p>
      )}
      {state.status === 'error' && (
        <p className="px-5 pb-4 text-[13px]" style={{ color: CORAL_TEXT }} role="alert">
          {state.message}
        </p>
      )}
      {state.status === 'ready' && (
        <>
          <Section title="Your correction">
            <ul className="flex flex-col gap-0.5 text-[13px]">
              <LeaderRow
                name="The model's guess"
                value={`${labelFor(state.detail.predicted_label)} · ${Math.round(state.detail.predicted_confidence * 100)}%`}
              />
              <LeaderRow name="Your answer" value={labelFor(state.detail.corrected_label)} />
              <LeaderRow name="Message ID" value={state.detail.provider_message_id} />
            </ul>
          </Section>
          <Section title="Words it counted">
            <Words words={state.receipt.words} />
          </Section>
          {state.receipt.pieces.length > 0 && (
            <Section title="Letter groups it counted">
              <div className="flex flex-wrap gap-1.5 font-mono text-xs">
                {state.receipt.pieces.slice(0, TOP_PIECES).map((p) => (
                  <span key={p.term} className="rounded px-1.5 py-0.5 whitespace-pre" style={{ background: LINE_SOFT }}>
                    {p.term}
                  </span>
                ))}
                {state.receipt.pieces.length > TOP_PIECES && (
                  <span className="px-1.5 py-0.5" style={{ color: MUTED }}>
                    + {state.receipt.pieces.length - TOP_PIECES} more
                  </span>
                )}
              </div>
              <p className="text-xs leading-relaxed" style={{ color: MUTED }}>
                Short pieces of words the model knows. They can hint at words that aren&apos;t in its word list.
              </p>
            </Section>
          )}
          {state.receipt.signals.length > 0 && (
            <Section title="Signals about the email">
              <ul className="flex flex-col gap-0.5 text-[13px]">
                {state.receipt.signals.map((s) => (
                  <LeaderRow key={s.label} name={s.label} value={s.value} />
                ))}
              </ul>
            </Section>
          )}
          <div className="m-3 mt-1 flex flex-col gap-2 rounded-lg px-4 py-3" style={{ background: TEAL_TINT }}>
            <h5 className="text-[13px] font-semibold" style={{ color: TEAL_TEXT }}>
              Never sent
            </h5>
            <ul className="grid grid-cols-1 gap-x-3 gap-y-1 text-[13px] sm:grid-cols-2" style={{ color: INK }}>
              {['The subject line', 'The email as written', 'Who sent it', 'Your other emails'].map((item) => (
                <li key={item} className="flex items-center gap-1.5">
                  <Check size={14} strokeWidth={2.25} style={{ color: TEAL_TEXT }} aria-hidden />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </section>
  )
}
```

(`React.ReactNode` needs `import type { ReactNode } from 'react'`; use `ReactNode` in `Section`'s props.)

- [ ] **Step 2: Wire it into the email view** (email-detail.tsx)

Imports: `RotateCcw` from lucide-react; `import { CorrectionReceipt } from './correction-receipt'`; `TEAL_TEXT, TEAL_TINT` already imported (add if not).

`VerdictCell` gains an optional `onShowSent?: () => void` and `showingSent: boolean` and renders the actions row:

```tsx
      {(onCorrect || onShowSent) && (
        <div className="mt-auto flex flex-wrap items-center gap-3">
          {onCorrect && ( /* the existing Correct button, minus its mt-auto */ )}
          {onShowSent && (
            <button
              type="button"
              onClick={onShowSent}
              aria-expanded={showingSent}
              data-testid={`${testId}-sent`}
              className={`h-8 rounded-sm text-[13px] font-medium hover:underline ${FOCUS_RING}`}
              style={{ color: CERULEAN_TEXT }}
            >
              What was sent?
            </button>
          )}
        </div>
      )}
```

In `EmailDetail`, read `activeVersions, resorted` from `useInbox()` and add state `const [receiptFor, setReceiptFor] = useState<{ id: string; model: PersonalizableModel } | null>(null)`. Then:

```tsx
  const showingReceipt = receiptFor?.id === row.id ? receiptFor.model : null
  const toggleReceipt = (model: PersonalizableModel) => {
    setPicking(null)
    setOptInFor(null)
    setReceiptFor(showingReceipt === model ? null : { id: row.id, model })
  }
```

`startCorrecting` also calls `setReceiptFor(null)` first. `toggleSorted`, when collapsing, also calls `setReceiptFor(null)`. In `correctionPanel()`, before `return null`:

```tsx
    if (showingReceipt) {
      return (
        <CorrectionReceipt
          key={`${row.id}:${showingReceipt}`}
          messageId={row.id}
          model={showingReceipt}
          title={MODEL_TITLES[showingReceipt]}
          labelFor={(value) => optionLabel(showingReceipt, value)}
          onClose={() => setReceiptFor(null)}
        />
      )
    }
```

Each `VerdictCell` gets, e.g. for category:
`onShowSent={canCorrect && row.corrected.category !== undefined ? () => toggleReceipt('category') : undefined}` and `showingSent={showingReceipt === 'category'}` (same for priority and spam).

Header: replace the "Sorted on this device" span with:

```tsx
            <span className="flex flex-wrap items-center justify-end gap-x-3 gap-y-1 text-xs" style={{ color: MUTED }}>
              {activeVersions.category && (
                <span
                  className="rounded-full px-2 py-0.5 text-[11px] font-semibold"
                  style={{ background: TEAL_TINT, color: TEAL_TEXT }}
                >
                  Your model · version {activeVersions.category}
                </span>
              )}
              <span className="flex items-center gap-1.5">
                <Cpu size={14} strokeWidth={1.75} aria-hidden />
                Sorted on this device
              </span>
            </span>
```

Category cell, after `<CategoryVerdict ... />`:

```tsx
                  {resorted?.from[row.id] && (
                    <span className="flex items-center gap-1 text-xs" style={{ color: TEAL_TEXT }}>
                      <RotateCcw size={12} strokeWidth={2.25} aria-hidden />
                      Was {resorted.from[row.id]} before this update
                    </span>
                  )}
```

- [ ] **Step 3: Type check, lint, test**

Run: `npx tsc --noEmit && npm run lint && npm test`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add app/frontend/src/app/inbox/correction-receipt.tsx app/frontend/src/app/inbox/email-detail.tsx
git commit -m "Show what a correction sent, and which model version sorted an email"
```

---

### Task 9: Verify in the running app and document

**Files:**
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Start both dev servers** (background): `npm run dev:be` and `npm run dev` from the repo root.

- [ ] **Step 2: Check without making new corrections** (a Playwright script in the scratchpad, against the user's connected session, read-only):
  - The inbox loads, and emails the user corrected earlier show the "Corrected" badge again after a reload.
  - "What was sent?" on a corrected category cell opens the receipt: guess, answer, message ID, words with weights, letter groups, signals, "Never sent".
  - The sidebar shows the report card for the latest attempt with its checks.
  - No page errors. Screenshots at 1440 and 390 wide.
  - The re-sorted banner needs a real retrain; it is covered by the unit tests. Do not trigger a retrain on the user's account without asking.

- [ ] **Step 3: Document** — add a "Show the work" section to `PROJECT_STATUS.md`: the receipt (and the letter-group privacy note), the report card, the re-sorted banner, and corrections surviving reloads.

- [ ] **Step 4: Full checks and commit**

Run: backend pytest, `npm test`, `npx tsc --noEmit`, `npm run lint`.

```bash
git add PROJECT_STATUS.md
git commit -m "Document the receipt, retrain report card and re-sorted view"
```

- [ ] **Step 5: Stop the dev servers.**
