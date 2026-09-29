// End-to-end personalization check against the real app (spec §12).
// Prereqs: Docker up; backend (npm run dev:be) and frontend (npm run dev:fe)
// running; Gmail connected at least once; training_data/ generated.
//
// Usage (from app/frontend):
//   node scripts/personalization_e2e.mjs <dmp_session cookie value> [--cleanup]
//
// Corrects 5 emails' category to Work through the UI, waits for the
// automatic retrain, reloads, and reports what changed. It asserts the
// mechanics (corrections saved, a retrain attempt recorded, the app reloads
// cleanly); how many other emails flip depends on the real inbox, so that
// part is reported, not asserted.
import { chromium } from 'playwright'

const [sessionId, ...flags] = process.argv.slice(2)
if (!sessionId) {
  console.error('usage: node scripts/personalization_e2e.mjs <dmp_session cookie value> [--cleanup]')
  process.exit(2)
}
const BACKEND = 'http://localhost:3011'
const INBOX = 'http://localhost:3010/inbox'
const LABEL = 'Work'
const N = 5

const browser = await chromium.launch()
const context = await browser.newContext()
await context.addCookies([{ name: 'dmp_session', value: sessionId, domain: 'localhost', path: '/', httpOnly: true }])
const page = await context.newPage()
const api = context.request

async function status() {
  const res = await api.get(`${BACKEND}/personalization/status`)
  if (!res.ok()) throw new Error(`status failed: ${res.status()}`)
  return res.json()
}

async function waitForClassifiedInbox() {
  await page.goto(INBOX)
  await page.waitForFunction(() => document.body.innerText.includes('GMAIL CONNECTED'), null, { timeout: 90_000 })
  await page.waitForFunction(
    () => {
      const rows = [...document.querySelectorAll('[data-testid="email-row"]')]
      return rows.length > 0 && rows.every((r) => r.dataset.category)
    },
    null,
    { timeout: 120_000 },
  )
}

async function categories() {
  return page.$$eval('[data-testid="email-row"]', (rows) =>
    Object.fromEntries(rows.map((r) => [r.dataset.emailId, r.dataset.category])),
  )
}

try {
  const optIn = await api.put(`${BACKEND}/personalization/settings`, { data: { enabled: true } })
  if (!optIn.ok()) throw new Error(`opt-in failed: ${optIn.status()}`)
  const attemptsBefore = (await status()).models.find((m) => m.model === 'category').last_attempt?.version ?? 0

  await waitForClassifiedInbox()
  const before = await categories()
  const targets = Object.keys(before).filter((id) => before[id] !== LABEL).slice(0, N)
  if (targets.length < N) throw new Error(`need ${N} emails not already ${LABEL}, found ${targets.length}`)

  for (const id of targets) {
    await page.locator(`[data-email-id="${id}"] button`).first().click()
    await page.getByTestId('correct-category').click()
    await page.getByTestId(`correction-option-${LABEL}`).click()
    await Promise.all([
      page.waitForResponse((r) => r.url().endsWith('/personalization/corrections') && r.ok()),
      page.getByTestId('correction-save').click(),
    ])
  }
  console.log(`Corrected ${targets.length} emails to ${LABEL}.`)

  let attempt = null
  for (let i = 0; i < 60 && !attempt; i++) {
    await page.waitForTimeout(2000)
    const category = (await status()).models.find((m) => m.model === 'category')
    if (!category.running && (category.last_attempt?.version ?? 0) > attemptsBefore) attempt = category.last_attempt
  }
  if (!attempt) throw new Error('no retrain attempt recorded within 120s')
  console.log(`Retrain v${attempt.version}: ${attempt.status}`, JSON.stringify(attempt.metrics))

  await waitForClassifiedInbox()
  const after = await categories()
  const correctedNow = targets.filter((id) => after[id] === LABEL).length
  const others = Object.keys(after).filter((id) => !targets.includes(id))
  const flipped = others.filter((id) => before[id] !== after[id])
  console.log(`After reload: ${correctedNow}/${targets.length} corrected emails now predicted ${LABEL} by the model.`)
  console.log(`Other emails whose category changed: ${flipped.length}/${others.length}`)
  for (const id of flipped) console.log(`  ${id}: ${before[id]} -> ${after[id]}`)

  if (flags.includes('--cleanup')) {
    await api.put(`${BACKEND}/personalization/settings`, { data: { enabled: false } })
    console.log('Cleaned up: personalization turned off, corrections and models deleted.')
  }
} catch (err) {
  console.error(`E2E FAILED: ${err instanceof Error ? err.message : err}`)
  await page.screenshot({ path: 'personalization_e2e_failure.png', fullPage: true })
  process.exitCode = 1
} finally {
  await browser.close()
}
