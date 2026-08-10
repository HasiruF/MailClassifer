// Headless-browser smoke check for the classify demo page — drives it with
// Playwright directly (no chromium-cli in this environment) since this is
// the one thing Node-based checks (e2e_check.mjs) can't catch: bugs that
// only manifest when onnxruntime-web actually runs in a browser context.
// Concretely, this caught a real bug that Node-side testing couldn't:
// classify()'s priority classifier + regressor .run() calls raced via
// Promise.all, which onnxruntime-web's WASM backend doesn't tolerate
// (throws "Session already started") — invisible to onnxruntime-node,
// which apparently doesn't enforce the same constraint.
//
// Usage: start the dev server first (npm run dev), then:
//   node scripts/browser_check.mjs
import { chromium } from 'playwright'
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const SHOT_DIR = join(dirname(fileURLToPath(import.meta.url)), '../.diag')
fs.mkdirSync(SHOT_DIR, { recursive: true })

const browser = await chromium.launch()
const page = await browser.newPage()

const consoleMessages = []
page.on('console', (msg) => consoleMessages.push(`[${msg.type()}] ${msg.text()}`))
page.on('pageerror', (err) => consoleMessages.push(`[pageerror] ${err.message}\n${err.stack ?? ''}`))

console.log('Navigating...')
await page.goto('http://localhost:3000', { waitUntil: 'domcontentloaded' })

// Wait for either the "ready" card or an error message to appear
await page.waitForFunction(
  () => document.body.innerText.includes('Classify an email') || document.body.innerText.includes('failed to load'),
  { timeout: 20000 },
).catch(() => console.log('TIMEOUT waiting for ready/error state'))

await page.screenshot({ path: `${SHOT_DIR}/01-loaded.png`, fullPage: true })
console.log('--- body text after load ---')
console.log(await page.evaluate(() => document.body.innerText))

console.log('\n--- console messages so far ---')
consoleMessages.forEach((m) => console.log(m))

const hasClassifyButton = await page.locator('button:has-text("Classify")').count()
if (hasClassifyButton > 0) {
  console.log('\nClicking Classify...')
  await page.locator('button:has-text("Classify")').click()
  await page.waitForFunction(
    () => document.body.innerText.includes('Priority:') || document.querySelector('.text-red-600'),
    { timeout: 30000 },
  ).catch(() => console.log('TIMEOUT waiting for classify result'))
  await page.screenshot({ path: `${SHOT_DIR}/02-after-classify.png`, fullPage: true })
  console.log('--- body text after classify ---')
  console.log(await page.evaluate(() => document.body.innerText))

  console.log('\n--- all console messages ---')
  consoleMessages.forEach((m) => console.log(m))
} else {
  console.log('\nNo Classify button found — page did not reach ready state.')
}

await browser.close()
