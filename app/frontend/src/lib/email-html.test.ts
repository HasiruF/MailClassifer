import { describe, expect, it } from 'vitest'
import { buildEmailDocument } from './email-html'

function csp(doc: string): string {
  const match = doc.match(/http-equiv="Content-Security-Policy" content="([^"]+)"/)
  if (!match) throw new Error('no CSP meta tag')
  return match[1]
}

describe('buildEmailDocument', () => {
  const html = '<table><tr><td><img src="https://cdn.example.com/logo.png">Hello</td></tr></table>'

  it('embeds the email HTML in a document of its own', () => {
    const doc = buildEmailDocument(html, { blockImages: false })
    expect(doc.startsWith('<!doctype html>')).toBe(true)
    expect(doc).toContain(html)
  })

  it('opens links in a new tab', () => {
    expect(buildEmailDocument(html, { blockImages: false })).toContain('<base target="_blank">')
  })

  it('forbids scripts, frames, plugins and form submission whatever the email contains', () => {
    const policy = csp(buildEmailDocument(html, { blockImages: false }))
    expect(policy).toContain("default-src 'none'")
    expect(policy).toContain("form-action 'none'")
    expect(policy).not.toMatch(/script-src/)
  })

  it('allows remote images by default and blocks them when asked', () => {
    const imgSrc = (blockImages: boolean) =>
      csp(buildEmailDocument(html, { blockImages }))
        .split('; ')
        .find((d) => d.startsWith('img-src'))
    expect(imgSrc(false)).toBe('img-src https: http: data: cid:')
    expect(imgSrc(true)).toBe('img-src data:')
  })
})
