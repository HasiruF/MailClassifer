import { describe, expect, it } from 'vitest'
import { extractBodies } from './gmail-fetch'

function b64url(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let binary = ''
  bytes.forEach((b) => (binary += String.fromCharCode(b)))
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

describe('extractBodies', () => {
  it('keeps the plain-text part for classification and the HTML part for display', () => {
    const payload = {
      mimeType: 'multipart/alternative',
      parts: [
        { mimeType: 'text/plain', body: { data: b64url('Hello there') } },
        { mimeType: 'text/html', body: { data: b64url('<p>Hello <b>there</b></p>') } },
      ],
    }
    expect(extractBodies(payload)).toEqual({ text: 'Hello there', html: '<p>Hello <b>there</b></p>' })
  })

  it('falls back to stripped HTML for the text when there is no plain part', () => {
    const payload = { mimeType: 'text/html', body: { data: b64url('<p>Only&nbsp;<i>HTML</i></p>') } }
    expect(extractBodies(payload)).toEqual({ text: 'Only HTML', html: '<p>Only&nbsp;<i>HTML</i></p>' })
  })

  it('has no HTML for a plain-text-only message', () => {
    const payload = { mimeType: 'text/plain', body: { data: b64url('Just text') } }
    expect(extractBodies(payload)).toEqual({ text: 'Just text', html: undefined })
  })

  it('finds parts nested inside multipart/related', () => {
    const payload = {
      mimeType: 'multipart/mixed',
      parts: [
        {
          mimeType: 'multipart/related',
          parts: [{ mimeType: 'text/html', body: { data: b64url('<h1>Deep</h1>') } }],
        },
      ],
    }
    expect(extractBodies(payload).html).toBe('<h1>Deep</h1>')
  })
})
