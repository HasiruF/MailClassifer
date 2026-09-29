import { describe, expect, it } from 'vitest'
import { shortLink, splitLinks, tidyPlainText, withShortLinks } from './links'

const TRACKING =
  'https://www.linkedin.com/comm/feed/?lipi=urn%3Ali%3Apage%3Aemail_email_weekly_analytics&midToken=AQFAzbJ9KhZ2jg'

describe('shortLink', () => {
  it('keeps the host and path, drops www and marks a dropped query with an ellipsis', () => {
    expect(shortLink(TRACKING)).toBe('linkedin.com/comm/feed…')
  })

  it('shows a bare host as-is', () => {
    expect(shortLink('https://example.com/')).toBe('example.com')
  })

  it('truncates a very long path', () => {
    expect(shortLink('https://example.com/a/very/long/path/that/keeps/going/and/going')).toBe(
      'example.com/a/very/long/path/that/k…',
    )
  })
})

describe('splitLinks', () => {
  it('separates links from the text around them', () => {
    expect(splitLinks(`See ${TRACKING} for details`)).toEqual([
      { type: 'text', text: 'See ' },
      { type: 'link', href: TRACKING, label: 'linkedin.com/comm/feed…' },
      { type: 'text', text: ' for details' },
    ])
  })

  it('leaves trailing sentence punctuation out of the link', () => {
    expect(splitLinks('Go to https://example.com/docs.')).toEqual([
      { type: 'text', text: 'Go to ' },
      { type: 'link', href: 'https://example.com/docs', label: 'example.com/docs' },
      { type: 'text', text: '.' },
    ])
  })

  it('returns plain text untouched', () => {
    expect(splitLinks('No links here.')).toEqual([{ type: 'text', text: 'No links here.' }])
  })
})

describe('withShortLinks', () => {
  it('rewrites every link in a snippet to its short form', () => {
    expect(withShortLinks(`Label: ${TRACKING} and https://example.com/`)).toBe(
      'Label: linkedin.com/comm/feed… and example.com',
    )
  })
})

describe('tidyPlainText', () => {
  it('collapses runs of blank lines into one and trims trailing spaces', () => {
    expect(tidyPlainText('Hi   \n\n\n\n  \nThanks\n\n\n')).toBe('Hi\n\nThanks')
  })

  it('handles CRLF line endings', () => {
    expect(tidyPlainText('Hi\r\n\r\n\r\n\r\nThanks')).toBe('Hi\n\nThanks')
  })
})
