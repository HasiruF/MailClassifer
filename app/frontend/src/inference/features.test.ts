import { describe, it, expect } from 'vitest'
import { extractHeaderFeatures, extractKeywordFeatures, cleanText } from './features'
import type { EmailInput } from '../types'

describe('extractHeaderFeatures', () => {
  it('counts recipients across to+cc, flags reply and automated sender', () => {
    const email: EmailInput = {
      id: '1',
      subject: 'Re: Meeting',
      body: '',
      to: 'a@b.com,c@d.com',
      cc: 'd@e.com',
      fromAddr: 'noreply@company.com',
      listUnsubscribe: true,
      precedence: 'bulk',
    }
    expect(extractHeaderFeatures(email)).toEqual({
      nRecipients: 3,
      isReplyOrForward: 1,
      senderAutomated: 1,
      hasListUnsubscribe: 1,
      hasPrecedenceBulk: 1,
    })
  })

  it('defaults to zero/false for a plain personal email', () => {
    const email: EmailInput = {
      id: '2',
      subject: 'Lunch?',
      body: '',
      to: 'friend@gmail.com',
      fromAddr: 'me@gmail.com',
    }
    expect(extractHeaderFeatures(email)).toEqual({
      nRecipients: 1,
      isReplyOrForward: 0,
      senderAutomated: 0,
      hasListUnsubscribe: 0,
      hasPrecedenceBulk: 0,
    })
  })
})

describe('extractKeywordFeatures', () => {
  it('counts high-urgency keywords', () => {
    expect(extractKeywordFeatures('URGENT', 'This is asap and urgent, deadline approaching')).toEqual({
      kwHigh: 4,
      kwMed: 0,
      kwLow: 0,
      exclSubj: 0,
    })
  })

  it('counts medium-urgency keywords and subject exclamations', () => {
    expect(extractKeywordFeatures('Please respond!', 'follow up needed, reminder')).toEqual({
      kwHigh: 0,
      kwMed: 3,
      kwLow: 0,
      exclSubj: 1,
    })
  })

  it('counts low-urgency keywords', () => {
    expect(
      extractKeywordFeatures('Newsletter', 'Please unsubscribe if not interested. This is fyi only.'),
    ).toEqual({ kwHigh: 0, kwMed: 0, kwLow: 3, exclSubj: 0 })
  })
})

describe('cleanText', () => {
  it('lowercases, strips punctuation/digits, collapses whitespace', () => {
    expect(cleanText('Hello, World! Call 555-1234.')).toBe('hello world call')
  })

  it('replaces URLs and emails with placeholder tokens', () => {
    expect(cleanText('Visit https://example.com or email me@x.com')).toBe(
      'visit url or email email',
    )
  })
})
