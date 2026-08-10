import { describe, it, expect } from 'vitest'
import { extractStylisticFeatures } from './stylistic'

describe('extractStylisticFeatures', () => {
  it('detects informal greeting, a caps-word, a question, an exclamation', () => {
    const subject = 'Hi'
    const body = 'Hey John, ETA is 5pm? See you soon!'
    const full = `${subject} ${body}`
    expect(extractStylisticFeatures(subject, body)).toEqual({
      text_len_log: Math.log1p(full.length),
      excl_density: (1 / full.length) * 1000,
      caps_word_count: 1,
      informal_greeting: 1,
      formal_greeting: 0,
      family_words: 0,
      formal_closing: 0,
      informal_closing: 0,
      question_count: 1,
    })
  })

  it('detects formal greeting, family word, formal closing', () => {
    const subject = 'Following up'
    const body = 'Dear Sir, mother asked me to relay this. Best regards, Alex'
    const full = `${subject} ${body}`
    expect(extractStylisticFeatures(subject, body)).toEqual({
      text_len_log: Math.log1p(full.length),
      excl_density: 0,
      caps_word_count: 0,
      informal_greeting: 0,
      formal_greeting: 1,
      family_words: 1,
      formal_closing: 1,
      informal_closing: 0,
      question_count: 0,
    })
  })

  it('clamps caps_word_count to 10 and question_count to 5', () => {
    const result = extractStylisticFeatures(
      '',
      'AAA BBB CCC DDD EEE FFF GGG HHH III JJJ KKK ???????',
    )
    expect(result.caps_word_count).toBe(10)
    expect(result.question_count).toBe(5)
  })
})
