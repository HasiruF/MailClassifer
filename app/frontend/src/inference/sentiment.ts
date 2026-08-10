import { SentimentIntensityAnalyzer } from 'vader-sentiment'

// scripts/classify_email.py runs VADER (Python's vaderSentiment) on the same
// cleaned text (lowercase, alpha + single spaces only — see cleanText in
// features.ts) used everywhere else in the pipeline, not the raw body. This
// JS port (vader-sentiment npm package) implements the same published VADER
// algorithm/lexicon independently, so scores should track closely but are
// not guaranteed bit-identical to the Python side.
export function vaderCompound(cleanedText: string): number {
  return SentimentIntensityAnalyzer.polarity_scores(cleanedText).compound
}
