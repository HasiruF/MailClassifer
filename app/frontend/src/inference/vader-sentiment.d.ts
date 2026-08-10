declare module 'vader-sentiment' {
  interface PolarityScores {
    neg: number
    neu: number
    pos: number
    compound: number
  }
  export const SentimentIntensityAnalyzer: {
    polarity_scores(text: string): PolarityScores
  }
}
