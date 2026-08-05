"""
eval_category_gold.py
──────────────────────
Evaluates the saved category_classifier_headers.joblib against the
human/LLM-labeled gold sample (gold_sample_labeled.csv, column
`true_category`), reporting accuracy, per-class precision/recall/F1, and a
confusion matrix. Builds the same text+header-feature input shape the model
was trained on (see train_category_classifier_headers.py) — this model does
NOT accept plain text, it needs the NUMERIC_COLS columns too.

Usage
  python eval_category_gold.py \
      --gold    ../../gold_sample_labeled.csv \
      --headers gold_header_features.csv \
      --model   ../../models/category_classifier_headers.joblib
"""

import argparse
import sys
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here)
sys.path.insert(0, _here + '\\..\\shared')
from train_category_classifier import clean_text
from train_category_classifier_headers import NUMERIC_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', default='../../gold_sample_labeled.csv')
    ap.add_argument('--headers', default='gold_header_features.csv')
    ap.add_argument('--model', default='../../models/category_classifier_headers.joblib')
    args = ap.parse_args()

    df = pd.read_csv(args.gold)
    df = df[df['true_category'].notna() & (df['true_category'].str.strip() != '')].copy()
    print(f"Loaded {len(df):,} gold-labeled rows from {args.gold}")

    df['text'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).apply(clean_text)
    headers = pd.read_csv(args.headers)
    df = df.merge(headers, on='file', how='left')
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)
    text = df[['text'] + NUMERIC_COLS]

    pipeline = joblib.load(args.model)
    preds = pipeline.predict(text)

    y_true = df['true_category']
    acc = accuracy_score(y_true, preds)

    print(f"\nOverall accuracy: {acc:.4f}  ({(y_true == preds).sum()}/{len(y_true)} correct)\n")

    labels = sorted(set(y_true) | set(preds))
    print(classification_report(y_true, preds, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, preds, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    header = '  ' + '  '.join(f'{c:>14s}' for c in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row_str = '  '.join(f'{v:>14d}' for v in cm[i])
        print(f"  {row_label:14s}  {row_str}")

    # Also compare against the model's own earlier prediction stored in predicted_category,
    # to see if it's stable/deterministic.
    if 'predicted_category' in df.columns:
        stale_match = (df['predicted_category'] == preds).mean()
        print(f"\nAgreement with originally-sampled 'predicted_category' column: {stale_match:.4f}")

    out_path = args.gold.replace('.csv', '_scored.csv') if args.gold.endswith('.csv') else args.gold + '_scored.csv'
    df_out = df.copy()
    df_out['model_prediction'] = preds
    df_out['correct'] = (y_true.values == preds)
    df_out.to_csv(out_path, index=False)
    print(f"\nPer-row scored output -> {out_path}")


if __name__ == '__main__':
    main()
