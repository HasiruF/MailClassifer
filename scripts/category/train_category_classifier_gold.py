"""
train_category_classifier_gold.py
──────────────────────────────────
Category classifier trained directly on the real gold labels
(gold_sample_labeled.csv, column `true_category`) instead of v1's
folder-name weak supervision.

v1 trained on which mail folder an email sat in (inbox -> Work, etc.).
Checked against the 540 hand-verified gold labels, that folder heuristic
was only 45% accurate. Training on the gold labels' actual text instead
gets 5-fold cross-validated accuracy to ~76%.

Outputs
  models/category_classifier_gold.joblib

Usage
  python train_category_classifier_gold.py \
      --gold ../gold_sample_labeled.csv \
      --out  ..
"""

import argparse
import os
import sys

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here)
sys.path.insert(0, _here + '\\..\\shared')
from train_category_classifier import clean_text


def make_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words='english',
        )),
        ('clf', LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            C=1.0,
        )),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', default='../../gold_sample_labeled.csv')
    ap.add_argument('--out', default='../..')
    args = ap.parse_args()

    df = pd.read_csv(args.gold)
    df = df[df['true_category'].notna() & (df['true_category'].str.strip() != '')].copy()
    df['text'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).apply(clean_text)
    print(f"Loaded {len(df):,} gold-labeled rows, classes: {sorted(df['true_category'].unique())}")

    print("\n[1/2] 5-fold cross-validated evaluation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = cross_val_predict(make_pipeline(), df['text'], df['true_category'], cv=cv)

    acc = accuracy_score(df['true_category'], oof_pred)
    print(f"    Out-of-fold accuracy: {acc:.4f}")
    labels = sorted(df['true_category'].unique())
    print(classification_report(df['true_category'], oof_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(df['true_category'], oof_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    header = '  ' + '  '.join(f'{c:>14s}' for c in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row_str = '  '.join(f'{v:>14d}' for v in cm[i])
        print(f"  {row_label:14s}  {row_str}")

    print("\n[2/2] Fitting final model on all 540 gold rows …")
    final_pipeline = make_pipeline()
    final_pipeline.fit(df['text'], df['true_category'])

    model_dir = os.path.join(args.out, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'category_classifier_gold.joblib')
    joblib.dump(final_pipeline, model_path)
    print(f"Model saved -> {model_path}")
    print("\nNote: the CV accuracy above (not resubstitution accuracy on the")
    print("saved model) is the honest estimate of how this model performs on unseen emails.")


if __name__ == '__main__':
    main()
