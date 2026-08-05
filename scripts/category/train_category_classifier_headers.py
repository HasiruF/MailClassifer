"""
train_category_classifier_headers.py
──────────────────────────────────────
The category classifier: TF-IDF word n-grams + TF-IDF char n-grams (captures
tone/register — informal spelling, punctuation habits, ALL-CAPS — that word
n-grams alone miss, which is exactly what separates Personal from Work) +
domain-agnostic header/metadata features (see header_features.py): recipient
count, reply/forward flag, automated-sender pattern, and List-Unsubscribe/
Precedence:bulk presence (the latter two are structurally absent from this
~2001 Enron corpus but included so the feature set is portable to modern mail
where they do appear).

5-fold CV accuracy on the 1000 gold-labeled rows, each addition on top of the
last:
  text only (word n-grams)              60.8%
  + header/metadata features            61.9%
  + char n-grams                        67.8%
  + taxonomy fix (Mailing List->Other)  73.6%
  + one verified label correction       74.1%
  + stylistic features                  76.8%   <- current

Usage
  python train_category_classifier_headers.py \
      --gold    ../gold_sample_labeled.csv \
      --headers gold_header_features.csv \
      --out     ..
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here)
sys.path.insert(0, _here + '\\..\\shared')
from train_category_classifier import clean_text
from header_features import log1p_recipients
from stylistic_features import extract_stylistic_features, STYLE_COLS

NUMERIC_COLS = ['n_recipients', 'is_reply_or_forward', 'sender_automated',
                 'has_list_unsubscribe', 'has_precedence_bulk'] + STYLE_COLS
# hour_sin/hour_cos/is_weekend/is_business_hours were tried and dropped — see
# PROJECT_STATUS.md. Enron employees sent personal chatter and work mail in
# the same 9-5 window, so time-of-day carried ~no separating signal on this
# corpus (is_business_hours was even *higher* for Other than for Work).
# header_features.py still computes them (harmless, might help on a modern
# Gmail-pattern mailbox with real evening/weekend usage split) — just not
# wired into this model's feature set.
#
# Sender-recipient relationship-frequency features (how often this From/To
# pair corresponds elsewhere in the corpus) were also tried and dropped —
# net negative when combined with stylistic features (75.7% vs 76.8%), only
# a small standalone gain (74.9%) that doesn't survive combination. See
# PROJECT_STATUS.md for the comparison table.


def make_pipeline():
    text_pipe = Pipeline([
        ('feats', FeatureUnion([
            ('word', TfidfVectorizer(
                max_features=20_000, ngram_range=(1, 2), min_df=2,
                sublinear_tf=True, stop_words='english',
            )),
            ('char', TfidfVectorizer(
                max_features=8_000, ngram_range=(3, 5), min_df=3,
                sublinear_tf=True, analyzer='char_wb',
            )),
        ])),
    ])
    numeric_pipe = Pipeline([
        ('log1p', FunctionTransformer(log1p_recipients)),
        ('scale', StandardScaler()),
    ])
    pre = ColumnTransformer([
        ('text', text_pipe, 'text'),
        ('numeric', numeric_pipe, NUMERIC_COLS),
    ])
    return Pipeline([
        ('features', pre),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=3.0)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', default='../../gold_sample_labeled.csv')
    ap.add_argument('--headers', default='gold_header_features.csv')
    ap.add_argument('--out', default='../..')
    args = ap.parse_args()

    gold = pd.read_csv(args.gold)
    gold = gold[gold['true_category'].notna() & (gold['true_category'].str.strip() != '')].copy()
    gold['subject'] = gold['subject'].fillna('')
    gold['body'] = gold['body'].fillna('')
    gold['text'] = (gold['subject'] + ' ' + gold['body']).apply(clean_text)

    style_feats = gold.apply(
        lambda row: pd.Series(extract_stylistic_features(row['subject'], row['body'])), axis=1)
    gold = pd.concat([gold, style_feats], axis=1)

    headers = pd.read_csv(args.headers)
    df = gold.merge(headers, on='file', how='left')
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    X = df[['text'] + NUMERIC_COLS]
    y = df['true_category']

    print(f"Loaded {len(df):,} rows with text + header features")
    print("\n[1/2] 5-fold CV: text + header features")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(make_pipeline(), X, y, cv=cv)
    acc = accuracy_score(y, pred)
    print(f"    Out-of-fold accuracy: {acc:.4f}")
    labels = sorted(y.unique())
    print(classification_report(y, pred, labels=labels, zero_division=0))
    cm = confusion_matrix(y, pred, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    header_row = '  ' + '  '.join(f'{c:>14s}' for c in labels)
    print(header_row)
    for i, row_label in enumerate(labels):
        row_str = '  '.join(f'{v:>14d}' for v in cm[i])
        print(f"  {row_label:14s}  {row_str}")

    print("\n[2/2] Fitting final model on all rows …")
    final_pipeline = make_pipeline()
    final_pipeline.fit(X, y)

    model_dir = os.path.join(args.out, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'category_classifier_headers.joblib')
    joblib.dump(final_pipeline, model_path)
    print(f"Model saved -> {model_path}")


if __name__ == '__main__':
    main()
