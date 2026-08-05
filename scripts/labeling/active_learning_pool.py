"""
active_learning_pool.py
────────────────────────
Scores every email in the Enron corpus (excluding ones already gold-labeled)
with the current category model, and writes out the N lowest-confidence rows
as a new labeling pool — same shape as gold_sample_unlabeled.csv so it drops
straight into export_batch.py.

Note: the current model (category_classifier_headers.joblib) needs text +
header/metadata features, not plain text — see ../category/train_category_classifier_headers.py.

Usage
  python active_learning_pool.py \
      --model  ../../models/category_classifier_headers.joblib \
      --gold   ../../gold_sample_labeled.csv \
      --enron  ../../Enron/emails.csv \
      --n      500 \
      --out    active_pool.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
import joblib

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here)
sys.path.insert(0, _here + '\\..\\shared')
sys.path.insert(0, _here + '\\..\\category')
from train_category_classifier import clean_text, parse_enron_message
from header_features import extract_header_features
from train_category_classifier_headers import NUMERIC_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='../../models/category_classifier_headers.joblib')
    ap.add_argument('--gold', default='../../gold_sample_labeled.csv')
    ap.add_argument('--enron', default='../../Enron/emails.csv')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--out', default='active_pool.csv')
    args = ap.parse_args()

    pipeline = joblib.load(args.model)

    gold = pd.read_csv(args.gold)
    already_labeled = set(gold['file'])
    print(f"Already gold-labeled: {len(already_labeled):,} files (excluded from scoring)")

    rows = []
    total = 0
    for chunk in pd.read_csv(args.enron, chunksize=50_000, on_bad_lines='skip'):
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        chunk = chunk[~chunk['file'].isin(already_labeled)]
        total += len(chunk)

        parsed = chunk['message'].apply(parse_enron_message)
        subj = parsed.apply(lambda p: p[0])
        body = parsed.apply(lambda p: p[1])
        texts = (subj + ' ' + body).apply(clean_text)

        feats = pd.DataFrame([
            extract_header_features(msg, subject=s) for msg, s in zip(chunk['message'], subj)
        ])
        model_input = pd.DataFrame({'text': texts.values})
        for col in NUMERIC_COLS:
            model_input[col] = feats[col].values

        proba = pipeline.predict_proba(model_input)
        classes = pipeline.classes_
        conf = proba.max(axis=1)
        pred = classes[np.argmax(proba, axis=1)]

        for f, s, b, c, p in zip(chunk['file'], subj, body, conf, pred):
            rows.append((f, s, b, c, p))

        print(f"    scored {total:>7,} rows", end='\r')

    print()
    pool = pd.DataFrame(rows, columns=['file', 'subject', 'body', 'confidence', 'predicted_category'])

    # The Enron corpus stores the same message repeatedly across a mailbox's
    # folders (sent/sent_items/all_documents/discussion_threads, ...). Dedupe
    # on content so the active-learning batch isn't spent re-labeling the
    # same email 3-4 times under different folder paths.
    before = len(pool)
    dedup_key = (pool['subject'].fillna('') + '||' + pool['body'].fillna('').str.slice(0, 300))
    pool = pool.loc[dedup_key.drop_duplicates(keep='first').index]
    print(f"Deduped {before:,} -> {len(pool):,} rows (removed same-content copies across folders)")

    pool = pool.sort_values('confidence', ascending=True)

    print("\nConfidence distribution of full scored pool:")
    print(pool['confidence'].describe())
    print("\nLowest-confidence rows: predicted-class breakdown (top", args.n, "):")
    lowest = pool.head(args.n).copy()
    print(lowest['predicted_category'].value_counts())

    lowest['body'] = lowest['body'].str.slice(0, 800)
    lowest['true_category'] = ''
    out_df = lowest[['file', 'predicted_category', 'subject', 'body', 'true_category']]
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df):,} lowest-confidence rows -> {args.out}")


if __name__ == '__main__':
    main()
