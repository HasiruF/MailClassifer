"""
sample_for_gold_labels.py
─────────────────────────
Pulls a stratified random sample from category_predictions.csv (N per predicted
category), then joins back to emails.csv to recover subject/body text, producing
a CSV ready for a human (or careful manual LLM pass) to fill in `true_category`.

Usage
  python sample_for_gold_labels.py \
      --predictions predictions/category_predictions.csv \
      --enron       Enron/emails.csv \
      --n           180 \
      --out         gold_sample_unlabeled.csv
"""

import argparse, re, sys
import pandas as pd

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here)
sys.path.insert(0, _here + '\\..\\shared')
from train_category_classifier import parse_enron_message

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--enron', required=True)
    ap.add_argument('--n', type=int, default=180)
    ap.add_argument('--out', default='gold_sample_unlabeled.csv')
    args = ap.parse_args()

    print(f"Loading predictions from {args.predictions} …")
    preds = pd.read_csv(args.predictions, usecols=['file', 'category'])

    sampled = (
        preds.groupby('category', group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), args.n), random_state=42))
        .reset_index(drop=True)
    )
    print("Sampled per predicted category:")
    for cat, cnt in sampled['category'].value_counts().items():
        print(f"  {cat:16s} {cnt:>4}")

    wanted_files = set(sampled['file'])
    found = {}

    print(f"\nScanning {args.enron} for matching rows …")
    for chunk in pd.read_csv(args.enron, chunksize=50_000, on_bad_lines='skip'):
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        hits = chunk[chunk['file'].isin(wanted_files)]
        for _, row in hits.iterrows():
            subj, body = parse_enron_message(row['message'])
            found[row['file']] = {'subject': subj, 'body': body[:800]}
        if len(found) >= len(wanted_files):
            break

    print(f"Matched {len(found):,} / {len(wanted_files):,} sampled files")

    out_rows = []
    for _, r in sampled.iterrows():
        info = found.get(r['file'], {'subject': '', 'body': ''})
        out_rows.append({
            'file': r['file'],
            'predicted_category': r['category'],
            'subject': info['subject'],
            'body': info['body'],
            'true_category': '',
        })

    out_df = pd.DataFrame(out_rows).sample(frac=1, random_state=7).reset_index(drop=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df):,} rows → {args.out}")

if __name__ == '__main__':
    main()
