"""
sample_for_gold_priority.py
────────────────────────────
Scans the Enron corpus, scores each message with the current rule-based
priority heuristic (+ spam/category context), and pulls a STRATIFIED sample
across the heuristic's own score buckets — not a pure random sample, which
would be dominated by routine low/medium mail and give a labeler almost no
high-priority examples to judge.

Output columns: file, subject, body (truncated), to, from, heuristic_score,
heuristic_reasons, spam_label, spam_conf, category_label, vader_compound,
true_priority (blank, for the labeling pass).

Usage
  python sample_for_gold_priority.py --n_per_bucket 34 --out gold_priority_unlabeled.csv
"""

import argparse, sys, random
import pandas as pd

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here + '\\..\\shared')
sys.path.insert(0, _here + '\\..')
from train_category_classifier import parse_enron_message
from header_features import _split_header_block, _parse_headers
from classify_email import EmailClassifier, _priority
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()


def bucket(score):
    if score < 0.35:
        return 'low'
    if score < 0.55:
        return 'medium'
    return 'high'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enron', default='../../Enron/emails.csv')
    ap.add_argument('--models', default='../../models')
    ap.add_argument('--scan_size', type=int, default=20000,
                     help='how many raw emails to scan/score before stratifying')
    ap.add_argument('--n_per_bucket', type=int, default=34)
    ap.add_argument('--out', default='../../gold_priority_unlabeled.csv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--exclude_csv', default=None,
                     help='CSV with a "file" column of already-labeled rows to skip')
    args = ap.parse_args()

    clf = EmailClassifier(args.models)

    exclude_files = set()
    if args.exclude_csv:
        exclude_files = set(pd.read_csv(args.exclude_csv, usecols=['file'])['file'])
        print(f"Excluding {len(exclude_files):,} already-labeled files")

    print(f"Scanning up to {args.scan_size:,} rows from {args.enron} …")
    rows = []
    random.seed(args.seed)
    for chunk in pd.read_csv(args.enron, chunksize=20_000, on_bad_lines='skip'):
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        if exclude_files:
            chunk = chunk[~chunk['file'].isin(exclude_files)]
        for _, r in chunk.sample(frac=1, random_state=args.seed).iterrows():
            if len(rows) >= args.scan_size:
                break
            subj, body = parse_enron_message(r['message'])
            header_block = _split_header_block(r['message'])
            fields = _parse_headers(header_block)
            to = fields.get('to', '')
            from_addr = fields.get('from', '')

            result = clf.classify(subject=subj, body=body, to=to, from_addr=from_addr)
            prio = result['priority']
            compound = _sia.polarity_scores((subj or '') + ' ' + (body or '')[:1500])['compound']

            rows.append({
                'file': r['file'],
                'subject': subj,
                'body': (body or '')[:700],
                'to': to,
                'from': from_addr,
                'heuristic_score': prio['score'],
                'heuristic_bucket': bucket(prio['score']),
                'heuristic_reasons': '; '.join(f"{x['signal']} ({x['weight']:+.2f})" for x in prio['reasons']),
                'spam_label': result['spam']['label'],
                'spam_conf': result['spam']['confidence'],
                'category_label': result['category']['label'],
                'vader_compound': round(compound, 3),
                'true_priority': '',
            })
        if len(rows) >= args.scan_size:
            break

    df = pd.DataFrame(rows)
    print(f"\nScored {len(df):,} emails. Heuristic bucket distribution:")
    print(df['heuristic_bucket'].value_counts())

    sampled = (
        df.groupby('heuristic_bucket', group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), args.n_per_bucket), random_state=args.seed))
        .sample(frac=1, random_state=args.seed + 1)
        .reset_index(drop=True)
    )
    print(f"\nStratified sample: {len(sampled)} rows")
    print(sampled['heuristic_bucket'].value_counts())

    sampled.to_csv(args.out, index=False)
    print(f"\nWrote {len(sampled)} rows -> {args.out}")


if __name__ == '__main__':
    main()
