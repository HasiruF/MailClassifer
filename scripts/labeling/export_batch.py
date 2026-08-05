"""
export_batch.py — dump rows [start:end) of gold_sample_unlabeled.csv as
plain readable text for manual labeling.

Usage: python export_batch.py --start 0 --end 60 --out batch.txt
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='../../gold_sample_unlabeled.csv')
ap.add_argument('--start', type=int, required=True)
ap.add_argument('--end', type=int, required=True)
ap.add_argument('--out', default='batch.txt')
args = ap.parse_args()

df = pd.read_csv(args.csv)
chunk = df.iloc[args.start:args.end]

lines = []
for i, row in chunk.iterrows():
    lines.append(f"=== [{i}] file={row['file']} (model said: {row['predicted_category']}) ===")
    lines.append(f"SUBJECT: {row['subject']}")
    body = str(row['body']).replace('\r\n', ' ').replace('\n', ' ')
    lines.append(f"BODY: {body[:500]}")
    lines.append("")

with open(args.out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Wrote {len(chunk)} rows ({args.start}:{args.end}) -> {args.out}")
