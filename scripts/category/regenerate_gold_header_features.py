"""
regenerate_gold_header_features.py
────────────────────────────────────
Rebuilds gold_header_features.csv by re-scanning Enron/emails.csv for the
`file` keys present in gold_sample_labeled.csv and calling
header_features.extract_header_features() on each raw message. Needed
whenever header_features.py's feature set changes (e.g. new columns added).

Usage
  python regenerate_gold_header_features.py \
      --gold  ../../gold_sample_labeled.csv \
      --enron ../../Enron/emails.csv \
      --out   gold_header_features.csv
"""
import argparse
import sys

import pandas as pd

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here + '\\..\\shared')
from header_features import extract_header_features

ap = argparse.ArgumentParser()
ap.add_argument('--gold', default='../../gold_sample_labeled.csv')
ap.add_argument('--enron', default='../../Enron/emails.csv')
ap.add_argument('--out', default='gold_header_features.csv')
args = ap.parse_args()

gold_files = set(pd.read_csv(args.gold, usecols=['file'])['file'])
print(f"Looking for {len(gold_files):,} gold-labeled files in the Enron corpus ...")

rows = []
found = 0
for chunk in pd.read_csv(args.enron, chunksize=50_000):
    hit = chunk[chunk['file'].isin(gold_files)]
    if len(hit) == 0:
        continue
    for _, r in hit.iterrows():
        feats = extract_header_features(r['message'])
        feats['file'] = r['file']
        rows.append(feats)
    found += len(hit)
    print(f"  ... matched {found:,}/{len(gold_files):,} so far")
    if found >= len(gold_files):
        break

out = pd.DataFrame(rows)
missing = gold_files - set(out['file'])
if missing:
    print(f"WARNING: {len(missing)} gold files not found in Enron corpus (will be filled with 0 at merge time)")

out.to_csv(args.out, index=False)
print(f"Wrote {len(out)} rows -> {args.out}")
print(out.head())
