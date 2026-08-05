"""
append_active_batch.py — append a freshly-labeled active-learning batch into
the master gold_sample_labeled.csv. Unlike merge_labels.py (which only fills
in true_category for rows already present in master), this appends brand-new
rows since active-learning candidates were never in the original sample.

Usage
  python append_active_batch.py \
      --pool   active_pool_round1.csv \
      --labels active_batch1_labels.csv \
      --master ../gold_sample_labeled.csv
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--pool', required=True, help='CSV with file/predicted_category/subject/body (the exported batch source)')
ap.add_argument('--labels', required=True, help='CSV with file,true_category (my labels for this batch)')
ap.add_argument('--master', default='../../gold_sample_labeled.csv')
args = ap.parse_args()

pool = pd.read_csv(args.pool)
labels = pd.read_csv(args.labels)
master = pd.read_csv(args.master)

already = set(master['file'])
new_labels = labels[~labels['file'].isin(already)]
skipped = len(labels) - len(new_labels)

merged = new_labels.merge(pool[['file', 'predicted_category', 'subject', 'body']], on='file', how='left')
missing = merged['predicted_category'].isna().sum()
if missing:
    print(f"WARNING: {missing} labeled files not found in pool CSV (subject/body will be blank)")

new_rows = merged[['file', 'predicted_category', 'subject', 'body', 'true_category']]
combined = pd.concat([master, new_rows], ignore_index=True)
combined.to_csv(args.master, index=False)

print(f"Appended {len(new_rows)} new rows ({skipped} already present, skipped).")
print(f"Master now has {len(combined)} rows, {combined['true_category'].notna().sum()} labeled.")
