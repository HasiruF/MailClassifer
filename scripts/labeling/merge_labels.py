"""
merge_labels.py — apply a batch of {file: true_category} labels (from a small
CSV with columns file,true_category) into the master gold sample CSV.

Usage: python merge_labels.py --labels batch_labels.csv --master gold_sample_labeled.csv
"""
import argparse
import os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True)
ap.add_argument('--master', default='../../gold_sample_labeled.csv')
ap.add_argument('--source', default='../../gold_sample_unlabeled.csv')
args = ap.parse_args()

if os.path.exists(args.master):
    master = pd.read_csv(args.master)
else:
    master = pd.read_csv(args.source)

labels = pd.read_csv(args.labels)
labels_map = dict(zip(labels['file'], labels['true_category']))

applied = 0
for idx, row in master.iterrows():
    if row['file'] in labels_map:
        master.at[idx, 'true_category'] = labels_map[row['file']]
        applied += 1

master.to_csv(args.master, index=False)
labeled_count = master['true_category'].notna().sum()
print(f"Applied {applied} labels this batch. Total labeled so far: {labeled_count}/{len(master)}")
