"""
build_synthetic_training_csv.py
─────────────────────────────────
Combines the 5 synthetic_priority_train_batchN.py files (500 hand-authored
rows total) into one CSV in the same shape as gold_priority_labeled.csv, then
concatenates with the existing 504-row Enron gold set to produce
gold_priority_labeled_v2.csv (1004 rows) — the combined training set used to
test whether adding non-Enron register diversity actually improves
out-of-distribution accuracy (see PROJECT_STATUS.md item 3e/3f).

category_label is computed via the CURRENT category classifier's own
prediction (not the hand-authored true_category) to match production, where
priority always receives a *predicted* category from the upstream model, not
ground truth. true_category is kept as an extra column for a possible future
category-classifier retraining round, unused by train_priority_classifier.py.

Guards against train/test leakage: asserts none of the 500 synthetic rows'
(subject, body) pairs match anything in adversarial_test_set.py, which must
stay held-out for eval.

Usage
  python build_synthetic_training_csv.py
"""

import sys
import pandas as pd

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here)
sys.path.insert(0, _here + '\\..\\shared')
sys.path.insert(0, _here + '\\..')

from synthetic_priority_train_batch1 import ROWS as B1
from synthetic_priority_train_batch2 import ROWS as B2
from synthetic_priority_train_batch3 import ROWS as B3
from synthetic_priority_train_batch4 import ROWS as B4
from synthetic_priority_train_batch5 import ROWS as B5
from adversarial_test_set import TEST_EMAILS

from classify_email import EmailClassifier, _clean, _sia

ALL_ROWS = B1 + B2 + B3 + B4 + B5
print(f"Loaded {len(ALL_ROWS)} synthetic rows across 5 batches")
assert len(ALL_ROWS) == 500, f"expected 500, got {len(ALL_ROWS)}"

ids = [r['id'] for r in ALL_ROWS]
assert len(set(ids)) == len(ids), "duplicate ids across batches"

# ── Leakage guard: no synthetic training row may match an eval row ──────────
eval_keys = {(e['subject'].strip().lower(), e['body'].strip().lower()) for e in TEST_EMAILS}
train_keys = {(r['subject'].strip().lower(), r['body'].strip().lower()) for r in ALL_ROWS}
overlap = eval_keys & train_keys
assert not overlap, f"LEAKAGE: {len(overlap)} synthetic training rows match adversarial eval rows: {overlap}"
print("No overlap between synthetic training rows and the adversarial eval set — clean.")

# ── Category balance check ───────────────────────────────────────────────
cat_counts = pd.Series([r['true_category'] for r in ALL_ROWS]).value_counts()
bucket_counts = pd.Series([r['true_bucket'] for r in ALL_ROWS]).value_counts()
print(f"\ntrue_category distribution:\n{cat_counts}")
print(f"\ntrue_bucket distribution:\n{bucket_counts}")

# ── Score each row with the current models to get spam_conf / category_label
# (predicted, not ground truth) / vader_compound, matching gold_priority_labeled.csv's schema ──
print("\nLoading models to compute spam_conf / predicted category_label / vader_compound ...")
clf = EmailClassifier('../../models')

out_rows = []
for r in ALL_ROWS:
    text = _clean(r['subject'] + ' ' + r['body'])
    spam_proba = clf.spam_pipeline.predict_proba([text])[0]
    spam_idx = list(clf.spam_pipeline.classes_).index(1)
    spam_conf = float(spam_proba[spam_idx])

    from header_features import extract_header_features_from_fields
    from stylistic_features import extract_stylistic_features
    from classify_email import NUMERIC_COLS as CAT_NUMERIC_COLS
    header_feats = extract_header_features_from_fields(subject=r['subject'], to=r['to'], from_addr=r['from_addr'])
    style_feats = extract_stylistic_features(r['subject'], r['body'])
    cat_input = pd.DataFrame([{**{'text': text}, **header_feats, **style_feats}])[['text'] + CAT_NUMERIC_COLS]
    cat_proba = clf.cat_pipeline.predict_proba(cat_input)[0]
    predicted_category = clf.cat_classes[cat_proba.argmax()]

    compound = _sia.polarity_scores(text)['compound']

    out_rows.append({
        'file': r['id'],
        'subject': r['subject'],
        'body': r['body'],
        'to': r['to'],
        'from': r['from_addr'],
        'spam_label': 'spam' if spam_conf >= 0.5 else 'ham',
        'spam_conf': round(spam_conf, 4),
        'category_label': predicted_category,
        'true_category': r['true_category'],
        'vader_compound': round(compound, 3),
        'true_priority': r['true_priority'],
        'true_bucket': r['true_bucket'],
    })

synthetic_df = pd.DataFrame(out_rows)
synthetic_df.to_csv('../../gold_priority_synthetic.csv', index=False)
print(f"\nWrote {len(synthetic_df)} rows -> ../../gold_priority_synthetic.csv")

cat_agree = (synthetic_df['category_label'] == synthetic_df['true_category']).mean()
print(f"Predicted-vs-true category agreement on synthetic rows: {cat_agree:.1%} "
      f"(expected to be imperfect — this is the same generalization gap from item 3e)")

# ── Combine with the existing Enron gold set ─────────────────────────────
enron_df = pd.read_csv('../../gold_priority_labeled.csv')
# align columns: enron_df has no true_category column, add as NaN
if 'true_category' not in enron_df.columns:
    enron_df['true_category'] = pd.NA

common_cols = ['file', 'subject', 'body', 'to', 'from', 'spam_label', 'spam_conf',
               'category_label', 'true_category', 'vader_compound', 'true_priority', 'true_bucket']
for c in common_cols:
    if c not in enron_df.columns:
        enron_df[c] = pd.NA

combined = pd.concat([enron_df[common_cols], synthetic_df[common_cols]], ignore_index=True)
combined.to_csv('../../gold_priority_labeled_v2.csv', index=False)
print(f"\nCombined training set: {len(combined)} rows "
      f"({len(enron_df)} Enron + {len(synthetic_df)} synthetic) -> ../../gold_priority_labeled_v2.csv")
print(f"true_bucket distribution (combined):\n{combined['true_bucket'].value_counts()}")
