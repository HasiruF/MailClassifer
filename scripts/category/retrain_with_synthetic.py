"""
retrain_with_synthetic.py
────────────────────────────
Same fix as priority (PROJECT_STATUS.md item 3f), applied to the category
classifier. Category scored 76.8% same-corpus CV but only 62.0% on the
100-email out-of-distribution adversarial test (adversarial_test_set.py) —
the same CV-vs-real-world gap priority had before its retrain, and for the
same reason: 1000 training rows, 100% sourced from one 2001 Enron corpus.

Reuses the 500 hand-authored synthetic rows already built for priority
(scripts/priority/synthetic_priority_train_batch{1..5}.py) — this time using
their true_category labels directly as ground truth (priority's retrain used
the *predicted* category to match production; category's own retrain uses
the *true* label, since here category IS the ground truth being learned).
Same leakage guard: asserts zero overlap with adversarial_test_set.py.

Trains the same architecture as train_category_classifier_headers.py
(word+char TF-IDF, header/style numeric features, LogisticRegression) on
1000 Enron + 500 synthetic = 1500 rows, for an apples-to-apples same-corpus
CV comparison, then separately evaluates against the adversarial set (the
metric that actually matters, per item 3e/3f).

Usage
  python retrain_with_synthetic.py
"""

import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here)
sys.path.insert(0, _here + '\\..\\shared')
sys.path.insert(0, _here + '\\..\\priority')
sys.path.insert(0, _here + '\\..')

from train_category_classifier_headers import make_pipeline, NUMERIC_COLS
from train_category_classifier import clean_text
from header_features import extract_header_features_from_fields
from stylistic_features import extract_stylistic_features

from synthetic_priority_train_batch1 import ROWS as B1
from synthetic_priority_train_batch2 import ROWS as B2
from synthetic_priority_train_batch3 import ROWS as B3
from synthetic_priority_train_batch4 import ROWS as B4
from synthetic_priority_train_batch5 import ROWS as B5
from adversarial_test_set import TEST_EMAILS

ALL_ROWS = B1 + B2 + B3 + B4 + B5
assert len(ALL_ROWS) == 500

# ── leakage guard ─────────────────────────────────────────────────────────
eval_keys = {(e['subject'].strip().lower(), e['body'].strip().lower()) for e in TEST_EMAILS}
train_keys = {(r['subject'].strip().lower(), r['body'].strip().lower()) for r in ALL_ROWS}
assert not (eval_keys & train_keys), "LEAKAGE: synthetic rows overlap with adversarial eval set"
print("No overlap between synthetic rows and adversarial eval set — clean.")

# ── Enron gold (existing) ────────────────────────────────────────────────
gold = pd.read_csv('../../gold_sample_labeled.csv')
gold = gold[gold['true_category'].notna() & (gold['true_category'].str.strip() != '')].copy()
gold['subject'] = gold['subject'].fillna('')
gold['body'] = gold['body'].fillna('')
gold['text'] = (gold['subject'] + ' ' + gold['body']).apply(clean_text)
style_feats = gold.apply(lambda row: pd.Series(extract_stylistic_features(row['subject'], row['body'])), axis=1)
gold = pd.concat([gold, style_feats], axis=1)
headers = pd.read_csv('gold_header_features.csv')
enron_df = gold.merge(headers, on='file', how='left')
enron_df[NUMERIC_COLS] = enron_df[NUMERIC_COLS].fillna(0)
enron_X = enron_df[['text'] + NUMERIC_COLS]
enron_y = enron_df['true_category']
print(f"Enron gold: {len(enron_df):,} rows")

# ── synthetic rows -> same feature shape ─────────────────────────────────
syn_rows = []
for r in ALL_ROWS:
    text = clean_text(r['subject'] + ' ' + r['body'])
    header_feats = extract_header_features_from_fields(subject=r['subject'], to=r['to'], from_addr=r['from_addr'])
    style = extract_stylistic_features(r['subject'], r['body'])
    row = {'text': text, 'true_category': r['true_category']}
    row.update({k: header_feats[k] for k in ['n_recipients', 'is_reply_or_forward', 'sender_automated']})
    row['has_list_unsubscribe'] = 0
    row['has_precedence_bulk'] = 0
    row.update(style)
    syn_rows.append(row)
syn_df = pd.DataFrame(syn_rows)
syn_X = syn_df[['text'] + NUMERIC_COLS]
syn_y = syn_df['true_category']
print(f"Synthetic: {len(syn_df):,} rows")
print(syn_y.value_counts())

# ── combine ───────────────────────────────────────────────────────────────
X = pd.concat([enron_X, syn_X], ignore_index=True)
y = pd.concat([enron_y, syn_y], ignore_index=True)
print(f"\nCombined: {len(X):,} rows")

print("\n[1/2] 5-fold CV on combined 1500 rows …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(make_pipeline(), X, y, cv=cv)
acc = accuracy_score(y, pred)
print(f"    Out-of-fold accuracy: {acc:.4f}")
labels = sorted(y.unique())
print(classification_report(y, pred, labels=labels, zero_division=0))
cm = confusion_matrix(y, pred, labels=labels)
print("Confusion matrix (rows=true, cols=predicted):")
print('  ' + '  '.join(f'{c:>14s}' for c in labels))
for i, row_label in enumerate(labels):
    print(f"  {row_label:14s}  " + '  '.join(f'{v:>14d}' for v in cm[i]))

print("\n[2/2] Fitting final model on all 1500 rows …")
final_pipeline = make_pipeline()
final_pipeline.fit(X, y)
joblib.dump(final_pipeline, '../../models_category_v2.joblib')
print("Saved -> ../../models_category_v2.joblib (not yet promoted to production)")
