"""
export_training_matrices.py
───────────────────────────
One-time precompute for per-user personalization retraining (see
docs/superpowers/specs/2026-09-29-personalization-design.md §4.1).

For spam, category and priority this rebuilds the exact rows each production
model was fit on, passes them through that production pipeline's
already-fitted feature transformer, and saves:

  <name>_X.npz            base training matrix (scipy sparse, CSR)
  <name>_gate_X.npz       gate evaluation matrix (not for priority_regressor)
  <name>_labels.npz       y (+ gate_y) as plain numpy arrays, no pickling
  <name>_estimator.joblib unfitted clone of the production estimator
  meta.json               per model: dim, classes, base_gate_accuracy

Row sources (each mirrors the trainer that produced the production model):
  spam      spam/train_spam_classifier.py __main__: 500 SpamAssassin spam +
            300 SpamAssassin ham + 700 Enron ham, shuffled with
            random_state=42; the saved model is refit on all rows.
            Gate: the stratified 15% split that script evaluates on. Both the
            base and personalized models have seen these rows, so this gate
            only detects regressions; it is not an accuracy estimate.
  category  category/retrain_with_synthetic.py: 1000 Enron gold rows + 500
            synthetic rows. Its data assembly is copied below because that
            script has no __main__ guard (importing it would retrain).
            Gate: the 100-email adversarial set.
  priority  priority/train_priority_classifier.py build_features() over
            gold_priority_labeled_v2.csv (1004 rows). Classifier target
            true_bucket, regressor target true_priority.
            Gate: the adversarial set, with category_label and spam_conf
            taken from the base category/spam models, as classify_email.py
            does at serving time.

Before writing anything, the script proves the rows match: refitting the
clone on the rebuilt matrix must reproduce the production estimator's
outputs on the gate matrix (max abs diff < 1e-6). If not, it exits non-zero
and writes nothing.

Usage (from scripts/):
  python export_training_matrices.py
"""

import argparse
import json
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ('', 'shared', 'spam', 'category', 'priority'):
    sys.path.insert(0, os.path.join(HERE, sub))

from train_category_classifier import clean_text as category_clean_text
from header_features import extract_header_features_from_fields
from stylistic_features import extract_stylistic_features
from priority_keyword_features import extract_keyword_features
from train_category_classifier_headers import NUMERIC_COLS as CATEGORY_NUMERIC_COLS
from train_priority_classifier import build_features as priority_build_features
from train_priority_classifier import NUMERIC_COLS as PRIORITY_NUMERIC_COLS
from train_spam_classifier import load_spamassassin, load_enron_ham
from classify_email import _clean as serving_clean, _sia
from synthetic_priority_train_batch1 import ROWS as B1
from synthetic_priority_train_batch2 import ROWS as B2
from synthetic_priority_train_batch3 import ROWS as B3
from synthetic_priority_train_batch4 import ROWS as B4
from synthetic_priority_train_batch5 import ROWS as B5
from adversarial_test_set import TEST_EMAILS

REPRO_TOLERANCE = 1e-6


def spam_rows(spam_csv, enron_csv):
    spam_all = load_spamassassin(spam_csv, sample_size=None)
    n_spam = int((spam_all['label'] == 1).sum())
    n_ham = int((spam_all['label'] == 0).sum())
    spam = spam_all[spam_all['label'] == 1].sample(min(500, n_spam), random_state=42)
    sa_ham = spam_all[spam_all['label'] == 0].sample(min(300, n_ham), random_state=42)
    enron_ham = load_enron_ham(enron_csv, 700)
    combined = pd.concat([
        spam[['text', 'label']],
        sa_ham[['text', 'label']],
        enron_ham[['text', 'label']],
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    _, gate_text, _, gate_y = train_test_split(
        combined['text'], combined['label'], test_size=0.15, stratify=combined['label'], random_state=42,
    )
    return (combined['text'], np.asarray(combined['label'], dtype=np.int64),
            gate_text, np.asarray(gate_y, dtype=np.int64))


def category_rows():
    gold = pd.read_csv(os.path.join(REPO, 'gold_sample_labeled.csv'))
    gold = gold[gold['true_category'].notna() & (gold['true_category'].str.strip() != '')].copy()
    gold['subject'] = gold['subject'].fillna('')
    gold['body'] = gold['body'].fillna('')
    gold['text'] = (gold['subject'] + ' ' + gold['body']).apply(category_clean_text)
    style = gold.apply(lambda row: pd.Series(extract_stylistic_features(row['subject'], row['body'])), axis=1)
    gold = pd.concat([gold, style], axis=1)
    headers = pd.read_csv(os.path.join(HERE, 'category', 'gold_header_features.csv'))
    enron = gold.merge(headers, on='file', how='left')
    enron[CATEGORY_NUMERIC_COLS] = enron[CATEGORY_NUMERIC_COLS].fillna(0)

    syn = []
    for r in B1 + B2 + B3 + B4 + B5:
        header = extract_header_features_from_fields(subject=r['subject'], to=r['to'], from_addr=r['from_addr'])
        row = {'text': category_clean_text(r['subject'] + ' ' + r['body']), 'true_category': r['true_category']}
        row.update({k: header[k] for k in ['n_recipients', 'is_reply_or_forward', 'sender_automated']})
        row['has_list_unsubscribe'] = 0
        row['has_precedence_bulk'] = 0
        row.update(extract_stylistic_features(r['subject'], r['body']))
        syn.append(row)
    syn = pd.DataFrame(syn)

    X = pd.concat([enron[['text'] + CATEGORY_NUMERIC_COLS], syn[['text'] + CATEGORY_NUMERIC_COLS]],
                  ignore_index=True)
    y = pd.concat([enron['true_category'], syn['true_category']], ignore_index=True)
    return X, np.asarray(y, dtype=str)


def priority_rows():
    raw = pd.read_csv(os.path.join(REPO, 'gold_priority_labeled_v2.csv'))
    df = priority_build_features(raw)
    X = df[['text', 'category_label'] + PRIORITY_NUMERIC_COLS]
    return X, np.asarray(df['true_bucket'], dtype=str), np.asarray(df['true_priority'], dtype=np.float64)


def adversarial_frames(spam_pipe, cat_pipe):
    cat_rows, prio_rows = [], []
    for e in TEST_EMAILS:
        text = serving_clean(e['subject'] + ' ' + e['body'])
        header = extract_header_features_from_fields(subject=e['subject'], to=e['to'], from_addr=e['from_addr'])
        style = extract_stylistic_features(e['subject'], e['body'])
        spam_proba = spam_pipe.predict_proba([text])[0]
        spam_conf = float(spam_proba[list(spam_pipe.classes_).index(1)])
        cat_rows.append({'text': text, **header, **style})
        prio_rows.append({
            'text': text,
            'n_recipients': header['n_recipients'],
            'is_reply_or_forward': header['is_reply_or_forward'],
            'sender_automated': header['sender_automated'],
            'spam_conf': spam_conf,
            'vader_compound': _sia.polarity_scores(text)['compound'],
            **style,
            **extract_keyword_features(e['subject'], e['body']),
        })
    cat_X = pd.DataFrame(cat_rows)[['text'] + CATEGORY_NUMERIC_COLS]
    prio_df = pd.DataFrame(prio_rows)
    prio_df['category_label'] = cat_pipe.predict(cat_X)
    prio_X = prio_df[['text', 'category_label'] + PRIORITY_NUMERIC_COLS]
    cat_y = np.asarray([e['expected_category'] for e in TEST_EMAILS], dtype=str)
    prio_y = np.asarray([e['expected_priority'] for e in TEST_EMAILS], dtype=str)
    return cat_X, cat_y, prio_X, prio_y


def refit_matches(name, prod_est, X, y, check_X):
    started = time.perf_counter()
    est = clone(prod_est)
    est.fit(X, y)
    seconds = time.perf_counter() - started
    if hasattr(prod_est, 'predict_proba'):
        diff = float(np.max(np.abs(est.predict_proba(check_X) - prod_est.predict_proba(check_X))))
    else:
        diff = float(np.max(np.abs(est.predict(check_X) - prod_est.predict(check_X))))
    ok = diff < REPRO_TOLERANCE
    print(f"[{name}] refit took {seconds:.1f}s; max diff vs production {diff:.2e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=os.path.join(REPO, 'models'))
    ap.add_argument('--spam', default=os.path.join(REPO, 'SpamAssasin', 'spamassassin.csv'))
    ap.add_argument('--enron', default=os.path.join(REPO, 'Enron', 'emails.csv'))
    ap.add_argument('--out', default=os.path.join(REPO, 'app', 'backend', 'training_data'))
    args = ap.parse_args()

    spam_pipe = joblib.load(os.path.join(args.models, 'spam_classifier.joblib'))
    cat_pipe = joblib.load(os.path.join(args.models, 'category_classifier_headers.joblib'))
    prio_clf_pipe = joblib.load(os.path.join(args.models, 'priority_classifier.joblib'))
    prio_reg_pipe = joblib.load(os.path.join(args.models, 'priority_regressor.joblib'))

    print("Rebuilding spam rows (scans Enron/emails.csv in chunks; takes a minute) ...")
    spam_text, spam_y, spam_gate_text, spam_gate_y = spam_rows(args.spam, args.enron)
    spam_tfidf = spam_pipe.named_steps['tfidf']
    spam_clf = spam_pipe.named_steps['clf']
    spam_X = sp.csr_matrix(spam_tfidf.transform(spam_text))
    spam_gate_X = sp.csr_matrix(spam_tfidf.transform(spam_gate_text))

    print("Rebuilding category rows ...")
    cat_frame, cat_y = category_rows()
    cat_features = cat_pipe.named_steps['features']
    cat_clf = cat_pipe.named_steps['clf']
    cat_X = sp.csr_matrix(cat_features.transform(cat_frame))

    print("Rebuilding priority rows ...")
    prio_frame, prio_y, prio_score = priority_rows()
    prio_clf = prio_clf_pipe.named_steps['clf']
    prio_reg = prio_reg_pipe.named_steps['reg']
    prio_X = sp.csr_matrix(prio_clf_pipe.named_steps['features'].transform(prio_frame))
    # Built with fit_transform on a clone, not the fitted transformer's
    # transform(): the two differ by ~1 ulp on a few values, and Ridge's
    # iterative sparse_cg solver (tol 1e-4) turns that into ~3e-5 of
    # coefficient drift. fit_transform is what production training used, so
    # it reproduces production exactly. The check below proves it's still
    # the same rows.
    prio_reg_features = prio_reg_pipe.named_steps['features']
    prio_reg_X = sp.csr_matrix(clone(prio_reg_features).fit_transform(prio_frame, prio_score))
    if abs(prio_reg_X - prio_reg_features.transform(prio_frame)).max() > 1e-12:
        sys.exit("priority_regressor: fit_transform rows differ from the production transform; nothing written.")

    print("Vectorizing the adversarial gate set ...")
    adv_cat_frame, adv_cat_y, adv_prio_frame, adv_prio_y = adversarial_frames(spam_pipe, cat_pipe)
    cat_gate_X = sp.csr_matrix(cat_features.transform(adv_cat_frame))
    prio_gate_X = sp.csr_matrix(prio_clf_pipe.named_steps['features'].transform(adv_prio_frame))
    prio_reg_gate_X = sp.csr_matrix(prio_reg_pipe.named_steps['features'].transform(adv_prio_frame))

    checks = [
        refit_matches('spam', spam_clf, spam_X, spam_y, spam_gate_X),
        refit_matches('category', cat_clf, cat_X, cat_y, cat_gate_X),
        refit_matches('priority', prio_clf, prio_X, prio_y, prio_gate_X),
        refit_matches('priority_regressor', prio_reg, prio_reg_X, prio_score, prio_reg_gate_X),
    ]
    if not all(checks):
        sys.exit("Rebuilt rows do not reproduce the production models; nothing written. "
                 "Find which trainer's data assembly differs before continuing.")

    os.makedirs(args.out, exist_ok=True)
    meta = {}

    def save(name, est, X, y, gate_X=None, gate_y=None):
        sp.save_npz(os.path.join(args.out, f'{name}_X.npz'), X)
        labels = {'y': y}
        if gate_X is not None:
            sp.save_npz(os.path.join(args.out, f'{name}_gate_X.npz'), gate_X)
            labels['gate_y'] = gate_y
        np.savez(os.path.join(args.out, f'{name}_labels.npz'), **labels)
        joblib.dump(clone(est), os.path.join(args.out, f'{name}_estimator.joblib'))
        entry = {'dim': int(X.shape[1])}
        if hasattr(est, 'classes_'):
            entry['classes'] = est.classes_.tolist()
        if gate_X is not None:
            entry['base_gate_accuracy'] = float(np.mean(est.predict(gate_X) == gate_y))
        meta[name] = entry
        extra = f", base gate accuracy {entry['base_gate_accuracy']:.1%}" if 'base_gate_accuracy' in entry else ''
        print(f"[{name}] saved {X.shape[0]} rows x {X.shape[1]} dims{extra}")

    save('spam', spam_clf, spam_X, spam_y, spam_gate_X, spam_gate_y)
    save('category', cat_clf, cat_X, cat_y, cat_gate_X, adv_cat_y)
    save('priority', prio_clf, prio_X, prio_y, prio_gate_X, adv_prio_y)
    save('priority_regressor', prio_reg, prio_reg_X, prio_score)
    with open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {os.path.join(args.out, 'meta.json')}")


if __name__ == '__main__':
    main()
