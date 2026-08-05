"""
train_priority_classifier.py
─────────────────────────────
Trains a real priority model on the 504-row hand-labeled gold set
(gold_priority_labeled.csv), replacing the rule-based heuristic in
classify_email.py._priority(). Ground truth (`true_priority`) was hand-judged
from subject/body/to/from ALONE — blind to the heuristic's own score/reasons,
spam_conf, category_label, or vader_compound — to avoid anchoring bias.

Motivation: the heuristic was checked against this same gold set and only
agreed with human judgment 40.3% of the time on a 3-bucket (low/med/high)
call, with a near-zero 0.215 correlation on the continuous score. Root cause
(see PROJECT_STATUS.md): the heuristic keys off literal words ("outage",
"emergency", "urgent") regardless of context — recurring automated outage
reports and spam that fakes urgency both scored high, while a real same-day
operational directive ("It is imperative... NO EXCUSES") scored low because
it used different words and got penalized by the ">6 recipients" rule.

Features (everything a trained model can learn combinations over, instead of
hard-coded keyword rules):
  - TF-IDF word (1-2gram) + char (3-5gram) on cleaned subject+body text
  - header features: n_recipients, is_reply_or_forward, sender_automated
    (has_list_unsubscribe/has_precedence_bulk unavailable in this sample —
    not scraped from raw headers during gold sampling, left as 0)
  - stylistic features (STYLE_COLS, shared with the category classifier)
  - spam_conf (from the now-fixed spam classifier)
  - category_label (one-hot: Work/Personal/Other)
  - vader_compound (sentiment)
  - explicit keyword-count features (priority_keyword_features.py) — urgent/
    deadline/bulk-mail word counts, kept as hand-authored counts rather than
    left to TF-IDF to rediscover, since 504 rows is too little data for a
    linear model to learn them reliably from raw text alone.

Two targets, both 5-fold CV:
  - true_priority (continuous, 0.1-1.0)  -> Ridge regression, report MAE
  - true_bucket (low/medium/high)        -> RandomForestClassifier, report
    accuracy. Tried first: plain LogisticRegression on text alone (70.0%),
    LogReg + keyword counts (69.0% — WORSE, a linear model can't exploit
    them), RandomForest on text alone (74.0%). RandomForest + keyword counts
    together (74.8%) beat all of those — the lift only appears when a model
    that captures interactions (a keyword count only signals urgency
    combined with certain header/style signals) gets to see the explicit
    counts; a linear model can't use them productively on its own. Smaller
    TF-IDF dimensionality + stronger regularization was also tried and made
    things WORSE (66.7-66.9%) — the dimensionality wasn't the bottleneck.
    RandomForest hyperparameters are tuned via nested cross-validation
    (GridSearchCV inside the outer 5-fold CV) so the reported accuracy isn't
    inflated by picking hyperparameters against the same folds they're
    scored on.

UPDATE (see PROJECT_STATUS.md items 3e/3f): the 504-row Enron-only set above
scored 73.0% nested-CV but only 46.0% on a held-out 100-email adversarial
test of out-of-distribution (non-Enron, modern/casual-register) text — CV
accuracy was measuring fit-to-corpus, not real generalization. Root cause
was training-data diversity, not feature/model tuning (five TF-IDF configs
were tried and none moved the adversarial number). Fixed by adding 500 more
hand-authored, balanced (category x priority) synthetic training rows
spanning personal/casual/emergency registers Enron never had, with a
leakage guard verifying zero overlap with the adversarial eval set. Default
gold set is now gold_priority_labeled_v2.csv (1004 rows = 504 Enron + 500
synthetic): nested-CV 85.1%, and — the number that actually matters —
adversarial accuracy 82.0% (up from 46.0%), high recall 64.5% (up from
3.2%). See scripts/priority/build_synthetic_training_csv.py and the
synthetic_priority_train_batch{1..5}.py files.

Usage
  python train_priority_classifier.py --gold ../../gold_priority_labeled_v2.csv --out ../../models
"""

import argparse
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here + '\\..\\shared')
from train_category_classifier import clean_text
from header_features import extract_header_features_from_fields, log1p_recipients
from stylistic_features import extract_stylistic_features, STYLE_COLS
from priority_keyword_features import extract_keyword_features, KEYWORD_COLS

HEADER_COLS = ['n_recipients', 'is_reply_or_forward', 'sender_automated']
NUMERIC_COLS = HEADER_COLS + STYLE_COLS + ['spam_conf', 'vader_compound'] + KEYWORD_COLS
CAT_COLS = ['category_label']

RF_PARAM_GRID = {
    'clf__n_estimators': [200, 400],
    'clf__max_depth': [None, 20],
    'clf__min_samples_leaf': [1, 3],
    'clf__max_features': ['sqrt', 0.3],
}


def build_features(df):
    df = df.copy()
    df['text'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).apply(clean_text)

    to_col = df['to'].fillna('').astype(str)
    from_col = df['from'].fillna('').astype(str)
    subj_col = df['subject'].fillna('').astype(str)
    header_rows = [
        extract_header_features_from_fields(subject=s, to=t, from_addr=f)
        for s, t, f in zip(subj_col, to_col, from_col)
    ]
    header_df = pd.DataFrame(header_rows)[HEADER_COLS]

    body_col = df['body'].fillna('').astype(str)
    style_rows = [
        extract_stylistic_features(s, b) for s, b in zip(subj_col, body_col)
    ]
    style_df = pd.DataFrame(style_rows)[STYLE_COLS]

    keyword_rows = [
        extract_keyword_features(s, b) for s, b in zip(subj_col, body_col)
    ]
    keyword_df = pd.DataFrame(keyword_rows)[KEYWORD_COLS]

    out = pd.concat([
        df[['text', 'spam_conf', 'vader_compound', 'category_label',
            'true_priority', 'true_bucket']].reset_index(drop=True),
        header_df.reset_index(drop=True),
        style_df.reset_index(drop=True),
        keyword_df.reset_index(drop=True),
    ], axis=1)
    return out


def make_column_transformer():
    text_pipe = Pipeline([
        ('feats', FeatureUnion([
            ('word', TfidfVectorizer(max_features=3_000, ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True, stop_words='english')),
            ('char', TfidfVectorizer(max_features=1_500, ngram_range=(3, 5), min_df=3,
                                      sublinear_tf=True, analyzer='char_wb')),
        ])),
    ])

    return ColumnTransformer([
        ('text', text_pipe, 'text'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
        ('numeric', Pipeline([
            ('log1p_recipients', FunctionTransformer(log1p_recipients)),
            ('scale', StandardScaler()),
        ]), ['n_recipients'] + [c for c in NUMERIC_COLS if c != 'n_recipients']),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', default='../../gold_priority_labeled_v2.csv')
    ap.add_argument('--out', default='../../models')
    args = ap.parse_args()

    print(f"Loading gold set from {args.gold} …")
    raw = pd.read_csv(args.gold)
    print(f"  {len(raw)} rows")

    df = build_features(raw)
    X = df[['text', 'category_label'] + NUMERIC_COLS]
    y_reg = df['true_priority'].values
    y_clf = df['true_bucket'].values

    # ── Regression: continuous priority score ──────────────────────────
    print("\n[1/2] Regression (Ridge) — predicting continuous true_priority …")
    reg_pipe = Pipeline([
        ('features', make_column_transformer()),
        ('reg', Ridge(alpha=5.0)),
    ])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_pred = cross_val_predict(reg_pipe, X, y_reg, cv=kf)
    mae = mean_absolute_error(y_reg, reg_pred)
    baseline_mae = mean_absolute_error(y_reg, np.full_like(y_reg, y_reg.mean()))
    print(f"  5-fold CV MAE: {mae:.3f}   (baseline — always predict the mean: {baseline_mae:.3f})")

    # ── Classification: low/medium/high bucket ──────────────────────────
    print("\n[2/2] Classification (RandomForest, hyperparameter-tuned) — predicting true_bucket …")
    base_clf_pipe = Pipeline([
        ('features', make_column_transformer()),
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ])
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    tuned_clf = GridSearchCV(base_clf_pipe, RF_PARAM_GRID, cv=inner_cv, scoring='accuracy', n_jobs=-1)

    # Nested CV: hyperparameters are re-selected inside each outer fold using
    # only that fold's training data, so the reported accuracy isn't
    # inflated by tuning against the same data it's evaluated on.
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_pred = cross_val_predict(tuned_clf, X, y_clf, cv=outer_cv)
    acc = accuracy_score(y_clf, clf_pred)
    majority_baseline = pd.Series(y_clf).value_counts(normalize=True).max()
    print(f"  5-fold nested-CV accuracy: {acc:.1%}   (baseline — always predict majority class: {majority_baseline:.1%})")
    print(f"  (for reference: the rule-based heuristic scored 40.3% bucket agreement on this same gold set,")
    print(f"   and a plain LogisticRegression on text-only features scored 70.0%)")
    print("\n" + classification_report(y_clf, clf_pred))
    print("Confusion matrix (rows=true, cols=predicted), labels=[high, low, medium]:")
    print(confusion_matrix(y_clf, clf_pred, labels=['high', 'low', 'medium']))

    # ── Fit final models on all data and save ───────────────────────────
    print("\nFitting final models on full gold set …")
    reg_pipe.fit(X, y_reg)
    tuned_clf.fit(X, y_clf)
    print(f"  Best RandomForest params (selected on full gold set): {tuned_clf.best_params_}")
    clf_pipe = tuned_clf.best_estimator_

    reg_path = f"{args.out}/priority_regressor.joblib"
    clf_path = f"{args.out}/priority_classifier.joblib"
    joblib.dump(reg_pipe, reg_path)
    joblib.dump(clf_pipe, clf_path)
    print(f"Saved -> {reg_path}")
    print(f"Saved -> {clf_path}")


if __name__ == '__main__':
    main()
