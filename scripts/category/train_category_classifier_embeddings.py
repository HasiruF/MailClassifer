"""
train_category_classifier_embeddings.py
─────────────────────────────────────────
Experimental variant of train_category_classifier_headers.py that swaps/augments
TF-IDF with sentence embeddings (all-MiniLM-L6-v2, 384-dim) to test whether
semantic features close the Work/Personal confusion gap that error analysis
found (see PROJECT_STATUS.md — 64% of errors are Personal<->Work, on content
like coworkers forwarding jokes/political opinions/personal reflections
through work email; TF-IDF word/char n-grams can't separate "topic" from
"social context" and neither can embeddings, but embeddings may pick up softer
tone/semantic signal current char n-grams miss).

Caches embeddings to disk (embeddings are slow to compute, deterministic per
text) so repeated CV/hyperparameter runs don't recompute them.

Usage
  python train_category_classifier_embeddings.py \
      --gold    ../../gold_sample_labeled.csv \
      --headers gold_header_features.csv \
      --out     ../..
"""

import argparse
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # Windows cp1252 console can't print the box-drawing chars below

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

_here = __file__.rsplit('\\', 1)[0] or '.'
sys.path.insert(0, _here)
sys.path.insert(0, _here + '\\..\\shared')
from train_category_classifier import clean_text
from header_features import log1p_recipients
from train_category_classifier_headers import NUMERIC_COLS, make_pipeline as make_tfidf_pipeline

EMBED_MODEL = 'all-MiniLM-L6-v2'  # small (~80MB), fast on CPU, 384 dims, strong general-purpose baseline


def get_embeddings(texts, cache_path):
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True).item()
        if cached['texts'] == list(texts):
            print(f"    (loaded {len(texts)} cached embeddings from {cache_path})")
            return cached['embeddings']
    from sentence_transformers import SentenceTransformer
    print(f"    Encoding {len(texts)} texts with {EMBED_MODEL} (first run downloads the model) ...")
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(list(texts), show_progress_bar=True, batch_size=32)
    np.save(cache_path, {'texts': list(texts), 'embeddings': emb}, allow_pickle=True)
    return emb


def make_embedding_pipeline(n_embed_dims):
    """Embeddings (already computed, passed as columns) + numeric header features -> LR."""
    embed_cols = [f'emb_{i}' for i in range(n_embed_dims)]
    numeric_pipe = Pipeline([('log1p', FunctionTransformer(log1p_recipients)), ('scale', StandardScaler())])
    pre = ColumnTransformer([
        ('embed', StandardScaler(), embed_cols),
        ('numeric', numeric_pipe, NUMERIC_COLS),
    ])
    return Pipeline([
        ('features', pre),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)),
    ]), embed_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', default='../../gold_sample_labeled.csv')
    ap.add_argument('--headers', default='gold_header_features.csv')
    ap.add_argument('--out', default='../..')
    ap.add_argument('--cache', default='gold_embeddings_cache.npy')
    args = ap.parse_args()

    gold = pd.read_csv(args.gold)
    gold = gold[gold['true_category'].notna() & (gold['true_category'].str.strip() != '')].copy()
    gold['text'] = (gold['subject'].fillna('') + ' ' + gold['body'].fillna('')).apply(clean_text)

    headers = pd.read_csv(args.headers)
    df = gold.merge(headers, on='file', how='left')
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)
    y = df['true_category']

    print("[1/4] Computing/loading sentence embeddings ...")
    emb = get_embeddings(df['text'].tolist(), args.cache)
    embed_cols = [f'emb_{i}' for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=embed_cols, index=df.index)
    df_emb = pd.concat([df[NUMERIC_COLS], emb_df], axis=1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[2/4] 5-fold CV: embeddings + header features (no TF-IDF)")
    pipe_emb, _ = make_embedding_pipeline(emb.shape[1])
    pred_emb = cross_val_predict(pipe_emb, df_emb, y, cv=cv)
    acc_emb = accuracy_score(y, pred_emb)
    print(f"    Accuracy: {acc_emb:.4f}   Macro-F1: {f1_score(y, pred_emb, average='macro'):.4f}")

    print("\n[3/4] 5-fold CV: TF-IDF word+char + header features (current baseline, for comparison)")
    X_tfidf = df[['text'] + NUMERIC_COLS]
    pred_tfidf = cross_val_predict(make_tfidf_pipeline(), X_tfidf, y, cv=cv)
    acc_tfidf = accuracy_score(y, pred_tfidf)
    print(f"    Accuracy: {acc_tfidf:.4f}   Macro-F1: {f1_score(y, pred_tfidf, average='macro'):.4f}")

    print("\n[4/4] 5-fold CV: TF-IDF word+char + embeddings + header features (combined)")
    from sklearn.pipeline import FeatureUnion
    text_pipe = Pipeline([
        ('feats', FeatureUnion([
            ('word', TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2,
                                       sublinear_tf=True, stop_words='english')),
            ('char', TfidfVectorizer(max_features=8_000, ngram_range=(3, 5), min_df=3,
                                       sublinear_tf=True, analyzer='char_wb')),
        ])),
    ])
    numeric_pipe = Pipeline([('log1p', FunctionTransformer(log1p_recipients)), ('scale', StandardScaler())])
    embed_pipe = StandardScaler()
    pre_combined = ColumnTransformer([
        ('text', text_pipe, 'text'),
        ('embed', embed_pipe, embed_cols),
        ('numeric', numeric_pipe, NUMERIC_COLS),
    ])
    pipe_combined = Pipeline([
        ('features', pre_combined),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)),
    ])
    X_combined = pd.concat([df[['text']], emb_df, df[NUMERIC_COLS]], axis=1)
    pred_combined = cross_val_predict(pipe_combined, X_combined, y, cv=cv)
    acc_combined = accuracy_score(y, pred_combined)
    print(f"    Accuracy: {acc_combined:.4f}   Macro-F1: {f1_score(y, pred_combined, average='macro'):.4f}")

    print("\n── Summary ─────────────────────────────────────────")
    print(f"  TF-IDF + headers (current)          {acc_tfidf:.4f}")
    print(f"  Embeddings + headers (no TF-IDF)     {acc_emb:.4f}")
    print(f"  TF-IDF + embeddings + headers        {acc_combined:.4f}")

    best_name, best_pred, best_acc = max(
        [('tfidf', pred_tfidf, acc_tfidf), ('embed', pred_emb, acc_emb), ('combined', pred_combined, acc_combined)],
        key=lambda t: t[2],
    )
    print(f"\nBest: {best_name} ({best_acc:.4f})")
    labels = sorted(y.unique())
    print(classification_report(y, best_pred, labels=labels, zero_division=0))
    cm = confusion_matrix(y, best_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    print('  ' + '  '.join(f'{c:>14s}' for c in labels))
    for i, row_label in enumerate(labels):
        print(f"  {row_label:14s}  " + '  '.join(f'{v:>14d}' for v in cm[i]))


if __name__ == '__main__':
    main()
