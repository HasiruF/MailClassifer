"""
export_onnx.py
──────────────
Converts the three trained sklearn pipelines used at serving time
(spam_classifier, category_classifier_headers, priority_classifier) to ONNX,
so the frontend can run inference entirely client-side via onnxruntime-web —
raw email text never leaves the browser.

spam_classifier is a plain Pipeline(TfidfVectorizer(word ngrams), LogisticRegression)
and converts to ONNX in one shot — the TF-IDF vectorizer is embedded directly
in the ONNX graph as native tokenizer/TF-IDF ops, so it takes raw cleaned
text straight in.

category_classifier_headers and priority_classifier are NOT converted whole.
Both use a char_wb-analyzer TfidfVectorizer (word-boundary character
n-grams), and skl2onnx's TfidfVectorizer converter explicitly does not
support analyzer='char_wb' (raises NotImplementedError — verified against
skl2onnx 1.20.0's operator_converters/text_vectoriser.py). Dropping the char
n-gram branch isn't an option either — it's a validated +5.9pp accuracy
contributor per train_category_classifier_headers.py's own ablation log, and
silently shipping a worse model than what was trained is exactly the kind
of shortcut this project's docs repeatedly flag as a mistake.

So instead, for these two, we export a *headless* ONNX graph — just the
final fitted classifier (LogisticRegression / RandomForestClassifier) on a
single dense float vector input, with a JSON sidecar (*.vocab.json)
containing everything needed to build that vector by hand: word/char
TF-IDF vocabulary + idf weights, the numeric StandardScaler's mean_/scale_,
and (priority only) the OneHotEncoder's category order. The JS port lives
in app/frontend/src/inference/features.ts and reproduces sklearn's TF-IDF
math exactly (tokenization, stopword removal, n-grams, sublinear tf, idf,
L2 norm — all standard, well-documented, and independent of any custom
Python function, unlike the log1p wrinkle below).

The one other custom-function wrinkle: both pipelines apply a Python
FunctionTransformer (log1p_recipients, in shared/header_features.py) to the
n_recipients column before scaling. It only touches column 0 of the numeric
block in place (see its docstring) — so JS applies Math.log1p(n_recipients)
itself before scaling, and we never need to convert that function at all.

Usage
  python export_onnx.py --models ../models --out ../models/onnx
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, (__file__.rsplit(os.sep, 1)[0] or '.') + os.sep + 'shared')
from header_features import extract_header_features_from_fields
from stylistic_features import extract_stylistic_features, STYLE_COLS
from priority_keyword_features import extract_keyword_features, KEYWORD_COLS

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType, FloatTensorType
import onnxruntime as rt

CATEGORY_NUMERIC_COLS = ['n_recipients', 'is_reply_or_forward', 'sender_automated',
                          'has_list_unsubscribe', 'has_precedence_bulk'] + STYLE_COLS
PRIORITY_NUMERIC_COLS = ['n_recipients', 'is_reply_or_forward', 'sender_automated'] \
    + STYLE_COLS + ['spam_conf', 'vader_compound'] + KEYWORD_COLS


def _clean(text):
    import re
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\b\d[\d\s\-().]{6,}\d\b', ' phone ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _onnx_predict_proba(sess, feed, classes):
    # zipmap=False: output_probability is a plain [N, n_classes] float tensor
    # (column order == classes, the same order predict_proba uses) — not a
    # sequence-of-maps. onnxruntime-node's native binding can't materialize
    # that "non-tensor" sequence type at all ("Non tensor type is temporarily
    # not supported"), and it's extra work to unwrap in onnxruntime-web too,
    # so all three classifiers are exported with zipmap=False and classes
    # order is carried separately in the JSON sidecar instead.
    out_names = [o.name for o in sess.get_outputs()]
    results = sess.run(out_names, feed)
    label = results[0][0]
    proba_row = results[1][0]
    return label, {cls: float(p) for cls, p in zip(classes, proba_row)}


def _tfidf_json(vectorizer):
    return {
        'vocabulary': {tok: int(idx) for tok, idx in vectorizer.vocabulary_.items()},
        'idf': vectorizer.idf_.tolist(),
        'ngram_range': list(vectorizer.ngram_range),
        'analyzer': vectorizer.analyzer,
        'stop_words': sorted(_english_stop_words()) if vectorizer.stop_words == 'english' else [],
    }


def _english_stop_words():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return list(ENGLISH_STOP_WORDS)


# ── spam: also exported headless (word-only TF-IDF, no char branch, no
# ColumnTransformer — the whole pipeline is just Pipeline(tfidf, clf)) ──
#
# Originally this converted whole (TfidfVectorizer word ngrams IS supported
# by skl2onnx, unlike char_wb), and it worked — but not bit-exactly:
# skl2onnx's own TfidfVectorizer converter docstring admits "the ONNX
# version does not produce the exact same results" (different regex engine
# — onnxruntime uses re2, Python uses its own re — plus other small
# tokenization differences). Measured up to a 0.016 proba gap on some
# inputs, same sklearn model vs its ONNX conversion, in pure Python with no
# JS involved (see the session that added this comment). Since the same
# word-ngram TF-IDF math is already ported and bit-exactly verified for
# category/priority (tfidf.ts, cross-checked against sklearn to 0.000000 —
# see app/frontend/scripts/verify_tfidf.ts), reusing it here instead of the
# in-graph TfidfVectorizer removes that drift entirely for free.

def export_spam(models_dir, out_dir):
    print("\n[spam] Loading models/spam_classifier.joblib …")
    pipeline = joblib.load(os.path.join(models_dir, 'spam_classifier.joblib'))
    word_vec = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']

    total_dim = len(word_vec.vocabulary_)
    assert total_dim == clf.coef_.shape[1], f"dim mismatch {total_dim} vs {clf.coef_.shape[1]}"

    onnx_model = convert_sklearn(
        clf,
        initial_types=[('features', FloatTensorType([None, total_dim]))],
        options={id(clf): {'zipmap': False}},
        target_opset=17,
    )
    out_path = os.path.join(out_dir, 'spam_classifier.onnx')
    with open(out_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"[spam] Saved -> {out_path}  (headless, dim={total_dim})")

    vocab_path = os.path.join(out_dir, 'spam_classifier.vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'word': _tfidf_json(word_vec),
            'classes': pipeline.classes_.tolist(),  # output_probability column order (0=ham, 1=spam)
        }, f)
    print(f"[spam] Saved -> {vocab_path}")

    texts = [
        _clean("URGENT: Approval needed by EOD Hi, please sign off on the attached contract before end of day. Legal is waiting."),
        _clean("Weekly newsletter - Energy market digest This is your weekly roundup. Unsubscribe at any time. No action needed."),
        _clean("You have won a prize!!! Congratulations! Click here to claim your $1000 reward. Limited time offer."),
        _clean("Lunch on Friday? Hey, are you free for lunch this Friday? Let me know."),
    ]
    sess = rt.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    max_diff = 0.0
    spam_idx = list(pipeline.classes_).index(1)
    for t in texts:
        sk_proba = pipeline.predict_proba([t])[0]
        vec = word_vec.transform([t]).toarray()[0].astype(np.float32)
        feed = {'features': vec.reshape(1, -1)}
        _, onnx_proba = _onnx_predict_proba(sess, feed, pipeline.classes_)
        diff = abs(sk_proba[spam_idx] - onnx_proba[1])
        max_diff = max(max_diff, diff)
    print(f"[spam] Max |sklearn - onnx| proba diff over {len(texts)} rows: {max_diff:.6f}")
    return max_diff


# ── category / priority: headless classifier + JSON feature-building sidecar ──

def export_category(models_dir, out_dir):
    print("\n[category] Loading models/category_classifier_headers.joblib …")
    pipeline = joblib.load(os.path.join(models_dir, 'category_classifier_headers.joblib'))

    ct = pipeline.named_steps['features']
    fu = ct.named_transformers_['text'].named_steps['feats']
    word_vec = dict(fu.transformer_list)['word']
    char_vec = dict(fu.transformer_list)['char']
    scaler = ct.named_transformers_['numeric'].named_steps['scale']
    clf = pipeline.named_steps['clf']

    total_dim = len(word_vec.vocabulary_) + len(char_vec.vocabulary_) + len(scaler.mean_)
    assert total_dim == clf.coef_.shape[1], f"dim mismatch {total_dim} vs {clf.coef_.shape[1]}"

    onnx_model = convert_sklearn(
        clf,
        initial_types=[('features', FloatTensorType([None, total_dim]))],
        options={id(clf): {'zipmap': False}},
        target_opset=17,
    )
    out_path = os.path.join(out_dir, 'category_classifier.onnx')
    with open(out_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"[category] Saved -> {out_path}  (headless, dim={total_dim})")

    vocab_path = os.path.join(out_dir, 'category_classifier.vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'word': _tfidf_json(word_vec),
            'char': _tfidf_json(char_vec),
            'numeric_cols': CATEGORY_NUMERIC_COLS,
            'numeric_mean': scaler.mean_.tolist(),
            'numeric_scale': scaler.scale_.tolist(),
            'classes': pipeline.classes_.tolist(),  # output_probability column order
        }, f)
    print(f"[category] Saved -> {vocab_path}")

    def build_vector(text, header_feats, style_feats):
        word_x = word_vec.transform([text]).toarray()[0]
        char_x = char_vec.transform([text]).toarray()[0]
        merged = {**header_feats, **style_feats}
        numeric_row = np.array([[merged[c] for c in CATEGORY_NUMERIC_COLS]], dtype=float)
        numeric_row[:, 0] = np.log1p(numeric_row[:, 0])  # n_recipients is col 0
        numeric_scaled = scaler.transform(numeric_row)[0]
        return np.concatenate([word_x, char_x, numeric_scaled]).astype(np.float32)

    test_rows = [
        dict(subject='URGENT: Approval needed by EOD',
             body='Hi, please sign off on the attached contract before end of day. Legal is waiting.',
             to='ceo@enron.com'),
        dict(subject='Weekly newsletter - Energy market digest',
             body='This is your weekly roundup. Unsubscribe at any time. No action needed.',
             to='all@enron.com', from_addr='digest@energymarketnews.com',
             list_unsubscribe=True, precedence='bulk'),
        dict(subject='Lunch on Friday?', body='Hey, are you free for lunch this Friday? Let me know.',
             to='friend@gmail.com'),
    ]
    sess = rt.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    max_diff_vector = 0.0  # manual vector build vs true ColumnTransformer output
    max_diff_onnx = 0.0    # onnx headless clf vs sklearn clf, on the manual vector
    for row in test_rows:
        text = _clean(row['subject'] + ' ' + row['body'])
        header_feats = extract_header_features_from_fields(
            subject=row['subject'], to=row.get('to', ''), cc=row.get('cc', ''),
            from_addr=row.get('from_addr', ''), list_unsubscribe=row.get('list_unsubscribe', False),
            precedence=row.get('precedence', ''),
        )
        style_feats = extract_stylistic_features(row['subject'], row['body'])

        sk_input = pd.DataFrame([{**{'text': text}, **header_feats, **style_feats}])[['text'] + CATEGORY_NUMERIC_COLS]
        true_proba = pipeline.predict_proba(sk_input)[0]

        vec = build_vector(text, header_feats, style_feats)
        sk_clf_proba = clf.predict_proba(vec.reshape(1, -1))[0]
        for i in range(len(pipeline.classes_)):
            max_diff_vector = max(max_diff_vector, abs(true_proba[i] - sk_clf_proba[i]))

        feed = {'features': vec.reshape(1, -1)}
        _, onnx_proba = _onnx_predict_proba(sess, feed, pipeline.classes_)
        for i, cls in enumerate(pipeline.classes_):
            max_diff_onnx = max(max_diff_onnx, abs(sk_clf_proba[i] - onnx_proba[cls]))

    print(f"[category] Max diff manual-vector vs true pipeline: {max_diff_vector:.6f}")
    print(f"[category] Max diff onnx vs sklearn clf (same vector): {max_diff_onnx:.6f}")
    return max(max_diff_vector, max_diff_onnx)


def export_priority(models_dir, out_dir):
    print("\n[priority] Loading models/priority_classifier.joblib …")
    pipeline = joblib.load(os.path.join(models_dir, 'priority_classifier.joblib'))

    ct = pipeline.named_steps['features']
    fu = ct.named_transformers_['text'].named_steps['feats']
    word_vec = dict(fu.transformer_list)['word']
    char_vec = dict(fu.transformer_list)['char']
    ohe = ct.named_transformers_['cat']
    categories = ohe.categories_[0].tolist()
    scaler = ct.named_transformers_['numeric'].named_steps['scale']
    clf = pipeline.named_steps['clf']

    total_dim = len(word_vec.vocabulary_) + len(char_vec.vocabulary_) + len(categories) + len(scaler.mean_)
    assert total_dim == clf.n_features_in_, f"dim mismatch {total_dim} vs {clf.n_features_in_}"

    onnx_model = convert_sklearn(
        clf,
        initial_types=[('features', FloatTensorType([None, total_dim]))],
        options={id(clf): {'zipmap': False}},
        target_opset=17,
    )
    out_path = os.path.join(out_dir, 'priority_classifier.onnx')
    with open(out_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"[priority] Saved -> {out_path}  (headless, dim={total_dim})")

    vocab_path = os.path.join(out_dir, 'priority_classifier.vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'word': _tfidf_json(word_vec),
            'char': _tfidf_json(char_vec),
            'categories': categories,
            'numeric_cols': PRIORITY_NUMERIC_COLS,
            'numeric_mean': scaler.mean_.tolist(),
            'numeric_scale': scaler.scale_.tolist(),
            'classes': pipeline.classes_.tolist(),  # output_probability column order
        }, f)
    print(f"[priority] Saved -> {vocab_path}")

    def build_vector(text, category_label, numeric_values):
        word_x = word_vec.transform([text]).toarray()[0]
        char_x = char_vec.transform([text]).toarray()[0]
        cat_x = np.array([1.0 if c == category_label else 0.0 for c in categories])
        numeric_row = np.array([[numeric_values[c] for c in PRIORITY_NUMERIC_COLS]], dtype=float)
        numeric_row[:, 0] = np.log1p(numeric_row[:, 0])  # n_recipients is col 0
        numeric_scaled = scaler.transform(numeric_row)[0]
        return np.concatenate([word_x, char_x, cat_x, numeric_scaled]).astype(np.float32)

    test_rows = [
        dict(subject='URGENT: Approval needed by EOD',
             body='Hi, please sign off on the attached contract before end of day. Legal is waiting.',
             to='ceo@enron.com', category_label='Work', spam_conf=0.02, vader_compound=0.1),
        dict(subject='Weekly newsletter - Energy market digest',
             body='This is your weekly roundup. Unsubscribe at any time. No action needed.',
             to='all@enron.com', from_addr='digest@energymarketnews.com',
             list_unsubscribe=True, precedence='bulk', category_label='Other',
             spam_conf=0.05, vader_compound=0.0),
        dict(subject='Lunch on Friday?', body='Hey, are you free for lunch this Friday? Let me know.',
             to='friend@gmail.com', category_label='Personal', spam_conf=0.01, vader_compound=0.4),
    ]
    sess = rt.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    max_diff_vector = 0.0
    max_diff_onnx = 0.0
    label_mismatches = 0
    for row in test_rows:
        text = _clean(row['subject'] + ' ' + row['body'])
        header_feats = extract_header_features_from_fields(
            subject=row['subject'], to=row.get('to', ''), cc=row.get('cc', ''),
            from_addr=row.get('from_addr', ''), list_unsubscribe=row.get('list_unsubscribe', False),
            precedence=row.get('precedence', ''),
        )
        style_feats = extract_stylistic_features(row['subject'], row['body'])
        keyword_feats = extract_keyword_features(row['subject'], row['body'])

        numeric_values = {
            'n_recipients': header_feats['n_recipients'],
            'is_reply_or_forward': header_feats['is_reply_or_forward'],
            'sender_automated': header_feats['sender_automated'],
            'spam_conf': row['spam_conf'],
            'vader_compound': row['vader_compound'],
            **style_feats,
            **keyword_feats,
        }

        sk_input = pd.DataFrame([{
            'text': text, 'category_label': row['category_label'], **numeric_values,
        }])
        true_proba = pipeline.predict_proba(sk_input)[0]
        true_label = pipeline.predict(sk_input)[0]

        vec = build_vector(text, row['category_label'], numeric_values)
        sk_clf_proba = clf.predict_proba(vec.reshape(1, -1))[0]
        for i in range(len(pipeline.classes_)):
            max_diff_vector = max(max_diff_vector, abs(true_proba[i] - sk_clf_proba[i]))

        feed = {'features': vec.reshape(1, -1)}
        onnx_label, onnx_proba = _onnx_predict_proba(sess, feed, pipeline.classes_)
        if onnx_label != true_label:
            label_mismatches += 1
            print(f"[priority]   LABEL MISMATCH true={true_label} onnx={onnx_label} (row: {row['subject']!r})")
        for i, cls in enumerate(pipeline.classes_):
            max_diff_onnx = max(max_diff_onnx, abs(sk_clf_proba[i] - onnx_proba[cls]))

    print(f"[priority] Max diff manual-vector vs true pipeline: {max_diff_vector:.6f}")
    print(f"[priority] Max diff onnx vs sklearn clf (same vector): {max_diff_onnx:.6f}  ({label_mismatches} label mismatches)")
    return max(max_diff_vector, max_diff_onnx)


# priority_regressor.joblib fits its own ColumnTransformer (make_column_transformer()
# is called once per pipeline in train_priority_classifier.py) but on the exact same
# X/y — verified numerically identical vocab/idf/scaler to priority_classifier's, so
# this reuses priority_classifier.vocab.json rather than duplicating a ~700KB sidecar.
def export_priority_regressor(models_dir, out_dir):
    print("\n[priority_regressor] Loading models/priority_regressor.joblib …")
    pipeline = joblib.load(os.path.join(models_dir, 'priority_regressor.joblib'))

    ct = pipeline.named_steps['features']
    fu = ct.named_transformers_['text'].named_steps['feats']
    word_vec = dict(fu.transformer_list)['word']
    char_vec = dict(fu.transformer_list)['char']
    ohe = ct.named_transformers_['cat']
    categories = ohe.categories_[0].tolist()
    scaler = ct.named_transformers_['numeric'].named_steps['scale']
    reg = pipeline.named_steps['reg']

    total_dim = len(word_vec.vocabulary_) + len(char_vec.vocabulary_) + len(categories) + len(scaler.mean_)
    assert total_dim == reg.coef_.shape[0], f"dim mismatch {total_dim} vs {reg.coef_.shape[0]}"

    onnx_model = convert_sklearn(
        reg,
        initial_types=[('features', FloatTensorType([None, total_dim]))],
        target_opset=17,
    )
    out_path = os.path.join(out_dir, 'priority_regressor.onnx')
    with open(out_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"[priority_regressor] Saved -> {out_path}  (headless, dim={total_dim}, reuses priority_classifier.vocab.json)")

    def build_vector(text, category_label, numeric_values):
        word_x = word_vec.transform([text]).toarray()[0]
        char_x = char_vec.transform([text]).toarray()[0]
        cat_x = np.array([1.0 if c == category_label else 0.0 for c in categories])
        numeric_row = np.array([[numeric_values[c] for c in PRIORITY_NUMERIC_COLS]], dtype=float)
        numeric_row[:, 0] = np.log1p(numeric_row[:, 0])
        numeric_scaled = scaler.transform(numeric_row)[0]
        return np.concatenate([word_x, char_x, cat_x, numeric_scaled]).astype(np.float32)

    row = dict(subject='URGENT: Approval needed by EOD',
               body='Hi, please sign off on the attached contract before end of day. Legal is waiting.',
               to='ceo@enron.com', category_label='Work', spam_conf=0.02, vader_compound=0.1)
    text = _clean(row['subject'] + ' ' + row['body'])
    header_feats = extract_header_features_from_fields(subject=row['subject'], to=row['to'])
    style_feats = extract_stylistic_features(row['subject'], row['body'])
    keyword_feats = extract_keyword_features(row['subject'], row['body'])
    numeric_values = {
        'n_recipients': header_feats['n_recipients'],
        'is_reply_or_forward': header_feats['is_reply_or_forward'],
        'sender_automated': header_feats['sender_automated'],
        'spam_conf': row['spam_conf'], 'vader_compound': row['vader_compound'],
        **style_feats, **keyword_feats,
    }
    sk_input = pd.DataFrame([{'text': text, 'category_label': row['category_label'], **numeric_values}])
    true_score = pipeline.predict(sk_input)[0]

    vec = build_vector(text, row['category_label'], numeric_values)
    sess = rt.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    onnx_score = sess.run(None, {'features': vec.reshape(1, -1)})[0][0][0]
    diff = abs(float(true_score) - float(onnx_score))
    print(f"[priority_regressor] sklearn={true_score:.4f} onnx={onnx_score:.4f} diff={diff:.6f}")
    return diff


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='../models')
    ap.add_argument('--out', default='../models/onnx')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    diffs = {
        'spam': export_spam(args.models, args.out),
        'category': export_category(args.models, args.out),
        'priority': export_priority(args.models, args.out),
        'priority_regressor': export_priority_regressor(args.models, args.out),
    }

    print("\n── Summary ──────────────────────────────────────────────")
    for name, diff in diffs.items():
        status = 'OK' if diff < 1e-3 else 'CHECK'
        print(f"  {name:10s} max diff = {diff:.6f}  [{status}]")
