"""
dump_tfidf_fixtures.py
───────────────────────
One-off cross-validation fixture generator: runs the REAL fitted
TfidfVectorizers (word + char_wb, from category_classifier_headers.joblib)
on a handful of test strings and dumps their exact output vectors to JSON,
so the JS port in app/frontend/src/inference/tfidf.ts can be checked against
ground truth via Node (see app/frontend/scripts/verify_tfidf.ts).

Usage
  python dump_tfidf_fixtures.py --models ../models --out /tmp/tfidf_fixtures.json
"""
import argparse
import json
import sys

import joblib

sys.path.insert(0, (__file__.rsplit('\\', 1)[0].rsplit('/', 1)[0] or '.') + '/shared')

TEST_TEXTS = [
    "urgent approval needed by eod hi please sign off on the attached contract before end of day legal is waiting",
    "weekly newsletter energy market digest this is your weekly roundup unsubscribe at any time no action needed",
    "lunch on friday hey are you free for lunch this friday let me know",
    "you have won a prize congratulations click here to claim your reward limited time offer",
    "re quarterly numbers hi team attached is the q reforecast let me know if you have questions thanks",
    "",
    "a",
    "urgent urgent urgent asap asap deadline",
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='../models')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    pipeline = joblib.load(args.models + '/category_classifier_headers.joblib')
    ct = pipeline.named_steps['features']
    fu = ct.named_transformers_['text'].named_steps['feats']
    word_vec = dict(fu.transformer_list)['word']
    char_vec = dict(fu.transformer_list)['char']

    fixtures = []
    for text in TEST_TEXTS:
        word_x = word_vec.transform([text]).toarray()[0]
        char_x = char_vec.transform([text]).toarray()[0]
        # sparse: only nonzero entries, to keep the fixture file small
        fixtures.append({
            'text': text,
            'word_nonzero': {int(i): float(v) for i, v in enumerate(word_x) if v != 0.0},
            'char_nonzero': {int(i): float(v) for i, v in enumerate(char_x) if v != 0.0},
        })

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(fixtures, f)
    print(f"Wrote {len(fixtures)} fixtures -> {args.out}")
