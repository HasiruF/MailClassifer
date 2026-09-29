"""
dump_sparse_parity_fixture.py
─────────────────────────────
Writes sklearn's own feature vectors (sparse) for a few emails to
app/frontend/scripts/fixtures/sparse_parity.json.
app/frontend/scripts/sparse_parity_check.mjs rebuilds the same vectors with
the TypeScript port + toSparse() and checks they match. These are the
vectors that land in the backend's base training matrices, so a match means
corrections and base rows live in the same feature space.

Usage (from scripts/):  python dump_sparse_parity_fixture.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'shared'))

from classify_email import _clean, _sia, NUMERIC_COLS as CATEGORY_NUMERIC_COLS, PRIORITY_NUMERIC_COLS
from header_features import extract_header_features_from_fields
from stylistic_features import extract_stylistic_features
from priority_keyword_features import extract_keyword_features

EMAILS = [
    dict(id='p1', subject='URGENT: Approval needed by EOD',
         body='Hi, please sign off on the attached contract before end of day. Legal is waiting.',
         to='ceo@company.com', fromAddr='pm@company.com'),
    dict(id='p2', subject='Lunch on Friday?', body='Hey, are you free for lunch this Friday? Let me know.',
         to='friend@gmail.com', fromAddr='me@gmail.com'),
    dict(id='p3', subject='You have won a prize!!!',
         body='Congratulations! Click here to claim your $1000 reward. Limited time offer.',
         to='a@b.com, c@d.com', fromAddr='noreply@promo.example'),
]


def sparse_row(matrix):
    row = sp.csr_matrix(matrix)
    row.eliminate_zeros()
    return {'indices': row.indices.tolist(), 'values': row.data.tolist()}


def main():
    models = os.path.join(REPO, 'models')
    spam_pipe = joblib.load(os.path.join(models, 'spam_classifier.joblib'))
    cat_pipe = joblib.load(os.path.join(models, 'category_classifier_headers.joblib'))
    prio_pipe = joblib.load(os.path.join(models, 'priority_classifier.joblib'))

    out = []
    for e in EMAILS:
        text = _clean(e['subject'] + ' ' + e['body'])
        header = extract_header_features_from_fields(subject=e['subject'], to=e['to'], from_addr=e['fromAddr'])
        style = extract_stylistic_features(e['subject'], e['body'])
        cat_input = pd.DataFrame([{'text': text, **header, **style}])[['text'] + CATEGORY_NUMERIC_COLS]
        category_label = cat_pipe.predict(cat_input)[0]
        spam_conf = float(spam_pipe.predict_proba([text])[0][list(spam_pipe.classes_).index(1)])
        compound = _sia.polarity_scores(text)['compound']
        prio_input = pd.DataFrame([{
            'text': text, 'category_label': category_label,
            'n_recipients': header['n_recipients'], 'is_reply_or_forward': header['is_reply_or_forward'],
            'sender_automated': header['sender_automated'], 'spam_conf': spam_conf, 'vader_compound': compound,
            **style, **extract_keyword_features(e['subject'], e['body']),
        }])[['text', 'category_label'] + PRIORITY_NUMERIC_COLS]
        out.append({
            'email': e,
            'spam': sparse_row(spam_pipe.named_steps['tfidf'].transform([text])),
            'category': sparse_row(cat_pipe.named_steps['features'].transform(cat_input)),
            'priority': {
                **sparse_row(prio_pipe.named_steps['features'].transform(prio_input)),
                'categoryLabel': str(category_label), 'spamConf': spam_conf, 'vaderCompound': compound,
            },
        })

    path = os.path.join(REPO, 'app', 'frontend', 'scripts', 'fixtures', 'sparse_parity.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {path}")


if __name__ == '__main__':
    main()
