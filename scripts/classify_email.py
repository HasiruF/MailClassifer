"""
classify_email.py
─────────────────
Lightweight inference wrapper — load the saved models and classify emails
one at a time. Import this into your Flask/FastAPI backend.

Usage (standalone test)
  python classify_email.py --models ../models

Usage (as a module)
  from classify_email import EmailClassifier
  clf = EmailClassifier('../models')
  result = clf.classify(subject="Urgent: approval needed", body="Please sign off by EOD.")
  print(result)
  # {
  #   'spam':     {'label': 'ham',  'confidence': 0.97},
  #   'category': {'label': 'Work', 'conf_work': 0.82, 'conf_personal': 0.04, 'conf_other': 0.14},
  #   'priority': {'score': 0.7,    'bucket': 'high'}
  # }
"""

import re, os, argparse, sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, (__file__.rsplit('\\', 1)[0] or '.') + '\\shared')
from header_features import extract_header_features_from_fields
from stylistic_features import extract_stylistic_features, STYLE_COLS
from priority_keyword_features import extract_keyword_features, KEYWORD_COLS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Module-level singleton — loading the lexicon per-call is expensive over
# batch runs (500K+ rows).
_sia = SentimentIntensityAnalyzer()

# Must match NUMERIC_COLS / column order in train_category_classifier_headers.py
# (hour_sin/hour_cos/is_weekend/is_business_hours tried and dropped — no real
# signal on the Enron corpus, see PROJECT_STATUS.md)
NUMERIC_COLS = ['n_recipients', 'is_reply_or_forward', 'sender_automated',
                 'has_list_unsubscribe', 'has_precedence_bulk'] + STYLE_COLS

# ── Text cleaning (must match training) ───────────────────────────────────────

def _clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\b\d[\d\s\-().]{6,}\d\b', ' phone ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Priority model (trained — replaces the old rule-based heuristic) ───────────
# The rule-based heuristic (keyword regex + hand-set weights) was checked
# against a 504-row hand-labeled Enron-only gold set (gold_priority_labeled.csv,
# labeled blind to the heuristic's own output) and only matched human judgment
# 40.3% of the time on the low/medium/high bucket call. A first trained model
# (LogReg on Enron-only data) reached 70.0% same-corpus CV — but a 100-email
# hand-authored ADVERSARIAL test (adversarial_test_set.py, deliberately
# out-of-distribution: modern/casual/personal-register text) exposed that this
# was fit-to-corpus, not real generalization: only 46.0% accuracy out of
# distribution, high recall 3.2% (see PROJECT_STATUS.md item 3e). Root cause
# was training-data diversity, not modeling — confirmed by adding 500 more
# hand-authored, balanced (category x priority), non-Enron training rows
# (gold_priority_labeled_v2.csv, 1004 rows total) with zero overlap with the
# adversarial eval set: adversarial accuracy jumped 46.0% -> 82.0%, high
# recall 3.2% -> 64.5%, same-corpus nested-CV 85.1%. See PROJECT_STATUS.md
# item 3f for the full retraining story and confusion matrices.
PRIORITY_NUMERIC_COLS = ['n_recipients', 'is_reply_or_forward', 'sender_automated'] \
    + STYLE_COLS + ['spam_conf', 'vader_compound'] + KEYWORD_COLS

# ── Classifier wrapper ────────────────────────────────────────────────────────

class EmailClassifier:
    def __init__(self, models_dir):
        spam_path = os.path.join(models_dir, 'spam_classifier.joblib')
        # category_classifier_headers.joblib = text + header/metadata features
        # (recipient count, reply/forward flag, automated-sender pattern,
        # List-Unsubscribe/Precedence:bulk presence). Beat the text-only model
        # (category_classifier_gold.joblib) 61.9% vs 60.8% CV accuracy, and
        # every feature is domain-agnostic (no company-specific signal), so it
        # should generalize to non-Enron mailboxes rather than just memorize
        # "enron.com == Work".
        cat_path  = os.path.join(models_dir, 'category_classifier_headers.joblib')
        prio_reg_path = os.path.join(models_dir, 'priority_regressor.joblib')
        prio_clf_path = os.path.join(models_dir, 'priority_classifier.joblib')

        if not os.path.exists(spam_path):
            raise FileNotFoundError(f"spam model not found: {spam_path}")
        if not os.path.exists(cat_path):
            raise FileNotFoundError(f"category model not found: {cat_path}")
        if not os.path.exists(prio_reg_path):
            raise FileNotFoundError(f"priority regressor not found: {prio_reg_path}")
        if not os.path.exists(prio_clf_path):
            raise FileNotFoundError(f"priority classifier not found: {prio_clf_path}")

        self.spam_pipeline = joblib.load(spam_path)
        self.cat_pipeline  = joblib.load(cat_path)
        self.cat_classes   = self.cat_pipeline.classes_.tolist()
        self.prio_reg      = joblib.load(prio_reg_path)
        self.prio_clf      = joblib.load(prio_clf_path)
        print(f"Models loaded from {models_dir}")

    def classify(self, subject='', body='', to='', cc='', from_addr='',
                 list_unsubscribe=False, precedence='', sent_at=None):
        text = _clean(subject + ' ' + body)

        # spam
        spam_proba = self.spam_pipeline.predict_proba([text])[0]
        spam_idx   = list(self.spam_pipeline.classes_).index(1)
        spam_conf  = float(spam_proba[spam_idx])
        spam_label = 'spam' if spam_conf >= 0.5 else 'ham'

        # category (text + header/metadata features — must match the column
        # shape the ColumnTransformer was fit on in train_category_classifier_headers.py)
        header_feats = extract_header_features_from_fields(
            subject=subject, to=to, cc=cc, from_addr=from_addr,
            list_unsubscribe=list_unsubscribe, precedence=precedence, sent_at=sent_at,
        )
        style_feats = extract_stylistic_features(subject, body)
        cat_input = pd.DataFrame([{**{'text': text}, **header_feats, **style_feats}])[['text'] + NUMERIC_COLS]
        cat_proba  = self.cat_pipeline.predict_proba(cat_input)[0]
        cat_label  = self.cat_classes[int(np.argmax(cat_proba))]
        cat_result = {'label': cat_label}
        for i, cls in enumerate(self.cat_classes):
            cat_result[f'conf_{cls.lower()}'] = round(float(cat_proba[i]), 4)

        # priority (trained model — see PRIORITY_NUMERIC_COLS comment above)
        compound = _sia.polarity_scores(text)['compound']
        keyword_feats = extract_keyword_features(subject, body)
        prio_input = pd.DataFrame([{
            'text': text,
            'category_label': cat_label,
            'n_recipients': header_feats['n_recipients'],
            'is_reply_or_forward': header_feats['is_reply_or_forward'],
            'sender_automated': header_feats['sender_automated'],
            'spam_conf': spam_conf,
            'vader_compound': compound,
            **style_feats,
            **keyword_feats,
        }])
        prio_score = float(np.clip(self.prio_reg.predict(prio_input)[0], 0.1, 1.0))
        prio_bucket = self.prio_clf.predict(prio_input)[0]
        prio = {'score': round(prio_score, 2), 'bucket': prio_bucket}

        # Spam suppresses priority — spam/phishing routinely fakes urgency
        # ("verify now or your account is suspended") specifically to game
        # attention, so a message already flagged as spam shouldn't be able
        # to rank as high-priority regardless of what language it uses.
        # Category is deliberately NOT wired in here: there's no principled
        # basis for "Work" being generically more urgent than "Personal"
        # (a personal emergency should outrank a routine status update), so
        # hard-coding that would be an unvalidated bias, not a fix.
        if spam_label == 'spam':
            prio = {'score': 0.1, 'bucket': 'low', 'note': f'classified as spam (conf={spam_conf:.2f}) — priority suppressed'}

        return {
            'spam':     {'label': spam_label, 'confidence': round(spam_conf, 4)},
            'category': cat_result,
            'priority': prio,
        }

    def classify_batch(self, emails):
        """
        emails: list of dicts with keys 'subject', 'body', 'to' and optionally
        'cc', 'from_addr', 'list_unsubscribe', 'precedence'
        returns: list of result dicts in the same order
        """
        return [self.classify(**e) for e in emails]

# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True, help='Path to models directory')
    args = parser.parse_args()

    clf = EmailClassifier(args.models)

    test_emails = [
        {
            'subject': 'URGENT: Approval needed by EOD',
            'body':    'Hi, please sign off on the attached contract before end of day. Legal is waiting.',
            'to':      'ceo@enron.com'
        },
        {
            'subject': 'Weekly newsletter - Energy market digest',
            'body':    'This is your weekly roundup. Unsubscribe at any time. No action needed.',
            'to':      'all@enron.com',
            'from_addr': 'digest@energymarketnews.com',
            'list_unsubscribe': True,
            'precedence': 'bulk',
        },
        {
            'subject': 'Lunch on Friday?',
            'body':    'Hey, are you free for lunch this Friday? Let me know.',
            'to':      'friend@gmail.com'
        },
        {
            'subject': 'You have won a prize!!!',
            'body':    'Congratulations! Click here to claim your $1000 reward. Limited time offer.',
            'to':      'victim@email.com'
        },
    ]

    print("\n── Test classifications ─────────────────────────────────────\n")
    for i, email in enumerate(test_emails, 1):
        result = clf.classify(**email)
        print(f"Email {i}: \"{email['subject']}\"")
        print(f"  Spam:     {result['spam']['label']:5s}  (conf: {result['spam']['confidence']:.2f})")
        print(f"  Category: {result['category']['label']}")
        conf_str = '  '.join(f"{k}: {v:.2f}" for k, v in result['category'].items() if k.startswith('conf_'))
        print(f"            [{conf_str}]")
        print(f"  Priority: {result['priority']['score']}  ({result['priority']['bucket']})")
        if 'note' in result['priority']:
            print(f"            {result['priority']['note']}")
        print()
