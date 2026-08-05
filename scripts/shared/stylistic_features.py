"""
stylistic_features.py
──────────────────────
Explicit tone/register markers extracted from subject+body text — informal
vs. formal greetings/closings, exclamation density, ALL-CAPS word count,
family/relationship words, question density. Tested against the category
classifier's dominant error mode (Personal<->Work confusion, ~64% of all
errors — see PROJECT_STATUS.md): these features carry real, verified signal
that char n-grams alone weren't capturing explicitly. 5-fold CV: 74.1% -> 76.8%.

Domain-agnostic like header_features.py — no company-specific patterns, just
generic style markers that hold in any mailbox.
"""

import re
import numpy as np
import pandas as pd

_INFORMAL_GREETING = re.compile(r'\b(hey|hi\s+\w+|hiya|yo|sup)\b', re.I)
_FORMAL_GREETING = re.compile(r'\b(dear\s+\w+|to\s+whom\s+it\s+may\s+concern|greetings)\b', re.I)
_FAMILY_WORDS = re.compile(r'\b(mom|dad|mother|father|honey|sweetie|love\s+you|xoxo|dearest)\b', re.I)
_FORMAL_CLOSING = re.compile(r'\b(regards|sincerely|best\s+regards|respectfully)\b', re.I)
_INFORMAL_CLOSING = re.compile(r'\b(cheers|talk\s+soon|luv|later|ttyl)\b', re.I)
_CAPS_WORD = re.compile(r'\b[A-Z]{3,}\b')

STYLE_COLS = [
    'text_len_log', 'excl_density', 'caps_word_count', 'informal_greeting',
    'formal_greeting', 'family_words', 'formal_closing', 'informal_closing',
    'question_count',
]


def extract_stylistic_features(subject, body):
    subject = subject or ''
    body = body or ''
    full = subject + ' ' + body
    text_len = len(full)
    excl_density = full.count('!') / max(text_len, 1) * 1000
    caps_words = len(_CAPS_WORD.findall(full))
    informal_greeting = bool(_INFORMAL_GREETING.search(body[:100]))
    formal_greeting = bool(_FORMAL_GREETING.search(body[:100]))
    family_words = bool(_FAMILY_WORDS.search(full))
    formal_closing = bool(_FORMAL_CLOSING.search(body[-200:]))
    informal_closing = bool(_INFORMAL_CLOSING.search(body[-200:]))
    question_count = full.count('?')

    return {
        'text_len_log': np.log1p(text_len),
        'excl_density': excl_density,
        'caps_word_count': min(caps_words, 10),
        'informal_greeting': int(informal_greeting),
        'formal_greeting': int(formal_greeting),
        'family_words': int(family_words),
        'formal_closing': int(formal_closing),
        'informal_closing': int(informal_closing),
        'question_count': min(question_count, 5),
    }


def add_stylistic_columns(df, subject_col='subject', body_col='body'):
    """Vectorized helper for training scripts: returns df with STYLE_COLS appended."""
    feats = df.apply(
        lambda row: pd.Series(extract_stylistic_features(row[subject_col], row[body_col])),
        axis=1,
    )
    return pd.concat([df, feats], axis=1)
