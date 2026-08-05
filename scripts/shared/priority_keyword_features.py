"""
priority_keyword_features.py
─────────────────────────────
Explicit urgency/deadline/bulk-mail keyword counts, kept as hand-authored
regex counts (not left to TF-IDF to rediscover) because a trained linear
model on only 504 gold rows has too little data to reliably learn these on
its own from raw text — a RandomForest combined with these explicit counts
scored 74.8% forced-accuracy vs 70.0% for a linear model on text alone (see
PROJECT_STATUS.md). Adding these same counts to the linear model alone (no
tree ensemble) made things worse (69.0%) — the lift only shows up when a
model that can use interaction effects (this keyword count only matters
combined with that header signal) gets to see them.
"""

import re

KEYWORD_COLS = ['kw_high', 'kw_med', 'kw_low', 'excl_subj']

_HIGH_KW = re.compile(
    r'\b(urgent|asap|emergency|critical|escalat\w*|breach|outage|'
    r'deadline|overdue|end of day|eod|cob|due by|expires?|right away|'
    r'time sensitive|crisis|disaster|lawsuit|legal action|'
    r'margin call|default|liquidat\w*)\b', re.I
)
_MED_KW = re.compile(
    r'\b(action required|please respond|please advise|need your (approval|input|sign[\s\-]?off)|'
    r'response needed|follow up|reminder)\b', re.I
)
_LOW_KW = re.compile(
    r'\b(unsubscribe|newsletter|fyi|for your information|no action needed|digest|'
    r'weekly update|monthly report|do not reply|automatic notification|'
    r'out of office|calendar invite|meeting reminder|rescheduled)\b', re.I
)


def extract_keyword_features(subject, body):
    subject = subject or ''
    text = subject + ' ' + (body or '')
    return {
        'kw_high': len(_HIGH_KW.findall(text)),
        'kw_med': len(_MED_KW.findall(text)),
        'kw_low': len(_LOW_KW.findall(text)),
        'excl_subj': subject.count('!'),
    }
