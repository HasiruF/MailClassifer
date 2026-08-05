"""
header_features.py
───────────────────
Domain-agnostic header/metadata features for email classification. Nothing
here references a specific company (no "enron.com" literal) — every feature
is a pattern or relationship that holds on any mailbox, corporate or a
personal Gmail account, so a model trained with these should generalize
beyond the Enron corpus it was built on.

Features
  n_recipients        — To + Cc address count (broadcast vs. 1:1)
  is_reply_or_forward — subject starts with Re:/Fw:/Fwd:
  sender_automated    — local-part of From matches noreply/no-reply/
                        donotreply/mailer-daemon/support/notifications/
                        alerts/updates/digest/newsletter/bounce
  has_list_unsubscribe — List-Unsubscribe header present (RFC 2369 standard
                        for mailing-list mail; near-universal on modern
                        Gmail-era mail, essentially absent on this ~2001
                        corpus predating wide adoption — included for
                        portability, not validated lift on Enron data)
  has_precedence_bulk — Precedence: bulk|list header present (older bulk-mail
                        marker, same caveat as above)
  hour_sin, hour_cos  — cyclical encoding of the local hour the message was
                        sent (from the Date header's own UTC offset, not
                        converted — sender's local-time behavior is the
                        signal, e.g. a personal email sent at 9pm their time).
                        Cyclical so 23:00 and 00:00 read as adjacent, not
                        maximally far apart.
  is_weekend          — sent on Sat/Sun (sender's local time)
  is_business_hours   — sent Mon-Fri, 08:00-18:00 (sender's local time)
"""

import re
import math
from email.utils import parsedate_to_datetime
import numpy as np

_AUTOMATED_LOCAL_PART = re.compile(
    r'^(no[\-_]?reply|do[\-_]?not[\-_]?reply|mailer[\-_]?daemon|support|'
    r'notifications?|alerts?|updates?|digest|newsletter|bounces?|postmaster)',
    re.I
)

_REPLY_FWD_PREFIX = re.compile(r'^\s*(re|fw|fwd)\s*:', re.I)


def _split_header_block(raw_message):
    if not isinstance(raw_message, str):
        return ""
    return raw_message.split('\n\n', 1)[0].split('\r\n\r\n', 1)[0]


def _parse_headers(header_block):
    """Fold continuation lines (start with whitespace) into the preceding field."""
    fields = {}
    current = None
    for line in header_block.splitlines():
        m = re.match(r'^([A-Za-z][A-Za-z0-9\-]*):\s?(.*)', line)
        if m:
            current = m.group(1).lower()
            fields[current] = fields.get(current, '') + m.group(2)
        elif current and line.startswith((' ', '\t')):
            fields[current] += ' ' + line.strip()
    return fields


def _count_addresses(field_value):
    if not field_value:
        return 0
    return len(re.findall(r'[\w.\-+]+@[\w.\-]+', field_value))


def _sender_local_part(from_field):
    m = re.search(r'([\w.\-+]+)@[\w.\-]+', from_field or '')
    return m.group(1) if m else ''


def _parse_date_local(date_str):
    """Parse an RFC 2822 Date header, preserving the sender's own UTC offset
    (never converted to UTC/another timezone) — we want the sender's local
    time-of-day behavior, not an absolute instant."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        return None


def extract_header_features(raw_message, subject=None):
    """Training-side: parse features out of a raw Enron-style message blob."""
    header_block = _split_header_block(raw_message)
    fields = _parse_headers(header_block)

    to_count = _count_addresses(fields.get('to', ''))
    cc_count = _count_addresses(fields.get('cc', ''))

    subj = subject if isinstance(subject, str) else fields.get('subject', '')
    from_addr = fields.get('from', '')
    has_list_unsubscribe = 'list-unsubscribe' in fields
    precedence = fields.get('precedence', '')
    sent_at = _parse_date_local(fields.get('date', ''))

    return _build_features(to_count + cc_count, subj, from_addr, has_list_unsubscribe, precedence, sent_at)


def extract_header_features_from_fields(subject='', to='', cc='', from_addr='',
                                          list_unsubscribe=False, precedence='', sent_at=None):
    """Serving-side: parse features out of already-structured fields (what a
    mail backend/API would hand you directly — no raw header block to split).
    sent_at: a datetime (ideally tz-aware, in the sender's local time) or an
    RFC 2822 / ISO 8601 date string."""
    n_recipients = _count_addresses(to) + _count_addresses(cc)
    if isinstance(sent_at, str):
        sent_at = _parse_date_local(sent_at) or _try_iso(sent_at)
    return _build_features(n_recipients, subject, from_addr, bool(list_unsubscribe), precedence, sent_at)


def _try_iso(date_str):
    try:
        from datetime import datetime
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None


def log1p_recipients(X):
    """FunctionTransformer target for the n_recipients column. Lives here (an
    importable module), not in the training script, so the pickled model can
    be loaded from any script — pickle needs the exact module a function was
    defined in to unpickle it, and a training script run directly gets
    recorded as "__main__", which breaks loading from anywhere else."""
    X = np.asarray(X, dtype=float)
    X[:, 0] = np.log1p(X[:, 0])
    return X


def _build_features(n_recipients, subject, from_addr, has_list_unsubscribe, precedence, sent_at=None):
    is_reply_or_forward = bool(_REPLY_FWD_PREFIX.match(subject or ''))
    local_part = _sender_local_part(from_addr)
    sender_automated = bool(_AUTOMATED_LOCAL_PART.match(local_part))
    has_precedence_bulk = (precedence or '').lower() in ('bulk', 'list')

    if sent_at is not None:
        hour = sent_at.hour + sent_at.minute / 60.0
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        weekday = sent_at.weekday()  # 0=Mon .. 6=Sun
        is_weekend = weekday >= 5
        is_business_hours = (weekday < 5) and (8 <= sent_at.hour < 18)
    else:
        # missing timestamp: neutral defaults (don't inject a false weekday/hour signal)
        hour_sin, hour_cos = 0.0, 0.0
        is_weekend, is_business_hours = False, True

    return {
        'n_recipients': n_recipients,
        'is_reply_or_forward': int(is_reply_or_forward),
        'sender_automated': int(sender_automated),
        'has_list_unsubscribe': int(has_list_unsubscribe),
        'has_precedence_bulk': int(has_precedence_bulk),
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'is_weekend': int(is_weekend),
        'is_business_hours': int(is_business_hours),
    }
