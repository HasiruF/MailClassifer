"""
train_spam_classifier.py
────────────────────────
Trains a spam / ham classifier on the SpamAssassin Kaggle dataset.

Outputs
  models/spam_classifier.joblib   — saved pipeline (TF-IDF + LR)
  predictions/spam_predictions.csv — Enron emails labelled spam/ham with confidence

Usage
  python train_spam_classifier.py \
      --spam   path/to/spam_assassin.csv \
      --enron  path/to/emails.csv \
      --out    output_dir          (default: current directory)

Expected SpamAssassin CSV columns:  text, label   (label = 'spam' or 'ham', or 0/1)
Expected Enron CSV columns:         file, message
"""

import argparse, re, os, sys
import pandas as pd
import numpy as np
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # Windows consoles default to cp1252, which can't print the arrows/± below
from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics           import classification_report, confusion_matrix

# ── Text cleaning ──────────────────────────────────────────────────────────────
#
# This SpamAssassin export flattens each raw mbox message to a single line
# with no newlines, so there's no blank-line boundary to cleanly split headers
# from body (the trick parse_enron_message uses below). Left alone, mail
# routing metadata (Received: chains, Message-Id, Mime-Version, ...) leaks
# into training as if it were content — and a model trained on that learns
# "this specific 2002 collection's mail servers" rather than actual spam
# language, which is exactly why the first run of this script flagged 88% of
# unrelated Enron corporate email as spam. _ROUTING_WORDS strips the
# mail-transport vocabulary (header field names + common MTA software names)
# that clean_text's normal punctuation/digit stripping doesn't touch, since
# they survive as ordinary lowercase words.

_ROUTING_WORDS = re.compile(
    r'\b(received|delivered|return|path|message|mime|version|content|type|'
    r'transfer|encoding|disposition|precedence|errors|references|reply|'
    r'organization|status|esmtp|smtp|postfix|sendmail|qmail|exim|fetchmail|'
    r'localhost|imap|pop3|'
    # Specific leaked hostnames/domains found via coefficient inspection —
    # this dataset's own routing/mailing-list infrastructure (mandark/webnote
    # are SpamAssassin corpus mail relays; xent/jmason are the SpamAssassin
    # project's own mailing-list host, jmason.org/xent.com — Justin Mason's
    # infrastructure). A whack-a-mole patch, not a general fix — the real fix
    # is mixing in genuine Enron ham so the model isn't relying on this
    # corpus's specific infrastructure fingerprints to define "ham" at all.
    r'mandark|webnote|xent|jmason)\b'
)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)          # URLs
    text = re.sub(r'\S+@\S+', ' email ', text)                  # email addresses
    text = re.sub(r'\b\d[\d\s\-().]{6,}\d\b', ' phone ', text) # phone numbers
    text = re.sub(r'[^a-z\s]', ' ', text)                       # non-alpha
    text = _ROUTING_WORDS.sub(' ', text)                        # mail-transport vocabulary
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_enron_message(raw):
    """Extract subject + body from a raw Enron email string."""
    if not isinstance(raw, str):
        return ""
    parts = re.split(r'\r?\n\r?\n', raw, maxsplit=1)
    header = parts[0]
    body   = parts[1].strip() if len(parts) > 1 else ""
    subject = ""
    for line in header.splitlines():
        m = re.match(r'^Subject:\s?(.*)', line, re.I)
        if m:
            subject = m.group(1)
            break
    return subject + " " + body[:3000]

# ── Load SpamAssassin ──────────────────────────────────────────────────────────

def load_spamassassin(path, sample_size=None, random_state=42):
    print(f"[1/5] Loading SpamAssassin: {path}")
    df = pd.read_csv(path)

    # normalise column names (Kaggle dataset varies slightly)
    df.columns = [c.lower().strip() for c in df.columns]

    text_col  = next((c for c in df.columns if 'text'    in c or 'message' in c or 'body' in c), None)
    label_col = next((c for c in df.columns if 'label'   in c or 'class'   in c or 'spam' in c or 'target' in c), None)

    if not text_col or not label_col:
        sys.exit(f"ERROR: could not identify text/label columns. Found: {df.columns.tolist()}")

    df = df[[text_col, label_col]].rename(columns={text_col:'text', label_col:'label'})
    df = df.dropna(subset=['text','label'])

    # normalise label to 0/1 (ham=0 spam=1)
    if df['label'].dtype == object:
        mapping = {}
        for v in df['label'].unique():
            v_str = str(v).lower().strip()
            if v_str in ('spam','1','yes','true'):  mapping[v] = 1
            else:                                    mapping[v] = 0
        df['label'] = df['label'].map(mapping)
    else:
        df['label'] = df['label'].astype(int).clip(0, 1)

    if sample_size and sample_size < len(df):
        # stratified sample: preserve the corpus's own spam:ham ratio rather
        # than artificially balancing, so CV numbers reflect a realistic mix
        frac = sample_size / len(df)
        df = df.groupby('label', group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=random_state)
        ).reset_index(drop=True)
        print(f"    Sampled {len(df):,}/{sample_size:,} rows (stratified by label)")

    df['text'] = df['text'].apply(clean_text)

    spam_count = df['label'].sum()
    ham_count  = len(df) - spam_count
    print(f"    → {len(df):,} emails  |  spam: {spam_count:,}  ham: {ham_count:,}")
    return df

# ── Load genuine Enron ham (the fix for SpamAssassin's narrow ham genre) ───────
#
# SpamAssassin's "ham" class turned out to be one specific technical mailing
# list (spamassassin-devel, hosted on jmason.org/xent.com) — quoted-reply
# threads, project jargon, a particular timestamp format. It is not a stand-in
# for "legitimate email" in general, and a classifier trained only on it
# learns that narrow genre's fingerprints rather than actual ham-vs-spam
# language. Mixing in real Enron messages (treated as ham — the corpus is
# overwhelmingly legitimate business/personal correspondence, not spam) gives
# the model a genuinely broad, deployment-realistic definition of "ham".

def load_enron_ham(path, sample_size, exclude_files=None, random_state=42, chunksize=50_000):
    print(f"[Enron ham] Sampling {sample_size:,} genuine Enron messages as ham examples ...")
    exclude_files = exclude_files or set()
    rng = np.random.RandomState(random_state)
    rows = []
    seen_keys = set()

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if len(rows) >= sample_size:
            break
        chunk = chunk[~chunk['file'].isin(exclude_files)]
        if len(chunk) == 0:
            continue
        sample = chunk.sample(min(500, len(chunk)), random_state=rng.randint(0, 1_000_000))
        for _, r in sample.iterrows():
            text = clean_text(parse_enron_message(r['message']))
            key = text[:150]
            if key in seen_keys or not text.strip():
                continue
            seen_keys.add(key)
            rows.append({'file': r['file'], 'text': text, 'label': 0})
            if len(rows) >= sample_size:
                break

    df = pd.DataFrame(rows)
    print(f"    -> {len(df):,} Enron messages sampled as ham")
    return df

# ── Train ──────────────────────────────────────────────────────────────────────

def train(df):
    print("[2/5] Building pipeline and running 5-fold cross-validation …")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features   = 30_000,
            ngram_range    = (1, 2),
            sublinear_tf   = True,
            min_df         = 3,
            stop_words     = 'english',
        )),
        ('clf', LogisticRegression(
            C              = 1.0,
            max_iter       = 1000,
            class_weight   = 'balanced',
            solver         = 'lbfgs',
            n_jobs         = -1,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, df['text'], df['label'], cv=cv, scoring='f1', n_jobs=-1)
    print(f"    Cross-val F1: {scores.mean():.4f} ± {scores.std():.4f}  (per fold: {[f'{s:.3f}' for s in scores]})")

    # final fit on full training split for held-out eval
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.15, stratify=df['label'], random_state=42
    )

    print("[3/5] Fitting final model …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n── Held-out evaluation (15%) ──────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=['ham','spam']))
    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  ham  predicted → ham: {cm[0,0]:5}  spam: {cm[0,1]:5}")
    print(f"  spam predicted → ham: {cm[1,0]:5}  spam: {cm[1,1]:5}")
    print("──────────────────────────────────────────────────────────\n")

    # refit on ALL data for the saved model
    pipeline.fit(df['text'], df['label'])
    return pipeline

# ── Save model ────────────────────────────────────────────────────────────────

def save_model(pipeline, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'spam_classifier.joblib')
    joblib.dump(pipeline, path)
    print(f"[4/5] Model saved → {path}")
    return path

# ── Held-out sanity check on FRESH, never-trained-on Enron mail ────────────────

def check_enron_false_positive_rate(pipeline, enron_path, exclude_files, sample_size=5000, random_state=99, chunksize=50_000):
    """The real test: genuine held-out Enron mail should mostly predict ham.
    Uses a disjoint sample from load_enron_ham's training rows (same
    exclude_files set) so this isn't just re-scoring what it trained on."""
    print(f"\n[Check] Sampling {sample_size:,} FRESH (never-trained-on) Enron messages to check false-positive rate ...")
    rng = np.random.RandomState(random_state)
    rows = []
    for chunk in pd.read_csv(enron_path, chunksize=chunksize):
        if len(rows) >= sample_size:
            break
        chunk = chunk[~chunk['file'].isin(exclude_files)]
        if len(chunk) == 0:
            continue
        sample = chunk.sample(min(500, len(chunk)), random_state=rng.randint(0, 1_000_000))
        for _, r in sample.iterrows():
            rows.append(r['message'])
            if len(rows) >= sample_size:
                break

    texts = [clean_text(parse_enron_message(m)) for m in rows]
    proba = pipeline.predict_proba(texts)[:, 1]
    flagged = (proba >= 0.5).sum()
    print(f"    {flagged}/{len(texts)} ({flagged/len(texts)*100:.1f}%) of fresh, genuine Enron mail flagged as spam")
    print(f"    (this is the number that matters — should look like a normal corporate mailbox's spam rate, not 85%+)")
    return flagged / len(texts)

# ── Run on Enron ──────────────────────────────────────────────────────────────

def predict_enron(pipeline, enron_path, out_dir, chunksize=50_000):
    print(f"[5/5] Running spam classifier over Enron dataset in chunks …")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'spam_predictions.csv')

    first_chunk = True
    total = 0
    spam_total = 0

    for chunk in pd.read_csv(enron_path, chunksize=chunksize, on_bad_lines='skip'):
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        if 'message' not in chunk.columns:
            continue
        chunk['clean_text'] = chunk['message'].apply(parse_enron_message).apply(clean_text)
        proba = pipeline.predict_proba(chunk['clean_text'])[:, 1]
        label = (proba >= 0.5).astype(int)

        out_chunk = pd.DataFrame({
            'file':           chunk.get('file', pd.Series([''] * len(chunk))),
            'spam_label':     label,
            'spam_confidence': proba.round(4),
        })

        out_chunk.to_csv(out_path, mode='a', header=first_chunk, index=False)
        first_chunk = False
        total       += len(chunk)
        spam_total  += label.sum()

        print(f"    processed {total:>7,} rows  |  spam found so far: {spam_total:,}")

    print(f"\n    Done. Total: {total:,}  spam detected: {spam_total:,}  ({spam_total/total*100:.1f}%)")
    print(f"    Predictions → {out_path}")

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train spam classifier on SpamAssassin spam + SpamAssassin ham + genuine Enron ham")
    parser.add_argument('--spam',  required=True, help='Path to SpamAssassin CSV')
    parser.add_argument('--enron', required=True, help='Path to Enron emails.csv')
    parser.add_argument('--out',   default='../..',   help='Output directory for model + predictions')
    parser.add_argument('--spam_sample', type=int, default=500, help='SpamAssassin spam examples to use')
    parser.add_argument('--sa_ham_sample', type=int, default=300, help='SpamAssassin ham examples to keep (for vocabulary diversity)')
    parser.add_argument('--enron_ham_sample', type=int, default=700, help='Genuine Enron messages to use as ham')
    parser.add_argument('--skip_full_predict', action='store_true', help='Skip the full 517K-row Enron predictions pass (just do the fast held-out check)')
    args = parser.parse_args()

    sa = pd.read_csv(args.spam)
    sa.columns = [c.lower().strip() for c in sa.columns]
    print(f"[1/6] SpamAssassin source: {len(sa):,} rows")

    spam_only = load_spamassassin(args.spam, sample_size=None)  # load full, then split by label below
    spam_rows = spam_only[spam_only['label'] == 1].sample(
        min(args.spam_sample, (spam_only['label'] == 1).sum()), random_state=42)
    sa_ham_rows = spam_only[spam_only['label'] == 0].sample(
        min(args.sa_ham_sample, (spam_only['label'] == 0).sum()), random_state=42)
    print(f"    -> using {len(spam_rows):,} SpamAssassin spam + {len(sa_ham_rows):,} SpamAssassin ham")

    enron_ham_df = load_enron_ham(args.enron, args.enron_ham_sample)
    exclude_files = set(enron_ham_df['file']) if 'file' in enron_ham_df.columns else set()

    combined = pd.concat([
        spam_rows[['text', 'label']],
        sa_ham_rows[['text', 'label']],
        enron_ham_df[['text', 'label']],
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n[2/6] Combined training set: {len(combined):,} rows "
          f"(spam: {(combined['label']==1).sum():,}, ham: {(combined['label']==0).sum():,})")

    pipeline = train(combined)
    save_model(pipeline, os.path.join(args.out, 'models'))

    check_enron_false_positive_rate(pipeline, args.enron, exclude_files)

    if not args.skip_full_predict:
        predict_enron(pipeline, args.enron, os.path.join(args.out, 'predictions'))
    print("\n✓  Spam pipeline complete.")
