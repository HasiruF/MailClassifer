"""
train_category_classifier.py
─────────────────────────────
Trains an email category classifier (Work / Personal / Other) on the Enron
dataset using folder names as weak supervision labels — no manual tagging needed.

The folder → category mapping is intentionally conservative: noisy or ambiguous
folders (deleted_items, all_documents, etc.) are skipped entirely so the model
trains on clean signal only.

Outputs
  models/category_classifier.joblib    — saved pipeline (TF-IDF + LR)
  predictions/category_predictions.csv — all Enron rows with predicted category
                                          and per-class confidence scores

Usage
  python train_category_classifier.py \
      --enron path/to/emails.csv \
      --out   output_dir          (default: current directory)

Expected Enron CSV columns: file, message
"""

import argparse, re, os, sys
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics           import classification_report, confusion_matrix

# ── Folder → category weak labels ─────────────────────────────────────────────
#
# Based on real Enron folder name patterns observed across the corpus.
# Folders not listed here are treated as UNLABELLED and excluded from training.
# This is intentional: training on clean folders only gives a stronger signal.

FOLDER_MAP = {
    # ── Work ──
    'inbox':               'Work',
    'sent':                'Work',
    '_sent_mail':          'Work',
    'sent_mail':           'Work',
    'sent_items':          'Work',
    'sent messages':       'Work',
    'discussion_threads':  'Work',
    'notes_inbox':         'Work',
    'calendar':            'Work',
    'meetings':            'Work',
    'schedule_crawler':    'Work',
    'logistics':           'Work',
    'wellhead':            'Work',
    'projects':            'Work',
    'contracts':           'Work',
    'legal':               'Work',
    'hr':                  'Work',
    'finance':             'Work',
    'trading':             'Work',
    'operations':          'Work',
    'administrative':      'Work',
    'to do':               'Work',
    'action items':        'Work',
    'tasks':               'Work',
    'follow up':           'Work',
    'important':           'Work',

    # ── Personal ──
    'personal':            'Personal',
    'personal folders':    'Personal',
    'family':              'Personal',
    'friends':             'Personal',
    'personal email':      'Personal',

    # ── News/Newsletter ──
    'enron announcements': 'News/Newsletter',
    'news':                'News/Newsletter',
    'newsletters':         'News/Newsletter',

    # ── Mailing List ──
    'mailing lists':       'Mailing List',
    'lists':               'Mailing List',
    'forums':              'Mailing List',
    'discussion':          'Mailing List',
    'ces':                 'Mailing List',
    '_americas':           'Mailing List',
}

# These are too noisy to use as training signal — skip them entirely.
SKIP_FOLDERS = {
    'all_documents', 'all documents',
    'deleted_items', 'deleted items',
    'trash', 'junk',
    'archiving', 'archive',
    'attachments',
    'c',                  # misc catch-all in some Enron accounts
}

# ── Text helpers ───────────────────────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\b\d[\d\s\-().]{6,}\d\b', ' phone ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_enron_message(raw):
    """Return (subject, body) parsed from a raw Enron email string."""
    if not isinstance(raw, str):
        return "", ""
    parts = re.split(r'\r?\n\r?\n', raw, maxsplit=1)
    header = parts[0]
    body   = parts[1].strip() if len(parts) > 1 else ""
    subject = ""
    for line in header.splitlines():
        m = re.match(r'^Subject:\s?(.*)', line, re.I)
        if m:
            subject = m.group(1).strip()
            break
    return subject, body[:3000]

def extract_folder(filepath):
    """Extract the folder segment from 'username/folder/email_id'."""
    if not isinstance(filepath, str):
        return None
    parts = filepath.strip().split('/')
    return parts[1].lower().strip() if len(parts) >= 2 else None

# ── Build labelled training set ────────────────────────────────────────────────

def build_training_set(enron_path, chunksize=50_000):
    print(f"[1/5] Scanning Enron dataset to build weak-label training set …")
    rows = []
    folder_counts = {}
    skipped = 0
    total = 0

    for chunk in pd.read_csv(enron_path, chunksize=chunksize, on_bad_lines='skip'):
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        if 'file' not in chunk.columns or 'message' not in chunk.columns:
            continue
        total += len(chunk)

        for _, row in chunk.iterrows():
            folder = extract_folder(row['file'])
            if folder is None:
                skipped += 1; continue
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

            if folder in SKIP_FOLDERS:
                skipped += 1; continue

            category = FOLDER_MAP.get(folder)
            if category is None:
                skipped += 1; continue

            subj, body = parse_enron_message(row['message'])
            text = clean_text(subj + ' ' + body)
            if len(text) < 10:
                skipped += 1; continue

            rows.append({'text': text, 'category': category})

    df = pd.DataFrame(rows)
    print(f"    Total rows scanned   : {total:,}")
    print(f"    Rows with weak labels: {len(df):,}")
    print(f"    Rows skipped (noisy/unlabelled): {skipped:,}")
    print(f"\n    Label distribution:")
    for cat, count in df['category'].value_counts().items():
        pct = count / len(df) * 100
        print(f"      {cat:12s}  {count:>7,}  ({pct:.1f}%)")

    # Top unlabelled folders (useful for extending FOLDER_MAP later)
    known = set(FOLDER_MAP.keys()) | SKIP_FOLDERS
    unknown_folders = {k: v for k, v in folder_counts.items() if k not in known}
    if unknown_folders:
        top_unknown = sorted(unknown_folders.items(), key=lambda x: -x[1])[:15]
        print(f"\n    Top unlabelled folders (consider adding to FOLDER_MAP):")
        for fname, cnt in top_unknown:
            print(f"      {fname:<30s}  {cnt:,}")
    print()
    return df

# ── Balance training set ───────────────────────────────────────────────────────

def cap_per_category(df, max_per_category=2000, random_state=42):
    print(f"    Capping each category at {max_per_category:,} rows (random sample) …")
    capped = [
        group.sample(n=min(len(group), max_per_category), random_state=random_state)
        for _, group in df.groupby('category')
    ]
    result = pd.concat(capped).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"    Balanced training set: {len(result):,} rows")
    for cat, count in result['category'].value_counts().items():
        print(f"      {cat:16s}  {count:>6,}")
    print()
    return result

# ── Train ──────────────────────────────────────────────────────────────────────

def train(df):
    print("[2/5] Building pipeline …")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features   = 40_000,
            ngram_range    = (1, 2),
            sublinear_tf   = True,
            min_df         = 4,
            stop_words     = 'english',
        )),
        ('clf', LogisticRegression(
            C              = 2.0,
            max_iter       = 1000,
            class_weight   = 'balanced',  # important: Personal is rare
            solver         = 'lbfgs',
            n_jobs         = -1,
        )),
    ])

    if len(df) < 500:
        sys.exit("ERROR: fewer than 500 labelled rows — check your FOLDER_MAP or dataset path.")

    print("[3/5] Running 5-fold stratified cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, df['text'], df['category'], cv=cv, scoring='f1_weighted', n_jobs=-1)
    print(f"    Cross-val F1 (weighted): {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"    Per fold: {[f'{s:.3f}' for s in scores]}")

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['category'], test_size=0.15, stratify=df['category'], random_state=42
    )

    print("[4/5] Fitting final model …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n── Held-out evaluation (15%) ──────────────────────────────")
    print(classification_report(y_test, y_pred))
    categories = sorted(df['category'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=categories)
    print("Confusion matrix (rows=actual, cols=predicted):")
    header = '  ' + '  '.join(f'{c:>10s}' for c in categories)
    print(header)
    for i, row_label in enumerate(categories):
        row_str = '  '.join(f'{v:>10d}' for v in cm[i])
        print(f"  {row_label:10s}  {row_str}")
    print("──────────────────────────────────────────────────────────\n")

    # refit on ALL data for the saved model
    pipeline.fit(df['text'], df['category'])
    return pipeline

# ── Save model ────────────────────────────────────────────────────────────────

def save_model(pipeline, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'category_classifier.joblib')
    joblib.dump(pipeline, path)
    print(f"Model saved → {path}")
    return path

# ── Predict full Enron dataset ────────────────────────────────────────────────

def predict_full_dataset(pipeline, enron_path, out_dir, chunksize=50_000):
    print(f"[5/5] Running category classifier over full Enron dataset …")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'category_predictions.csv')
    classes  = pipeline.classes_.tolist()

    first_chunk = True
    total = 0
    label_counts = {}

    for chunk in pd.read_csv(enron_path, chunksize=chunksize, on_bad_lines='skip'):
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        if 'message' not in chunk.columns:
            continue

        texts = chunk['message'].apply(
            lambda m: clean_text(' '.join(parse_enron_message(m)))
        )
        proba = pipeline.predict_proba(texts)
        preds = pipeline.classes_[np.argmax(proba, axis=1)]

        out_chunk = pd.DataFrame({
            'file':     chunk.get('file', pd.Series([''] * len(chunk))),
            'category': preds,
        })
        for i, cls in enumerate(classes):
            out_chunk[f'conf_{cls.lower()}'] = proba[:, i].round(4)

        out_chunk.to_csv(out_path, mode='a', header=first_chunk, index=False)
        first_chunk = False
        total += len(chunk)
        for p in preds:
            label_counts[p] = label_counts.get(p, 0) + 1

        print(f"    processed {total:>7,} rows", end='\r')

    print(f"\n    Done. Total: {total:,}")
    print(f"    Category breakdown:")
    for cat, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"      {cat:12s}  {cnt:>7,}  ({cnt/total*100:.1f}%)")
    print(f"    Predictions → {out_path}")

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train category classifier on Enron with folder weak labels")
    parser.add_argument('--enron', required=True, help='Path to Enron emails.csv')
    parser.add_argument('--out',   default='.',   help='Output directory for model + predictions')
    args = parser.parse_args()

    df       = build_training_set(args.enron)
    df       = cap_per_category(df)
    pipeline = train(df)
    save_model(pipeline, os.path.join(args.out, 'models'))
    predict_full_dataset(pipeline, args.enron, os.path.join(args.out, 'predictions'))
    print("\n✓  Category pipeline complete.")
