# Personalization (Corrections + Per-User Retraining) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an opted-in user correct a label on an email (including inventing a custom category), and retrain a personal copy of the affected model on those corrections, so future similar emails come out the way they labeled them.

**Architecture:** An offline script precomputes each production model's exact training matrix once. The FastAPI backend stores corrections as sparse feature vectors, refits a `clone()` of the production classifier head on base rows plus weighted corrections in a background task, and promotes the result only if it passes a validation gate. The backend exports passing models to ONNX and keeps them versioned in Postgres. The Next.js client loads the user's active personalized ONNX graphs in place of the shared base ones, and still classifies entirely in the browser.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2, Alembic, Postgres 16, scikit-learn 1.9.0, skl2onnx 1.20.0, scipy, pytest; Next.js 16, React 19, onnxruntime-web 1.27.0, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-29-personalization-design.md`

## Global Constraints

- Raw subject/body never leave the browser. The only email-derived data sent to the backend is a sparse feature vector, and only after the user opts in.
- Only the classifier head is refit. TF-IDF vocabulary, idf, scaler, and one-hot order stay fixed. The refit estimator is `sklearn.base.clone()` of the production estimator (same hyperparameters, no grid search).
- `RETRAIN_THRESHOLD = 5` new corrections per model triggers an automatic retrain; a manual "retrain now" also exists.
- `CORRECTION_WEIGHT = 10.0` (starting value; base rows weight 1).
- Gate: `OWN_CORRECTION_MIN = 0.80`, `MAX_GATE_DROP = 0.03`, `MAX_CUSTOM_LABEL_FRACTION = 0.10`.
- Priority bucket to regressor target: low 0.2, medium 0.5, high 0.8.
- Custom labels are allowed for category only. Priority labels are `low`/`medium`/`high`; spam labels are `spam`/`ham`.
- One retrain job at a time per (user, model).
- onnxruntime-web: never let two session operations (`create` or `run`) overlap.
- Backend Python deps pinned to the versions that produced the models: `scikit-learn==1.9.0`, `skl2onnx==1.20.0`.
- Ports: backend 3011, frontend 3010, Postgres 5433. Backend tests use database `dmp_test`. Docker (Postgres) must be running for backend tests.
- Windows: npm scripts run under `cmd.exe`, so no `./` path prefixes in scripts.
- Next.js 16: read the relevant guide in `app/frontend/node_modules/next/dist/docs/` before using any Next-specific API (per `app/frontend/AGENTS.md`). This plan only touches client components and plain modules.
- Commit messages never include `Co-Authored-By` or any other AI-attribution trailer.

## File Map

**Offline (`scripts/`)**
- Create `scripts/export_training_matrices.py`: rebuilds base training rows, vectorizes them with the fitted production transformers, proves they reproduce production, writes `app/backend/training_data/`.
- Create `scripts/dump_sparse_parity_fixture.py`: writes sklearn's own vectors for 3 emails, for the client parity check.

**Backend (`app/backend/`)**
- Modify `pyproject.toml`, `src/config.py`, `src/main.py`
- Modify `src/models/user.py`, `src/models/correction.py`, `src/models/__init__.py`; create `src/models/personalized_model.py`
- Create migration under `alembic/versions/`
- Create `src/personalization/__init__.py`, `sparse.py`, `base_data.py`, `trainer.py`, `gate.py`, `jobs.py`, `measure.py`
- Create `src/schemas/personalization.py`, `src/routers/personalization.py`
- Create `tests/__init__.py`, `tests/conftest.py`, `tests/fakes.py`, and one test module per unit

**Frontend (`app/frontend/`)**
- Modify `src/types/index.ts`, `src/inference/engine.ts`
- Create `src/lib/corrections.ts`, `src/lib/sparse.ts`, `src/lib/personalization-api.ts` (+ tests)
- Modify `src/app/inbox/inbox-context.tsx`, `email-detail.tsx`, `sidebar.tsx`, `inbox-list.tsx`; create `src/app/inbox/correction-picker.tsx`
- Create `scripts/sparse_parity_check.mjs`, `scripts/personalization_e2e.mjs`, `scripts/fixtures/sparse_parity.json`

**Docs**
- Modify `.gitignore`, `PROJECT_STATUS.md`

---

### Task 1: Precompute base training matrices

**Files:**
- Create: `scripts/export_training_matrices.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces, in `app/backend/training_data/` (gitignored): for each name in `spam`, `category`, `priority`, `priority_regressor`:
  - `<name>_X.npz`: scipy CSR base matrix, float64
  - `<name>_labels.npz`: arrays `y`, plus `gate_y` except for `priority_regressor`
  - `<name>_gate_X.npz`: CSR gate matrix, except for `priority_regressor`
  - `<name>_estimator.joblib`: unfitted clone of the production estimator
  - `meta.json`: `{name: {"dim": int, "classes": [...], "base_gate_accuracy": float}}`. `priority_regressor` has only `dim`.

- [ ] **Step 1: Write the script**

Create `scripts/export_training_matrices.py`:

```python
"""
export_training_matrices.py
───────────────────────────
One-time precompute for per-user personalization retraining (see
docs/superpowers/specs/2026-09-29-personalization-design.md §4.1).

For spam, category and priority this rebuilds the exact rows each production
model was fit on, passes them through that production pipeline's
already-fitted feature transformer, and saves:

  <name>_X.npz            base training matrix (scipy sparse, CSR)
  <name>_gate_X.npz       gate evaluation matrix (not for priority_regressor)
  <name>_labels.npz       y (+ gate_y) as plain numpy arrays, no pickling
  <name>_estimator.joblib unfitted clone of the production estimator
  meta.json               per model: dim, classes, base_gate_accuracy

Row sources (each mirrors the trainer that produced the production model):
  spam      spam/train_spam_classifier.py __main__: 500 SpamAssassin spam +
            300 SpamAssassin ham + 700 Enron ham, shuffled with
            random_state=42; the saved model is refit on all rows.
            Gate: the stratified 15% split that script evaluates on. Both the
            base and personalized models have seen these rows, so this gate
            only detects regressions; it is not an accuracy estimate.
  category  category/retrain_with_synthetic.py: 1000 Enron gold rows + 500
            synthetic rows. Its data assembly is copied below because that
            script has no __main__ guard (importing it would retrain).
            Gate: the 100-email adversarial set.
  priority  priority/train_priority_classifier.py build_features() over
            gold_priority_labeled_v2.csv (1004 rows). Classifier target
            true_bucket, regressor target true_priority.
            Gate: the adversarial set, with category_label and spam_conf
            taken from the base category/spam models, as classify_email.py
            does at serving time.

Before writing anything, the script proves the rows match: refitting the
clone on the rebuilt matrix must reproduce the production estimator's
outputs on the gate matrix (max abs diff < 1e-6). If not, it exits non-zero
and writes nothing.

Usage (from scripts/):
  python export_training_matrices.py
"""

import argparse
import json
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ('', 'shared', 'spam', 'category', 'priority'):
    sys.path.insert(0, os.path.join(HERE, sub))

from train_category_classifier import clean_text as category_clean_text
from header_features import extract_header_features_from_fields
from stylistic_features import extract_stylistic_features
from priority_keyword_features import extract_keyword_features
from train_category_classifier_headers import NUMERIC_COLS as CATEGORY_NUMERIC_COLS
from train_priority_classifier import build_features as priority_build_features
from train_priority_classifier import NUMERIC_COLS as PRIORITY_NUMERIC_COLS
from train_spam_classifier import load_spamassassin, load_enron_ham
from classify_email import _clean as serving_clean, _sia
from synthetic_priority_train_batch1 import ROWS as B1
from synthetic_priority_train_batch2 import ROWS as B2
from synthetic_priority_train_batch3 import ROWS as B3
from synthetic_priority_train_batch4 import ROWS as B4
from synthetic_priority_train_batch5 import ROWS as B5
from adversarial_test_set import TEST_EMAILS

REPRO_TOLERANCE = 1e-6


def spam_rows(spam_csv, enron_csv):
    spam_all = load_spamassassin(spam_csv, sample_size=None)
    n_spam = int((spam_all['label'] == 1).sum())
    n_ham = int((spam_all['label'] == 0).sum())
    spam = spam_all[spam_all['label'] == 1].sample(min(500, n_spam), random_state=42)
    sa_ham = spam_all[spam_all['label'] == 0].sample(min(300, n_ham), random_state=42)
    enron_ham = load_enron_ham(enron_csv, 700)
    combined = pd.concat([
        spam[['text', 'label']],
        sa_ham[['text', 'label']],
        enron_ham[['text', 'label']],
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    _, gate_text, _, gate_y = train_test_split(
        combined['text'], combined['label'], test_size=0.15, stratify=combined['label'], random_state=42,
    )
    return (combined['text'], np.asarray(combined['label'], dtype=np.int64),
            gate_text, np.asarray(gate_y, dtype=np.int64))


def category_rows():
    gold = pd.read_csv(os.path.join(REPO, 'gold_sample_labeled.csv'))
    gold = gold[gold['true_category'].notna() & (gold['true_category'].str.strip() != '')].copy()
    gold['subject'] = gold['subject'].fillna('')
    gold['body'] = gold['body'].fillna('')
    gold['text'] = (gold['subject'] + ' ' + gold['body']).apply(category_clean_text)
    style = gold.apply(lambda row: pd.Series(extract_stylistic_features(row['subject'], row['body'])), axis=1)
    gold = pd.concat([gold, style], axis=1)
    headers = pd.read_csv(os.path.join(HERE, 'category', 'gold_header_features.csv'))
    enron = gold.merge(headers, on='file', how='left')
    enron[CATEGORY_NUMERIC_COLS] = enron[CATEGORY_NUMERIC_COLS].fillna(0)

    syn = []
    for r in B1 + B2 + B3 + B4 + B5:
        header = extract_header_features_from_fields(subject=r['subject'], to=r['to'], from_addr=r['from_addr'])
        row = {'text': category_clean_text(r['subject'] + ' ' + r['body']), 'true_category': r['true_category']}
        row.update({k: header[k] for k in ['n_recipients', 'is_reply_or_forward', 'sender_automated']})
        row['has_list_unsubscribe'] = 0
        row['has_precedence_bulk'] = 0
        row.update(extract_stylistic_features(r['subject'], r['body']))
        syn.append(row)
    syn = pd.DataFrame(syn)

    X = pd.concat([enron[['text'] + CATEGORY_NUMERIC_COLS], syn[['text'] + CATEGORY_NUMERIC_COLS]],
                  ignore_index=True)
    y = pd.concat([enron['true_category'], syn['true_category']], ignore_index=True)
    return X, np.asarray(y, dtype=str)


def priority_rows():
    raw = pd.read_csv(os.path.join(REPO, 'gold_priority_labeled_v2.csv'))
    df = priority_build_features(raw)
    X = df[['text', 'category_label'] + PRIORITY_NUMERIC_COLS]
    return X, np.asarray(df['true_bucket'], dtype=str), np.asarray(df['true_priority'], dtype=np.float64)


def adversarial_frames(spam_pipe, cat_pipe):
    cat_rows, prio_rows = [], []
    for e in TEST_EMAILS:
        text = serving_clean(e['subject'] + ' ' + e['body'])
        header = extract_header_features_from_fields(subject=e['subject'], to=e['to'], from_addr=e['from_addr'])
        style = extract_stylistic_features(e['subject'], e['body'])
        spam_proba = spam_pipe.predict_proba([text])[0]
        spam_conf = float(spam_proba[list(spam_pipe.classes_).index(1)])
        cat_rows.append({'text': text, **header, **style})
        prio_rows.append({
            'text': text,
            'n_recipients': header['n_recipients'],
            'is_reply_or_forward': header['is_reply_or_forward'],
            'sender_automated': header['sender_automated'],
            'spam_conf': spam_conf,
            'vader_compound': _sia.polarity_scores(text)['compound'],
            **style,
            **extract_keyword_features(e['subject'], e['body']),
        })
    cat_X = pd.DataFrame(cat_rows)[['text'] + CATEGORY_NUMERIC_COLS]
    prio_df = pd.DataFrame(prio_rows)
    prio_df['category_label'] = cat_pipe.predict(cat_X)
    prio_X = prio_df[['text', 'category_label'] + PRIORITY_NUMERIC_COLS]
    cat_y = np.asarray([e['expected_category'] for e in TEST_EMAILS], dtype=str)
    prio_y = np.asarray([e['expected_priority'] for e in TEST_EMAILS], dtype=str)
    return cat_X, cat_y, prio_X, prio_y


def refit_matches(name, prod_est, X, y, check_X):
    started = time.perf_counter()
    est = clone(prod_est)
    est.fit(X, y)
    seconds = time.perf_counter() - started
    if hasattr(prod_est, 'predict_proba'):
        diff = float(np.max(np.abs(est.predict_proba(check_X) - prod_est.predict_proba(check_X))))
    else:
        diff = float(np.max(np.abs(est.predict(check_X) - prod_est.predict(check_X))))
    ok = diff < REPRO_TOLERANCE
    print(f"[{name}] refit took {seconds:.1f}s; max diff vs production {diff:.2e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=os.path.join(REPO, 'models'))
    ap.add_argument('--spam', default=os.path.join(REPO, 'SpamAssasin', 'spamassassin.csv'))
    ap.add_argument('--enron', default=os.path.join(REPO, 'Enron', 'emails.csv'))
    ap.add_argument('--out', default=os.path.join(REPO, 'app', 'backend', 'training_data'))
    args = ap.parse_args()

    spam_pipe = joblib.load(os.path.join(args.models, 'spam_classifier.joblib'))
    cat_pipe = joblib.load(os.path.join(args.models, 'category_classifier_headers.joblib'))
    prio_clf_pipe = joblib.load(os.path.join(args.models, 'priority_classifier.joblib'))
    prio_reg_pipe = joblib.load(os.path.join(args.models, 'priority_regressor.joblib'))

    print("Rebuilding spam rows (scans Enron/emails.csv in chunks; takes a minute) ...")
    spam_text, spam_y, spam_gate_text, spam_gate_y = spam_rows(args.spam, args.enron)
    spam_tfidf = spam_pipe.named_steps['tfidf']
    spam_clf = spam_pipe.named_steps['clf']
    spam_X = sp.csr_matrix(spam_tfidf.transform(spam_text))
    spam_gate_X = sp.csr_matrix(spam_tfidf.transform(spam_gate_text))

    print("Rebuilding category rows ...")
    cat_frame, cat_y = category_rows()
    cat_features = cat_pipe.named_steps['features']
    cat_clf = cat_pipe.named_steps['clf']
    cat_X = sp.csr_matrix(cat_features.transform(cat_frame))

    print("Rebuilding priority rows ...")
    prio_frame, prio_y, prio_score = priority_rows()
    prio_clf = prio_clf_pipe.named_steps['clf']
    prio_reg = prio_reg_pipe.named_steps['reg']
    prio_X = sp.csr_matrix(prio_clf_pipe.named_steps['features'].transform(prio_frame))
    prio_reg_X = sp.csr_matrix(prio_reg_pipe.named_steps['features'].transform(prio_frame))

    print("Vectorizing the adversarial gate set ...")
    adv_cat_frame, adv_cat_y, adv_prio_frame, adv_prio_y = adversarial_frames(spam_pipe, cat_pipe)
    cat_gate_X = sp.csr_matrix(cat_features.transform(adv_cat_frame))
    prio_gate_X = sp.csr_matrix(prio_clf_pipe.named_steps['features'].transform(adv_prio_frame))
    prio_reg_gate_X = sp.csr_matrix(prio_reg_pipe.named_steps['features'].transform(adv_prio_frame))

    checks = [
        refit_matches('spam', spam_clf, spam_X, spam_y, spam_gate_X),
        refit_matches('category', cat_clf, cat_X, cat_y, cat_gate_X),
        refit_matches('priority', prio_clf, prio_X, prio_y, prio_gate_X),
        refit_matches('priority_regressor', prio_reg, prio_reg_X, prio_score, prio_reg_gate_X),
    ]
    if not all(checks):
        sys.exit("Rebuilt rows do not reproduce the production models; nothing written. "
                 "Find which trainer's data assembly differs before continuing.")

    os.makedirs(args.out, exist_ok=True)
    meta = {}

    def save(name, est, X, y, gate_X=None, gate_y=None):
        sp.save_npz(os.path.join(args.out, f'{name}_X.npz'), X)
        labels = {'y': y}
        if gate_X is not None:
            sp.save_npz(os.path.join(args.out, f'{name}_gate_X.npz'), gate_X)
            labels['gate_y'] = gate_y
        np.savez(os.path.join(args.out, f'{name}_labels.npz'), **labels)
        joblib.dump(clone(est), os.path.join(args.out, f'{name}_estimator.joblib'))
        entry = {'dim': int(X.shape[1])}
        if hasattr(est, 'classes_'):
            entry['classes'] = est.classes_.tolist()
        if gate_X is not None:
            entry['base_gate_accuracy'] = float(np.mean(est.predict(gate_X) == gate_y))
        meta[name] = entry
        extra = f", base gate accuracy {entry['base_gate_accuracy']:.1%}" if 'base_gate_accuracy' in entry else ''
        print(f"[{name}] saved {X.shape[0]} rows x {X.shape[1]} dims{extra}")

    save('spam', spam_clf, spam_X, spam_y, spam_gate_X, spam_gate_y)
    save('category', cat_clf, cat_X, cat_y, cat_gate_X, adv_cat_y)
    save('priority', prio_clf, prio_X, prio_y, prio_gate_X, adv_prio_y)
    save('priority_regressor', prio_reg, prio_reg_X, prio_score)
    with open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {os.path.join(args.out, 'meta.json')}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Gitignore the output**

Append to `.gitignore` (repo root):

```
# Personalization base training matrices (scripts/export_training_matrices.py)
app/backend/training_data/
```

- [ ] **Step 3: Run it**

Run (from `scripts/`): `python export_training_matrices.py`

Expected: four lines ending `-> OK`, one per model, each with a refit time. Then four `saved ... rows x ... dims` lines: spam 1500 rows, category 1500 rows, priority 1004 rows, priority_regressor 1004 rows. Then `Wrote ...meta.json`.

If any line says `MISMATCH`, stop and report it. It means that model's rebuilt rows don't match what production was trained on, so every later task would train on the wrong base. Do not loosen `REPRO_TOLERANCE`.

Record the printed refit time for `priority` (spec §13 asks for it).

- [ ] **Step 4: Commit**

```bash
git add scripts/export_training_matrices.py .gitignore
git commit -m "Add export_training_matrices.py: precompute base matrices for personalization retraining"
```

---

### Task 2: Backend dependencies, config, and test harness

**Files:**
- Modify: `app/backend/pyproject.toml`
- Modify: `app/backend/src/config.py`
- Create: `app/backend/tests/__init__.py`, `app/backend/tests/conftest.py`, `app/backend/tests/test_health.py`

**Interfaces:**
- Produces: `settings.training_data_dir: str` (default `"training_data"`, relative to `app/backend`).
- Produces pytest fixtures: `db` (SQLAlchemy session on `dmp_test`), `client` (`TestClient`), and `connected` (a `SimpleNamespace` with `.user`, `.session`, `.connection`; the client already carries its `dmp_session` cookie). Every test gets a freshly created schema.

- [ ] **Step 1: Replace `app/backend/pyproject.toml`**

```toml
[project]
name = "email-classifier-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.118",
    "uvicorn[standard]>=0.38",
    "sqlalchemy>=2.0",
    "alembic>=1.16",
    "psycopg[binary]>=3.2",
    "pydantic-settings>=2.10",
    "httpx>=0.28",
    "numpy>=2.3",
    "scipy>=1.16",
    # Pinned to the versions that trained/exported models/*.joblib; the
    # estimator clones in training_data/ are unpickled with this version.
    "scikit-learn==1.9.0",
    "joblib>=1.5",
    "skl2onnx==1.20.0",
]

[dependency-groups]
dev = [
    "pytest>=8.4",
    "onnxruntime>=1.28",
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Install**

Run (from `app/backend`): `.venv/Scripts/python.exe -m pip install -e . --group dev -q`
Expected: exits 0. Then `.venv/Scripts/python.exe -c "import sklearn, skl2onnx; print(sklearn.__version__, skl2onnx.__version__)"` prints `1.9.0 1.20.0`.

- [ ] **Step 3: Add the training data setting**

In `app/backend/src/config.py`, add after `frontend_url: str`:

```python
    training_data_dir: str = "training_data"
```

- [ ] **Step 4: Write the harness**

Create `app/backend/tests/__init__.py` (empty file).

Create `app/backend/tests/conftest.py`:

```python
import os

# Must be set before anything imports src.config: Settings() reads the
# environment (which overrides .env) at import time.
os.environ["POSTGRES_DB"] = "dmp_test"

from types import SimpleNamespace  # noqa: E402

import psycopg  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.config import settings  # noqa: E402


def _ensure_test_database() -> None:
    with psycopg.connect(
        host="localhost",
        port=settings.database_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname="postgres",
        autocommit=True,
    ) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("dmp_test",)).fetchone()
        if not exists:
            conn.execute("CREATE DATABASE dmp_test")


_ensure_test_database()

from src.database import Base, SessionLocal, engine  # noqa: E402
from src.main import app  # noqa: E402
from src.models import EmailConnection, EmailProvider, User  # noqa: E402
from src.models import Session as SessionModel  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_schema():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def connected(db, client):
    user = User()
    db.add(user)
    db.flush()
    session = SessionModel(user_id=user.id, remember_me=True)
    connection = EmailConnection(
        user_id=user.id,
        provider=EmailProvider.gmail,
        provider_account_id="google-sub-1",
        provider_email="me@example.com",
    )
    db.add_all([session, connection])
    db.commit()
    client.cookies.set("dmp_session", str(session.id))
    return SimpleNamespace(user=user, session=session, connection=connection)
```

Create `app/backend/tests/test_health.py`:

```python
def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 5: Run the tests**

Run (from `app/backend`, Docker running): `.venv/Scripts/pytest.exe -q`
Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add app/backend/pyproject.toml app/backend/src/config.py app/backend/tests
git commit -m "Add backend ML deps and a pytest harness on a dedicated dmp_test database"
```

---

### Task 3: Schema changes and migration

**Files:**
- Modify: `app/backend/src/models/user.py`, `app/backend/src/models/correction.py`, `app/backend/src/models/__init__.py`
- Create: `app/backend/src/models/personalized_model.py`
- Create: `app/backend/alembic/versions/<generated>_personalization.py`
- Test: `app/backend/tests/test_models.py`

**Interfaces:**
- Produces: `User.personalization_enabled: bool` (default False); `Correction.provider_message_id: str`; `Correction.feature_vector: dict` (sparse JSON); unique constraint `uq_corrections_message_model` on (`email_connection_id`, `provider_message_id`, `model`).
- Produces: `PersonalizedModel` with `id, user_id, model: PersonalizedModelKind, version: int, status: PersonalizedModelStatus, onnx: bytes | None, classes: list | None, correction_count: int, metrics: dict, created_at`. Enums: `PersonalizedModelKind` (`spam`, `category`, `priority`, `priority_regressor`) and `PersonalizedModelStatus` (`active`, `rejected`, `superseded`). All exported from `src.models`.

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_models.py`:

```python
import pytest
from sqlalchemy.exc import IntegrityError

from src.models import Correction, CorrectionModel, User


def test_personalization_is_off_by_default(db):
    user = User()
    db.add(user)
    db.commit()
    db.refresh(user)
    assert user.personalization_enabled is False


def _correction(connection_id, label):
    return Correction(
        email_connection_id=connection_id,
        provider_message_id="msg-1",
        model=CorrectionModel.category,
        feature_vector={"dim": 3, "indices": [0], "values": [1.0]},
        predicted_label="Other",
        predicted_confidence=0.6,
        corrected_label=label,
    )


def test_one_correction_per_message_and_model(db, connected):
    db.add(_correction(connected.connection.id, "Work"))
    db.commit()
    db.add(_correction(connected.connection.id, "Personal"))
    with pytest.raises(IntegrityError):
        db.commit()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_models.py -q`
Expected: FAIL (`personalization_enabled` does not exist, and the second commit succeeds).

- [ ] **Step 3: Update the models**

Replace `app/backend/src/models/user.py`:

```python
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, false, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    personalization_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

Replace `app/backend/src/models/correction.py`:

```python
import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class CorrectionModel(str, enum.Enum):
    category = "category"
    priority = "priority"
    spam = "spam"


class Correction(Base):
    __tablename__ = "corrections"
    __table_args__ = (
        UniqueConstraint(
            "email_connection_id", "provider_message_id", "model", name="uq_corrections_message_model"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("email_connections.id"), nullable=False
    )
    provider_message_id: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[CorrectionModel] = mapped_column(Enum(CorrectionModel, name="correction_model"), nullable=False)
    # Sparse {"dim", "indices", "values"} form (src/personalization/sparse.py)
    # of the dense vector the client builds for ONNX inference. Never raw
    # subject/body text, per the app's privacy boundary.
    feature_vector: Mapped[dict] = mapped_column(JSONB, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String, nullable=False)
    predicted_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    corrected_label: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

Create `app/backend/src/models/personalized_model.py`:

```python
import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, LargeBinary, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class PersonalizedModelKind(str, enum.Enum):
    spam = "spam"
    category = "category"
    priority = "priority"
    priority_regressor = "priority_regressor"


class PersonalizedModelStatus(str, enum.Enum):
    active = "active"
    rejected = "rejected"
    superseded = "superseded"


class PersonalizedModel(Base):
    __tablename__ = "personalized_models"
    __table_args__ = (UniqueConstraint("user_id", "model", "version", name="uq_personalized_models_version"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    model: Mapped[PersonalizedModelKind] = mapped_column(
        Enum(PersonalizedModelKind, name="personalized_model_kind"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[PersonalizedModelStatus] = mapped_column(
        Enum(PersonalizedModelStatus, name="personalized_model_status"), nullable=False
    )
    # Null for rejected attempts: the gate runs before export.
    onnx: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    classes: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    correction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

Replace `app/backend/src/models/__init__.py`:

```python
from src.models.classification_event import ClassificationEvent
from src.models.correction import Correction, CorrectionModel
from src.models.email_connection import EmailConnection, EmailProvider
from src.models.interaction_event import InteractionAction, InteractionEvent
from src.models.personalized_model import PersonalizedModel, PersonalizedModelKind, PersonalizedModelStatus
from src.models.session import Session
from src.models.user import User

__all__ = [
    "ClassificationEvent",
    "Correction",
    "CorrectionModel",
    "EmailConnection",
    "EmailProvider",
    "InteractionAction",
    "InteractionEvent",
    "PersonalizedModel",
    "PersonalizedModelKind",
    "PersonalizedModelStatus",
    "Session",
    "User",
]
```

In `app/backend/alembic/env.py`, add `PersonalizedModel` to the `from src.models import (...)` list so autogenerate sees the new table.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/pytest.exe tests/test_models.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Generate the migration**

First confirm the dev `corrections` table is empty, because the new `NOT NULL` column has no default:
`docker exec backend-postgres-1 psql -U dmp_user -d dmp -c "SELECT count(*) FROM corrections;"` → `0`.

Run (from `app/backend`): `.venv/Scripts/alembic.exe revision --autogenerate -m "personalization"`

Open the generated file. Its `upgrade()` must contain exactly these operations. Fix it by hand if autogenerate differs:

```python
def upgrade() -> None:
    op.create_table('personalized_models',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('model', sa.Enum('spam', 'category', 'priority', 'priority_regressor', name='personalized_model_kind'), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('status', sa.Enum('active', 'rejected', 'superseded', name='personalized_model_status'), nullable=False),
        sa.Column('onnx', sa.LargeBinary(), nullable=True),
        sa.Column('classes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('correction_count', sa.Integer(), nullable=False),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'model', 'version', name='uq_personalized_models_version'),
    )
    op.add_column('corrections', sa.Column('provider_message_id', sa.String(), nullable=False))
    op.create_unique_constraint('uq_corrections_message_model', 'corrections', ['email_connection_id', 'provider_message_id', 'model'])
    op.add_column('users', sa.Column('personalization_enabled', sa.Boolean(), server_default=sa.text('false'), nullable=False))
```

Autogenerate's `downgrade()` drops the table but leaves the two enum types behind, so a later upgrade would fail with "type already exists". Make the end of `downgrade()` read:

```python
    op.drop_column('users', 'personalization_enabled')
    op.drop_constraint('uq_corrections_message_model', 'corrections', type_='unique')
    op.drop_column('corrections', 'provider_message_id')
    op.drop_table('personalized_models')
    sa.Enum(name='personalized_model_status').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='personalized_model_kind').drop(op.get_bind(), checkfirst=True)
```

- [ ] **Step 6: Apply, and prove it reverses cleanly**

Run (from `app/backend`):
```
.venv/Scripts/alembic.exe upgrade head
.venv/Scripts/alembic.exe downgrade -1
.venv/Scripts/alembic.exe upgrade head
```
Expected: all three succeed with no errors. Then `docker exec backend-postgres-1 psql -U dmp_user -d dmp -c "\d personalized_models"` shows the table.

- [ ] **Step 7: Commit**

```bash
git add app/backend/src/models app/backend/alembic app/backend/tests/test_models.py
git commit -m "Add personalization schema: opt-in flag, correction message id, personalized_models"
```

---

### Task 4: Sparse vectors and request/response schemas

**Files:**
- Create: `app/backend/src/personalization/__init__.py` (empty), `app/backend/src/personalization/sparse.py`
- Create: `app/backend/src/schemas/personalization.py`
- Test: `app/backend/tests/test_sparse.py`

**Interfaces:**
- Produces: `SparseVector(dim: int, indices: list[int], values: list[float])` (pydantic; validates equal lengths, in-range, and unique indices) and `to_csr(vectors: list[SparseVector], dim: int) -> scipy.sparse.csr_matrix`, which raises `ValueError` on a dim mismatch.
- Produces schemas: `CorrectionIn`, `CorrectionOut(retrain_scheduled: bool, corrections_until_retrain: int)`, `SettingsIn(enabled: bool)`, `SettingsOut(enabled: bool)`, `LastAttempt(version, status, metrics, created_at)`, `ModelStatus(model, correction_count, corrections_until_retrain, running, last_attempt)`, `StatusOut(enabled, models, custom_labels)`, `ManifestEntry(model, version, classes)`, `RetrainOut(scheduled: list[str])`.

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_sparse.py`:

```python
import numpy as np
import pytest
from pydantic import ValidationError

from src.personalization.sparse import SparseVector, to_csr
from src.schemas.personalization import CorrectionIn


def test_to_csr_places_values_at_indices():
    vectors = [
        SparseVector(dim=4, indices=[1, 3], values=[0.5, -2.0]),
        SparseVector(dim=4, indices=[], values=[]),
    ]
    matrix = to_csr(vectors, 4)
    assert matrix.shape == (2, 4)
    np.testing.assert_array_equal(matrix.toarray(), [[0, 0.5, 0, -2.0], [0, 0, 0, 0]])


def test_to_csr_of_nothing_is_an_empty_matrix():
    assert to_csr([], 4).shape == (0, 4)


def test_to_csr_rejects_wrong_dim():
    with pytest.raises(ValueError, match="dim"):
        to_csr([SparseVector(dim=3, indices=[0], values=[1.0])], 4)


def test_sparse_vector_rejects_bad_shapes():
    with pytest.raises(ValidationError):
        SparseVector(dim=3, indices=[0, 1], values=[1.0])
    with pytest.raises(ValidationError):
        SparseVector(dim=3, indices=[3], values=[1.0])
    with pytest.raises(ValidationError):
        SparseVector(dim=3, indices=[1, 1], values=[1.0, 2.0])


def _payload(model, label):
    return {
        "model": model,
        "provider_message_id": "m1",
        "feature_vector": {"dim": 2, "indices": [0], "values": [1.0]},
        "predicted_label": "x",
        "predicted_confidence": 0.5,
        "corrected_label": label,
    }


def test_priority_and_spam_labels_are_fixed_sets():
    with pytest.raises(ValidationError):
        CorrectionIn(**_payload("priority", "urgent"))
    with pytest.raises(ValidationError):
        CorrectionIn(**_payload("spam", "junk"))
    assert CorrectionIn(**_payload("priority", "high")).corrected_label == "high"


def test_category_accepts_a_custom_label_and_trims_it():
    assert CorrectionIn(**_payload("category", "  Finance ")).corrected_label == "Finance"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_sparse.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.personalization'`.

- [ ] **Step 3: Implement**

Create `app/backend/src/personalization/__init__.py` (empty).

Create `app/backend/src/personalization/sparse.py`:

```python
import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, model_validator


class SparseVector(BaseModel):
    dim: int
    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def _check_shape(self) -> "SparseVector":
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")
        if any(i < 0 or i >= self.dim for i in self.indices):
            raise ValueError("index out of range for dim")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("duplicate indices")
        return self


def to_csr(vectors: list[SparseVector], dim: int) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row, vector in enumerate(vectors):
        if vector.dim != dim:
            raise ValueError(f"vector dim {vector.dim} != expected dim {dim}")
        rows.extend([row] * len(vector.indices))
        cols.extend(vector.indices)
        vals.extend(vector.values)
    return sp.csr_matrix(
        (np.array(vals, dtype=np.float64), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(len(vectors), dim),
    )
```

Create `app/backend/src/schemas/personalization.py`:

```python
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from src.models import CorrectionModel
from src.personalization.sparse import SparseVector

FIXED_LABELS = {
    CorrectionModel.priority: {"low", "medium", "high"},
    CorrectionModel.spam: {"spam", "ham"},
}


class CorrectionIn(BaseModel):
    model: CorrectionModel
    provider_message_id: str = Field(min_length=1)
    feature_vector: SparseVector
    predicted_label: str
    predicted_confidence: float = Field(ge=0.0, le=1.0)
    corrected_label: str = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def _label_allowed(self) -> "CorrectionIn":
        self.corrected_label = self.corrected_label.strip()
        if not self.corrected_label:
            raise ValueError("corrected_label is blank")
        allowed = FIXED_LABELS.get(self.model)
        if allowed is not None and self.corrected_label not in allowed:
            raise ValueError(f"{self.model.value} corrections must be one of {sorted(allowed)}")
        return self


class CorrectionOut(BaseModel):
    retrain_scheduled: bool
    corrections_until_retrain: int


class SettingsIn(BaseModel):
    enabled: bool


class SettingsOut(BaseModel):
    enabled: bool


class LastAttempt(BaseModel):
    version: int
    status: str
    metrics: dict
    created_at: datetime


class ModelStatus(BaseModel):
    model: str
    correction_count: int
    corrections_until_retrain: int
    running: bool
    last_attempt: LastAttempt | None


class StatusOut(BaseModel):
    enabled: bool
    models: list[ModelStatus]
    custom_labels: list[str]


class ManifestEntry(BaseModel):
    model: str
    version: int
    classes: list[str | int]


class RetrainOut(BaseModel):
    scheduled: list[str]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/pytest.exe tests/test_sparse.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add app/backend/src/personalization app/backend/src/schemas/personalization.py app/backend/tests/test_sparse.py
git commit -m "Add sparse feature-vector handling and personalization request/response schemas"
```

---

### Task 5: Base data loader and test fakes

**Files:**
- Create: `app/backend/src/personalization/base_data.py`
- Create: `app/backend/tests/fakes.py`
- Test: `app/backend/tests/test_base_data.py`

**Interfaces:**
- Consumes: the `training_data/` layout from Task 1.
- Produces: `BaseBundle(X: csr_matrix, y: np.ndarray, gate_X: csr_matrix | None, gate_y: np.ndarray | None, estimator, base_gate_accuracy: float | None, classes: list)` (frozen dataclass); `load_base_bundle(name: str) -> BaseBundle` (cached); `load_meta() -> dict` (cached); `feature_dim(name: str) -> int`. The cached `estimator` is a template: callers must `clone()` it and never fit it directly.
- Produces (tests): `tests.fakes.DIM = 6`, `make_bundle(name) -> BaseBundle`, `patch_base_data(monkeypatch)`. Fake base rows use only features 0-4, so a correction that sets feature 5 is learnable without contradicting base data.

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_base_data.py`:

```python
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.personalization import base_data

REAL_DIR = Path(__file__).resolve().parents[1] / "training_data"


@pytest.fixture
def fresh_cache():
    base_data.load_meta.cache_clear()
    base_data.load_base_bundle.cache_clear()
    yield
    base_data.load_meta.cache_clear()
    base_data.load_base_bundle.cache_clear()


def test_loads_a_bundle_from_the_training_data_dir(tmp_path, monkeypatch, fresh_cache):
    sp.save_npz(tmp_path / "category_X.npz", sp.csr_matrix(np.eye(3)))
    sp.save_npz(tmp_path / "category_gate_X.npz", sp.csr_matrix(np.eye(3)))
    np.savez(
        tmp_path / "category_labels.npz",
        y=np.array(["Work", "Other", "Work"]),
        gate_y=np.array(["Work", "Other", "Other"]),
    )
    joblib.dump(LogisticRegression(C=3.0), tmp_path / "category_estimator.joblib")
    (tmp_path / "meta.json").write_text(
        json.dumps({"category": {"dim": 3, "classes": ["Other", "Work"], "base_gate_accuracy": 0.66}})
    )
    monkeypatch.setattr(settings, "training_data_dir", str(tmp_path))

    bundle = base_data.load_base_bundle("category")

    assert bundle.X.shape == (3, 3)
    assert list(bundle.y) == ["Work", "Other", "Work"]
    assert bundle.gate_X.shape == (3, 3)
    assert list(bundle.gate_y) == ["Work", "Other", "Other"]
    assert bundle.estimator.C == 3.0
    assert bundle.base_gate_accuracy == 0.66
    assert bundle.classes == ["Other", "Work"]
    assert base_data.feature_dim("category") == 3


@pytest.mark.skipif(not (REAL_DIR / "meta.json").exists(), reason="run scripts/export_training_matrices.py first")
@pytest.mark.parametrize("name", ["spam", "category", "priority", "priority_regressor"])
def test_real_bundles_are_internally_consistent(name, monkeypatch, fresh_cache):
    monkeypatch.setattr(settings, "training_data_dir", str(REAL_DIR))
    bundle = base_data.load_base_bundle(name)
    assert bundle.X.shape[1] == base_data.feature_dim(name)
    assert bundle.X.shape[0] == len(bundle.y)
    if bundle.gate_X is not None:
        assert bundle.gate_X.shape[1] == bundle.X.shape[1]
        assert bundle.gate_X.shape[0] == len(bundle.gate_y)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_base_data.py -q`
Expected: FAIL with `ImportError: cannot import name 'base_data'`.

- [ ] **Step 3: Implement the loader**

Create `app/backend/src/personalization/base_data.py`:

```python
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp

from src.config import settings


@dataclass(frozen=True)
class BaseBundle:
    X: sp.csr_matrix
    y: np.ndarray
    gate_X: sp.csr_matrix | None
    gate_y: np.ndarray | None
    # Unfitted clone of the production estimator. A shared template:
    # callers clone() it and never fit it directly.
    estimator: object
    base_gate_accuracy: float | None
    classes: list


def _dir() -> Path:
    return Path(settings.training_data_dir)


@lru_cache(maxsize=None)
def load_meta() -> dict:
    return json.loads((_dir() / "meta.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_base_bundle(name: str) -> BaseBundle:
    directory = _dir()
    meta = load_meta()[name]
    labels = np.load(directory / f"{name}_labels.npz")
    gate_path = directory / f"{name}_gate_X.npz"
    return BaseBundle(
        X=sp.load_npz(directory / f"{name}_X.npz").tocsr(),
        y=labels["y"],
        gate_X=sp.load_npz(gate_path).tocsr() if gate_path.exists() else None,
        gate_y=labels["gate_y"] if "gate_y" in labels.files else None,
        estimator=joblib.load(directory / f"{name}_estimator.joblib"),
        base_gate_accuracy=meta.get("base_gate_accuracy"),
        classes=meta.get("classes", []),
    )


def feature_dim(name: str) -> int:
    return int(load_meta()[name]["dim"])
```

- [ ] **Step 4: Write the fakes used by later tasks**

Create `app/backend/tests/fakes.py`:

```python
"""Small synthetic stand-ins for app/backend/training_data/ bundles.

Base rows only ever use features 0-4; feature 5 is always 0. A correction
that sets feature 5 is therefore learnable without contradicting base data,
the same situation as a user's emails containing words the base training
set rarely sees."""
import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge

from src.personalization.base_data import BaseBundle

DIM = 6
CLASSES = {
    "category": ["Other", "Personal", "Work"],
    "priority": ["high", "low", "medium"],
    "spam": [0, 1],
}
SEEDS = {"category": 0, "priority": 1, "spam": 2}


def _base_matrix(n: int, seed: int) -> np.ndarray:
    dense = np.zeros((n, DIM))
    dense[:, :5] = np.random.RandomState(seed).normal(size=(n, 5))
    return dense


def _labels(dense: np.ndarray, classes: list) -> np.ndarray:
    return np.asarray([classes[int(np.argmax(row[: len(classes)]))] for row in dense])


def make_bundle(name: str) -> BaseBundle:
    if name == "priority_regressor":
        X = _base_matrix(300, 3)
        y = np.clip(0.5 + 0.2 * X[:, 0], 0.1, 1.0)
        return BaseBundle(
            X=sp.csr_matrix(X), y=y, gate_X=None, gate_y=None,
            estimator=Ridge(alpha=1.0), base_gate_accuracy=None, classes=[],
        )
    classes = CLASSES[name]
    X = _base_matrix(300, SEEDS[name])
    gate_X = _base_matrix(60, 99)
    y = _labels(X, classes)
    gate_y = _labels(gate_X, classes)
    estimator = LogisticRegression(max_iter=1000)
    fitted = clone(estimator).fit(sp.csr_matrix(X), y)
    base_accuracy = float(np.mean(fitted.predict(sp.csr_matrix(gate_X)) == gate_y))
    return BaseBundle(
        X=sp.csr_matrix(X), y=y, gate_X=sp.csr_matrix(gate_X), gate_y=gate_y,
        estimator=estimator, base_gate_accuracy=base_accuracy, classes=list(classes),
    )


def patch_base_data(monkeypatch) -> None:
    from src.personalization import base_data

    monkeypatch.setattr(base_data, "load_base_bundle", make_bundle)
    monkeypatch.setattr(base_data, "feature_dim", lambda name: DIM)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/pytest.exe tests/test_base_data.py -q`
Expected: `5 passed` (1 synthetic + 4 real-data cases, since Task 1 generated `training_data/`).

- [ ] **Step 6: Commit**

```bash
git add app/backend/src/personalization/base_data.py app/backend/tests/fakes.py app/backend/tests/test_base_data.py
git commit -m "Add base training data loader for personalization retraining"
```

---

### Task 6: Trainer, validation gate, ONNX export

**Files:**
- Create: `app/backend/src/personalization/trainer.py`, `app/backend/src/personalization/gate.py`
- Test: `app/backend/tests/test_trainer.py`, `app/backend/tests/test_gate.py`

**Interfaces:**
- Consumes: `BaseBundle` and `tests.fakes` from Task 5.
- Produces (`trainer`): `CORRECTION_WEIGHT = 10.0`; `BUCKET_SCORE = {"low": 0.2, "medium": 0.5, "high": 0.8}`; `correction_targets(model: str, labels: list[str]) -> np.ndarray`, where `model` is one of `spam`, `category`, `priority`, `priority_regressor`; `fit_personalized(base_X, base_y, estimator, corr_X, corr_y, weight=CORRECTION_WEIGHT)`, which returns a fitted clone; `export_onnx(fitted, dim: int) -> bytes`.
- Produces (`gate`): `OWN_CORRECTION_MIN = 0.80`, `MAX_GATE_DROP = 0.03`, `MAX_CUSTOM_LABEL_FRACTION = 0.10`; `GateResult(passed, own_correction_accuracy, gate_accuracy, base_gate_accuracy, custom_label_fraction, reason)` with `.as_metrics() -> dict`; `evaluate_gate(fitted, corr_X, corr_y, gate_X, gate_y, base_gate_accuracy, base_classes) -> GateResult`.

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_trainer.py`:

```python
import numpy as np
import onnxruntime as ort
import scipy.sparse as sp

from src.personalization import trainer
from tests.fakes import DIM, make_bundle

# Base models call this point "Other" (feature 0 wins); feature 5 is a
# signal only the corrections carry.
POINT = np.array([1.0, 0, 0, 0, 0, 5.0])
BASE_ONLY_POINT = np.array([1.0, 0, 0, 0, 0, 0])


def corrections_at(point, label, n=5, model="category"):
    return sp.csr_matrix(np.tile(point, (n, 1))), trainer.correction_targets(model, [label] * n)


def test_weighted_corrections_flip_the_corrected_region_only():
    bundle = make_bundle("category")
    corr_X, corr_y = corrections_at(POINT, "Work")
    fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    assert fitted.predict(sp.csr_matrix([POINT]))[0] == "Work"
    assert fitted.predict(sp.csr_matrix([BASE_ONLY_POINT]))[0] == "Other"


def test_a_custom_label_becomes_a_new_class():
    bundle = make_bundle("category")
    corr_X, corr_y = corrections_at(POINT, "Finance")
    fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    assert "Finance" in fitted.classes_
    assert fitted.predict(sp.csr_matrix([POINT]))[0] == "Finance"


def test_the_template_estimator_is_never_fitted():
    bundle = make_bundle("category")
    corr_X, corr_y = corrections_at(POINT, "Work")
    trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    assert not hasattr(bundle.estimator, "classes_")


def test_targets_are_mapped_per_model():
    assert trainer.correction_targets("spam", ["spam", "ham"]).tolist() == [1, 0]
    assert trainer.correction_targets("priority_regressor", ["low", "medium", "high"]).tolist() == [0.2, 0.5, 0.8]
    assert trainer.correction_targets("category", ["Work"]).tolist() == ["Work"]
    assert trainer.correction_targets("priority", ["high"]).tolist() == ["high"]


def test_exported_classifier_matches_sklearn():
    bundle = make_bundle("category")
    corr_X, corr_y = corrections_at(POINT, "Finance")
    fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    session = ort.InferenceSession(trainer.export_onnx(fitted, DIM), providers=["CPUExecutionProvider"])
    rows = np.vstack([POINT, BASE_ONLY_POINT]).astype(np.float32)
    label, proba = session.run(None, {"features": rows})
    np.testing.assert_array_equal(label, fitted.predict(sp.csr_matrix(rows)))
    np.testing.assert_allclose(proba, fitted.predict_proba(sp.csr_matrix(rows)), atol=1e-5)


def test_exported_regressor_matches_sklearn():
    bundle = make_bundle("priority_regressor")
    corr_X, corr_y = corrections_at(POINT, "high", model="priority_regressor")
    fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
    session = ort.InferenceSession(trainer.export_onnx(fitted, DIM), providers=["CPUExecutionProvider"])
    rows = np.vstack([POINT]).astype(np.float32)
    (out,) = session.run(None, {"features": rows})
    np.testing.assert_allclose(out.ravel(), fitted.predict(sp.csr_matrix(rows)), atol=1e-5)
```

Create `app/backend/tests/test_gate.py`:

```python
import numpy as np
import scipy.sparse as sp

from src.personalization.gate import evaluate_gate


class Scripted:
    """Returns canned predictions keyed by how many rows it's asked about."""

    def __init__(self, by_rows):
        self.by_rows = by_rows

    def predict(self, X):
        return np.asarray(self.by_rows[X.shape[0]])


BASE_CLASSES = ["Other", "Personal", "Work"]
CORR_X = sp.csr_matrix(np.zeros((5, 2)))
CORR_Y = np.array(["Work"] * 5)
GATE_X = sp.csr_matrix(np.zeros((10, 2)))
GATE_Y = np.array(["Other"] * 10)


def gate_for(own_pred, gate_pred, base_accuracy=0.9):
    fitted = Scripted({5: own_pred, 10: gate_pred})
    return evaluate_gate(fitted, CORR_X, CORR_Y, GATE_X, GATE_Y, base_accuracy, BASE_CLASSES)


def test_passes_when_corrections_took_and_nothing_regressed():
    result = gate_for(["Work"] * 5, ["Other"] * 9 + ["Work"])
    assert result.passed and result.reason is None
    assert result.own_correction_accuracy == 1.0
    assert result.gate_accuracy == 0.9


def test_rejects_when_corrections_did_not_take():
    result = gate_for(["Work"] * 3 + ["Other"] * 2, ["Other"] * 10)
    assert not result.passed
    assert "corrections" in result.reason


def test_rejects_an_accuracy_drop_over_three_points():
    result = gate_for(["Work"] * 5, ["Other"] * 8 + ["Work"] * 2, base_accuracy=0.9)
    assert not result.passed
    assert "accuracy" in result.reason


def test_custom_label_predictions_are_left_out_of_accuracy():
    result = gate_for(["Work"] * 5, ["Other"] * 9 + ["Finance"])
    assert result.passed
    assert result.custom_label_fraction == 0.1
    assert result.gate_accuracy == 1.0


def test_rejects_when_custom_labels_swallow_the_test_set():
    result = gate_for(["Work"] * 5, ["Other"] * 8 + ["Finance"] * 2)
    assert not result.passed
    assert "custom labels" in result.reason


def test_metrics_are_json_ready():
    metrics = gate_for(["Work"] * 5, ["Other"] * 10).as_metrics()
    assert metrics["passed"] is True
    assert set(metrics) == {
        "passed", "own_correction_accuracy", "gate_accuracy",
        "base_gate_accuracy", "custom_label_fraction", "reason",
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_trainer.py tests/test_gate.py -q`
Expected: FAIL with `ImportError` for `trainer` and `gate`.

- [ ] **Step 3: Implement the trainer**

Create `app/backend/src/personalization/trainer.py`:

```python
import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Starting value; the spec's §8 measurement picks the final one.
CORRECTION_WEIGHT = 10.0
# Regressor target for a corrected priority bucket.
BUCKET_SCORE = {"low": 0.2, "medium": 0.5, "high": 0.8}
SPAM_LABEL_TO_TARGET = {"ham": 0, "spam": 1}


def correction_targets(model: str, labels: list[str]) -> np.ndarray:
    if model == "spam":
        return np.array([SPAM_LABEL_TO_TARGET[label] for label in labels], dtype=np.int64)
    if model == "priority_regressor":
        return np.array([BUCKET_SCORE[label] for label in labels], dtype=np.float64)
    return np.array(labels, dtype=str)


def fit_personalized(base_X, base_y, estimator, corr_X, corr_y, weight: float = CORRECTION_WEIGHT):
    """Refit a clone of the production estimator on base rows plus corrections.

    Corrections are up-weighted, otherwise a handful of them against ~1,500
    base rows would barely move the model. Never fits `estimator` itself:
    it's the shared, cached template from base_data."""
    X = sp.vstack([base_X, corr_X]).tocsr()
    y = np.concatenate([np.asarray(base_y), np.asarray(corr_y)])
    sample_weight = np.concatenate([np.ones(base_X.shape[0]), np.full(corr_X.shape[0], weight)])
    fitted = clone(estimator)
    fitted.fit(X, y, sample_weight=sample_weight)
    return fitted


def export_onnx(fitted, dim: int) -> bytes:
    """Same headless export scripts/export_onnx.py uses: dense float vector
    in, classifier only, zipmap off so probabilities are a plain tensor."""
    is_classifier = hasattr(fitted, "classes_")
    options = {id(fitted): {"zipmap": False}} if is_classifier else None
    onnx_model = convert_sklearn(
        fitted,
        initial_types=[("features", FloatTensorType([None, dim]))],
        options=options,
        target_opset=17,
    )
    return onnx_model.SerializeToString()
```

- [ ] **Step 4: Implement the gate**

Create `app/backend/src/personalization/gate.py`:

```python
from dataclasses import asdict, dataclass

import numpy as np

OWN_CORRECTION_MIN = 0.80
MAX_GATE_DROP = 0.03
MAX_CUSTOM_LABEL_FRACTION = 0.10


@dataclass
class GateResult:
    passed: bool
    own_correction_accuracy: float
    gate_accuracy: float
    base_gate_accuracy: float
    custom_label_fraction: float
    reason: str | None

    def as_metrics(self) -> dict:
        return asdict(self)


def evaluate_gate(fitted, corr_X, corr_y, gate_X, gate_y, base_gate_accuracy, base_classes) -> GateResult:
    own = float(np.mean(fitted.predict(corr_X) == corr_y)) if corr_X.shape[0] else 0.0

    predictions = fitted.predict(gate_X)
    known = set(base_classes)
    custom_mask = np.array([p not in known for p in predictions], dtype=bool)
    custom_fraction = float(custom_mask.mean()) if len(predictions) else 0.0
    # The gate set's ground truth only knows the base classes, so emails
    # predicted as a custom label can't be scored right or wrong.
    scored = ~custom_mask
    gate_accuracy = float(np.mean(predictions[scored] == gate_y[scored])) if scored.any() else 0.0

    reason = None
    if own < OWN_CORRECTION_MIN:
        reason = f"only {own:.0%} of your corrections are predicted as corrected (need {OWN_CORRECTION_MIN:.0%})"
    elif custom_fraction > MAX_CUSTOM_LABEL_FRACTION:
        reason = (f"{custom_fraction:.0%} of the test set was pulled into custom labels "
                  f"(max {MAX_CUSTOM_LABEL_FRACTION:.0%})")
    elif gate_accuracy < base_gate_accuracy - MAX_GATE_DROP:
        reason = (f"test-set accuracy fell from {base_gate_accuracy:.1%} to {gate_accuracy:.1%} "
                  f"(max drop {MAX_GATE_DROP:.0%})")

    return GateResult(
        passed=reason is None,
        own_correction_accuracy=own,
        gate_accuracy=gate_accuracy,
        base_gate_accuracy=float(base_gate_accuracy),
        custom_label_fraction=custom_fraction,
        reason=reason,
    )
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/pytest.exe tests/test_trainer.py tests/test_gate.py -q`
Expected: `12 passed`.

- [ ] **Step 6: Commit**

```bash
git add app/backend/src/personalization/trainer.py app/backend/src/personalization/gate.py app/backend/tests/test_trainer.py app/backend/tests/test_gate.py
git commit -m "Add personalized retrain, validation gate, and ONNX export"
```

---

### Task 7: Retrain job

**Files:**
- Create: `app/backend/src/personalization/jobs.py`
- Test: `app/backend/tests/test_retrain_job.py`

**Interfaces:**
- Consumes: `base_data.load_base_bundle`, `trainer.*`, `evaluate_gate`, `SparseVector`, `to_csr`, and the models from Task 3.
- Produces: `RETRAIN_THRESHOLD = 5`; `scoped_corrections(stmt, user_id, model)`, which adds the user/model join and filter to a `select`; `correction_count(db, user_id, model) -> int`; `last_attempt(db, user_id, kind) -> PersonalizedModel | None`; `corrections_until_retrain(db, user_id, model) -> int`; `has_changes_since_last_attempt(db, user_id, model) -> bool`; `is_running(user_id, model) -> bool`; `run_retrain(user_id, model) -> None`, which is synchronous and opens its own DB session. Internal `_claim(user_id, model)` and `_release(user_id, model)` are used by tests. Throughout, `model` is a `CorrectionModel` and `kind` is a `PersonalizedModelKind`.

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_retrain_job.py`:

```python
from sqlalchemy import select

from src.models import (
    Correction,
    CorrectionModel,
    PersonalizedModel,
    PersonalizedModelKind,
    PersonalizedModelStatus,
)
from src.personalization import jobs
from src.personalization.gate import GateResult
from tests.fakes import DIM, patch_base_data

POINT = {"dim": DIM, "indices": [0, 5], "values": [1.0, 5.0]}


def add_corrections(db, connection, model, label, n=5, start=0):
    for i in range(start, start + n):
        db.add(Correction(
            email_connection_id=connection.id,
            provider_message_id=f"msg-{i}",
            model=model,
            feature_vector=POINT,
            predicted_label="Other",
            predicted_confidence=0.7,
            corrected_label=label,
        ))
    db.commit()


def attempts(db, user_id, kind):
    db.expire_all()
    return db.scalars(
        select(PersonalizedModel)
        .where(PersonalizedModel.user_id == user_id, PersonalizedModel.model == kind)
        .order_by(PersonalizedModel.version)
    ).all()


def test_retrain_activates_a_personalized_category_model(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    add_corrections(db, connected.connection, CorrectionModel.category, "Finance")

    jobs.run_retrain(connected.user.id, CorrectionModel.category)

    (attempt,) = attempts(db, connected.user.id, PersonalizedModelKind.category)
    assert attempt.status == PersonalizedModelStatus.active
    assert attempt.version == 1 and attempt.correction_count == 5
    assert "Finance" in attempt.classes
    assert attempt.onnx and attempt.metrics["passed"] is True


def test_second_retrain_supersedes_the_first(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work")
    jobs.run_retrain(connected.user.id, CorrectionModel.category)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work", start=5)
    jobs.run_retrain(connected.user.id, CorrectionModel.category)

    first, second = attempts(db, connected.user.id, PersonalizedModelKind.category)
    assert (first.version, first.status) == (1, PersonalizedModelStatus.superseded)
    assert (second.version, second.status) == (2, PersonalizedModelStatus.active)
    assert second.correction_count == 10


def test_gate_failure_is_recorded_without_activating(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    failing = GateResult(
        passed=False, own_correction_accuracy=0.2, gate_accuracy=0.9, base_gate_accuracy=0.9,
        custom_label_fraction=0.0, reason="only 20% of your corrections are predicted as corrected (need 80%)",
    )
    monkeypatch.setattr(jobs, "evaluate_gate", lambda *args, **kwargs: failing)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work")

    jobs.run_retrain(connected.user.id, CorrectionModel.category)

    (attempt,) = attempts(db, connected.user.id, PersonalizedModelKind.category)
    assert attempt.status == PersonalizedModelStatus.rejected
    assert attempt.onnx is None
    assert attempt.metrics["reason"].startswith("only 20%")


def test_a_training_error_is_recorded_as_a_rejection(db, connected, monkeypatch):
    patch_base_data(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(jobs.trainer, "fit_personalized", boom)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work")

    jobs.run_retrain(connected.user.id, CorrectionModel.category)

    (attempt,) = attempts(db, connected.user.id, PersonalizedModelKind.category)
    assert attempt.status == PersonalizedModelStatus.rejected
    assert attempt.metrics == {"error": "RuntimeError: disk full"}


def test_priority_retrain_activates_classifier_and_regressor_together(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    add_corrections(db, connected.connection, CorrectionModel.priority, "high")

    jobs.run_retrain(connected.user.id, CorrectionModel.priority)

    (classifier,) = attempts(db, connected.user.id, PersonalizedModelKind.priority)
    (regressor,) = attempts(db, connected.user.id, PersonalizedModelKind.priority_regressor)
    assert classifier.status == regressor.status == PersonalizedModelStatus.active
    assert regressor.classes is None and regressor.onnx


def test_a_second_job_for_the_same_user_and_model_is_skipped(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    add_corrections(db, connected.connection, CorrectionModel.category, "Work")
    assert jobs._claim(connected.user.id, CorrectionModel.category)
    try:
        jobs.run_retrain(connected.user.id, CorrectionModel.category)
    finally:
        jobs._release(connected.user.id, CorrectionModel.category)
    assert attempts(db, connected.user.id, PersonalizedModelKind.category) == []


def test_threshold_counts_only_corrections_since_the_last_attempt(db, connected, monkeypatch):
    patch_base_data(monkeypatch)
    user_id = connected.user.id
    add_corrections(db, connected.connection, CorrectionModel.category, "Work", n=3)
    assert jobs.corrections_until_retrain(db, user_id, CorrectionModel.category) == 2
    add_corrections(db, connected.connection, CorrectionModel.category, "Work", n=2, start=3)
    assert jobs.corrections_until_retrain(db, user_id, CorrectionModel.category) == 0

    jobs.run_retrain(user_id, CorrectionModel.category)

    db.expire_all()
    assert jobs.corrections_until_retrain(db, user_id, CorrectionModel.category) == 5
    assert jobs.has_changes_since_last_attempt(db, user_id, CorrectionModel.category) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_retrain_job.py -q`
Expected: FAIL with `ImportError: cannot import name 'jobs'`.

- [ ] **Step 3: Implement**

Create `app/backend/src/personalization/jobs.py`:

```python
"""Per-user retrain jobs (spec §4.2). run_retrain is synchronous: FastAPI
runs it in its background-task threadpool, one job at a time per
(user, model)."""
import threading
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from src.database import SessionLocal
from src.models import (
    Correction,
    CorrectionModel,
    EmailConnection,
    PersonalizedModel,
    PersonalizedModelKind,
    PersonalizedModelStatus,
)
from src.personalization import base_data, trainer
from src.personalization.gate import evaluate_gate
from src.personalization.sparse import SparseVector, to_csr

RETRAIN_THRESHOLD = 5

_running: set[tuple[uuid.UUID, str]] = set()
_running_lock = threading.Lock()


def is_running(user_id: uuid.UUID, model: CorrectionModel) -> bool:
    with _running_lock:
        return (user_id, model.value) in _running


def _claim(user_id: uuid.UUID, model: CorrectionModel) -> bool:
    with _running_lock:
        key = (user_id, model.value)
        if key in _running:
            return False
        _running.add(key)
        return True


def _release(user_id: uuid.UUID, model: CorrectionModel) -> None:
    with _running_lock:
        _running.discard((user_id, model.value))


def scoped_corrections(stmt, user_id: uuid.UUID, model: CorrectionModel):
    return stmt.join(EmailConnection, Correction.email_connection_id == EmailConnection.id).where(
        EmailConnection.user_id == user_id, Correction.model == model
    )


def correction_count(db: OrmSession, user_id: uuid.UUID, model: CorrectionModel) -> int:
    return db.scalar(scoped_corrections(select(func.count(Correction.id)), user_id, model)) or 0


def last_attempt(db: OrmSession, user_id: uuid.UUID, kind: PersonalizedModelKind) -> PersonalizedModel | None:
    return db.scalars(
        select(PersonalizedModel)
        .where(PersonalizedModel.user_id == user_id, PersonalizedModel.model == kind)
        .order_by(PersonalizedModel.version.desc())
        .limit(1)
    ).first()


def corrections_until_retrain(db: OrmSession, user_id: uuid.UUID, model: CorrectionModel) -> int:
    last = last_attempt(db, user_id, PersonalizedModelKind(model.value))
    new = correction_count(db, user_id, model) - (last.correction_count if last else 0)
    return max(0, RETRAIN_THRESHOLD - new)


def has_changes_since_last_attempt(db: OrmSession, user_id: uuid.UUID, model: CorrectionModel) -> bool:
    latest = db.scalar(scoped_corrections(select(func.max(Correction.created_at)), user_id, model))
    if latest is None:
        return False
    last = last_attempt(db, user_id, PersonalizedModelKind(model.value))
    return last is None or latest > last.created_at


def _next_version(db: OrmSession, user_id: uuid.UUID, kind: PersonalizedModelKind) -> int:
    current = db.scalar(
        select(func.max(PersonalizedModel.version)).where(
            PersonalizedModel.user_id == user_id, PersonalizedModel.model == kind
        )
    )
    return (current or 0) + 1


def _record_rejection(db, user_id, kind, count: int, metrics: dict) -> None:
    db.add(PersonalizedModel(
        user_id=user_id,
        model=kind,
        version=_next_version(db, user_id, kind),
        status=PersonalizedModelStatus.rejected,
        onnx=None,
        classes=None,
        correction_count=count,
        metrics=metrics,
    ))
    db.commit()


def _activate(db, user_id, kind, fitted, dim: int, count: int, metrics: dict) -> None:
    previous_active = db.scalars(
        select(PersonalizedModel).where(
            PersonalizedModel.user_id == user_id,
            PersonalizedModel.model == kind,
            PersonalizedModel.status == PersonalizedModelStatus.active,
        )
    ).all()
    for previous in previous_active:
        previous.status = PersonalizedModelStatus.superseded
    db.add(PersonalizedModel(
        user_id=user_id,
        model=kind,
        version=_next_version(db, user_id, kind),
        status=PersonalizedModelStatus.active,
        onnx=trainer.export_onnx(fitted, dim),
        classes=fitted.classes_.tolist() if hasattr(fitted, "classes_") else None,
        correction_count=count,
        metrics=metrics,
    ))


def run_retrain(user_id: uuid.UUID, model: CorrectionModel) -> None:
    if not _claim(user_id, model):
        return
    try:
        with SessionLocal() as db:
            _retrain(db, user_id, model)
    finally:
        _release(user_id, model)


def _retrain(db: OrmSession, user_id: uuid.UUID, model: CorrectionModel) -> None:
    rows = db.scalars(scoped_corrections(select(Correction), user_id, model).order_by(Correction.created_at)).all()
    if not rows:
        return
    kind = PersonalizedModelKind(model.value)
    labels = [row.corrected_label for row in rows]
    try:
        bundle = base_data.load_base_bundle(model.value)
        dim = bundle.X.shape[1]
        corr_X = to_csr([SparseVector.model_validate(row.feature_vector) for row in rows], dim)
        corr_y = trainer.correction_targets(model.value, labels)
        fitted = trainer.fit_personalized(bundle.X, bundle.y, bundle.estimator, corr_X, corr_y)
        gate = evaluate_gate(
            fitted, corr_X, corr_y, bundle.gate_X, bundle.gate_y, bundle.base_gate_accuracy, bundle.classes
        )
        if not gate.passed:
            _record_rejection(db, user_id, kind, len(rows), gate.as_metrics())
            return

        outputs = [(kind, fitted)]
        if model == CorrectionModel.priority:
            # The score bar comes from the regressor; retraining only the
            # bucket classifier would let the two disagree. Promoted together.
            reg_bundle = base_data.load_base_bundle("priority_regressor")
            if reg_bundle.X.shape[1] != dim:
                raise ValueError(f"priority_regressor dim {reg_bundle.X.shape[1]} != priority dim {dim}")
            reg_y = trainer.correction_targets("priority_regressor", labels)
            outputs.append((
                PersonalizedModelKind.priority_regressor,
                trainer.fit_personalized(reg_bundle.X, reg_bundle.y, reg_bundle.estimator, corr_X, reg_y),
            ))
        for out_kind, out_fitted in outputs:
            _activate(db, user_id, out_kind, out_fitted, dim, len(rows), gate.as_metrics())
        db.commit()
    except Exception as exc:  # a background job has no caller to raise to; record it instead
        db.rollback()
        _record_rejection(db, user_id, kind, len(rows), {"error": f"{type(exc).__name__}: {exc}"})
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/pytest.exe tests/test_retrain_job.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add app/backend/src/personalization/jobs.py app/backend/tests/test_retrain_job.py
git commit -m "Add per-user retrain job with versioned activation and rejection records"
```

---

### Task 8: Personalization API routes

**Files:**
- Create: `app/backend/src/routers/personalization.py`
- Modify: `app/backend/src/main.py`
- Test: `app/backend/tests/test_personalization_routes.py`

**Interfaces:**
- Consumes: `jobs.*`, `base_data.feature_dim`, the schemas from Task 4, and `get_current_session`.
- Produces HTTP routes (all need the `dmp_session` cookie; corrections and retrain return 403 unless opted in):
  - `PUT /personalization/settings` `{enabled}` → `{enabled}`. Setting `false` deletes the user's corrections and personalized models.
  - `POST /personalization/corrections` (`CorrectionIn`) → `{retrain_scheduled, corrections_until_retrain}`
  - `POST /personalization/retrain` → `{scheduled: [model names]}`
  - `GET /personalization/status` → `StatusOut`, with models ordered `category`, `priority`, `spam`
  - `GET /personalization/models` → `[{model, version, classes}]` for active models
  - `GET /personalization/models/{model}/{version}.onnx` → raw bytes

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_personalization_routes.py`:

```python
import onnxruntime as ort

from tests.fakes import DIM, patch_base_data

VECTOR = {"dim": DIM, "indices": [0, 5], "values": [1.0, 5.0]}


def correction(message_id, label="Work", model="category"):
    return {
        "model": model,
        "provider_message_id": message_id,
        "feature_vector": VECTOR,
        "predicted_label": "Other",
        "predicted_confidence": 0.7,
        "corrected_label": label,
    }


def enable(client):
    assert client.put("/personalization/settings", json={"enabled": True}).json() == {"enabled": True}


def test_status_requires_a_session(client):
    assert client.get("/personalization/status").status_code == 401


def test_status_starts_empty_and_disabled(client, connected):
    body = client.get("/personalization/status").json()
    assert body["enabled"] is False
    assert [m["model"] for m in body["models"]] == ["category", "priority", "spam"]
    assert all(m["correction_count"] == 0 and m["corrections_until_retrain"] == 5 for m in body["models"])
    assert body["custom_labels"] == []


def test_corrections_are_refused_until_opted_in(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    assert client.post("/personalization/corrections", json=correction("m1")).status_code == 403


def test_recorrecting_an_email_replaces_the_earlier_correction(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    client.post("/personalization/corrections", json=correction("m1", "Work"))
    response = client.post("/personalization/corrections", json=correction("m1", "Personal"))
    assert response.status_code == 200
    assert response.json()["corrections_until_retrain"] == 4
    assert client.get("/personalization/status").json()["models"][0]["correction_count"] == 1


def test_a_vector_of_the_wrong_dimension_is_rejected(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    bad = correction("m1")
    bad["feature_vector"] = {"dim": DIM + 1, "indices": [0], "values": [1.0]}
    assert client.post("/personalization/corrections", json=bad).status_code == 422


def test_the_fifth_correction_retrains_and_publishes_a_model(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    for i in range(4):
        body = client.post("/personalization/corrections", json=correction(f"m{i}", "Finance")).json()
        assert body["retrain_scheduled"] is False
    fifth = client.post("/personalization/corrections", json=correction("m4", "Finance")).json()
    assert fifth["retrain_scheduled"] is True

    # TestClient runs background tasks before returning, so the job is done.
    status = client.get("/personalization/status").json()
    assert status["models"][0]["last_attempt"]["status"] == "active"
    assert status["custom_labels"] == ["Finance"]
    manifest = client.get("/personalization/models").json()
    assert manifest == [{"model": "category", "version": 1, "classes": ["Finance", "Other", "Personal", "Work"]}]
    onnx_bytes = client.get("/personalization/models/category/1.onnx").content
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[1] == DIM


def test_manual_retrain_only_schedules_models_with_new_corrections(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    client.post("/personalization/corrections", json=correction("m1", "Work"))
    assert client.post("/personalization/retrain").json() == {"scheduled": ["category"]}
    assert client.post("/personalization/retrain").json() == {"scheduled": []}


def test_opting_out_deletes_corrections_and_models(client, connected, monkeypatch):
    patch_base_data(monkeypatch)
    enable(client)
    for i in range(5):
        client.post("/personalization/corrections", json=correction(f"m{i}", "Work"))
    assert client.get("/personalization/models").json() != []

    assert client.put("/personalization/settings", json={"enabled": False}).json() == {"enabled": False}

    status = client.get("/personalization/status").json()
    assert status["models"][0]["correction_count"] == 0
    assert status["models"][0]["last_attempt"] is None
    assert client.get("/personalization/models").json() == []


def test_unknown_model_version_is_404(client, connected):
    assert client.get("/personalization/models/category/9.onnx").status_code == 404
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_personalization_routes.py -q`
Expected: FAIL (all `/personalization/*` requests return 404).

- [ ] **Step 3: Implement the router**

Create `app/backend/src/routers/personalization.py`:

```python
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session as OrmSession

from src.database import get_db
from src.dependencies import get_current_session
from src.models import (
    Correction,
    CorrectionModel,
    EmailConnection,
    PersonalizedModel,
    PersonalizedModelKind,
    PersonalizedModelStatus,
    User,
)
from src.models import Session as SessionModel
from src.personalization import base_data, jobs
from src.schemas.personalization import (
    CorrectionIn,
    CorrectionOut,
    LastAttempt,
    ManifestEntry,
    ModelStatus,
    RetrainOut,
    SettingsIn,
    SettingsOut,
    StatusOut,
)

router = APIRouter(prefix="/personalization", tags=["personalization"])

BASE_CATEGORIES = ("Work", "Personal", "Other")


def _current_user(
    session: SessionModel = Depends(get_current_session),
    db: OrmSession = Depends(get_db),
) -> User:
    user = db.get(User, session.user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user


def _require_enabled(user: User) -> None:
    if not user.personalization_enabled:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Personalization is not enabled")


def _current_connection(db: OrmSession, user_id: uuid.UUID) -> EmailConnection:
    connection = db.scalars(
        select(EmailConnection)
        .where(EmailConnection.user_id == user_id)
        .order_by(EmailConnection.last_used_at.desc())
        .limit(1)
    ).first()
    if connection is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No email connection for this session")
    return connection


@router.put("/settings", response_model=SettingsOut)
def update_settings(
    body: SettingsIn, user: User = Depends(_current_user), db: OrmSession = Depends(get_db)
) -> SettingsOut:
    user.personalization_enabled = body.enabled
    if not body.enabled:
        connection_ids = select(EmailConnection.id).where(EmailConnection.user_id == user.id)
        db.execute(
            delete(Correction)
            .where(Correction.email_connection_id.in_(connection_ids))
            .execution_options(synchronize_session=False)
        )
        db.execute(
            delete(PersonalizedModel)
            .where(PersonalizedModel.user_id == user.id)
            .execution_options(synchronize_session=False)
        )
    db.commit()
    return SettingsOut(enabled=user.personalization_enabled)


@router.post("/corrections", response_model=CorrectionOut)
def submit_correction(
    body: CorrectionIn,
    background: BackgroundTasks,
    user: User = Depends(_current_user),
    db: OrmSession = Depends(get_db),
) -> CorrectionOut:
    _require_enabled(user)
    expected_dim = base_data.feature_dim(body.model.value)
    if body.feature_vector.dim != expected_dim:
        raise HTTPException(422, f"feature_vector.dim must be {expected_dim} for {body.model.value}")
    connection = _current_connection(db, user.id)
    values = {
        "predicted_label": body.predicted_label,
        "predicted_confidence": body.predicted_confidence,
        "corrected_label": body.corrected_label,
        "feature_vector": body.feature_vector.model_dump(),
    }
    db.execute(
        pg_insert(Correction)
        .values(
            id=uuid.uuid4(),
            email_connection_id=connection.id,
            provider_message_id=body.provider_message_id,
            model=body.model,
            **values,
        )
        .on_conflict_do_update(
            constraint="uq_corrections_message_model",
            set_={**values, "created_at": func.now()},
        )
    )
    db.commit()
    remaining = jobs.corrections_until_retrain(db, user.id, body.model)
    scheduled = remaining == 0 and not jobs.is_running(user.id, body.model)
    if scheduled:
        background.add_task(jobs.run_retrain, user.id, body.model)
    return CorrectionOut(retrain_scheduled=scheduled, corrections_until_retrain=remaining)


@router.post("/retrain", response_model=RetrainOut)
def retrain_now(
    background: BackgroundTasks, user: User = Depends(_current_user), db: OrmSession = Depends(get_db)
) -> RetrainOut:
    _require_enabled(user)
    scheduled = []
    for model in CorrectionModel:
        if jobs.has_changes_since_last_attempt(db, user.id, model) and not jobs.is_running(user.id, model):
            background.add_task(jobs.run_retrain, user.id, model)
            scheduled.append(model.value)
    return RetrainOut(scheduled=scheduled)


@router.get("/status", response_model=StatusOut)
def get_status(user: User = Depends(_current_user), db: OrmSession = Depends(get_db)) -> StatusOut:
    models = []
    for model in CorrectionModel:
        last = jobs.last_attempt(db, user.id, PersonalizedModelKind(model.value))
        models.append(ModelStatus(
            model=model.value,
            correction_count=jobs.correction_count(db, user.id, model),
            corrections_until_retrain=jobs.corrections_until_retrain(db, user.id, model),
            running=jobs.is_running(user.id, model),
            last_attempt=LastAttempt(
                version=last.version, status=last.status.value, metrics=last.metrics, created_at=last.created_at
            ) if last else None,
        ))
    custom_labels = db.scalars(
        jobs.scoped_corrections(select(Correction.corrected_label).distinct(), user.id, CorrectionModel.category)
        .where(Correction.corrected_label.not_in(BASE_CATEGORIES))
    ).all()
    return StatusOut(enabled=user.personalization_enabled, models=models, custom_labels=sorted(custom_labels))


@router.get("/models", response_model=list[ManifestEntry])
def get_manifest(user: User = Depends(_current_user), db: OrmSession = Depends(get_db)) -> list[ManifestEntry]:
    active = db.scalars(
        select(PersonalizedModel)
        .where(PersonalizedModel.user_id == user.id, PersonalizedModel.status == PersonalizedModelStatus.active)
        .order_by(PersonalizedModel.model)
    ).all()
    return [ManifestEntry(model=m.model.value, version=m.version, classes=m.classes or []) for m in active]


@router.get("/models/{model}/{version}.onnx")
def get_model_bytes(
    model: PersonalizedModelKind,
    version: int,
    user: User = Depends(_current_user),
    db: OrmSession = Depends(get_db),
) -> Response:
    row = db.scalars(
        select(PersonalizedModel).where(
            PersonalizedModel.user_id == user.id,
            PersonalizedModel.model == model,
            PersonalizedModel.version == version,
            PersonalizedModel.onnx.is_not(None),
        )
    ).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    return Response(
        content=row.onnx,
        media_type="application/octet-stream",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )
```

In `app/backend/src/main.py`, add the import and include the router next to `auth_router`:

```python
from src.routers.personalization import router as personalization_router
```

```python
app.include_router(auth_router)
app.include_router(personalization_router)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/pytest.exe tests/test_personalization_routes.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Run the full backend suite**

Run: `.venv/Scripts/pytest.exe -q`
Expected: all tests pass (`42 passed`).

- [ ] **Step 6: Commit**

```bash
git add app/backend/src/routers/personalization.py app/backend/src/main.py app/backend/tests/test_personalization_routes.py
git commit -m "Add personalization API: settings, corrections, retrain, status, model download"
```

---

### Task 9: Measurement script for the report

**Files:**
- Create: `app/backend/src/personalization/measure.py`
- Test: `app/backend/tests/test_measure.py`

**Interfaces:**
- Consumes: `trainer.fit_personalized`, `trainer.correction_targets`, `jobs.scoped_corrections`, `base_data.load_base_bundle`, `to_csr`.
- Produces: `measure(bundle, corr_X, corr_y, weights: list[float], seed: int = 0) -> list[dict]`. Each dict has keys `weight`, `held_out_before`, `held_out_after`, `gate_before`, `gate_after`. Also a CLI: `python -m src.personalization.measure --user-id <uuid> --model category [--weights 1 3 10 30]`.

- [ ] **Step 1: Write the failing test**

Create `app/backend/tests/test_measure.py`:

```python
import numpy as np
import pytest
import scipy.sparse as sp

from src.personalization import trainer
from src.personalization.measure import measure
from tests.fakes import make_bundle

POINT = np.array([1.0, 0, 0, 0, 0, 5.0])


def test_reports_before_and_after_on_held_out_corrections():
    bundle = make_bundle("category")
    corr_X = sp.csr_matrix(np.tile(POINT, (8, 1)))
    corr_y = trainer.correction_targets("category", ["Work"] * 8)

    rows = measure(bundle, corr_X, corr_y, weights=[1.0, 10.0])

    assert [r["weight"] for r in rows] == [1.0, 10.0]
    assert rows[1]["held_out_before"] == 0.0
    assert rows[1]["held_out_after"] == 1.0
    assert rows[1]["gate_before"] == bundle.base_gate_accuracy
    assert 0.0 <= rows[1]["gate_after"] <= 1.0


def test_needs_enough_corrections_to_split():
    bundle = make_bundle("category")
    corr_X = sp.csr_matrix(np.tile(POINT, (3, 1)))
    with pytest.raises(ValueError, match="at least 4"):
        measure(bundle, corr_X, trainer.correction_targets("category", ["Work"] * 3), weights=[10.0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/pytest.exe tests/test_measure.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.personalization.measure'`.

- [ ] **Step 3: Implement**

Create `app/backend/src/personalization/measure.py`:

```python
"""Before/after accuracy on held-out corrections (spec §8).

Trains on half of a user's corrections and measures the other half, before
(base model) and after (personalized), for several correction weights.
Also reports gate-set accuracy so the report shows nothing else regressed.

  python -m src.personalization.measure --user-id <uuid> --model category
"""
import argparse
import uuid

import numpy as np
from sklearn.base import clone
from sqlalchemy import select

from src.database import SessionLocal
from src.models import Correction, CorrectionModel
from src.personalization import base_data, jobs, trainer
from src.personalization.sparse import SparseVector, to_csr


def measure(bundle, corr_X, corr_y, weights: list[float], seed: int = 0) -> list[dict]:
    n = corr_X.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 corrections to split, got {n}")
    order = np.random.RandomState(seed).permutation(n)
    train_idx, test_idx = order[: n // 2], order[n // 2:]

    base = clone(bundle.estimator).fit(bundle.X, bundle.y)
    held_out_before = float(np.mean(base.predict(corr_X[test_idx]) == corr_y[test_idx]))

    results = []
    for weight in weights:
        fitted = trainer.fit_personalized(
            bundle.X, bundle.y, bundle.estimator, corr_X[train_idx], corr_y[train_idx], weight=weight
        )
        results.append({
            "weight": weight,
            "held_out_before": held_out_before,
            "held_out_after": float(np.mean(fitted.predict(corr_X[test_idx]) == corr_y[test_idx])),
            "gate_before": bundle.base_gate_accuracy,
            "gate_after": (float(np.mean(fitted.predict(bundle.gate_X) == bundle.gate_y))
                           if bundle.gate_X is not None else None),
        })
    return results


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user-id", required=True, type=uuid.UUID)
    parser.add_argument("--model", required=True, choices=[m.value for m in CorrectionModel])
    parser.add_argument("--weights", nargs="+", type=float, default=[1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = CorrectionModel(args.model)
    with SessionLocal() as db:
        rows = db.scalars(
            jobs.scoped_corrections(select(Correction), args.user_id, model).order_by(Correction.created_at)
        ).all()
    bundle = base_data.load_base_bundle(model.value)
    corr_X = to_csr([SparseVector.model_validate(r.feature_vector) for r in rows], bundle.X.shape[1])
    corr_y = trainer.correction_targets(model.value, [r.corrected_label for r in rows])

    print(f"{len(rows)} corrections: trained on {len(rows) // 2}, measured on {len(rows) - len(rows) // 2}")
    print(f"{'weight':>8} {'held-out before':>16} {'held-out after':>15} {'gate before':>12} {'gate after':>11}")
    for r in measure(bundle, corr_X, corr_y, args.weights, args.seed):
        print(f"{r['weight']:>8.1f} {_pct(r['held_out_before']):>16} {_pct(r['held_out_after']):>15} "
              f"{_pct(r['gate_before']):>12} {_pct(r['gate_after']):>11}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/pytest.exe tests/test_measure.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add app/backend/src/personalization/measure.py app/backend/tests/test_measure.py
git commit -m "Add held-out before/after measurement script for personalization"
```

---

### Task 10: Frontend types, classification inputs, correction helpers

**Files:**
- Modify: `app/frontend/src/types/index.ts`
- Modify: `app/frontend/src/inference/engine.ts` (classify's return value only)
- Create: `app/frontend/src/lib/corrections.ts`
- Test: `app/frontend/src/lib/corrections.test.ts`

**Interfaces:**
- Produces types: `PersonalizableModel = 'spam' | 'category' | 'priority'`; `BackendModelName = PersonalizableModel | 'priority_regressor'`; `PersonalizedModelArtifact { model: BackendModelName; version: number; classes: (string | number)[]; bytes: ArrayBuffer }`; `ClassificationInputs { spamConf: number; vaderCompound: number; categoryLabel: string; priorityBucket: PriorityBucket; priorityConfidences: Record<string, number> }`. `ClassificationResult` gains `inputs: ClassificationInputs`, which holds the raw values the models saw, before rounding, spam suppression, or any user correction.
- Produces (`corrections.ts`): `BUCKET_SCORE`; `predictedFor(model, result) -> { label: string; confidence: number }`; `applyCorrection(result, model, label) -> ClassificationResult`; `applyCorrections(result, corrected: Partial<Record<PersonalizableModel, string>>) -> ClassificationResult`.

- [ ] **Step 1: Write the failing test**

Create `app/frontend/src/lib/corrections.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import type { ClassificationResult } from '../types'
import { applyCorrection, applyCorrections, predictedFor } from './corrections'

const result: ClassificationResult = {
  spam: { label: 'ham', confidence: 0.2 },
  category: { label: 'Other', confidences: { Work: 0.3, Personal: 0.1, Other: 0.6 } },
  priority: { score: 0.1, bucket: 'low', note: 'classified as spam (conf=0.20) — priority suppressed' },
  inputs: {
    spamConf: 0.2,
    vaderCompound: -0.1,
    categoryLabel: 'Other',
    priorityBucket: 'medium',
    priorityConfidences: { high: 0.1, low: 0.3, medium: 0.6 },
  },
}

describe('predictedFor', () => {
  it('reports what each model actually predicted, before suppression or corrections', () => {
    expect(predictedFor('category', result)).toEqual({ label: 'Other', confidence: 0.6 })
    expect(predictedFor('priority', result)).toEqual({ label: 'medium', confidence: 0.6 })
    expect(predictedFor('spam', result)).toEqual({ label: 'ham', confidence: 0.8 })
  })
})

describe('applyCorrection', () => {
  it('overrides the category label, including a custom one', () => {
    expect(applyCorrection(result, 'category', 'Finance').category.label).toBe('Finance')
  })

  it('moves the priority score to the bucket representative and drops the suppression note', () => {
    expect(applyCorrection(result, 'priority', 'high').priority).toEqual({ score: 0.8, bucket: 'high' })
  })

  it('flips the spam label without touching its confidence', () => {
    expect(applyCorrection(result, 'spam', 'spam').spam).toEqual({ label: 'spam', confidence: 0.2 })
  })

  it('never mutates the model result', () => {
    applyCorrection(result, 'category', 'Work')
    expect(result.category.label).toBe('Other')
  })
})

describe('applyCorrections', () => {
  it('applies every stored correction', () => {
    const shown = applyCorrections(result, { category: 'Work', priority: 'high' })
    expect(shown.category.label).toBe('Work')
    expect(shown.priority.bucket).toBe('high')
    expect(shown.spam.label).toBe('ham')
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run (from `app/frontend`): `npx vitest run src/lib/corrections.test.ts`
Expected: FAIL (cannot find module `./corrections`, and `inputs` is not a known property).

- [ ] **Step 3: Extend the types**

In `app/frontend/src/types/index.ts`, replace the `ClassificationResult` interface with:

```ts
export type PersonalizableModel = 'spam' | 'category' | 'priority'
export type BackendModelName = PersonalizableModel | 'priority_regressor'

// A user's personalized ONNX graph, as downloaded from the backend.
export interface PersonalizedModelArtifact {
  model: BackendModelName
  version: number
  classes: (string | number)[]
  bytes: ArrayBuffer
}

// What the models actually saw and said, before rounding, spam suppression,
// or any user correction. Corrections are built from these values, so the
// backend trains on exactly what production computed.
export interface ClassificationInputs {
  spamConf: number
  vaderCompound: number
  categoryLabel: string
  priorityBucket: PriorityBucket
  priorityConfidences: Record<string, number>
}

export interface ClassificationResult {
  spam: { label: SpamLabel; confidence: number }
  category: { label: CategoryLabel | string; confidences: Record<string, number> }
  priority: { score: number; bucket: PriorityBucket; note?: string }
  inputs: ClassificationInputs
}
```

- [ ] **Step 4: Populate `inputs` in the engine**

In `app/frontend/src/inference/engine.ts`, directly after the line `const priorityBucket = (priorityClfOut.label.data as string[])[0] as PriorityBucket`, add:

```ts
  const priorityProba = priorityClfOut.probabilities.data as Float32Array
  const priorityConfidences: Record<string, number> = {}
  priorityVocab.classes.forEach((cls, i) => {
    priorityConfidences[cls] = priorityProba[i]
  })
```

Replace the final `return { ... }` of `classify` with:

```ts
  return {
    spam: { label: spamLabel, confidence: Math.round(spamConf * 10000) / 10000 },
    category: { label: categoryLabel, confidences: categoryConfidences },
    priority,
    inputs: {
      spamConf,
      vaderCompound: compound,
      categoryLabel,
      priorityBucket,
      priorityConfidences,
    },
  }
```

- [ ] **Step 5: Write the helpers**

Create `app/frontend/src/lib/corrections.ts`:

```ts
import type { ClassificationResult, PersonalizableModel, PriorityBucket, SpamLabel } from '../types'

// Representative score for a corrected bucket. Same values the backend
// trains the priority regressor on (trainer.BUCKET_SCORE).
export const BUCKET_SCORE: Record<PriorityBucket, number> = { low: 0.2, medium: 0.5, high: 0.8 }

export function predictedFor(
  model: PersonalizableModel,
  result: ClassificationResult,
): { label: string; confidence: number } {
  const { inputs } = result
  if (model === 'category') {
    return { label: inputs.categoryLabel, confidence: result.category.confidences[inputs.categoryLabel] ?? 0 }
  }
  if (model === 'priority') {
    return { label: inputs.priorityBucket, confidence: inputs.priorityConfidences[inputs.priorityBucket] ?? 0 }
  }
  const label: SpamLabel = inputs.spamConf >= 0.5 ? 'spam' : 'ham'
  return { label, confidence: label === 'spam' ? inputs.spamConf : 1 - inputs.spamConf }
}

// Display-only: a user's correction always wins for the email it was made
// on, whatever the model says.
export function applyCorrection(
  result: ClassificationResult,
  model: PersonalizableModel,
  label: string,
): ClassificationResult {
  if (model === 'category') return { ...result, category: { ...result.category, label } }
  if (model === 'priority') {
    const bucket = label as PriorityBucket
    return { ...result, priority: { score: BUCKET_SCORE[bucket], bucket } }
  }
  return { ...result, spam: { ...result.spam, label: label as SpamLabel } }
}

export function applyCorrections(
  result: ClassificationResult,
  corrected: Partial<Record<PersonalizableModel, string>>,
): ClassificationResult {
  return (Object.entries(corrected) as [PersonalizableModel, string][]).reduce(
    (shown, [model, label]) => applyCorrection(shown, model, label),
    result,
  )
}
```

- [ ] **Step 6: Run tests, type check, lint**

Run (from `app/frontend`):
```
npx vitest run src/lib/corrections.test.ts
npx tsc --noEmit
npm run lint
```
Expected: `6 passed`; `tsc` reports no errors; lint is clean.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/src/types/index.ts app/frontend/src/inference/engine.ts app/frontend/src/lib/corrections.ts app/frontend/src/lib/corrections.test.ts
git commit -m "Keep raw model inputs on classification results; add correction helpers"
```

---

### Task 11: Sparse encoding and personalization API client

**Files:**
- Create: `app/frontend/src/lib/sparse.ts`, `app/frontend/src/lib/personalization-api.ts`
- Test: `app/frontend/src/lib/sparse.test.ts`, `app/frontend/src/lib/personalization-api.test.ts`

**Interfaces:**
- Consumes: `PersonalizableModel`, `BackendModelName`, and `PersonalizedModelArtifact` from Task 10.
- Produces (`sparse.ts`): `SparseVector { dim; indices; values }`, `toSparse(dense: Float32Array): SparseVector`, `toDense(v: SparseVector): Float32Array`.
- Produces (`personalization-api.ts`): types `LastAttempt`, `ModelStatus`, `PersonalizationStatus`, `CorrectionPayload`, `CorrectionResponse`. Functions: `getStatus()`, `setEnabled(enabled)`, `submitCorrection(payload)`, `retrainNow()`, `fetchActiveModels(): Promise<PersonalizedModelArtifact[]>`. All send `credentials: 'include'` and throw `Error` with the status code on a non-2xx response.

- [ ] **Step 1: Write the failing tests**

Create `app/frontend/src/lib/sparse.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { toDense, toSparse } from './sparse'

describe('toSparse', () => {
  it('keeps only non-zero entries, including negative ones', () => {
    expect(toSparse(new Float32Array([0, 0.5, 0, -1.25]))).toEqual({ dim: 4, indices: [1, 3], values: [0.5, -1.25] })
  })

  it('round-trips through toDense exactly', () => {
    const dense = new Float32Array([0.1, 0, 0, 3.7, 0, -0.2])
    expect(Array.from(toDense(toSparse(dense)))).toEqual(Array.from(dense))
  })

  it('encodes an all-zero vector as empty', () => {
    expect(toSparse(new Float32Array(3))).toEqual({ dim: 3, indices: [], values: [] })
  })
})
```

Create `app/frontend/src/lib/personalization-api.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from 'vitest'
import { submitCorrection, type CorrectionPayload } from './personalization-api'

const payload: CorrectionPayload = {
  model: 'category',
  provider_message_id: 'm1',
  feature_vector: { dim: 2, indices: [0], values: [1] },
  predicted_label: 'Other',
  predicted_confidence: 0.6,
  corrected_label: 'Work',
}

describe('submitCorrection', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('posts the correction with credentials and returns the parsed body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ retrain_scheduled: false, corrections_until_retrain: 4 }), { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(submitCorrection(payload)).resolves.toEqual({ retrain_scheduled: false, corrections_until_retrain: 4 })

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:3011/personalization/corrections')
    expect(init.method).toBe('POST')
    expect(init.credentials).toBe('include')
    expect(JSON.parse(init.body)).toEqual(payload)
  })

  it('throws with the status code when the backend refuses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 403 })))
    await expect(submitCorrection(payload)).rejects.toThrow('403')
  })
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/lib/sparse.test.ts src/lib/personalization-api.test.ts`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement**

Create `app/frontend/src/lib/sparse.ts`:

```ts
// Dense model vectors are thousands of dimensions and nearly all zero, so
// corrections travel as (index, value) pairs. Lossless: toDense(toSparse(v))
// reproduces v exactly.
export interface SparseVector {
  dim: number
  indices: number[]
  values: number[]
}

export function toSparse(dense: Float32Array): SparseVector {
  const indices: number[] = []
  const values: number[] = []
  for (let i = 0; i < dense.length; i++) {
    if (dense[i] !== 0) {
      indices.push(i)
      values.push(dense[i])
    }
  }
  return { dim: dense.length, indices, values }
}

export function toDense(vector: SparseVector): Float32Array {
  const out = new Float32Array(vector.dim)
  vector.indices.forEach((index, k) => {
    out[index] = vector.values[k]
  })
  return out
}
```

Create `app/frontend/src/lib/personalization-api.ts`:

```ts
import type { BackendModelName, PersonalizableModel, PersonalizedModelArtifact } from '../types'
import type { SparseVector } from './sparse'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:3011'

export interface LastAttempt {
  version: number
  status: 'active' | 'rejected' | 'superseded'
  metrics: { reason?: string | null; error?: string; passed?: boolean; [key: string]: unknown }
  created_at: string
}

export interface ModelStatus {
  model: PersonalizableModel
  correction_count: number
  corrections_until_retrain: number
  running: boolean
  last_attempt: LastAttempt | null
}

export interface PersonalizationStatus {
  enabled: boolean
  models: ModelStatus[]
  custom_labels: string[]
}

export interface CorrectionPayload {
  model: PersonalizableModel
  provider_message_id: string
  feature_vector: SparseVector
  predicted_label: string
  predicted_confidence: number
  corrected_label: string
}

export interface CorrectionResponse {
  retrain_scheduled: boolean
  corrections_until_retrain: number
}

interface ManifestEntry {
  model: BackendModelName
  version: number
  classes: (string | number)[]
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const headers: Record<string, string> = init.body ? { 'Content-Type': 'application/json' } : {}
  const res = await fetch(`${BACKEND_URL}${path}`, { ...init, headers, credentials: 'include' })
  if (!res.ok) throw new Error(`${init.method ?? 'GET'} ${path} failed (${res.status})`)
  return res
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  return (await request(path, init)).json() as Promise<T>
}

export function getStatus(): Promise<PersonalizationStatus> {
  return json('/personalization/status')
}

export function setEnabled(enabled: boolean): Promise<{ enabled: boolean }> {
  return json('/personalization/settings', { method: 'PUT', body: JSON.stringify({ enabled }) })
}

export function submitCorrection(payload: CorrectionPayload): Promise<CorrectionResponse> {
  return json('/personalization/corrections', { method: 'POST', body: JSON.stringify(payload) })
}

export function retrainNow(): Promise<{ scheduled: string[] }> {
  return json('/personalization/retrain', { method: 'POST' })
}

export async function fetchActiveModels(): Promise<PersonalizedModelArtifact[]> {
  const manifest = await json<ManifestEntry[]>('/personalization/models')
  const artifacts: PersonalizedModelArtifact[] = []
  for (const entry of manifest) {
    const res = await request(`/personalization/models/${entry.model}/${entry.version}.onnx`)
    artifacts.push({ ...entry, bytes: await res.arrayBuffer() })
  }
  return artifacts
}
```

- [ ] **Step 4: Run tests and type check**

Run: `npx vitest run src/lib/sparse.test.ts src/lib/personalization-api.test.ts && npx tsc --noEmit`
Expected: `5 passed`; no type errors.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/src/lib/sparse.ts app/frontend/src/lib/sparse.test.ts app/frontend/src/lib/personalization-api.ts app/frontend/src/lib/personalization-api.test.ts
git commit -m "Add sparse vector encoding and personalization API client"
```

---

### Task 12: Sparse parity check (client vectors vs sklearn)

The spec (§12) asks for a parity fixture. This check runs as a dev script, not a Vitest test, because it needs the generated `public/models/*.vocab.json` files, which are gitignored. That matches the existing `scripts/e2e_check.mjs` convention.

**Files:**
- Create: `scripts/dump_sparse_parity_fixture.py`
- Create: `app/frontend/scripts/fixtures/sparse_parity.json` (generated, committed)
- Create: `app/frontend/scripts/sparse_parity_check.mjs`

**Interfaces:**
- Consumes: `toSparse` (Task 11), `buildSpamVector`, `buildCategoryVector`, `buildPriorityVector`, `cleanText` (existing, `src/inference/features.ts`).
- Produces: a script that exits 1 on any index or value mismatch.

- [ ] **Step 1: Write the Python fixture dumper**

Create `scripts/dump_sparse_parity_fixture.py`:

```python
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
```

- [ ] **Step 2: Generate the fixture**

Run (from `scripts/`): `python dump_sparse_parity_fixture.py`
Expected: `Wrote ...app\frontend\scripts\fixtures\sparse_parity.json`.

- [ ] **Step 3: Write the Node check**

Create `app/frontend/scripts/sparse_parity_check.mjs`:

```js
// Dev-only check (spec §12): the vectors the client sends as corrections
// must match, index for index, the sklearn vectors in the backend's base
// training matrices. Regenerate the fixture with
// scripts/dump_sparse_parity_fixture.py (repo root) if the models retrain.
//
// Usage (from app/frontend): npx tsx scripts/sparse_parity_check.mjs
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { buildSpamVector, buildCategoryVector, buildPriorityVector, cleanText } from '../src/inference/features.ts'
import { toSparse } from '../src/lib/sparse.ts'

const here = dirname(fileURLToPath(import.meta.url))
const vocab = (name) => JSON.parse(readFileSync(join(here, '../public/models', `${name}.vocab.json`), 'utf-8'))
const spamVocab = vocab('spam_classifier')
const categoryVocab = vocab('category_classifier')
const priorityVocab = vocab('priority_classifier')
const fixture = JSON.parse(readFileSync(join(here, 'fixtures/sparse_parity.json'), 'utf-8'))

const TOLERANCE = 1e-5
let failures = 0

function compare(label, got, want) {
  const sameIndices =
    got.indices.length === want.indices.length && got.indices.every((index, k) => index === want.indices[k])
  const maxDiff = sameIndices ? Math.max(0, ...got.values.map((v, k) => Math.abs(v - want.values[k]))) : Infinity
  const ok = sameIndices && maxDiff <= TOLERANCE
  if (!ok) failures++
  console.log(`${ok ? 'OK  ' : 'FAIL'} ${label}: ${got.indices.length} non-zeros, max value diff ${maxDiff.toExponential(2)}`)
}

for (const { email, spam, category, priority } of fixture) {
  const input = { id: email.id, subject: email.subject, body: email.body, to: email.to, fromAddr: email.fromAddr }
  const text = cleanText(`${email.subject} ${email.body}`)
  compare(`${email.id} spam`, toSparse(buildSpamVector(text, spamVocab)), spam)
  compare(`${email.id} category`, toSparse(buildCategoryVector(input, text, categoryVocab)), category)
  compare(
    `${email.id} priority`,
    toSparse(buildPriorityVector(input, text, priority.categoryLabel, priority.spamConf, priority.vaderCompound, priorityVocab)),
    priority,
  )
}

if (failures > 0) {
  console.error(`\n${failures} mismatch(es)`)
  process.exit(1)
}
console.log('\nAll vectors match.')
```

- [ ] **Step 4: Run it**

Run (from `app/frontend`): `npx tsx scripts/sparse_parity_check.mjs`
Expected: nine `OK` lines and `All vectors match.`

If any line fails, stop and report it. A failure means corrections would be trained in a different feature space from the base rows.

- [ ] **Step 5: Commit**

```bash
git add scripts/dump_sparse_parity_fixture.py app/frontend/scripts/fixtures/sparse_parity.json app/frontend/scripts/sparse_parity_check.mjs
git commit -m "Add client-vs-sklearn sparse vector parity check"
```

---

### Task 13: Engine support for personalized models

**Files:**
- Modify: `app/frontend/src/inference/engine.ts`

**Interfaces:**
- Consumes: `PersonalizedModelArtifact`, `PersonalizableModel`, `ClassificationResult` (Task 10).
- Produces: `applyPersonalizedModels(artifacts: PersonalizedModelArtifact[]): Promise<void>`. Passing `[]` reverts to base models. It is serialized with every `classify()` call. `buildCorrectionVector(email: EmailInput, model: PersonalizableModel, result: ClassificationResult): Float32Array`. `classify()` keeps its signature but now runs exclusively, and it maps confidences using the personalized `classes` when present.

- [ ] **Step 1: Add the serialization chain, session bookkeeping, and model swap**

In `app/frontend/src/inference/engine.ts`:

Change the types import to:

```ts
import type {
  EmailInput,
  ClassificationResult,
  CategoryLabel,
  PersonalizableModel,
  PersonalizedModelArtifact,
  PriorityBucket,
} from '../types'
```

Replace `let sessions: Partial<Record<ModelKey, ort.InferenceSession>> = {}` with:

```ts
type Sessions = Partial<Record<ModelKey, ort.InferenceSession>>

// Backend model names (personalized_models.model) → this module's keys.
const BACKEND_MODEL_KEYS: Record<PersonalizedModelArtifact['model'], ModelKey> = {
  spam: 'spam',
  category: 'category',
  priority: 'priority',
  priority_regressor: 'priorityRegressor',
}

let baseSessions: Sessions = {}
let sessions: Sessions = {}
let personalSessions: ort.InferenceSession[] = []
// Output class order for personalized graphs (a personalized category
// model can have custom classes the shared vocab.json doesn't know about).
let classesOverride: Partial<Record<ModelKey, (string | number)[]>> = {}

// onnxruntime-web's WASM backend tolerates only one session operation
// (create or run) in flight at a time. classify() already awaits its own
// .run() calls in sequence; this chain extends that guarantee across
// callers, so a personalized-model swap can never overlap a classify().
let engineChain: Promise<unknown> = Promise.resolve()
function exclusive<T>(fn: () => Promise<T>): Promise<T> {
  const run = engineChain.then(fn)
  engineChain = run.catch(() => undefined)
  return run
}
```

In `loadModels()`, replace `sessions = newSessions` with:

```ts
    baseSessions = newSessions
    sessions = { ...newSessions }
```

Replace `modelsLoaded()`'s first condition `Object.keys(sessions).length` with `Object.keys(baseSessions).length`.

Add after `modelsLoaded()`:

```ts
export function applyPersonalizedModels(artifacts: PersonalizedModelArtifact[]): Promise<void> {
  return exclusive(async () => {
    const created: ort.InferenceSession[] = []
    const next: Sessions = { ...baseSessions }
    const nextClasses: typeof classesOverride = {}
    try {
      for (const artifact of artifacts) {
        const key = BACKEND_MODEL_KEYS[artifact.model]
        const session = await ort.InferenceSession.create(new Uint8Array(artifact.bytes))
        created.push(session)
        next[key] = session
        nextClasses[key] = artifact.classes
      }
    } catch (err) {
      for (const session of created) await session.release()
      throw err
    }
    const retired = personalSessions
    sessions = next
    classesOverride = nextClasses
    personalSessions = created
    for (const session of retired) await session.release()
  })
}

export function buildCorrectionVector(
  email: EmailInput,
  model: PersonalizableModel,
  result: ClassificationResult,
): Float32Array {
  if (!spamVocab || !categoryVocab || !priorityVocab) {
    throw new Error('Models not loaded — call loadModels() first.')
  }
  const cleanedText = cleanText(`${email.subject} ${email.body}`)
  if (model === 'spam') return buildSpamVector(cleanedText, spamVocab)
  if (model === 'category') return buildCategoryVector(email, cleanedText, categoryVocab)
  const { categoryLabel, spamConf, vaderCompound } = result.inputs
  return buildPriorityVector(email, cleanedText, categoryLabel, spamConf, vaderCompound, priorityVocab)
}
```

- [ ] **Step 2: Make `classify` exclusive and class-override aware**

Rename the existing `export async function classify(email: EmailInput): Promise<ClassificationResult> {` to `async function classifyNow(email: EmailInput): Promise<ClassificationResult> {` and add above it:

```ts
export function classify(email: EmailInput): Promise<ClassificationResult> {
  return exclusive(() => classifyNow(email))
}
```

Inside `classifyNow`, make three replacements.

Replace `const spamIdx = spamVocab.classes.indexOf(1)` with:

```ts
  const spamIdx = ((classesOverride.spam ?? spamVocab.classes) as number[]).indexOf(1)
```

Replace `categoryVocab.classes.forEach((cls, i) => {` with:

```ts
  ;((classesOverride.category ?? categoryVocab.classes) as string[]).forEach((cls, i) => {
```

Replace `priorityVocab.classes.forEach((cls, i) => {` (added in Task 10) with:

```ts
  ;((classesOverride.priority ?? priorityVocab.classes) as string[]).forEach((cls, i) => {
```

- [ ] **Step 3: Verify**

Run (from `app/frontend`):
```
npx tsc --noEmit
npm run lint
npm run test
```
Expected: no type errors; lint clean; all Vitest tests pass.

Then run `npm run dev` from the repo root with the backend up. Load `http://localhost:3010/inbox` and confirm the inbox still classifies (sidebar shows `● GMAIL CONNECTED`, and rows show category pills). This proves the exclusive chain didn't deadlock `classify()`. Stop the dev server afterwards.

- [ ] **Step 4: Commit**

```bash
git add app/frontend/src/inference/engine.ts
git commit -m "Let the engine swap in personalized ONNX models and build correction vectors"
```

---

### Task 14: Inbox state: corrections, sync, retrain watching

**Files:**
- Modify: `app/frontend/src/app/inbox/inbox-context.tsx` (full replacement below)

**Interfaces:**
- Consumes: Tasks 10, 11, 13.
- Produces (via `useInbox()`):
  - `Row` gains `modelResult: ClassificationResult | null` and `corrected: Partial<Record<PersonalizableModel, string>>`. `result` is now the model output with corrections applied.
  - `filter: string` and `setFilter(f: string)`, so custom labels can be filters.
  - `personalization: PersonalizationStatus | null`, `personalizationError: string | null`, `retraining: boolean`.
  - `correctEmail(id, model, label): Promise<void>`, `enablePersonalization(): Promise<boolean>`, `disablePersonalization(): Promise<void>`, `retrainPersonalization(): Promise<void>`.

- [ ] **Step 1: Replace `app/frontend/src/app/inbox/inbox-context.tsx`**

```tsx
'use client'

import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { applyPersonalizedModels, buildCorrectionVector, classify, loadModels, modelsLoaded } from '@/inference/engine'
import {
  startGmailConnect,
  fetchGmailAccessToken,
  loadStoredGmailToken,
  saveGmailToken,
  clearStoredGmailToken,
} from '@/lib/gmail-auth'
import { fetchRecentInboxEmails } from '@/lib/gmail-fetch'
import { applyCorrections, predictedFor } from '@/lib/corrections'
import {
  fetchActiveModels,
  getStatus,
  retrainNow,
  setEnabled,
  submitCorrection,
  type CorrectionResponse,
  type PersonalizationStatus,
} from '@/lib/personalization-api'
import { toSparse } from '@/lib/sparse'
import type { ClassificationResult, InboxEmail, PersonalizableModel, PersonalizedModelArtifact } from '@/types'
import { SAMPLE_EMAILS } from '@/data/sample-emails'

export type Row = InboxEmail & {
  // What the UI renders: the model's output with this user's corrections
  // applied on top (a correction always wins for the email it was made on).
  result: ClassificationResult | null
  // Raw model output, never overwritten. Corrections are built from this.
  modelResult: ClassificationResult | null
  corrected: Partial<Record<PersonalizableModel, string>>
}
export type GmailStatus = 'disconnected' | 'connecting' | 'fetching' | 'connected' | 'error'

interface InboxState {
  rows: Row[]
  status: 'loading' | 'ready' | 'error'
  error: string | null
  filter: string
  setFilter: (f: string) => void
  onlyHigh: boolean
  setOnlyHigh: (v: boolean) => void
  showArchived: boolean
  setShowArchived: (v: boolean) => void
  query: string
  setQuery: (q: string) => void
  archivedIds: Set<string>
  archiveEmail: (id: string) => void
  unarchiveEmail: (id: string) => void
  toggleRead: (id: string) => void
  selectedId: string | null
  selectEmail: (id: string | null) => void
  gmailStatus: GmailStatus
  gmailError: string | null
  connectGmail: (rememberMe: boolean) => void
  refreshInbox: () => Promise<void>
  personalization: PersonalizationStatus | null
  personalizationError: string | null
  retraining: boolean
  correctEmail: (id: string, model: PersonalizableModel, label: string) => Promise<void>
  enablePersonalization: () => Promise<boolean>
  disablePersonalization: () => Promise<void>
  retrainPersonalization: () => Promise<void>
}

const InboxContext = createContext<InboxState | null>(null)

// Module-level guard against React Strict Mode's dev-only double-invoke of
// the mount effect below: without it, two concurrent invocations each
// independently mint a Gmail access token and fetch the inbox, doubling
// concurrent Gmail API requests past its per-user rate limit (confirmed via
// a headless-browser repro — both runs failed with 403, and the resume
// path's catch swallowed it silently, freezing the UI at "FETCHING
// INBOX..." forever). Same pattern as engine.ts's loadModels() memoized
// promise, and same single-instance assumption (one InboxProvider per app).
let fetchAndClassifyPromise: Promise<void> | null = null

const RETRAIN_POLL_MS = 2000
const RETRAIN_POLL_LIMIT = 60

function emptyRow(email: InboxEmail): Row {
  return { ...email, result: null, modelResult: null, corrected: {} }
}

function withResult(row: Row, result: ClassificationResult): Row {
  return { ...row, modelResult: result, result: applyCorrections(result, row.corrected) }
}

function versionKey(artifacts: PersonalizedModelArtifact[]): string {
  return artifacts
    .map((a) => `${a.model}:${a.version}`)
    .sort()
    .join(',')
}

// Changes whenever any model records a new retrain attempt (active or
// rejected), which is how the watcher knows a job finished.
function attemptKey(status: PersonalizationStatus | null): string {
  return (status?.models ?? []).map((m) => `${m.model}:${m.last_attempt?.version ?? 0}`).join(',')
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

export function InboxProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [rows, setRows] = useState<Row[]>(SAMPLE_EMAILS.map(emptyRow))
  const [filter, setFilter] = useState<string>('All')
  const [onlyHigh, setOnlyHigh] = useState(false)
  const [showArchived, setShowArchived] = useState(false)
  const [query, setQuery] = useState('')
  // Archive/read state is client-only — real, working actions for the
  // current session, not wired to Gmail itself (archiving here never
  // touches the real inbox).
  const [archivedIds, setArchivedIds] = useState<Set<string>>(new Set())
  // Detail view is client-side state, not a /inbox/[id] route — Gmail
  // message IDs only exist at runtime (after fetch), so a dynamic
  // filesystem route could never satisfy generateStaticParams() under
  // output: 'export', which pre-renders every route at build time.
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [gmailStatus, setGmailStatus] = useState<GmailStatus>('disconnected')
  const [gmailError, setGmailError] = useState<string | null>(null)
  const [personalization, setPersonalization] = useState<PersonalizationStatus | null>(null)
  const [personalizationError, setPersonalizationError] = useState<string | null>(null)
  const [retraining, setRetraining] = useState(false)

  // Async flows below outlive the render they started in; refs let them
  // read current state instead of a stale closure.
  const rowsRef = useRef<Row[]>(rows)
  const personalizationRef = useRef<PersonalizationStatus | null>(null)
  const appliedVersions = useRef('')

  useEffect(() => {
    rowsRef.current = rows
  }, [rows])

  function updatePersonalization(next: PersonalizationStatus) {
    personalizationRef.current = next
    setPersonalization(next)
  }

  function archiveEmail(id: string) {
    setArchivedIds((prev) => new Set(prev).add(id))
  }
  function unarchiveEmail(id: string) {
    setArchivedIds((prev) => {
      const next = new Set(prev)
      next.delete(id)
      return next
    })
  }
  function toggleRead(id: string) {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, unread: !r.unread } : r)))
  }

  // Loads this user's active personalized models into the engine, or
  // reverts to base models when there are none or personalization is off.
  // Returns whether the loaded set changed. Never throws: personalization
  // is strictly additive, so a failure leaves the current models in place.
  async function syncPersonalization(): Promise<boolean> {
    try {
      const next = await getStatus()
      updatePersonalization(next)
      const artifacts = next.enabled ? await fetchActiveModels() : []
      const key = versionKey(artifacts)
      if (key === appliedVersions.current) return false
      await applyPersonalizedModels(artifacts)
      appliedVersions.current = key
      return true
    } catch (err) {
      setPersonalizationError(`Personalization unavailable, using base models: ${message(err)}`)
      return false
    }
  }

  async function reclassify(): Promise<void> {
    for (const row of rowsRef.current) {
      if (!row.modelResult) continue
      const result = await classify(row)
      setRows((prev) => prev.map((r) => (r.id === row.id ? withResult(r, result) : r)))
    }
  }

  function fetchAndClassify(token: string): Promise<void> {
    if (fetchAndClassifyPromise) return fetchAndClassifyPromise
    fetchAndClassifyPromise = (async () => {
      const emails = await fetchRecentInboxEmails(token)

      await loadModels()
      if (!modelsLoaded()) throw new Error('models did not finish loading')
      await syncPersonalization()

      setRows(emails.map(emptyRow))
      // Same WASM single-flight constraint as the sample-email loop below —
      // classify() calls go one at a time, not Promise.all.
      for (const email of emails) {
        const result = await classify(email)
        setRows((prev) => prev.map((r) => (r.id === email.id ? withResult(r, result) : r)))
      }
    })()
    // Clear once settled (success or failure) so the next real call starts a
    // fresh fetch. Handled on both paths: a bare .finally() would re-reject
    // into an unhandled rejection whenever the fetch fails.
    const clear = () => {
      fetchAndClassifyPromise = null
    }
    fetchAndClassifyPromise.then(clear, clear)
    return fetchAndClassifyPromise
  }

  // Navigates away to the backend's OAuth start route — nothing after this
  // runs in this tab. The page that loads on return (/inbox) picks the
  // connection up silently via the mount effect below, since the session
  // cookie is already set by the time Google redirects back.
  function connectGmail(rememberMe: boolean) {
    setGmailStatus('connecting')
    setGmailError(null)
    startGmailConnect(rememberMe)
  }

  // Always asks the backend for a fresh token rather than reusing the
  // cached one — that's what makes this silent instead of erroring once the
  // cached access token expires.
  async function refreshInbox() {
    setGmailStatus('fetching')
    setGmailError(null)
    try {
      const token = await fetchGmailAccessToken()
      saveGmailToken(token)
      await fetchAndClassify(token.accessToken)
      setGmailStatus('connected')
    } catch (err) {
      clearStoredGmailToken()
      setGmailError(message(err))
      setGmailStatus('error')
    }
  }

  function setCorrected(id: string, model: PersonalizableModel, label: string | undefined) {
    setRows((prev) =>
      prev.map((r) => {
        if (r.id !== id) return r
        const corrected = { ...r.corrected }
        if (label === undefined) delete corrected[model]
        else corrected[model] = label
        return { ...r, corrected, result: r.modelResult ? applyCorrections(r.modelResult, corrected) : r.result }
      }),
    )
  }

  // `before` is the attempt snapshot taken before the request that kicked
  // off the retrain, so a job that finishes before the first poll still
  // registers as a change.
  async function watchRetrain(before: string): Promise<void> {
    setRetraining(true)
    try {
      for (let i = 0; i < RETRAIN_POLL_LIMIT; i++) {
        await new Promise((resolve) => setTimeout(resolve, RETRAIN_POLL_MS))
        const next = await getStatus()
        updatePersonalization(next)
        if (!next.models.some((m) => m.running) && attemptKey(next) !== before) break
      }
      if (await syncPersonalization()) await reclassify()
    } catch (err) {
      setPersonalizationError(message(err))
    } finally {
      setRetraining(false)
    }
  }

  async function correctEmail(id: string, model: PersonalizableModel, label: string): Promise<void> {
    const row = rowsRef.current.find((r) => r.id === id)
    if (!row?.modelResult || row.source !== 'gmail') return
    const before = attemptKey(personalizationRef.current)
    const previous = row.corrected[model]
    const predicted = predictedFor(model, row.modelResult)
    setCorrected(id, model, label)
    setPersonalizationError(null)

    let response: CorrectionResponse
    try {
      response = await submitCorrection({
        model,
        provider_message_id: id,
        feature_vector: toSparse(buildCorrectionVector(row, model, row.modelResult)),
        predicted_label: predicted.label,
        predicted_confidence: Math.min(1, Math.max(0, predicted.confidence)),
        corrected_label: label,
      })
    } catch (err) {
      setCorrected(id, model, previous)
      setPersonalizationError(`Correction not saved: ${message(err)}`)
      return
    }
    try {
      updatePersonalization(await getStatus())
    } catch (err) {
      setPersonalizationError(message(err))
    }
    if (response.retrain_scheduled) void watchRetrain(before)
  }

  async function enablePersonalization(): Promise<boolean> {
    setPersonalizationError(null)
    try {
      await setEnabled(true)
      updatePersonalization(await getStatus())
      return true
    } catch (err) {
      setPersonalizationError(`Could not turn on personalization: ${message(err)}`)
      return false
    }
  }

  async function disablePersonalization(): Promise<void> {
    setPersonalizationError(null)
    try {
      await setEnabled(false)
    } catch (err) {
      setPersonalizationError(`Could not turn off personalization: ${message(err)}`)
      return
    }
    setRows((prev) => prev.map((r) => ({ ...r, corrected: {}, result: r.modelResult })))
    if (await syncPersonalization()) await reclassify()
  }

  async function retrainPersonalization(): Promise<void> {
    const before = attemptKey(personalizationRef.current)
    setPersonalizationError(null)
    try {
      const { scheduled } = await retrainNow()
      if (scheduled.length > 0) await watchRetrain(before)
    } catch (err) {
      setPersonalizationError(`Retrain did not start: ${message(err)}`)
    }
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      // Cosmetic only — the session cookie from /auth/gmail/callback is
      // already set by the time this page loads.
      if (window.location.search.includes('connected=1')) {
        window.history.replaceState({}, '', window.location.pathname)
      }

      const stored = loadStoredGmailToken()
      if (stored) {
        setGmailStatus('fetching')
        try {
          await fetchAndClassify(stored.accessToken)
          if (!cancelled) {
            setGmailStatus('connected')
            setStatus('ready')
          }
          return
        } catch (err) {
          clearStoredGmailToken()
          if (!cancelled) {
            setGmailError(message(err))
            setGmailStatus('error')
          }
          // fall through to the sample-email view below
        }
      } else {
        // No cached access token — try the backend session cookie before
        // giving up. Succeeds silently whenever a prior connection is still
        // alive.
        try {
          const token = await fetchGmailAccessToken()
          saveGmailToken(token)
          setGmailStatus('fetching')
          await fetchAndClassify(token.accessToken)
          if (!cancelled) {
            setGmailStatus('connected')
            setStatus('ready')
          }
          return
        } catch {
          // Not connected yet — stay 'disconnected' (the normal first-visit
          // state) and fall through to the sample-email view below.
        }
      }

      try {
        await loadModels()
        if (!modelsLoaded()) throw new Error('models did not finish loading')
        for (const email of SAMPLE_EMAILS) {
          if (cancelled) return
          const result = await classify(email)
          setRows((prev) => prev.map((r) => (r.id === email.id ? withResult(r, result) : r)))
        }
        if (!cancelled) setStatus('ready')
      } catch (err) {
        if (!cancelled) {
          setError(message(err))
          setStatus('error')
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <InboxContext.Provider
      value={{
        rows,
        status,
        error,
        filter,
        setFilter,
        onlyHigh,
        setOnlyHigh,
        showArchived,
        setShowArchived,
        query,
        setQuery,
        archivedIds,
        archiveEmail,
        unarchiveEmail,
        toggleRead,
        selectedId,
        selectEmail: setSelectedId,
        gmailStatus,
        gmailError,
        connectGmail,
        refreshInbox,
        personalization,
        personalizationError,
        retraining,
        correctEmail,
        enablePersonalization,
        disablePersonalization,
        retrainPersonalization,
      }}
    >
      {children}
    </InboxContext.Provider>
  )
}

export function useInbox() {
  const ctx = useContext(InboxContext)
  if (!ctx) throw new Error('useInbox must be used within InboxProvider')
  return ctx
}
```

- [ ] **Step 2: Fix call sites that assumed a narrow filter type**

Run: `npx tsc --noEmit`
Expected errors appear only in `sidebar.tsx`, from `goToCategory(c: CategoryLabel)` and `countFor(cat: CategoryLabel | 'All')`. Those are replaced in Task 16. To get a clean build now, change both parameter types in `sidebar.tsx` to `string`, then rerun `npx tsc --noEmit` and confirm there are no errors.

- [ ] **Step 3: Lint and test**

Run: `npm run lint && npm run test`
Expected: lint clean; all Vitest tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/frontend/src/app/inbox/inbox-context.tsx app/frontend/src/app/inbox/sidebar.tsx
git commit -m "Track corrections and personalized models in inbox state; fix unhandled rejection in fetch memo"
```

---

### Task 15: Correction picker and opt-in prompt in the email detail view

**Files:**
- Create: `app/frontend/src/app/inbox/correction-picker.tsx`
- Modify: `app/frontend/src/app/inbox/email-detail.tsx`

**Interfaces:**
- Consumes: `useInbox()` additions from Task 14.
- Produces test ids for the e2e script: `correct-category`, `correct-priority`, `correct-spam`, `correction-picker`, `correction-option-<label>`, `opt-in-prompt`, `opt-in-enable`.

- [ ] **Step 1: Create the picker and prompt**

Create `app/frontend/src/app/inbox/correction-picker.tsx`:

```tsx
'use client'

import { useState } from 'react'
import { DETAIL_BORDER, DETAIL_FAINT, DETAIL_MUTED, INK } from './tokens'

export function CorrectionPicker({
  options,
  allowNew,
  current,
  onPick,
  onCancel,
}: {
  options: string[]
  allowNew: boolean
  current: string
  onPick: (label: string) => void
  onCancel: () => void
}) {
  const [draft, setDraft] = useState('')
  const trimmed = draft.trim()

  return (
    <div
      className="mt-2 flex flex-col gap-2 rounded-sm p-2"
      style={{ border: `1px solid ${DETAIL_BORDER}` }}
      data-testid="correction-picker"
    >
      <div className="flex flex-wrap gap-1.5">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onPick(option)}
            data-testid={`correction-option-${option}`}
            className="rounded-sm px-2 py-1 font-mono text-[11px] tracking-wide uppercase"
            style={{
              color: option === current ? INK : DETAIL_MUTED,
              border: `1px solid ${option === current ? DETAIL_MUTED : DETAIL_BORDER}`,
            }}
          >
            {option}
          </button>
        ))}
      </div>
      {allowNew && (
        <form
          className="flex gap-1.5"
          onSubmit={(e) => {
            e.preventDefault()
            if (trimmed) onPick(trimmed)
          }}
        >
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            maxLength={40}
            placeholder="new label…"
            aria-label="New label"
            className="min-w-0 flex-1 bg-transparent px-2 py-1 font-mono text-[11px] outline-none"
            style={{ color: INK, border: `1px solid ${DETAIL_BORDER}` }}
          />
          <button
            type="submit"
            disabled={!trimmed}
            className="px-2 font-mono text-[11px] uppercase disabled:opacity-40"
            style={{ color: DETAIL_MUTED }}
          >
            add
          </button>
        </form>
      )}
      <button
        type="button"
        onClick={onCancel}
        className="self-start font-mono text-[10px] tracking-wide uppercase"
        style={{ color: DETAIL_FAINT }}
      >
        cancel
      </button>
    </div>
  )
}

// Wording follows the spec's §10: say plainly what leaves the browser.
export function OptInPrompt({ busy, onEnable, onCancel }: { busy: boolean; onEnable: () => void; onCancel: () => void }) {
  return (
    <div
      className="mt-2 flex flex-col gap-2 rounded-sm p-3"
      style={{ border: `1px solid ${DETAIL_BORDER}` }}
      data-testid="opt-in-prompt"
    >
      <p className="font-serif text-sm leading-relaxed" style={{ color: INK }}>
        Corrections train a copy of the model that only you use. To do that, this app sends its server a word-level
        summary of each email you correct: which words from its fixed vocabulary appear and how strongly, plus a few
        numbers like the recipient count. It does not send the email text or word order. You can turn this off at any
        time, which deletes everything it stored.
      </p>
      <div className="flex gap-3">
        <button
          type="button"
          onClick={onEnable}
          disabled={busy}
          data-testid="opt-in-enable"
          className="font-mono text-[11px] tracking-wide uppercase disabled:opacity-40"
          style={{ color: INK }}
        >
          {busy ? 'turning on…' : 'turn on personalization'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="font-mono text-[11px] tracking-wide uppercase"
          style={{ color: DETAIL_FAINT }}
        >
          not now
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Update the imports in `email-detail.tsx`**

Replace the import block at the top of `app/frontend/src/app/inbox/email-detail.tsx` with:

```tsx
'use client'

import { useState, type ReactNode } from 'react'
import { Archive, ArchiveRestore, ExternalLink, Mail, MailOpen } from 'lucide-react'
import type { CategoryLabel, PersonalizableModel, PriorityBucket, SpamLabel } from '@/types'
import { useInbox } from './inbox-context'
import { Avatar } from './avatar'
import { CorrectionPicker, OptInPrompt } from './correction-picker'
import { CATEGORY_COLOR, DETAIL_BG, DETAIL_BORDER, DETAIL_FAINT, DETAIL_MUTED, HIGH, INK, MEDIUM } from './tokens'
```

- [ ] **Step 3: Show a custom-label winner in `CategoryStrip`**

In `CategoryStrip`, directly before its outermost closing `</div>`, add:

```tsx
      {!order.includes(winner as CategoryLabel) && (
        <p className="mt-2 font-mono text-[11px] tracking-wide uppercase" style={{ color: INK }}>
          → {winner}
          {confidences[winner] !== undefined && ` ${Math.round(confidences[winner] * 100)}%`}
        </p>
      )}
```

- [ ] **Step 4: Add the section label helper**

Directly above `export function EmailDetail`, add:

```tsx
const FIXED_OPTIONS: Record<Exclude<PersonalizableModel, 'category'>, string[]> = {
  priority: ['low', 'medium', 'high'],
  spam: ['spam', 'ham'],
}

function SectionLabel({
  title,
  corrected,
  onCorrect,
  testId,
}: {
  title: string
  corrected: boolean
  onCorrect?: () => void
  testId: string
}) {
  return (
    <p className="mb-2 flex items-center gap-2 font-mono text-[10px] tracking-widest" style={{ color: DETAIL_FAINT }}>
      {title}
      {corrected && <span style={{ color: DETAIL_MUTED }}>· CORRECTED</span>}
      {onCorrect && (
        <button
          type="button"
          onClick={onCorrect}
          data-testid={testId}
          className="tracking-wide underline-offset-2 hover:underline"
          style={{ color: DETAIL_MUTED }}
        >
          CORRECT
        </button>
      )}
    </p>
  )
}
```

- [ ] **Step 5: Replace the `EmailDetail` function**

Replace the whole `export function EmailDetail(...) { ... }` with:

```tsx
export function EmailDetail({ id }: { id: string }) {
  const {
    rows,
    selectEmail,
    archivedIds,
    archiveEmail,
    unarchiveEmail,
    toggleRead,
    personalization,
    personalizationError,
    correctEmail,
    enablePersonalization,
  } = useInbox()
  const [picking, setPicking] = useState<PersonalizableModel | null>(null)
  const [optInFor, setOptInFor] = useState<PersonalizableModel | null>(null)
  const [enabling, setEnabling] = useState(false)
  const row = rows.find((r) => r.id === id)

  if (!row) {
    return (
      <main className="p-6" style={{ color: INK }}>
        <button onClick={() => selectEmail(null)} className="font-mono text-xs tracking-wide" style={{ color: DETAIL_MUTED }}>
          ← INBOX
        </button>
        <p className="mt-4 font-mono text-sm">Unknown message.</p>
      </main>
    )
  }

  const archived = archivedIds.has(row.id)
  // Corrections need a real message id and a backend connection, so the
  // hand-authored sample emails can't be corrected.
  const canCorrect = row.source === 'gmail' && row.modelResult !== null

  const startCorrecting = (model: PersonalizableModel) => {
    if (personalization?.enabled) {
      setOptInFor(null)
      setPicking(model)
    } else {
      setPicking(null)
      setOptInFor(model)
    }
  }

  const confirmOptIn = async () => {
    if (!optInFor) return
    setEnabling(true)
    const ok = await enablePersonalization()
    setEnabling(false)
    if (ok) {
      setPicking(optInFor)
      setOptInFor(null)
    }
  }

  const pick = (model: PersonalizableModel, label: string) => {
    setPicking(null)
    void correctEmail(row.id, model, label)
  }

  const optionsFor = (model: PersonalizableModel): string[] =>
    model === 'category' ? ['Work', 'Personal', 'Other', ...(personalization?.custom_labels ?? [])] : FIXED_OPTIONS[model]

  const correctionUi = (model: PersonalizableModel, current: string) => {
    if (optInFor === model) {
      return <OptInPrompt busy={enabling} onEnable={() => void confirmOptIn()} onCancel={() => setOptInFor(null)} />
    }
    if (picking === model) {
      return (
        <CorrectionPicker
          options={optionsFor(model)}
          allowNew={model === 'category'}
          current={current}
          onPick={(label) => pick(model, label)}
          onCancel={() => setPicking(null)}
        />
      )
    }
    return null
  }

  return (
    <main className="min-h-screen" style={{ background: DETAIL_BG, color: INK }}>
      <div className="mx-auto max-w-2xl px-6 py-6">
        <button onClick={() => selectEmail(null)} className="font-mono text-xs tracking-wide" style={{ color: DETAIL_MUTED }}>
          ← INBOX
        </button>

        <div className="mt-4 flex items-start gap-3">
          <Avatar name={row.fromName} size="md" />
          <div className="min-w-0 flex-1">
            <h1 className="font-serif text-2xl font-medium" style={{ color: INK }}>
              {row.subject}
            </h1>
            <p className="mt-1 font-mono text-[11px]" style={{ color: DETAIL_MUTED }}>
              FROM {row.fromName} &lt;{row.fromAddr}&gt; → {row.to} · {row.receivedAt}
            </p>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
          <ActionButton
            icon={row.unread ? <MailOpen size={13} /> : <Mail size={13} />}
            label={row.unread ? 'MARK READ' : 'MARK UNREAD'}
            onClick={() => toggleRead(row.id)}
          />
          <ActionButton
            icon={archived ? <ArchiveRestore size={13} /> : <Archive size={13} />}
            label={archived ? 'MOVE TO INBOX' : 'ARCHIVE'}
            onClick={() => (archived ? unarchiveEmail(row.id) : archiveEmail(row.id))}
          />
          {row.source === 'gmail' && (
            <ActionButton
              icon={<ExternalLink size={13} />}
              label="OPEN IN GMAIL"
              href={`https://mail.google.com/mail/u/0/#all/${row.id}`}
            />
          )}
        </div>

        <div className="my-5 h-px" style={{ background: DETAIL_BORDER }} />

        <p className="font-serif text-base leading-relaxed whitespace-pre-wrap" style={{ color: INK }}>
          {row.body}
        </p>

        <div className="my-6 h-px" style={{ background: DETAIL_BORDER }} />

        <h2 className="mb-3 font-mono text-[11px] tracking-[0.2em]" style={{ color: DETAIL_MUTED }}>
          MODEL OUTPUT
        </h2>

        {!row.result ? (
          <p className="font-mono text-xs" style={{ color: DETAIL_MUTED }}>
            classifying…
          </p>
        ) : (
          <div className="flex flex-col gap-5">
            <div>
              <SectionLabel
                title="CATEGORY"
                corrected={row.corrected.category !== undefined}
                onCorrect={canCorrect ? () => startCorrecting('category') : undefined}
                testId="correct-category"
              />
              <CategoryStrip confidences={row.result.category.confidences} winner={row.result.category.label} />
              {correctionUi('category', row.result.category.label)}
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div>
                <SectionLabel
                  title="PRIORITY"
                  corrected={row.corrected.priority !== undefined}
                  onCorrect={canCorrect ? () => startCorrecting('priority') : undefined}
                  testId="correct-priority"
                />
                <PriorityGauge score={row.result.priority.score} bucket={row.result.priority.bucket} />
                {row.result.priority.note && (
                  <p className="mt-1 text-center font-mono text-[10px]" style={{ color: DETAIL_MUTED }}>
                    {row.result.priority.note}
                  </p>
                )}
                {correctionUi('priority', row.result.priority.bucket)}
              </div>

              <div>
                <SectionLabel
                  title="SPAM CHECK"
                  corrected={row.corrected.spam !== undefined}
                  onCorrect={canCorrect ? () => startCorrecting('spam') : undefined}
                  testId="correct-spam"
                />
                <SpamBadge label={row.result.spam.label} confidence={row.result.spam.confidence} />
                {correctionUi('spam', row.result.spam.label)}
              </div>
            </div>

            {personalizationError && (
              <p className="font-mono text-[11px]" style={{ color: HIGH }} role="alert">
                {personalizationError}
              </p>
            )}
          </div>
        )}
      </div>
    </main>
  )
}
```

- [ ] **Step 6: Verify**

Run: `npx tsc --noEmit && npm run lint && npm run test`
Expected: all clean and passing.

Manual check (backend and frontend running, Gmail connected, personalization not yet enabled):
1. Open any email and click `CORRECT` next to CATEGORY. The opt-in prompt appears with the privacy wording.
2. Click `turn on personalization`. The picker appears.
3. Pick `Work`. The strip's winner changes and `· CORRECTED` appears.
4. Open a sample email (disconnected view). There's no `CORRECT` button.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/src/app/inbox/correction-picker.tsx app/frontend/src/app/inbox/email-detail.tsx
git commit -m "Add correction picker and personalization opt-in prompt to the email detail view"
```

---

### Task 16: Sidebar status, custom-label nav, list selectors

**Files:**
- Modify: `app/frontend/src/app/inbox/sidebar.tsx`
- Modify: `app/frontend/src/app/inbox/inbox-list.tsx`

**Interfaces:**
- Consumes: `useInbox()` additions from Task 14.
- Produces test ids: `personalization-status`, `retrain-now`, and on each list row `data-testid="email-row"`, `data-email-id`, `data-category`.

- [ ] **Step 1: Custom labels in the category nav**

In `sidebar.tsx`:
- Remove the `import type { CategoryLabel } from '@/types'` line.
- Change `const CATEGORIES: CategoryLabel[] = ['Work', 'Personal', 'Other']` to `const CATEGORIES = ['Work', 'Personal', 'Other']`.
- Add `personalization`, `personalizationError`, `retraining`, `retrainPersonalization`, and `disablePersonalization` to the `useInbox()` destructuring.
- Directly after the `useInbox()` call, add:

```tsx
  const categories = [...CATEGORIES, ...(personalization?.custom_labels ?? [])]
```

- Make sure `countFor` and `goToCategory` take `string` (done in Task 14).
- In the first `<nav>`, change `{CATEGORIES.map((c) => (` to `{categories.map((c) => (`.

- [ ] **Step 2: Personalization status block**

Directly above the `const live = ...` line, add:

```tsx
  const models = personalization?.models ?? []
  const totalCorrections = models.reduce((n, m) => n + m.correction_count, 0)
  const nextRetrainIn = models.length ? Math.min(...models.map((m) => m.corrections_until_retrain)) : 5
  const lastAttempt = models
    .filter((m) => m.last_attempt)
    .sort((a, b) => (a.last_attempt!.created_at < b.last_attempt!.created_at ? 1 : -1))[0]
  const lastAttemptText = lastAttempt?.last_attempt
    ? lastAttempt.last_attempt.status === 'rejected'
      ? `LAST RETRAIN REJECTED: ${lastAttempt.last_attempt.metrics.reason ?? lastAttempt.last_attempt.metrics.error ?? 'unknown reason'}`
      : `LAST RETRAIN: ${lastAttempt.model.toUpperCase()} V${lastAttempt.last_attempt.version}`
    : null

  function turnOffPersonalization() {
    if (window.confirm('Turning off personalization deletes your corrections and personalized models. Continue?')) {
      void disablePersonalization()
    }
  }
```

Directly after the closing `</div>` of the Gmail connect/remember-me block (the `<div className="flex flex-col gap-1.5">` wrapper), add:

```tsx
      {gmailStatus === 'connected' && personalization?.enabled && (
        <div
          className="flex flex-col gap-1 px-1 font-mono text-[10px] tracking-wide"
          style={{ color: FAINT }}
          data-testid="personalization-status"
        >
          <span>
            {totalCorrections} CORRECTIONS · {retraining ? 'RETRAINING…' : `RETRAIN IN ${nextRetrainIn}`}
          </span>
          {lastAttemptText && (
            <span style={{ color: lastAttempt?.last_attempt?.status === 'rejected' ? HIGH : MUTED }}>{lastAttemptText}</span>
          )}
          {personalizationError && <span style={{ color: HIGH }}>{personalizationError}</span>}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => void retrainPersonalization()}
              disabled={retraining || totalCorrections === 0}
              data-testid="retrain-now"
              className="disabled:opacity-40"
              style={{ color: MUTED }}
            >
              ↻ RETRAIN NOW
            </button>
            <button type="button" onClick={turnOffPersonalization} style={{ color: FAINT }}>
              TURN OFF
            </button>
          </div>
        </div>
      )}
```

- [ ] **Step 3: List row selectors**

In `inbox-list.tsx`, on the `<li` that opens `EmailRow`, add these attributes:

```tsx
      data-testid="email-row"
      data-email-id={row.id}
      data-category={row.result?.category.label ?? ''}
```

- [ ] **Step 4: Verify**

Run: `npx tsc --noEmit && npm run lint && npm run test`
Expected: all clean and passing.

Manual check (personalization enabled from Task 15):
1. The sidebar shows `1 CORRECTIONS · RETRAIN IN 4` (or your current count).
2. Correct a category to a new label such as `Finance`. `FINANCE` appears in the category nav and filters correctly.
3. Click `↻ RETRAIN NOW`. The line shows `RETRAINING…`, then `LAST RETRAIN: CATEGORY V1` or `LAST RETRAIN REJECTED: <reason>`.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/src/app/inbox/sidebar.tsx app/frontend/src/app/inbox/inbox-list.tsx
git commit -m "Show personalization status, retrain controls, and custom labels in the sidebar"
```

---

### Task 17: End-to-end check and project status

**Files:**
- Create: `app/frontend/scripts/personalization_e2e.mjs`
- Modify: `PROJECT_STATUS.md`

**Interfaces:**
- Consumes: all test ids from Tasks 15 and 16, and the backend routes from Task 8.

- [ ] **Step 1: Write the e2e script**

Create `app/frontend/scripts/personalization_e2e.mjs`:

```js
// End-to-end personalization check against the real app (spec §12).
// Prereqs: Docker up; backend (npm run dev:be) and frontend (npm run dev:fe)
// running; Gmail connected at least once; training_data/ generated.
//
// Usage (from app/frontend):
//   node scripts/personalization_e2e.mjs <dmp_session cookie value> [--cleanup]
//
// Corrects 5 emails' category to Work through the UI, waits for the
// automatic retrain, reloads, and reports what changed. It asserts the
// mechanics (corrections saved, a retrain attempt recorded, the app reloads
// cleanly); how many other emails flip depends on the real inbox, so that
// part is reported, not asserted.
import { chromium } from 'playwright'

const [sessionId, ...flags] = process.argv.slice(2)
if (!sessionId) {
  console.error('usage: node scripts/personalization_e2e.mjs <dmp_session cookie value> [--cleanup]')
  process.exit(2)
}
const BACKEND = 'http://localhost:3011'
const INBOX = 'http://localhost:3010/inbox'
const LABEL = 'Work'
const N = 5

const browser = await chromium.launch()
const context = await browser.newContext()
await context.addCookies([{ name: 'dmp_session', value: sessionId, domain: 'localhost', path: '/', httpOnly: true }])
const page = await context.newPage()
const api = context.request

async function status() {
  const res = await api.get(`${BACKEND}/personalization/status`)
  if (!res.ok()) throw new Error(`status failed: ${res.status()}`)
  return res.json()
}

async function waitForClassifiedInbox() {
  await page.goto(INBOX)
  await page.waitForFunction(() => document.body.innerText.includes('GMAIL CONNECTED'), null, { timeout: 90_000 })
  await page.waitForFunction(
    () => {
      const rows = [...document.querySelectorAll('[data-testid="email-row"]')]
      return rows.length > 0 && rows.every((r) => r.dataset.category)
    },
    null,
    { timeout: 120_000 },
  )
}

async function categories() {
  return page.$$eval('[data-testid="email-row"]', (rows) =>
    Object.fromEntries(rows.map((r) => [r.dataset.emailId, r.dataset.category])),
  )
}

try {
  const optIn = await api.put(`${BACKEND}/personalization/settings`, { data: { enabled: true } })
  if (!optIn.ok()) throw new Error(`opt-in failed: ${optIn.status()}`)
  const attemptsBefore = (await status()).models.find((m) => m.model === 'category').last_attempt?.version ?? 0

  await waitForClassifiedInbox()
  const before = await categories()
  const targets = Object.keys(before).filter((id) => before[id] !== LABEL).slice(0, N)
  if (targets.length < N) throw new Error(`need ${N} emails not already ${LABEL}, found ${targets.length}`)

  for (const id of targets) {
    await page.locator(`[data-email-id="${id}"] button`).first().click()
    await page.getByTestId('correct-category').click()
    await Promise.all([
      page.waitForResponse((r) => r.url().endsWith('/personalization/corrections') && r.ok()),
      page.getByTestId(`correction-option-${LABEL}`).click(),
    ])
    await page.getByText('← INBOX').click()
  }
  console.log(`Corrected ${targets.length} emails to ${LABEL}.`)

  let attempt = null
  for (let i = 0; i < 60 && !attempt; i++) {
    await page.waitForTimeout(2000)
    const category = (await status()).models.find((m) => m.model === 'category')
    if (!category.running && (category.last_attempt?.version ?? 0) > attemptsBefore) attempt = category.last_attempt
  }
  if (!attempt) throw new Error('no retrain attempt recorded within 120s')
  console.log(`Retrain v${attempt.version}: ${attempt.status}`, JSON.stringify(attempt.metrics))

  await waitForClassifiedInbox()
  const after = await categories()
  const correctedNow = targets.filter((id) => after[id] === LABEL).length
  const others = Object.keys(after).filter((id) => !targets.includes(id))
  const flipped = others.filter((id) => before[id] !== after[id])
  console.log(`After reload: ${correctedNow}/${targets.length} corrected emails now predicted ${LABEL} by the model.`)
  console.log(`Other emails whose category changed: ${flipped.length}/${others.length}`)
  for (const id of flipped) console.log(`  ${id}: ${before[id]} -> ${after[id]}`)

  if (flags.includes('--cleanup')) {
    await api.put(`${BACKEND}/personalization/settings`, { data: { enabled: false } })
    console.log('Cleaned up: personalization turned off, corrections and models deleted.')
  }
} catch (err) {
  console.error(`E2E FAILED: ${err instanceof Error ? err.message : err}`)
  await page.screenshot({ path: 'personalization_e2e_failure.png', fullPage: true })
  process.exitCode = 1
} finally {
  await browser.close()
}
```

- [ ] **Step 2: Run it**

Start the backend and frontend (`npm run dev:be` and `npm run dev:fe` from the repo root). Get the session id with:
`docker exec backend-postgres-1 psql -U dmp_user -d dmp -t -c "SELECT id FROM sessions ORDER BY last_used_at DESC LIMIT 1;"`

Run (from `app/frontend`): `node scripts/personalization_e2e.mjs <session id> --cleanup`

Expected, in order:
1. `Corrected 5 emails to Work.`
2. `Retrain v1: active` (or `rejected` with its gate metrics)
3. The `After reload` counts
4. `Cleaned up: ...`
5. Exit code 0

Record the printed numbers for the report. A `rejected` result is a legitimate outcome of the gate, not a script failure; include its reason in the report.

- [ ] **Step 3: Update `PROJECT_STATUS.md`**

In `PROJECT_STATUS.md`, add under the "Application layer (`app/`)" section (after the Gmail integration subsection):

```markdown
### Personalization (corrections + per-user retraining) — built

Spec: `docs/superpowers/specs/2026-09-29-personalization-design.md`.
Plan: `docs/superpowers/plans/2026-09-29-personalization.md`.

- Opt-in per user. When on, correcting a label in the email detail view
  sends the email's **sparse feature vector** (never the text) to the
  backend. Custom labels are allowed for category only.
- After 5 new corrections for a model (or "retrain now"), the backend
  refits a `clone()` of the production classifier head on the base rows plus
  the corrections (weight 10). It keeps the TF-IDF vocabulary fixed, so
  nothing on the client's feature side changes. Priority retrains its
  classifier and regressor together.
- A validation gate runs before a new version goes active: at least 80% of
  the user's own corrections predicted as corrected, at most a 3-point drop
  on the adversarial set, and at most 10% of that set pulled into custom
  labels. Otherwise the attempt is stored as `rejected` with the reason.
- Versions live in `personalized_models`. The client loads the user's
  active ONNX graphs in place of the base ones and still classifies in the
  browser.
- **Privacy caveat (stated honestly, including in the opt-in prompt):** a
  TF-IDF vector plus the public `vocab.json` reveals which vocabulary words
  appeared (a bag of words), though not the text or word order.
- Regenerate the base matrices whenever the base models retrain:
  `python scripts/export_training_matrices.py` (it refuses to write unless a
  zero-correction refit reproduces each production model).
- Measure for the report:
  `app/backend: .venv\Scripts\python.exe -m src.personalization.measure --user-id <uuid> --model category`
  trains on half of that user's corrections and reports held-out accuracy
  before/after for several weights, plus gate-set accuracy.
- Checks: backend `npm run test:be`; client/sklearn vector parity
  `npx tsx scripts/sparse_parity_check.mjs`; end to end
  `node scripts/personalization_e2e.mjs <session id>` (both from
  `app/frontend`).
```

Also, in its "Not started" list, remove "Personalization" from the Phase 2 bullet.

- [ ] **Step 4: Final full verification**

Run:
```
npm run test:be
npm --prefix app/frontend run test
npm --prefix app/frontend run lint
```
from the repo root, then `npx tsc --noEmit` from `app/frontend`.
Expected: all pass and no type errors.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/scripts/personalization_e2e.mjs PROJECT_STATUS.md
git commit -m "Add personalization end-to-end check and document the feature in PROJECT_STATUS"
```
