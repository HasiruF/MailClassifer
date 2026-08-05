# Backend

FastAPI + SQLAlchemy + Alembic + Postgres. Exists only to support the
opt-in personalization feature (custom categories, per-user model
refinement) — the base spam/category/priority models run entirely
client-side via `onnxruntime-web` and never touch this service. See
`app/services/personalization.py` for the centroid/kNN-vs-retrain strategy
and why a naive "retrain on the user's corrections" approach won't work at
the data volumes a single user will realistically produce.

## Local dev

Via docker-compose (from `app/`):

```bash
docker compose up -d postgres adminer   # postgres:5432, adminer UI:8080
cp backend/.env.example backend/.env
cd backend
python -m venv .venv && .venv\Scripts\activate   # (Windows)
pip install -r requirements.txt
alembic revision --autogenerate -m "init"
alembic upgrade head
uvicorn app.main:app --reload
```

Or run everything in containers: `docker compose up --build`.

## Layout

- `app/models/` — SQLAlchemy ORM (User, CustomCategory, Correction, ModelVersion)
- `app/schemas/` — Pydantic request/response shapes
- `app/api/routes/` — `corrections` (receive feature-vector + label from the
  client), `categories` (custom category CRUD), `personalization`
  (trigger/fetch a personalized model)
- `app/services/personalization.py` — the actual personalization logic
- `alembic/` — migrations; nothing is auto-created against a real DB
  without running `alembic upgrade head`

## Auth

`app/api/deps.py` currently reads a plain `X-User-Id` header — a
placeholder so the rest of the stack can be built without blocking on the
real auth flow (verifying the Gmail OAuth token, or a session of our own).
Replace before this goes anywhere near production.
