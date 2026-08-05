# App

The application layer — separate from `../scripts/` (the offline ML
training pipeline) and `../models/` (the trained `.joblib` artifacts it
produces). See `../PROJECT_STATUS.md` for the full classifier history.

```
app/
├── frontend/   Vite + React + TS. Runs all inference client-side via
│               onnxruntime-web — see frontend/README.md
├── backend/    FastAPI + Postgres. Opt-in personalization only — never
│               sees raw email content, see backend/README.md
└── docker-compose.yml   postgres + adminer + backend, for local dev
```

## Current state

Scaffolded, not yet functional end-to-end. Concretely:

- [ ] **Model export** (`scripts/*/export_model.py`, not written yet) —
      convert `models/*.joblib` to ONNX via `skl2onnx`, drop into
      `frontend/public/models/`. Nothing in the frontend can classify a
      real email until this exists — it's the next real blocker.
- [x] Frontend scaffold: types, IndexedDB storage, feature extraction
      (ported from `scripts/shared/*.py`, needs the TF-IDF vocabulary once
      export exists), `onnxruntime-web` wrapper (throws until models load),
      API client for the opt-in sync
- [x] Backend scaffold: Postgres schema (users, custom_categories,
      corrections, model_versions), FastAPI routes, centroid/kNN
      personalization logic (retrain-on-threshold path is a stub)
- [ ] Real auth (placeholder `X-User-Id` header currently)
- [ ] Gmail OAuth integration on the frontend (keep the consent screen in
      "Testing" status — see PROJECT_STATUS.md, avoids Google's
      verification process which doesn't fit a course timeline)
- [ ] UI — no pages/components built yet beyond the loading-state shell

## Quickstart

```bash
docker compose up -d postgres adminer
cd backend && pip install -r requirements.txt && alembic upgrade head && uvicorn app.main:app --reload
cd ../frontend && npm install && npm run dev
```
