# Frontend

Vite + React + TypeScript. Runs all classification inference client-side
via `onnxruntime-web` — email content never leaves the browser unless the
user explicitly opts into personalization sync, and even then only an
extracted feature vector is sent (see `src/api/client.ts`).

## Setup

```bash
npm install
cp .env.example .env   # fill in VITE_GOOGLE_CLIENT_ID if using live Gmail
npm run dev
```

Requires `public/models/*.onnx` to exist before inference works — see
`public/models/README.md`. Until then the app loads but shows a
"models not available" state (`src/App.tsx`).

## Layout

- `src/inference/` — feature extraction (mirrors `scripts/shared/*.py`,
  must stay numerically identical) + the `onnxruntime-web` wrapper
- `src/storage/` — IndexedDB: emails, corrections, custom categories, all
  local-first
- `src/api/` — talks to the backend, only for the opt-in personalization
  sync and fetching a personalized model back down
- `src/pages/`, `src/components/` — UI, not yet built out
