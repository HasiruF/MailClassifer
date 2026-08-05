# Model artifacts go here

`spam_classifier.onnx`, `category_classifier.onnx`, `priority_classifier.onnx`
— exported from the trained sklearn pipelines in `models/*.joblib` via
`skl2onnx`. Not committed (binary build artifacts); regenerate with the
(not-yet-written) `scripts/*/export_model.py` and copy the output here.

Until these exist, `src/inference/engine.ts` throws instead of returning
wrong predictions from missing models — see `App.tsx` for the resulting
UI state.
