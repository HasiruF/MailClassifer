# Show the work — design

**Date:** 2026-10-08
**Status:** approved (mockups on the "Inbox redesign" canvas, artboards
"1 · What a correction sends", "2 · Retrain report card", "3 · Re-sorted by
your model")

## Goal

The app already sorts mail on the device, retrains a personal model from
corrections behind a three-check gate, and keeps email text off the server.
None of that is visible. This work makes three of those facts visible in the
places they happen, using data the system already has.

## 1 · Re-sorted by your model

When a retrain produces a new active model and the inbox is re-classified,
the user sees which emails moved.

- Before `watchRetrain` re-classifies, snapshot each row's raw model category
  (`modelResult.category`). After re-classification, an email is *re-sorted*
  when its raw category changed **and** the user has not corrected its
  category (a correction already decides what they see).
- Only category changes are tracked. Priority and spam flips are not tagged.
- State lives in the inbox context: `resorted: { version: number | null;
  from: Record<emailId, string> } | null`, plus `showResortedOnly: boolean`.
  `version` is the newly active category model version (null when the
  category model did not change).
- The list shows a teal banner "Your model (version N) re-sorted K emails."
  with **Show only these** (sets `showResortedOnly`) and a dismiss button
  (clears both). Rows in `from` carry a "was X" tag after their category chip.
  When `showResortedOnly` is set, the list shows only those rows.
- Turning personalization off, reloading, or a retrain that changes nothing
  clears the state. A retrain that is rejected does not re-classify, so it
  never sets it.
- The sorting panel header shows "Your model · version N" next to "Sorted on
  this device" whenever a personalized category model is active
  (`activeVersions.category`), and the category cell of a re-sorted email
  shows "Was X before this update".

## 2 · Retrain report card

The sidebar's "last attempt" block becomes a report card for the most recent
attempt across models.

- Backend: `LastAttempt` gains `correction_count: int` (the attempt's
  `personalized_models.correction_count`, already stored).
- Checks, from `metrics` (already stored by `gate.as_metrics()`):
  1. **Learned your corrections** — `round(own_correction_accuracy ×
     correction_count)` of `correction_count`, with a bar and a threshold
     mark at 80%. Caption "Needs N of M".
  2. **Still sorts other mail well** — `gate_accuracy`, caption
     "`base_gate_accuracy` before · may drop 3 points at most". Passes when
     `gate_accuracy ≥ base_gate_accuracy − 0.03`.
  3. **Your labels stay specific** — category model only:
     `custom_label_fraction` "of test mail went to your labels · 10% at most".
- The check rules mirror `gate.py` (OWN_CORRECTION_MIN 0.80, MAX_GATE_DROP
  0.03, MAX_CUSTOM_LABEL_FRACTION 0.10); each row is marked passed/failed.
- Active: "<Model> model updated" + "Version N passed all checks and is
  sorting your inbox." + the three checks + "See the K emails it re-sorted"
  (only when `resorted` has entries; sets `showResortedOnly`).
- Rejected: "Version N not applied" + "Version M is still sorting your
  inbox." when an active version exists, else "The standard model is still
  sorting your inbox." + the checks with the failing one in coral.
- When metrics lack the gate numbers (a job that errored records
  `{error: ...}`), fall back to today's text: the stored reason or error.

## 3 · What a correction sends

A receipt that shows exactly what the server holds for one correction,
decoded back into readable terms on the device.

- Backend:
  - `GET /personalization/corrections` → `[{provider_message_id, model,
    corrected_label}]` for the current user (all their connections). Used on
    inbox load to restore `row.corrected`, so corrections survive a reload.
  - `GET /personalization/corrections/{provider_message_id}` → the stored
    rows for that message: `[{model, provider_message_id, predicted_label,
    predicted_confidence, corrected_label, feature_vector, created_at}]`.
    Empty list when none. Both require a session (401 otherwise) and are
    scoped to the user.
- Decoding (`src/lib/receipt.ts`, pure): given the model, its vocab, and the
  sparse vector, split indices by the vector layout the client builds:
  - spam: word TF-IDF
  - category: word | char_wb | numeric (scaled)
  - priority: word | char_wb | category one-hot | numeric (scaled)
  Words and letter groups map index → term by inverting `vocabulary`, sorted
  by weight. Numeric values are unscaled (`v × scale + mean`; `n_recipients`
  and `text_len_log` also `expm1`) and rendered as plain signals; a numeric
  column missing from the sparse vector is a stored zero (= the column mean).
- Engine exposes the loaded vocabs (`getVocabs()`).
- UI: corrected verdict cells get a **What was sent?** link that opens the
  receipt in the same slot as the correction picker. The receipt shows: the
  model's guess + confidence, the user's answer, the message ID, top 8 words
  (with "Show all N words"), up to 12 letter groups (+ "N more") with the
  caveat "Short pieces of words the model knows. They can hint at words that
  aren't in its word list.", the signals, and a "Never sent" box (subject
  line, the email as written, who sent it, your other emails). Loading and
  error states use the existing panel styles.

## Out of scope

- Changing what gets sent (e.g. dropping letter groups). Noted as a privacy
  follow-up; the receipt states it honestly.
- Tagging priority/spam re-sorts; persisting the re-sorted state.
- The "Needs you today" view (separate, designed earlier).

## Testing

- Backend: `correction_count` on `last_attempt`; list + detail endpoints
  (401 without session, scoped to the user, empty for unknown ids).
- Frontend unit tests: `receipt.ts` decoding (each layout, unscaling,
  ordering), `resort.ts` diffing (corrected rows excluded, unchanged rows
  excluded), `retrain-report.ts` (passed, rejected, category-only third
  check, error fallback).
- Full suites, type check, lint; then the running app: open a receipt for an
  existing correction, check the report card for the last attempt, and check
  that corrections show after a reload.
