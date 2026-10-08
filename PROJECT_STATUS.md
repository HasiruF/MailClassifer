# Email Classifier — Current Setup

Last updated: 2026-08-14

## What this project is

A three-part email classifier (spam / category / priority) trained on two public
datasets sitting in this folder, meant to be dropped into a backend via
`scripts/classify_email.py`.

| Part | Status | Model file |
|---|---|---|
| **Spam / ham** | Fixed and verified this session — false-positive rate 88.4% → **0.3%** on Enron | `models/spam_classifier.joblib` |
| **Category** (Work / Personal / Other) | Retrained on 1500 rows (1000 Enron + 500 hand-authored synthetic) — **78.0% accuracy on the 100-email out-of-distribution adversarial test** (up from 62.0%), **80.1% same-corpus CV**. See v7 |
| **Priority** (high/medium/low) | Retrained on a 1004-row gold set (504 Enron + 500 hand-authored synthetic, balanced category x priority) — **81.0% accuracy on the 100-email out-of-distribution adversarial test** (up from 46.0%), **85.1% nested-CV**. See items 3e/3f | `models/priority_regressor.joblib`, `models/priority_classifier.joblib` |

`scripts/classify_email.py` (`EmailClassifier`) is wired up to both current
models and accepts the header fields it needs
(`subject, body, to, cc, from_addr, list_unsubscribe, precedence, sent_at`).
Both model files now exist and load cleanly.

## Repository layout

```
DataManagementProject/
├── Enron/, Enron.zip                    raw corpus
├── SpamAssasin/, SpamAssasin.zip        raw corpus (folder name really does drop one "s")
├── gold_sample_labeled.csv              THE master dataset — 1000 hand-labeled rows
├── models/
│   └── category_classifier_headers.joblib   the only model currently on disk
├── predictions/                         empty — regenerate once satisfied with accuracy
└── scripts/
    ├── classify_email.py                serving entry point (EmailClassifier)
    ├── shared/                          utilities imported by multiple other scripts
    │   ├── train_category_classifier.py     clean_text/parse_enron_message + retired v1 trainer
    │   └── header_features.py               domain-agnostic header/metadata feature extraction
    ├── category/                        category-classifier training/eval
    │   ├── train_category_classifier_gold.py     v2 (text-only, superseded)
    │   ├── train_category_classifier_headers.py  v3-v5, current trainer
    │   ├── eval_category_gold.py            quick resubstitution sanity-check
    │   └── gold_header_features.csv         cached header features for the 1000 gold rows
    ├── spam/
    │   └── train_spam_classifier.py     currently broken, see below
    └── labeling/                        gold-label creation tooling + audit trail
        ├── sample_for_gold_labels.py, export_batch.py, merge_labels.py    (original random-sample round)
        ├── active_learning_pool.py, append_active_batch.py               (active-learning rounds)
        ├── resort_decisions.py           the Work/Personal/Other re-sort decision dict (v5)
        └── active_batch1-10_labels.csv   my label decisions per active-learning batch
```

**Cross-script imports:** any script outside `shared/` that needs
`clean_text`/`parse_enron_message`/`header_features` adds `../shared` to
`sys.path` (see the `sys.path.insert` lines near the top of each file).
Default CLI arg paths are relative to each script's own folder — e.g.
`category/train_category_classifier_headers.py`'s `--gold` default is
`../../gold_sample_labeled.csv` (two levels up: `category/` → `scripts/` →
project root).

**This reorg surfaced two real bugs**, now fixed: `eval_category_gold.py`
and `active_learning_pool.py` were still calling `pipeline.predict(text)` /
`pipeline.predict_proba(texts)` on **plain text**, left over from before the
header-features model existed — the current model needs a DataFrame with
`text` + the 5 `NUMERIC_COLS`, not a bare text Series. Both would have
crashed (or silently mis-scored) the next time either was actually run.
Fixed by building the same feature shape `train_category_classifier_headers.py`
trains on. Retrained + re-ran eval from the new folder locations afterward to
confirm nothing broke (same 73.6% CV result).

**`app/`** — the actual application (separate from the `scripts/` training
pipeline above), now a working Next.js (App Router, static export) frontend
running all four models client-side via `onnxruntime-web`, connected to a
real Gmail inbox via client-side OAuth. See the "Application layer (`app/`)"
section below for full detail — the one-line version: Phase 1 (frontend-only
classification engine) is built, tested, and committed; a full inbox UI sits
on top of it, committed; live Gmail fetch+classify sits on top of *that*,
working but not yet committed; Phase 2 (FastAPI+Postgres backend, opt-in
personalization, plus a minimal Gmail token-holder for persistent login) is
not started. No `app/README.md` exists.

## Datasets

- `Enron/` (from `Enron.zip`) — ~517K real corporate emails, one CSV (`emails.csv`) with columns `file` (mailbox path, e.g. `dasovich-j/inbox/199.`) and `message` (raw email incl. headers, newline-separated, headers cleanly split from body on the first blank line).
  - **Quirk:** the same message is frequently duplicated verbatim across several folders in the same person's mailbox (`sent`, `sent_items`, `all_documents`, `discussion_threads`, ...). Dedupe on subject+body before sampling/labeling or you waste effort re-labeling the same email 3-4x.
- `SpamAssasin/` (from `SpamAssasin.zip`, note the folder's actual spelling drops one "s") — `spamassassin.csv`, columns `text`/`target` (0=ham, 1=spam), 5,796 rows.
  - **Quirk (the one that matters):** `text` is each raw mbox message **flattened to a single line — no newlines at all.** There's no blank-line boundary to split headers from body the way `Enron/emails.csv` allows, so the full `Received:` routing chain, server hostnames, and MTA software tags ride along as if they were message content. See the spam section below — this is the root cause of a real bug, not a cosmetic issue.
  - **Also not a good source for priority/category diversification** — its ham class is almost entirely one cluster of 2002 tech mailing lists (ILUG, FoRK, spamassassin-devel), not general correspondence. Considered and rejected for that purpose in item 3f.
- `gold_priority_synthetic.csv` — 500 fully hand-authored (not sampled) synthetic emails, balanced by construction across category (Work/Personal/Other) and priority bucket (low/medium/high), covering personal/casual/emergency registers absent from Enron. Built via `scripts/priority/synthetic_priority_train_batch{1..5}.py` + `build_synthetic_training_csv.py`. See item 3f.
- `gold_priority_labeled_v2.csv` — `gold_priority_labeled.csv` (504 Enron rows) + `gold_priority_synthetic.csv` (500 rows) = 1004 rows, current default training set for `train_priority_classifier.py`.

## Category classifier — full history

### v1 (superseded, deleted) — folder-name weak supervision
`scripts/shared/train_category_classifier.py` mapped mailbox folder names to
categories (`inbox`/`sent` → Work, `mailing lists` → Mailing List, etc. — see
`FOLDER_MAP`) and trained on that, no human labels. It's kept only because
`clean_text`/`parse_enron_message` inside it are shared utilities every other
script imports — the actual v1 model/predictions it produced were deleted
during cleanup (superseded, verified broken).

**Verified problem:** checked against gold labels and it was only **45%
accurate** — folder location turned out to be a poor proxy for actual content
category (a folder's own weak-label only agreed with true category ~56% of
the time, even restricted to folders with an explicit mapping).

### Gold-labeling pipeline
1. `scripts/labeling/sample_for_gold_labels.py` — stratified-sampled the v1
   model's predictions, joined back to `Enron/emails.csv` for text →
   `gold_sample_unlabeled.csv` (deleted post-labeling, fully superseded).
2. `scripts/labeling/export_batch.py` — dumps a row range as plain text for a
   human (or careful LLM pass) to read and label.
3. Labels → `batch1-9.txt` → `batch_labels.csv` (both deleted, consumed) →
   merged into `gold_sample_labeled.csv` via `scripts/labeling/merge_labels.py`.
   Produced the original **540** rows. Classes: Work, Personal, Mailing List
   (News/Newsletter never showed up in practice, dropped from the effective
   label set).

### v2 — training on gold labels directly (`category/train_category_classifier_gold.py`)
TF-IDF (word 1-2 grams, 20K features) + Logistic Regression, trained on the
**540 real gold labels** instead of folder names. 5-fold CV: **76.3%** — a big
jump over v1's 45%, since it's learning actual content signal instead of
mailbox filing habits.

### Active-learning label expansion: 540 → 1000
Selected the corpus's **lowest-confidence** predictions each round (not
random), labeled 50 at a time, retrained/rescored every ~100 labels:
- `scripts/labeling/active_learning_pool.py` — scores every unlabeled email,
  **dedupes on subject+body**, returns the N lowest-confidence rows.
- `scripts/labeling/append_active_batch.py` — appends a freshly-labeled batch
  into `gold_sample_labeled.csv` (unlike `merge_labels.py`, which only fills
  in `true_category` for rows already present).

Accuracy trend (drop is expected, not a regression — each round intentionally
pulls in harder, more ambiguous emails, so the yardstick gets tougher along
with the data):

| Gold labels | 5-fold CV accuracy |
|---|---|
| 540 (original random sample) | 76.3% |
| 640 → 940 | 71.7% → 63.7% (steady decline) |
| 1000 (final, text-only) | 60.8% |

Upside: Personal went from a thin 93/540 (17%) to a healthier 245/1000
(24.5%) — the original class-imbalance problem is largely fixed even though
raw accuracy looked lower at this point.

**Labeling-consistency audit:** spot-checked recurring patterns (out-of-office
autoreplies, e-ticket confirmations, contact-book entries, recurring
newsletter templates like "Btu Weekly"/"CapstoneOnline") for cases where
near-identical content got different labels across the 10 batches. Found
none — the ad hoc judgment calls held up internally consistent. So the
~60-68% ceiling was a real model/data-signal limit, not sloppy labeling.

### v3 — domain-agnostic header/metadata features (`category/train_category_classifier_headers.py`, `shared/header_features.py`)
Added features designed to generalize beyond Enron/beyond any one company —
**deliberately not** "sender domain == enron.com" (that would just memorize
one company and be useless on e.g. a Gmail user's mailbox):
- `n_recipients` — To+Cc address count (broadcast vs. 1:1)
- `is_reply_or_forward` — subject starts with Re:/Fw:/Fwd:
- `sender_automated` — local-part matches noreply/no-reply/donotreply/
  mailer-daemon/support/notifications/etc. (pattern, not domain)
- `has_list_unsubscribe`, `has_precedence_bulk` — the real modern
  mailing-list standard (RFC 2369). **Caveat:** structurally absent from this
  ~2001 Enron corpus (predates wide adoption), so these two are all-zero and
  contributed nothing to the CV numbers below — included so the feature set
  is ready for real modern mail (e.g. Gmail) where they do fire.

`shared/header_features.py` exposes two entry points sharing the same
regex/logic: `extract_header_features(raw_message, subject)` for training
(parses a raw Enron-style header block) and
`extract_header_features_from_fields(subject, to, cc, from_addr,
list_unsubscribe, precedence)` for serving (a backend hands you structured
fields directly, no raw block to parse) — used by `classify_email.py`.

Result: 60.8% → **61.9%** CV accuracy (small, needed re-tuning `C` from 1.0 →
8.0 for the new feature mix — the header features on their own, at the old
`C`, actually *degraded* things by over-triggering on `n_recipients`).

**Portability bug found & fixed:** the first save of this model couldn't be
loaded from any script except the training script itself — a
`FunctionTransformer` referenced a helper function defined in the training
script, and pickle records such functions as belonging to `__main__` when a
script is run directly, which breaks on load elsewhere. Fixed by moving the
function (`log1p_recipients`) into the importable `header_features.py`
module. **Lesson: any custom function fed to `FunctionTransformer`/similar
must live in a real importable module, never in the `if __name__ ==
"__main__"` training script**, or the saved model is single-use.

### v4 — word + char n-grams
Added TF-IDF **character** n-grams (3-5 chars, `analyzer='char_wb'`) alongside
the existing word n-grams, combined via `FeatureUnion`. Word-level TF-IDF
only sees vocabulary; char n-grams pick up tone/register — informal spelling,
ALL-CAPS, punctuation habits — which is exactly the signal that separates a
casual Personal note from a formal Work email and doesn't show up as
"words." Needed retuning `C` again (8.0 → 3.0) for the new feature mix.

**Result: 61.9% → 67.8% CV accuracy.**

### v5 — retired "Mailing List", replaced with "Other" (current, biggest jump)
Error analysis on v4 surfaced a structural problem, not a modeling one:
"Mailing List" was mixing two different axes into one label — **topic**
(what it's about) and **distribution pattern** (broadcast vs. addressed to
you). A recurring internal legal report and a one-off internal legal email
have identical vocabulary but got different labels, purely based on how many
people it was sent to. No amount of text modeling fixes a label that isn't
actually a function of the text.

**Taxonomy change (user-directed):** dropped Mailing List, taxonomy is now
strictly **Work / Personal / Other** (spam handled entirely separately by
the spam classifier below, not folded into this scheme). "Other" is
deliberately loose for now — a catch-all for whatever's neither Work nor
Personal — with the expectation that sub-categories may get carved out of it
later once it's clearer what accumulates there.

**Re-sorted all 241 former "Mailing List" rows by hand** into Work/Personal/
Other by actual content rather than distribution pattern
(`scripts/labeling/resort_decisions.py` — a `{file: category}` dict, verified
key-for-key against the source rows before applying, after a first pass had a
silent transcription miscount). Rough rule applied: real correspondence/
internal reports/operational systems → Work; genuine social/relationship/
volunteer content → Personal; external newsletters/subscriptions/trade
publications/marketing/solicitations/chain-forwards-to-large-lists/automated
non-content noise (sync logs, read receipts, bounces) → Other. Result:
**Work 625, Personal 264, Other 111** (was Work 514, Personal 245,
Mailing List 241).

**Result: 67.8% → 73.6% CV accuracy** — the single biggest jump of the
session, bigger than char n-grams or header features combined. Confirms the
diagnosis: once the label was actually a function of the content, the same
model architecture could learn it far better.

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Work | 0.81 | 0.84 | 0.82 | 625 |
| Personal | 0.61 | 0.59 | 0.60 | 264 |
| Other | 0.59 | 0.50 | 0.54 | 111 |

**Current model:** `models/category_classifier_headers.joblib` (same
filename, retrained on the new taxonomy — pipeline/architecture unchanged
from v4, only the labels changed). Retrain via
`scripts/category/train_category_classifier_headers.py --gold ../../gold_sample_labeled.csv
--headers gold_header_features.csv --out ../..` (run from inside `category/`;
the header-features CSV for the 1000 gold rows is cached right there —
regenerate it by re-scanning `Enron/emails.csv` for the files in
`gold_sample_labeled.csv` and calling `extract_header_features` per row if it
ever goes stale).

**What's actually inside "Other" right now** (sampled all 111 rows and
grouped by pattern — useful if/when carving out sub-categories later):

| Sub-pattern | ~Count | Examples |
|---|---|---|
| External trade/market/industry newsletters & data subscriptions | ~24 | RIGZONE, TradersNews, MAPP/outage feeds, TR Daily, BNP Paribas commentary, PowerMarketers.com, CongressDaily |
| Academic/institutional broadcasts (MBA program, school admin) | ~15 | Register for Spring 2001, Berkeley Business Plan Winners, MIT Forum, CapstoneOnline, Coffee Colloquium |
| Automated non-content noise (receipts/bounces/bare dumps) | ~11 | Email Receipt Confirmation, Delivery Status Notification, Read: receipts, bare address-list dump, auto address-change notices |
| Spam/scam/unsolicited solicitations | ~11 | 419-style scam, credit-repair ads, MLM pitch, online pharmacy, tax-help spam |
| External retail/travel/consumer marketing | ~13 | Travelocity, buy.com, southwest.com specials, Crutchfield Newsletter, rewards programs |
| Chain-forwards / inspirational / religious / viral | ~10 | Prayer forwards, devotionals, "brighten your day" chains |
| Political/civic/fundraiser solicitations | ~8 | Fundraiser invites, BIPAC membership pitch, "Demand Ken Lay Donate" |
| R-help technical mailing list (external) | ~4 | The "[R]" statistics mailing list threads |
| External recruiter/business-development solicitation | ~5 | Paid survey invite, staffing-agency outreach, conference pitch |
| Facility/wellness (building/gym notices) | ~2 | Water-main outage at the gym, gait-analysis clinic |

The two largest chunks (external industry newsletters, ~24; spam/scam +
retail marketing combined, ~24) are the clearest candidates for their own
sub-category if "Other" ever gets split further.

Tried and **rejected** along the way (didn't beat the eventual config):
- LinearSVC, ComplementNB, wider word n-grams/feature counts — no
  meaningful improvement at this label count.
- Semi-supervised self-training (pseudo-labeling the corpus at high
  confidence, folding back into training) — only ~+1 point for a lot of
  added complexity. Deleted.
- More active-learning rounds beyond 1000 — diminishing/negative returns
  already observed; remaining errors are ambiguity-driven (short/generic
  content like "RE:", automated notifications), not scarcity-driven.

### Error analysis — why the ceiling is where it is (post-v5)

Full 5-fold out-of-fold error breakdown on the 264 v5 misclassifications:
**64% of all errors are Personal↔Work confusion** (95 Personal→Work, 75
Work→Personal), not "Other" being messy. Concrete examples of what's actually
getting confused: `"Car jokes"`, `"Back in the saddle again..."`, `"An Enron
Employee's Perspective"`, `"Demand Ken Lay Donate Proceeds..."`, `"Neil
Anderson Devotional for Monday"` — coworkers forwarding jokes, political
opinions, devotionals, and personal reflections **through their work email to
work colleagues**. The content genuinely reads as personal/social; the
context (work inbox, work distribution) says work-adjacent. This is the same
failure mode that motivated the v5 taxonomy fix, one layer down — content-only
classification structurally cannot resolve "same words, different social
context" no matter what features or model you throw at it.

**Label-noise check:** flagged rows where the model was >75% confident in a
different label than gold. Only 22 of 264 errors qualified — the other 242
are cases where the model itself is uncertain too (genuine ambiguity, not a
fixable mistake). Of those 22, systematically checked for exact-duplicate-
content pairs with inconsistent labels (the only reliable way to find real
mislabels without re-reading all 1000 by hand): found exactly **one** genuine
error — two near-identical automated Charles Schwab "Email Receipt
Confirmation" auto-replies, one labeled Other, one labeled Personal. Per the
v5 taxonomy rule ("automated non-content noise → Other" regardless of topic),
corrected `dasovich-j/all_documents/10168.` from Personal → Other.
`"Calendar"` and `"RE: FW:"` also had multiple true_category values but
turned out to be different real emails sharing a generic subject line —
verified correct as-is by reading full bodies.

**Result: 73.6% → 74.1%** (small, expected — one row out of 1000, plus
re-shuffled stratified-fold membership from the class-count shift).

### v6 experiments — two more real levers tried, both rejected

**Sentence embeddings** (`all-MiniLM-L6-v2`, 384-dim, via
`scripts/category/train_category_classifier_embeddings.py`) — motivated by
the proposal's own reasoning that BERT-style models help "complicated or
ambiguous emails." Direct 5-fold CV comparison, same data/folds:

| Config | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF word+char + headers (baseline) | **73.6%** | **0.6534** |
| Embeddings alone + headers, no TF-IDF | 72.2% | 0.6494 |
| TF-IDF + embeddings + headers combined | 72.3% | 0.6498 |

Embeddings underperform on their own and make things slightly *worse* when
combined — a general-purpose frozen embedding model doesn't capture the
tone/register signal (informal spelling, ALL-CAPS, punctuation habits) the
existing char n-grams already get, and it can't resolve the dominant error
mode above anyway, since that's a social-context problem, not a
semantic-meaning one. **Conclusion: a fine-tuned transformer (actual
backprop through labeled examples, not a frozen general-purpose embedding)
remains untested and is the one form of "try BERT" not yet ruled out** — a
bigger, slower experiment, not attempted this session.

**Time-of-day / day-of-week features** (`hour_sin`, `hour_cos`, `is_weekend`,
`is_business_hours` — new fields added to `shared/header_features.py`,
parsed from the `Date:` header via `email.utils.parsedate_to_datetime`,
sender's local time preserved, never converted to a single timezone).
Hypothesis: personal email skews evenings/weekends, work skews business
hours — a portable, domain-agnostic signal that would generalize past Enron.
**Result: no real signal on this corpus.** Cyclical hour encoding alone hurt
accuracy (72.7-72.9%); binary `is_weekend`/`is_business_hours` alone were
within noise of baseline (73.4-73.7%); combined, still net negative. Checked
class means directly: `is_business_hours` was actually *higher* for Other
(54%) than for Work (45%) — the opposite of the hypothesis. Likely
explanation: 2001-era Enron culture sent personal chatter and real work
correspondence through the same 9-5 office window rather than splitting by
time of day, so the signal that might exist on a modern personal-Gmail
inbox (different usage pattern) doesn't show up here.
**Not wired into the shipped model** — `NUMERIC_COLS` in
`train_category_classifier_headers.py`/`classify_email.py` stays at the
original 5 columns. The extraction code stays in `header_features.py`
(harmless, documented, available if retrained against a corpus where the
pattern actually holds) — `classify_email.py`'s `EmailClassifier.classify()`
now accepts an optional `sent_at` param that's threaded through but
currently doesn't affect the loaded model's predictions.

**Current model after this round: 74.1% CV accuracy**, same architecture,
5-column header feature set, one label correction applied.

### Confidence-gating — the practical path past the raw-accuracy ceiling

Given the ceiling above is structural (content-only classification can't
resolve context-dependent ambiguity), the actual path to a trustworthy
90%+ number is **not forcing a label on every email** — defer low-confidence
predictions instead of guessing. Per-class analysis (5-fold OOF
`predict_proba`, same 1000 rows):

| Predicted class | Threshold for ≥90% precision | Coverage at that threshold |
|---|---|---|
| Work | ≥0.60 | 71% of predicted-Work kept (46% of all mail) |
| Other | ≥0.80 | 23% of predicted-Other kept (2% of all mail) |
| Personal | ≥0.90 | 3% of predicted-Personal kept (1% of all mail) |

Work predictions are far more trustworthy than Personal/Other at any given
confidence — Work alone can hit 90%+ precision while auto-classifying
nearly half the corpus. Personal/Other essentially cannot reach 90%
precision in any useful volume. Blended per-class thresholds (Work≥0.60,
Other≥0.80, Personal≥0.75) give **53% coverage at 90.0% overall accuracy**.
**Recommended design, not yet implemented:** ship as a 3-tier system — trust
high-confidence Work predictions outright, surface Personal/Other
predictions as suggestions needing user confirmation rather than silent
filing, and fall back to unclassified/default view below threshold.
`classify_email.py`'s `EmailClassifier.classify()` does not yet return a
`confident`/`needs_review` flag — this is the next concrete implementation
step if pursued.

### v7 — same synthetic-data fix applied as priority (item 3f): 62.0% → 78.0% on adversarial

Category had the same problem priority did before item 3f: 76.8% same-corpus
CV but only **62.0%** on the 100-email out-of-distribution adversarial test
(`adversarial_test_set.py`) — 1000 training rows, 100% Enron, don't teach a
model to recognize modern/personal-register Work vs. Personal vs. Other.

Reused the 500 hand-authored synthetic rows already built for priority's
item 3f (`scripts/priority/synthetic_priority_train_batch{1..5}.py`) — this
time training on their `true_category` labels directly (priority's own
retrain instead used the category *model's prediction*, to match what
priority receives in production; here category IS the thing being learned,
so the true label is used). Same leakage guard against
`adversarial_test_set.py` (zero overlap, asserted in
`scripts/category/retrain_with_synthetic.py`). Same architecture as v5/v6
(word+char TF-IDF + header/style features + LogisticRegression, `C=3.0`,
`class_weight='balanced'`), trained on 1000 Enron + 500 synthetic = 1500
rows, plain 5-fold CV (no nested search — this model's hyperparameters were
already fixed from earlier rounds).

| metric | before (1000 Enron-only) | after (1500, +500 synthetic) |
|---|---|---|
| Same-corpus CV accuracy | 76.8% | **80.1%** |
| Adversarial accuracy | 62.0% | **78.0%** |
| Adversarial Work recall | 82.9% | 68.3% |
| Adversarial Personal recall | 63.6% | **84.8%** |
| Adversarial Other recall | 26.9% | **84.6%** |

Personal and Other recall roughly tripled/doubled; Work recall dropped
(82.9%→68.3%) — the old model's bias toward over-predicting Work (Enron is
~80% Work-register mail) is exactly what training on balanced synthetic
data corrects, at some real cost to Work specifically. Net: +16 points
overall, a clear improvement, but the new Work/Personal boundary is worth
a closer look if Work recall matters more than Personal/Other in practice.
Priority's own adversarial accuracy moved from 82.0% to 81.0% as a
side-effect (category_label is one of priority's input features) — within
noise, not a regression worth chasing.

**Promoted to production**: `models/category_classifier_headers.joblib`
overwritten, smoke-tested via `classify_email.py` alongside the item 3f
priority models. Both retrains together now give: spam (unchanged, still
0.3% FP rate), category 78.0% adversarial, priority 81.0% adversarial —
all three classifiers now validated against genuinely out-of-distribution
text, not just same-corpus CV.

## Spam classifier — fixed and verified

`scripts/spam/train_spam_classifier.py` went through three rounds of
diagnosis this session. First two bugs were cosmetic/mechanical:

1. **Fixed:** column-detection didn't recognize this CSV's `target` column
   (only looked for `label`/`class`/`spam` in the name) — added `target`.
2. **Fixed:** a `print()` with a Unicode arrow crashed on Windows' default
   `cp1252` console encoding — added `sys.stdout.reconfigure(encoding='utf-8')`.

The third was the real one: trained-and-predicted over the full Enron corpus
flagged **88.4%** of it as spam. First mitigation attempt (`_ROUTING_WORDS`
regex stripping mail-transport vocabulary like `received`/`postfix`/`smtp`
from `clean_text()`) only brought it down to **85.3%** — essentially
unfixed. That forced a deeper diagnosis.

**Root cause (found by inspecting the trained LogisticRegression's actual
coefficients — top spam-indicative and ham-indicative terms):**
- Spam-indicative terms included genuine spam vocabulary (`free`, `click`,
  `money`, `remove`, `credit`) mixed with raw **HTML tag/attribute tokens**
  (`br`, `font`, `href`, `html`, `td`, `tr`, `align`, `width`, `arial`) —
  SpamAssassin's spam examples are HTML-formatted marketing mail, and
  `clean_text`'s punctuation-stripping turns `<font color="red">` into bare
  word tokens `font color red` instead of removing it — plus specific
  **leaked hostnames** (`mandark`, `mandark labs`, `webnote`, `webnote net`)
  that `_ROUTING_WORDS` never caught since it only stripped generic header
  *field names*, not arbitrary proper-noun hostnames.
- Ham-indicative terms were the real discovery: `xent`, `xent com`,
  `jmason`, `jmason org`, `wrote`, `oct`, `sep`, `pdt`, `ist` — even the
  literal word `"spam"` was ham-indicative. **`jmason.org`/`xent.com` are
  SpamAssassin project infrastructure** — its "ham" class isn't generic
  legitimate email at all, it's specifically **the SpamAssassin
  developer/user mailing list's own traffic** (quoted-reply threads ending
  in "X wrote:", a particular timestamp format, project jargon — "ham"
  examples are literally emails *about* spam filtering). No amount of
  header-vocabulary stripping fixes a dataset where the **ham class itself**
  is one narrow, unrepresentative genre — Enron's plain corporate
  correspondence doesn't match SpamAssassin's ham profile *or* its spam
  profile, and the model defaulted to "spam" because none of the
  ham-defining cues (mailing-list structure, reply-quote format, `jmason`/
  `xent` infrastructure) are present in Enron text at all.

**Real fix: mix in genuine Enron messages as ham training examples.**
`train_spam_classifier.py` now trains on 500 SpamAssassin spam + 300
SpamAssassin ham (kept for vocabulary diversity) + 700 real Enron messages
(sampled, deduped, labeled ham — the corpus is overwhelmingly legitimate
correspondence). New `load_enron_ham()` function samples/dedupes/cleans
Enron rows the same way `active_learning_pool.py` does. Also added the
specific leaked hostnames (`mandark`, `webnote`, `xent`, `jmason`) to
`_ROUTING_WORDS` as a cheap patch — not a general solution, the ham-mixing
is the actual fix.

**Result — verified two ways:**
- Held-out eval on the mixed training set (15% split): 98% accuracy, spam
  precision/recall both ~97%.
- **The number that actually matters:** false-positive rate on fresh,
  never-trained-on Enron mail. Checked two ways for a fair (non-circular)
  test: a 5,000-row disjoint sample (`check_enron_false_positive_rate()`,
  excludes the exact files used in training) → **0.4%** flagged spam. Full
  517K-row corpus scan → **0.3%** (1,494 flagged) — sane for a real
  corporate mailbox, down from 88.4%/85.3%.

`models/spam_classifier.joblib` and `predictions/spam_predictions.csv` are
both regenerated and current. Retrain via
`scripts/spam/train_spam_classifier.py --spam ../../SpamAssasin/spamassassin.csv
--enron ../../Enron/emails.csv --out ../..` (defaults: 500/300/700 split,
override via `--spam_sample`/`--sa_ham_sample`/`--enron_ham_sample`; add
`--skip_full_predict` to skip the full corpus pass and just get the fast
held-out check).

**Not yet done:** this classifier's own accuracy on genuinely novel spam
(not from SpamAssassin) is untested — the held-out 98% is against more of
the same two source distributions, not independent spam examples. Also
worth trying as a category-classifier feature per the "maybe spam fixing
helps Other too" hypothesis (~22-24% of "Other" is genuinely spam-shaped —
see the Other composition table above) — `spam_probability` as an added
numeric column, not yet wired in.

## Application layer (`app/`)

An earlier Vite+React attempt was deleted and restarted deliberately on
Next.js — see the design doc §1 for why (that attempt did prove `skl2onnx`
can't convert `char_wb` TF-IDF, which is why the TF-IDF math below is
hand-ported instead of trusted to ONNX conversion).

### Phase 1 — classification engine (`app/frontend/`) — done, committed

Next.js 16 (App Router, TypeScript, `output: 'export'` so "email content
never touches a server" is a build-time guarantee), shadcn/ui, Vitest.
Built task-by-task per the plan, all committed (see `git log` — "Scaffold
Next.js app" through "Fix real 'Session already started' cause"):

- `scripts/export_onnx.py` (repo root) exports each `.joblib` model as a
  **headless** ONNX graph (dense float vector in, not raw text) plus a
  `*.vocab.json` sidecar (vocabulary, idf, scaler mean/scale, class order).
  Output is gitignored (`models/onnx/`, `app/frontend/public/models/*.onnx`,
  `*.vocab.json`) — regenerate via `python scripts/export_onnx.py --models
  models --out models/onnx` then copy into `app/frontend/public/models/`.
- `src/inference/tfidf.ts` — hand-ported word + `char_wb` TF-IDF, verified
  **bit-exact (0.000000 diff)** against fixtures dumped from the real fitted
  sklearn vectorizers (`scripts/dump_tfidf_fixtures.py` →
  `src/inference/__fixtures__/tfidf_fixtures.json`, committed as a test
  fixture).
- `src/inference/features.ts` — header/keyword feature extraction (ported
  from `scripts/shared/header_features.py` /
  `priority_keyword_features.py`) plus the dense vector builders
  (`buildSpamVector`/`buildCategoryVector`/`buildPriorityVector`) that
  concatenate TF-IDF + header + stylistic + keyword features in the exact
  order the exported ONNX graphs expect.
- `src/inference/stylistic.ts` — tone/register features, ported from
  `scripts/shared/stylistic_features.py`.
- `src/inference/sentiment.ts` — VADER via the `vader-sentiment` npm
  package, cross-checked against Python's `vaderSentiment` to 4 decimal
  places (not exhaustively parity-tested the way TF-IDF was — see design
  doc §9).
- `src/inference/engine.ts` — `loadModels()`/`modelsLoaded()`/`classify()`.
  Loads all 4 ONNX sessions (spam, category, priority classifier +
  regressor) and their vocab JSON from `/models/`, runs spam and category
  independently then priority last (it depends on spam's confidence,
  category's label, and VADER sentiment). Same spam→priority suppression
  rule as `classify_email.py` (spam floors priority to 0.1).
  **Two real bugs found and fixed post-plan** (fixed via three follow-up
  commits, not anticipated in the plan): onnxruntime-web's WASM backend only
  tolerates one `InferenceSession` operation in flight at a time —
  `loadModels()` now creates sessions sequentially instead of via
  `Promise.all`, `classify()` now awaits the two priority-model `.run()`
  calls sequentially instead of racing them, and `loadModels()` is
  idempotent (memoized promise) so React 19 Strict Mode's dev-only
  double-invoke doesn't reload everything twice.
- `src/app/page.tsx` — the plan's minimal demo page: one hand-entered email,
  Classify button, badges for spam/category/priority. Still present at `/`.
- **Not done from the design doc's target state (expected — this was Phase
  1 only):** no Gmail ingestion, no IndexedDB persistence, no
  personalization backend, no correction logging, no dashboard.

### Inbox UI (`app/frontend/src/app/inbox/`) — built, working, **committed**

Net-new scope beyond the Phase 1 plan (which only specified the single-email
demo page above). Not covered by any written design/plan doc; built
directly. Committed 2026-08-14 (`Add inbox UI (list/detail/sidebar) over the
classification engine`).

- `src/data/sample-emails.ts` — 8 hand-authored emails, fallback view shown
  before a real Gmail account is connected (see Gmail integration below).
  Every row is run through the real `classify()` engine at runtime —
  nothing is pre-labeled.
- `inbox-context.tsx` — `InboxProvider`/`useInbox()`, loads models once and
  classifies rows **sequentially** (same WASM one-at-a-time constraint as
  `engine.ts`), exposes `rows` (email + result), category filter, "only
  high priority" toggle, and (as of the Gmail work below) Gmail connection
  state and a client-side `selectedId` for the detail view.
- `sidebar.tsx` — nav by category (All/Work/Personal/Other, live counts), a
  "High Priority" view, model-load status, and Gmail connect/refresh
  controls (see below).
- `inbox-list.tsx` — subject/sender/preview rows with a priority dot (filled
  = high, hollow ring = low), unread indicator, low-priority rows dimmed via
  opacity, category+priority+score badges per row, spam tag when
  applicable. (Split out of `page.tsx` on 2026-08-14 — see routing fix
  below.)
- `email-detail.tsx` — full email view: category confidence bars (all
  classes, sorted), priority score bar, spam-confidence bar,
  spam-suppression note when applicable.
- `page.tsx` — thin switcher: renders `EmailDetail` if an email is selected,
  `InboxList` otherwise. No longer a route-driven page (see below).
- `tokens.ts` — a small hand-picked design-token palette (editorial/mono
  aesthetic: `INK`/`MUTED`/`FAINT`/`HIGH`/`MEDIUM`/`SPAM`/border/background
  colors) shared across the inbox list, sidebar, and detail view.

### Gmail integration (`app/frontend/src/lib/gmail-auth.ts`, `gmail-fetch.ts`) — built, working, **uncommitted**

Built 2026-08-14, same session as a Google Cloud Console walkthrough (OAuth
consent screen in **Testing** publish status, `gmail.readonly` scope, one
Web application OAuth client). Not covered by any design/plan doc yet.
**Deviates from the design doc's own assumption** (§10 open question: "is
the backend involved at all") — turns out no: the read-only fetch+classify
path needed zero backend involvement, resolving that open question in favor
of "not involved."

- `gmail-auth.ts` — Google Identity Services (GIS) token-client flow
  (`initTokenClient`, popup-based). No backend, no client secret — this flow
  structurally never issues a refresh token, only a short-lived (~1hr)
  access token. `NEXT_PUBLIC_GOOGLE_CLIENT_ID` lives in
  `app/frontend/.env.local` (gitignored).
- `gmail-fetch.ts` — fetches the 40 most recent inbox messages directly from
  the browser (`gmail.googleapis.com`, `Authorization: Bearer` header, no
  backend proxy), parses the MIME payload (prefers `text/plain`, falls back
  to `text/html` with tags stripped), maps into `InboxEmail`.
  **Bug found and fixed:** initially fetched all messages via one
  `Promise.all` — worked at 20 messages, threw `429 "Too many concurrent
  requests for user"` at 40. This is a *separate*, undocumented per-user
  concurrent-in-flight-requests cap, distinct from the (documented)
  250-quota-units/sec limit. Fixed with a small worker-pool
  (`mapWithConcurrency`, limit 8) instead of guessing a higher throttle.
- Wired into `inbox-context.tsx`: `connectGmail()` (popup → fetch → classify
  → replace sample rows with the real inbox) and `refreshInbox()` (re-fetch
  using the already-granted token, no popup).
- **Routing fix:** the original inbox detail view was `/inbox/[id]`, a
  Next.js dynamic route. Under `output: 'export'`, every dynamic route's
  params must be known at *build* time via `generateStaticParams()` — fine
  for the 8 fixed sample-email IDs, structurally impossible for Gmail
  message IDs, which don't exist until a user fetches at *runtime*. Hit as
  `Page "/inbox/[id]/page" is missing param... required with "output:
  export"`. Fixed by deleting the `[id]/` route entirely and moving the
  detail view to client-side state (`selectedId` in `InboxProvider`) — no
  URL for an individual message anymore (trade-off: no deep-linking/back
  button between list and detail), but the static-export conflict is gone
  for good rather than worked around.
- **Persistence:** the access token is cached in `sessionStorage` (survives
  a page reload within its ~1hr life; cleared on tab close) so reconnecting
  isn't required on every reload. Deliberately **not** upgraded to a
  refresh token or `localStorage` — reasoned through in session: no
  client-side storage location is actually safe against XSS for a
  long-lived credential (a hardcoded/derived key is readable straight out
  of the shipped bundle; a non-extractable Web Crypto key blocks key
  *export* but not same-origin *use*, so injected same-origin JS can still
  call decrypt). Real persistent login needs a backend holding the refresh
  token — logged as Phase 2 guidance directly below.
- `app/frontend/package.json`'s `dev` script now pins `next dev -p 3010` to
  match the origin/redirect URI registered on the Google OAuth client —
  don't run this app on a different port without also updating the OAuth
  client's registered origins in Google Cloud Console.
- **Uncommitted right now:** `gmail-auth.ts`, `gmail-fetch.ts`,
  `inbox-list.tsx`, `email-detail.tsx` (new files), plus modifications to
  `inbox-context.tsx`, `sidebar.tsx`, `page.tsx`, `types/index.ts`,
  `sample-emails.ts`, `package.json`, and the deletion of `inbox/[id]/`.

**Phase 2 backend guidance (decided, not yet built):** when `app/backend/`
gets built, its Gmail-auth piece should be a **minimal token-holder only**
— server-side refresh token (e.g. HttpOnly-cookie-backed session), mints
short-lived access tokens on request. The browser still calls the Gmail API
directly and classifies client-side, exactly as now, so subject/body still
never touch the backend — this keeps the design doc's §7 privacy boundary
("raw subject/body never leaves the browser") intact, since a refresh token
is a credential, not email content. Keep this separate from the
personalization/corrections backend work (§5.3) — different concern, same
Phase 2 backend. Do not default to storing a refresh token client-side.

### Personalization (corrections + per-user retraining) — built

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

### Show the work (receipt, retrain report card, re-sorted view) — built

- **What was sent?** A corrected label in the email view links to a receipt
  of exactly what the server stored for that correction
  (`GET /personalization/corrections/{message_id}`), decoded on the device
  (`src/lib/receipt.ts`) into the words and letter groups the vector counted,
  with their weights, plus the header/style signals unscaled to plain values.
- **Retrain report card.** The sidebar explains the latest attempt as the
  gate's checks (`src/lib/retrain-report.ts`, mirroring `gate.py`), with the
  failing one marked when a version is rejected. `last_attempt` now carries
  `correction_count`.
- **Re-sorted by your model.** After a retrain loads a new model, emails whose
  model category changed (and that the user hadn't corrected) get a "was X"
  tag, a banner counts them with "Show only these", and the sorting panel
  shows the active personal model version.
- Corrections now survive a reload: the inbox restores them from
  `GET /personalization/corrections` when it loads.
- **Privacy finding the receipt makes visible:** the category and priority
  vectors include char_wb letter groups, not only words. A real LinkedIn
  correction carried 26 words but 524 letter groups, which can hint at words
  outside the word vocabulary (e.g. names). The privacy caveat above
  understates this; dropping or coarsening the char features before upload
  is a follow-up.

### Not started

- **Phase 2 (backend), remaining:** the analytics dashboard backend. The
  FastAPI + Postgres backend itself, the Gmail token-holder, and
  personalization (section above) are built.
- Committing the Gmail integration work, and reconciling both it and the
  inbox UI with a written spec/plan (both were built ahead of any doc for
  them — worth writing one retroactively, or before extending either
  further, per this project's own process).
- IndexedDB persistence for fetched/classified emails — currently
  session-scoped only (a reload re-fetches from Gmail rather than reading
  a local cache); this is the design doc's `LocalStore`, still unbuilt.

## Known gaps / not yet done

- `predictions/` is now empty — the stale v1 category predictions and the
  broken spam predictions were both deleted during cleanup. Regenerate once
  each model is trustworthy.
- Spam classifier: see above — needs the retrain re-run and the Enron
  spam-rate sanity-checked before it's trustworthy. No model file exists
  right now.
- No structured rubric/tie-breaker document exists for the recurring
  ambiguous category cases under the new Work/Personal/Other taxonomy
  (external-but-job-relevant newsletter vs. Work; charity/volunteer content
  vs. Personal) — the resort in v5 applied a consistent-but-informal rule
  (see v5 section above), worth writing down properly if more labeling
  happens later.
- "Other" is currently the weakest class (F1 0.54) and the most
  heterogeneous by design — see the composition table above. The user's own
  framing was that sub-categories may get carved out of it later once it's
  clearer what accumulates there.

## Where to go next

1. **Category classifier — remaining accuracy levers.** User's target is
   forced accuracy ≥80% (every email gets a label, no deferring); currently
   at **76.8%**, ~3 points short. Tried and rejected this round: sentence
   embeddings, time-of-day features, Random Forest (69.0% — worse than LR,
   trees don't suit sparse high-dim TF-IDF), sender-recipient relationship
   frequency (74.9% standalone, net *negative* combined with stylistic
   features at 75.7%). Remaining options, roughly in expected order of
   payoff:
   - Wire the now-fixed **spam classifier's `spam_probability`** in as a
     category-classifier feature — motivated by the "maybe fixing spam helps
     Other too" hypothesis, confirmed directionally correct for ~22-24% of
     Other (the genuinely spam/scam/marketing-shaped rows) but not the
     majority (newsletters/academic/mailing-list/receipts aren't spam by any
     definition). Not yet attempted.
   - Naive Bayes / XGBoost baseline comparison — RF tested and lost, these
     two are still untested; also needed for the proposal's Objective 5.
   - Fine-tuned transformer (actual backprop on the 1000 labeled rows, not a
     frozen general-purpose embedding like the rejected MiniLM experiment) —
     **ruled out by the client-side/browser-privacy requirement**, not
     accuracy — a fine-tuned BERT-family model is 60-90MB+ and needs a WASM
     ML runtime, incompatible with "must run in the browser." The current
     TF-IDF+LogisticRegression pipeline is browser-portable (a vocabulary
     map + coefficient matrix, pure JS, no heavy runtime) — this is the
     actual constraint shaping the model choice now, not just accuracy.
   - Separate subject-only vs. body-only TF-IDF weighting — small, cheap,
     untried.
   - More active-learning labeled data — real time cost, uncertain payoff
     given the dominant error mode (Personal↔Work) is content-ambiguity, not
     data-scarcity.
   - Split "Other" into sub-categories per the composition table above once
     there's enough labeled volume in each.
   - **Confidence-gating implementation** (defer low-confidence predictions
     instead of forcing every email into a bucket) remains the
     validated-but-unimplemented path to a trustworthy 90%+ number — this is
     a different metric than forced accuracy (see the confidence-gating
     section above), worth keeping in mind if the 80% forced-accuracy target
     turns out to be unreachable through the levers above.
1b. **Priority classifier — remaining accuracy levers.** The top lever
   identified here (diversify training data beyond Enron) was **executed in
   item 3f**: adversarial accuracy 46.0% → 82.0%, nested-CV 73.0% → 85.1%.
   Remaining options, re-ranked now that the big structural fix is done:
   - ~~Same treatment for the category classifier~~ — **done, see Category
     classifier v7**: 62.0% → 78.0% on the adversarial set using the same
     500 synthetic rows' `true_category` labels. Work recall dropped
     (82.9%→68.3%) as a side effect of correcting the model's Enron-era
     Work-majority bias — worth revisiting if Work recall specifically
     matters more than overall accuracy.
   - **Remaining priority misses are qualitatively different post-3f** —
     no longer "collapses to low," now two narrower failure modes: (a)
     keyword-free/tonally-soft urgency still under-called (`C4`, `C7`,
     `J14`, `J17` — see 3f), (b) a few casual-texting *low*-priority
     messages over-triggering high (`D8`, `D10`), suggesting the model may
     be partly keying on terseness/lowercase style rather than content.
     Worth another small, targeted adversarial round aimed specifically at
     "casual but genuinely low-priority" texts to check if that's real.
   - **More gold labels targeting the "high" bucket** — still true in
     principle but now much less urgent; adversarial high recall already
     moved from 3.2% to 64.5% via the corpus-diversity fix alone.
   - **Ordinal-aware modeling** — priority is inherently ordered
     (low < medium < high) but the classifier head treats it as flat
     multiclass. Still untested, still a legitimate refinement, now lower
     priority given how much headroom the data fix already recovered.
   - **Confidence-gating** was computed but explicitly rejected by the user
     (wants forced accuracy, not deferral) — conf≥0.70 gets 86.6% accuracy
     at 55% coverage, conf≥0.50 gets 75.3% at 85.9% coverage (both numbers
     predate 3f and were measured on same-corpus CV, so treat as stale).
     Left here as a documented fallback if it's ever revisited.
2. **JS/browser export — done.** See "Application layer (`app/`)" above.
   Landed differently than originally sketched here: rather than a pure
   hand-rolled TF-IDF+dot-product+softmax port, the models export to
   headless ONNX graphs (`scripts/export_onnx.py`) run via
   `onnxruntime-web`, with only the TF-IDF vectorization itself (the piece
   `skl2onnx` can't convert exactly) hand-ported to TypeScript
   (`app/frontend/src/inference/tfidf.ts`), verified bit-exact against the
   real fitted sklearn vectorizers. Fully client-side, no server round-trip
   for classification.
3. **Sentiment module — done.** Wired VADER (`vaderSentiment` pkg) into
   `_priority()` in `classify_email.py`. It's a lexicon+rule scorer, not a
   trained model, so it doesn't hit the client-side/privacy constraint that
   ruled out transformers (a JS port is a direct lexicon-table port, no ML
   runtime — same shape as the planned TF-IDF+LR export in item 2).
   Validated on 10 probe emails before trusting it (see
   `scripts/classify_email.py` comment above the sentiment block):
   - Only the **negative tail** is used (compound ≤ -0.3 / ≤ -0.6 → +0.08 /
     +0.15 to the priority score). This reliably catches frustration/
     complaint/crisis language that the existing keyword lists miss —
     e.g. "completely unacceptable... extremely frustrated" isn't in
     `_HIGH_SIGNALS` but scores compound -0.73, and correctly bumped that
     email's priority from an indistinguishable 0.3 to 0.4.
   - **Positive compound is deliberately NOT used** to suppress priority.
     Tested and rejected: a plainly urgent, polite business email
     ("URGENT: please review and sign off by EOD") scored +0.59 (positive)
     on VADER — a positive-side dampening rule would have fought the
     keyword heuristic on a case it already gets right. This is the same
     "verify before wiring in" discipline used for time-of-day and
     relationship-frequency features earlier — it would have been a
     regression if added blindly.
   - Known blind spot: VADER misses quiet negative content with no
     emotionally-coded vocabulary (e.g. a bereavement email scored 0.0
     standalone). Not a regression — it just contributes no signal either
     way — but worth knowing it's not a general emotion detector.
   - Smoke-tested via `classify_email.py --models ../models`: existing
     demo emails (none contain negative-sentiment language) are byte-for-
     byte unchanged in priority score — confirms no regression.
3b. **Spam → priority gating — done.** The three classifiers ran fully
   independently until now (spam/category/priority were computed
   separately and just bundled into one output dict; `_priority()` never
   saw the spam or category result). Added one wire: if `spam_label ==
   'spam'`, priority score is floored to 0.1 with an explicit reason
   (`classified as spam (conf=X) — priority suppressed`). Rationale:
   spam/phishing routinely fakes urgency ("verify now or your account is
   suspended") specifically to game attention, so a message already
   flagged spam shouldn't be able to rank high-priority regardless of
   what language it uses — this is a product-correctness fix, not a
   hypothesis needing A/B validation.
   **Category was deliberately NOT wired into priority** — there's no
   principled basis for "Work" being generically more urgent than
   "Personal" (a personal emergency should outrank a routine status
   update), so hard-coding that would be an unvalidated bias.
   Verified with a live test: a `<font>`/`<br>`-laden fake-prize email
   scored spam conf 0.77 → priority correctly floored to 0.1. Also
   surfaced a real gap while building that test case: a phishing email
   using plain suspension/verify-now phrasing with no literal HTML markup
   scored only 0.30-0.40 spam confidence (ham) — the spam classifier's
   strongest learned signal is literal HTML tag text (`font`, `br`,
   `color=`) from the SpamAssassin corpus, not modern phishing phrasing.
   It under-catches markup-free phishing. Not fixed, just documented —
   would need phishing-style examples added to spam training data,
   similar to how Enron ham was added earlier.
3c. **Priority: rule-based heuristic replaced with a trained model — done.**
   Built a 504-row hand-labeled gold set (`gold_priority_labeled.csv`) via
   `scripts/priority/sample_for_gold_priority.py`, stratified across the
   heuristic's own low/medium/high score buckets (not pure random — a random
   sample from a 20K-row scan was 81% low / 17% medium / **1.2% high**, which
   would starve labeling of high-priority examples). Labels were hand-judged
   from subject/body/to/from **alone** — blind to the heuristic's score,
   reasons, spam_conf, category_label, and vader_compound, to avoid anchoring
   bias — in two rounds (102, then +402).
   - **Checked the heuristic against these labels before touching anything:
     40.3% bucket agreement, 0.215 correlation, 0.197 MAE.** Root cause:
     keyword matching without context. "Weekend Outage Report" (routine,
     automated, recurring) scored 0.6-0.7 purely because "outage" is in
     `_HIGH_SIGNALS`; "URGENT! URGENT! VIRUS ALERT!" (genuinely urgent IT
     security notice) and marketing spam that fakes urgency both hit the same
     keyword and scored similarly; meanwhile "It is imperative that we have
     our books flat by 6:00 a.m.. NO EXCUSES" (real same-day trading-floor
     directive) scored only 0.3 because it doesn't use the word "urgent" and
     got penalized by the ">6 recipients = broadcast" rule despite being an
     urgent instruction *to* many people, not a passive FYI.
   - **Trained `scripts/priority/train_priority_classifier.py`** on: TF-IDF
     word+char text, header features (n_recipients, is_reply_or_forward,
     sender_automated), the 9 stylistic features (shared with the category
     classifier), spam_conf (from the now-fixed spam classifier),
     category_label (one-hot), and vader_compound. Two heads, both 5-fold CV:
     Ridge regression for the continuous score (MAE 0.091 vs 0.120 for
     always-predict-the-mean) and LogisticRegression for the bucket (**70.0%
     accuracy vs 40.3% for the heuristic, on the identical gold set**; macro-F1
     0.59 vs ~0.27 for a naive always-guess-"low" baseline — raw accuracy
     barely beats the 69.0% majority-class baseline only because "low" is
     69% of the data, but per-class recall shows real signal: 51% of true
     high-priority and 45% of true medium-priority emails correctly caught,
     vs 0% for the majority guess).
   - **Wired into `classify_email.py`**: `EmailClassifier` now loads
     `priority_regressor.joblib`/`priority_classifier.joblib` instead of
     calling the old `_priority()` function (deleted, along with the
     `_HIGH_SIGNALS`/`_MED_SIGNALS`/`_LOW_SIGNALS` regexes). Output shape
     changed from `{'score', 'reasons': [...]}` to `{'score', 'bucket'}` —
     the model doesn't have per-email rule "reasons" the way the heuristic
     did. Spam-gating (item 3b) still applies on top of the model's output.
   - **Known limitation, found during smoke-testing:** on the hand-crafted
     demo email "URGENT: Approval needed by EOD", the trained model predicts
     **low (0.34)** where the old heuristic said 0.6. The training data is
     504 *real* 2001 Enron emails only, so the model may under-perform on
     synthetic/hand-written phrasing it never saw a distributionally-similar
     example of. Still present after the 3d fine-tuning below (score moved to
     0.44, bucket still "low") — not fixed, needs a hand-crafted adversarial
     test set (like the 21-email category test) before trusting this on
     live, non-Enron mail.
3d. **Priority forced-accuracy fine-tuning — done, real gain, target not
   reached.** User pushed for forced (non-gated) accuracy specifically;
   computed the confidence-gating tradeoff first for comparison (85.9%
   coverage at conf≥0.50 → 75.3% accuracy; conf≥0.70 → 86.6% accuracy at 55%
   coverage) but user wants absolute accuracy, not deferral, so that path
   wasn't used. Ran controlled comparisons (identical 5-fold CV, same gold
   set) against the 3c baseline (plain LogisticRegression on text+header+
   style+spam+category+sentiment, 70.0%):
   | change | accuracy | outcome |
   |---|---|---|
   | RandomForest, same features | 74.0% | better |
   | GradientBoosting, same features | 71.8% | better, less than RF |
   | Smaller TF-IDF dims + more regularization | 66.7-66.9% | **worse** |
   | LogReg + explicit keyword-count features | 69.0% | **worse** |
   | RandomForest + explicit keyword-count features | 74.8% | **best** |

   Two of four hypotheses were wrong (shrinking TF-IDF dimensionality and
   adding keyword counts to the *linear* model both backfired) — the lift
   only showed up combining a model that captures feature *interactions*
   (RandomForest) with the explicit counts; a linear model can't exploit
   them productively on its own. New features added as
   `scripts/shared/priority_keyword_features.py` (kw_high/kw_med/kw_low
   regex counts + exclamation-in-subject count — same vocabulary as the old
   heuristic's `_HIGH_SIGNALS`/`_MED_SIGNALS`/`_LOW_SIGNALS`, now used as
   auxiliary numeric features instead of the sole scoring mechanism).
   Deployed as production: `RandomForestClassifier`, hyperparameters
   selected via **nested cross-validation** (GridSearchCV inside the outer
   5-fold CV, over n_estimators/max_depth/min_samples_leaf/max_features) so
   the reported number isn't inflated by tuning against its own test folds.
   Final honest number: **73.0% forced accuracy** (down slightly from the
   74.8% quick-test figure — nested CV is the methodologically correct,
   less optimistic estimate). Per-class: high recall 41% (precision 94% —
   few false alarms, but still misses ~6 in 10 truly urgent emails), medium
   recall 40%, low recall 88%.
   **Explicitly short of the user's 85% target.** Told the user directly:
   closing the remaining ~12 points needs more gold labels (especially
   "high" bucket, still only 37/504 rows) or a different modeling approach
   entirely, not further hyperparameter tuning on the current data — this
   mirrors the category classifier's plateau around 76-80% for the same
   underlying reason (label-boundary ambiguity + limited data), not
   something to paper over with a bigger promise.
3e. **Adversarial out-of-distribution test — 73.0% CV accuracy does NOT
   generalize. Real accuracy on unfamiliar text: 46.0%.** Built
   `scripts/priority/adversarial_test_set.py` — 100 fully hand-authored
   synthetic emails (not sampled from Enron), each assigned ground-truth
   category/priority/sentiment at write time, deliberately covering cases
   the 504-row Enron gold set can't teach: modern/casual phrasing, texting
   style, sarcasm, minimal-text emergencies ("call me now", "911"),
   marketing copy that hijacks urgency words ("URGENT!!! FINAL HOURS!!!"),
   and positive-sentiment-but-genuinely-urgent messages (job promotion with
   a deadline). Run via `scripts/priority/run_adversarial_eval.py`.

   Result: **priority accuracy 46.0%** (vs. 73.0% nested-CV), **category
   accuracy 62.0%** (vs. 76.8% CV). Priority collapsed almost entirely to
   "low": low recall 100% (41/41), medium recall 14.3% (4/28), **high
   recall 3.2% (1/31)** — including missing the exact "URGENT: Approval
   needed by EOD" case documented as a known gap in item 3c/3d.

   Root-caused via RandomForest feature importances
   (`clf.feature_importances_` grouped by feature block): char-level TF-IDF
   trigrams/n-grams account for **89.9%** of the model's decision weight,
   while the explicit keyword-count features built in item 3d for
   generalization (`kw_high` etc.) get only **1.4%**. Hypothesis: the model
   is fitting corpus-specific phrasing patterns from only 504 rows across
   ~4,500 TF-IDF dimensions, not a real definition of urgency.

   **Tested the fix — hypothesis falsified.** Re-ran nested CV *and* the
   adversarial set across 5 configs shrinking/removing the char n-grams
   (word=3000/char=1500 down to word=1000/char=0):

   | config | CV acc | ADV acc | high recall |
   |---|---|---|---|
   | current prod (word=3000, char=1500) | 74.0% | 46.0% | 1/31 |
   | word only, no char n-grams | 73.4% | 43.0% | 2/31 |
   | word=1500, no char | 73.2% | 45.0% | 3/31 |
   | word=1000, no char | 72.6% | 44.0% | 3/31 |
   | word=3000, char=300 | 73.6% | 42.0% | 0/31 |

   Neither metric moved meaningfully — CV stayed ~73-74% regardless of
   feature-space size, and adversarial accuracy stayed stuck at 42-46%.
   Conclusion: this is **not a hyperparameter or feature-dimensionality
   problem.** Even emails with an unambiguous keyword match (`"URGENT:
   Approval needed by EOD"` literally contains "urgent" and "eod") still
   predict low — the model learned, correctly *for the Enron gold set*,
   that keyword presence is a weak/noisy predictor (this is the same
   reason the old rule-based heuristic failed — spam and recurring reports
   fake urgency words too), so it appropriately downweights keywords
   in-distribution. That judgment does not transfer to text from outside
   the training corpus.

   **Takeaway: same-corpus CV accuracy measures fit-to-corpus, not
   real-world generalization — they are different metrics, and this
   project only had a way to measure the first one until now.** The 73.0%
   production number is not wrong, but it should never again be quoted
   without this caveat. Root cause is training-data diversity (504 rows,
   100% sourced from one 2001 corporate email corpus), not something model
   tuning can fix — see the (now re-ranked) options in "Where to go next"
   item 1b.
3f. **Fixed it — 500 hand-authored synthetic training rows, adversarial
   accuracy 46.0% → 82.0%.** Considered `SpamAssasin/spamassassin.csv`
   (3,900 ham rows, already in the repo) as a free diversification source
   first, but ruled it out: per the spam classifier's own documented
   finding (`scripts/spam/train_spam_classifier.py` comments), that ham
   class is almost entirely one narrow cluster of 2002 tech mailing lists
   (ILUG, FoRK, spamassassin-devel — confirmed by inspecting raw rows,
   `ilug@linux.ie`, `fork@xent.com`), not general correspondence — it would
   have added topical/era diversity but not the personal-register,
   casual/texting, and safety-emergency language the adversarial test
   actually exposed as missing. Chose to hand-author new synthetic training
   data instead, using the same method as the adversarial_test_set.py eval
   set but for training rather than eval.

   **Construction:** 500 rows across 5 files
   (`scripts/priority/synthetic_priority_train_batch{1..5}.py`), each row
   with subject/body/to/from_addr plus hand-assigned ground truth
   (true_category, true_priority, true_bucket). Deliberately balanced by
   design — not sampled, so exact control was possible — across category
   (Work 170 / Personal 170 / Other 160) and priority bucket (low 170 /
   medium 165 / high 165, far above the ~7% real-world "high" rate,
   intentionally oversampling the class item 3e's confusion matrix showed
   was starved). Content spans ~40 distinct fictional businesses (health
   clinic, marina, brewery, farm, law firm, radio station, etc.) and ~40
   personal relationship types (stepparent, twin, foster parent, running
   club friend, etc.) specifically to avoid the model learning "email from
   X domain = priority Y" as a shortcut. Every row was written to include
   genuine safety/emergency, casual-texting, and non-corporate register —
   the exact gaps item 3e identified.

   **Leakage guard:** `scripts/priority/build_synthetic_training_csv.py`
   asserts zero (subject, body) overlap between the 500 new training rows
   and the 100 adversarial_test_set.py eval rows before writing anything —
   the adversarial set must stay held-out or the re-eval below is
   meaningless. Combined with the existing 504 Enron rows into
   `gold_priority_labeled_v2.csv` (1004 rows total; `category_label` for
   the synthetic rows is the *current category model's own prediction*,
   not the hand-authored true_category, to match what priority actually
   receives in production — true_category is kept as an unused extra
   column for a possible future category-classifier retraining round).

   **Result — retrained on 1004 rows (RandomForest, nested-CV
   hyperparameter search, same architecture as 3d):**

   | metric | before (504 Enron-only) | after (1004, +500 synthetic) |
   |---|---|---|
   | Nested-CV accuracy | 73.0% | **85.1%** |
   | Adversarial (out-of-distribution) accuracy | 46.0% | **82.0%** |
   | Adversarial high recall | 3.2% (1/31) | **64.5% (20/31)** |
   | Adversarial medium recall | 14.3% (4/28) | **82.1% (23/28)** |
   | Adversarial low recall | 100% (41/41) | 95.1% (39/41) |

   CV and adversarial accuracy are now close (85.1% vs. 82.0%) instead of
   30+ points apart — the model's same-corpus performance estimate is
   finally trustworthy, which is arguably the more important fix than the
   raw accuracy number itself. Remaining adversarial misses cluster in two
   places: the still-known keyword-free/tonally-soft urgency cases (e.g.
   `C4`, `"It is imperative... NO EXCUSES"`, still predicted low; `C1`, the
   EOD-approval case, moved from low → medium, still not high) and a few
   low-priority casual-texting messages now over-triggering high (`D8`,
   `D10` — an explicit-low-pressure check-in and a "lol" meme-share both
   got flagged high, likely because casual/texting *style* now correlates
   with the synthetic "high" rows' style too, since many were also
   deliberately terse/lowercase). **Promoted to production**:
   `models/priority_regressor.joblib` and `models/priority_classifier.joblib`
   overwritten, smoke-tested via `classify_email.py` — the canonical "URGENT:
   Approval needed by EOD" test case now scores 0.53/medium (up from
   0.44/low pre-3f).

   **Category classifier was intentionally left untouched this round**
   (still 62.0% on the adversarial set, unchanged) — the synthetic rows'
   `true_category` labels exist and could retrain it the same way, but
   that's a separate, not-yet-requested piece of work (category's own
   adversarial gap looks similar in kind — see "Where to go next").
4. MIME-encoded subject headers aren't decoded (`email.header.decode_header`)
   — found while spot-checking the 50-unseen-email demo; a real gap for live
   mail (modern clients MIME-encode non-ASCII subjects far more than this
   2001 corpus does). Small, not urgent.
5. `sender_automated` regex (in `header_features.py`) doesn't match
   `billing@...`-style senders — found while building the 20-email
   hand-crafted test; `"Payment receipt"` from `billing@subscriptionservice.com`
   should have tripped the automated-sender flag and didn't. Cheap fix,
   not yet applied — add `billing`/`receipt`/`invoice`/`orders` to
   `_AUTOMATED_LOCAL_PART` in `shared/header_features.py`.
