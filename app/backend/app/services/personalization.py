"""
Personalization strategy, per the design in PROJECT_STATUS.md.

This project's own training history (spam/category/priority each needed
500-1000+ labeled rows before a retrained model generalized — see
PROJECT_STATUS.md items 3e/3f) rules out "just retrain on the user's
corrections" for anything but the most active users. So:

  - < retrain_threshold_per_category examples for a (user, label) pair:
    nearest-centroid / kNN over the stored feature vectors. Cheap, works
    from a handful of examples, doesn't overfit the way a fresh classifier
    would at this scale.
  - >= threshold: promote to a properly retrained classifier head for that
    user, reusing the same pipeline shape as scripts/*/train_*.py — just
    scoped to their corrections (optionally bootstrapped with the shared
    gold set for cold start).

Neither path is implemented yet — this module is the seam where that work
attaches. Both paths consume Correction.feature_vector rows and produce a
ModelVersion.artifact for the frontend to fetch and blend with the base
model at inference time (see src/api/client.ts, src/inference/engine.ts).
"""

from collections import defaultdict

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.correction import Correction


def corrections_by_label(db: Session, user_id, label_type: str) -> dict[str, list[Correction]]:
    rows = (
        db.query(Correction)
        .filter(Correction.user_id == user_id, Correction.label_type == label_type)
        .all()
    )
    grouped: dict[str, list[Correction]] = defaultdict(list)
    for row in rows:
        grouped[row.corrected_label].append(row)
    return grouped


def choose_strategy(example_count: int) -> str:
    return "retrained" if example_count >= settings.retrain_threshold_per_category else "centroid"


def compute_centroid(corrections: list[Correction]) -> dict[str, float]:
    """Average the numeric feature vectors for one label into a single
    centroid. TODO: extend to the sparse text_features once the TF-IDF
    vocabulary is finalized by the ONNX export step — averaging sparse term
    weights needs a shared vocabulary to average over."""
    if not corrections:
        return {}
    names = corrections[0].feature_vector["numeric_feature_names"]
    sums = [0.0] * len(names)
    for c in corrections:
        for i, v in enumerate(c.feature_vector["numeric_features"]):
            sums[i] += v
    return {name: total / len(corrections) for name, total in zip(names, sums)}


def build_model_version_artifact(db: Session, user_id, label_type: str) -> list[dict]:
    """Returns one artifact entry per label: strategy + representation.
    Caller (an API route or scheduled job) wraps this into a ModelVersion row."""
    grouped = corrections_by_label(db, user_id, label_type)
    artifacts = []
    for label, corrections in grouped.items():
        strategy = choose_strategy(len(corrections))
        if strategy == "centroid":
            artifacts.append(
                {
                    "label": label,
                    "strategy": strategy,
                    "example_count": len(corrections),
                    "centroid": compute_centroid(corrections),
                }
            )
        else:
            # TODO: kick off an actual retrain job (reuse scripts/*/train_*.py
            # pipeline shape) rather than falling back to centroid silently.
            artifacts.append(
                {
                    "label": label,
                    "strategy": "centroid",  # placeholder until retrain job exists
                    "example_count": len(corrections),
                    "centroid": compute_centroid(corrections),
                    "note": "retrain not yet implemented, using centroid despite crossing threshold",
                }
            )
    return artifacts
