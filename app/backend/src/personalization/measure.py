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
