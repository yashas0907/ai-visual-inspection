"""Approved human feedback -> training-data export (human-in-the-loop gate).

This script exports *approved* inspector feedback into a manifest so it can
be audited before being merged into the training set. It is deliberately
**not** wired into any automatic retraining: a human maintainer decides what
enters the next training run.

Usage:
    PYTHONPATH=. python scripts/export_feedback.py --out data/feedback_export

The exported manifest maps each corrected inspection to its stored image and
the human-provided label, ready for manual dataset curation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.db.session import get_session_local  # noqa: E402
from app.db.models import Feedback, Inspection  # noqa: E402
from sqlalchemy import select  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/feedback_export")
    parser.add_argument("--status", default="approved", choices=["pending", "approved", "rejected", "all"])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    session = get_session_local()()
    q = select(Feedback, Inspection).join(Inspection, Feedback.inspection_id == Inspection.id)
    if args.status != "all":
        q = q.where(Feedback.review_status == args.status)
    rows = session.execute(q).all()

    manifest = []
    for fb, insp in rows:
        label = fb.corrected_label if not fb.is_correct else (insp.prediction.predicted_label if insp.prediction else None)
        if label is None:
            continue
        entry = {
            "feedback_id": fb.id,
            "inspection_id": insp.id,
            "image_path": str(Path("backend/storage") / insp.image_path),
            "human_label": label,
            "model_label": insp.prediction.predicted_label if insp.prediction else None,
            "model_version": fb.model_version_at_feedback,
            "review_status": fb.review_status,
            "notes": fb.notes,
        }
        manifest.append(entry)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported {len(manifest)} feedback rows ({args.status}) to {out/'manifest.json'}")


if __name__ == "__main__":
    main()
