"""Database-level tests: persistence, cascade deletes, constraints."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import Feedback, Inspection, Prediction
from app.db.session import get_session_local


def _session():
    return get_session_local()()


class TestPersistence:
    def test_inspection_with_prediction_roundtrip(self, fake_settings):
        s = _session()
        insp = Inspection(
            image_path="uploads/x.jpg",
            original_filename="x.jpg",
            image_width=200,
            image_height=200,
            image_sha256="a" * 64,
        )
        pred = Prediction(
            model_version="v1.0.0",
            predicted_label="crazing",
            predicted_index=0,
            confidence=0.91,
            confidence_tier="high",
            probabilities={"crazing": 0.91},
            needs_review=False,
            evidence_coverage=0.2,
            evidence_bbox={"x": 10, "y": 10, "w": 50, "h": 40},
            severity="major",
            severity_score=4,
            severity_components={"defect_risk": 3, "confidence_tier": 0, "evidence_coverage": 1},
            recommended_action="quarantine",
            latency_ms=42.0,
        )
        insp.prediction = pred
        s.add(insp)
        s.commit()

        loaded = s.get(Inspection, insp.id)
        assert loaded.prediction.predicted_label == "crazing"
        assert loaded.prediction.severity == "major"
        assert loaded.prediction.evidence_bbox["w"] == 50
        assert loaded.created_at is not None
        s.close()

    def test_feedback_unique_per_inspection(self, fake_settings):
        s = _session()
        insp = Inspection(
            image_path="uploads/y.jpg",
            original_filename="y.jpg",
            image_width=200,
            image_height=200,
            image_sha256="b" * 64,
        )
        s.add(insp)
        s.commit()
        f1 = Feedback(
            inspection_id=insp.id,
            is_correct=True,
            model_version_at_feedback="v1.0.0",
        )
        s.add(f1)
        s.commit()
        f2 = Feedback(
            inspection_id=insp.id,
            is_correct=False,
            model_version_at_feedback="v1.0.0",
        )
        s.add(f2)
        try:
            s.commit()
            assert False, "unique constraint should have raised"
        except IntegrityError:
            s.rollback()
        s.close()

    def test_cascade_delete(self, fake_settings):
        from sqlalchemy import func

        s = _session()
        insp = Inspection(
            image_path="uploads/z.jpg",
            original_filename="z.jpg",
            image_width=200,
            image_height=200,
            image_sha256="c" * 64,
        )
        insp.prediction = Prediction(
            model_version="v1.0.0",
            predicted_label="patches",
            predicted_index=2,
            confidence=0.5,
            confidence_tier="medium",
            probabilities={},
            severity="minor",
            severity_score=2,
            severity_components={},
            recommended_action="ok",
        )
        s.add(insp)
        s.commit()
        iid = insp.id
        s.delete(insp)
        s.commit()
        assert s.get(Inspection, iid) is None
        n = s.scalar(select(func.count()).select_from(Prediction))
        assert n == 0
        s.close()

    def test_feedback_defaults_pending(self, fake_settings):
        s = _session()
        insp = Inspection(
            image_path="uploads/w.jpg",
            original_filename="w.jpg",
            image_width=200,
            image_height=200,
            image_sha256="d" * 64,
        )
        s.add(insp)
        s.commit()
        f = Feedback(
            inspection_id=insp.id,
            is_correct=False,
            corrected_label="inclusion",
            model_version_at_feedback="v1.0.0",
        )
        s.add(f)
        s.commit()
        assert f.review_status == "pending"
        assert f.created_at is not None
        s.close()
