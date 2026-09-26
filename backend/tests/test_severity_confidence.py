"""Unit tests: severity engine (pure business rules) + confidence tiering."""
from __future__ import annotations

import pytest

from app.services.severity import (
    Severity,
    SeverityInput,
    assess_severity,
    coverage_points,
    risk_points,
)

from ml.inference.predictor import ConfidenceTiers, confidence_tier


class TestConfidenceTier:
    def test_high(self):
        assert confidence_tier(0.95, ConfidenceTiers()) == "high"

    def test_medium(self):
        assert confidence_tier(0.75, ConfidenceTiers()) == "medium"

    def test_low(self):
        assert confidence_tier(0.40, ConfidenceTiers()) == "low"

    def test_boundaries(self):
        t = ConfidenceTiers(high=0.80, medium=0.60)
        assert confidence_tier(0.80, t) == "high"
        assert confidence_tier(0.799, t) == "medium"
        assert confidence_tier(0.60, t) == "medium"
        assert confidence_tier(0.599, t) == "low"

    def test_custom_tiers(self):
        t = ConfidenceTiers(high=0.90, medium=0.70)
        assert confidence_tier(0.85, t) == "medium"


class TestSeverity:
    def test_low_risk_high_conf_small_coverage_is_minor(self):
        a = assess_severity(SeverityInput("patches", "high", 0.05))
        assert a.severity is Severity.minor
        assert a.components == {"defect_risk": 1, "confidence_tier": 0, "evidence_coverage": 0}
        assert a.recommended_action

    def test_high_risk_low_conf_large_coverage_is_critical(self):
        # risk 3 + tier 2 + coverage 2 (0.5 > 0.30) = 7 -> critical
        a = assess_severity(SeverityInput("inclusion", "low", 0.5))
        assert a.severity is Severity.critical
        assert a.score == 7

    def test_mid_case_is_major(self):
        a = assess_severity(SeverityInput("scratches", "medium", 0.2))
        # risk 2 + tier 1 + coverage 1 = 4 -> major
        assert a.severity is Severity.major
        assert a.score == 4

    def test_unknown_label_defaults_to_moderate_risk(self):
        assert risk_points("mystery_defect") == 2

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            assess_severity(SeverityInput("patches", "ultra", 0.1))

    def test_negative_coverage_raises(self):
        with pytest.raises(ValueError):
            coverage_points(-0.01)

    def test_coverage_band_edges(self):
        # bands: [0,0.10] -> 0 pts, (0.10,0.30] -> 1 pt, (0.30,1.0] -> 2 pts
        # exact boundaries take the more severe band (documented choice).
        assert coverage_points(0.0) == 0
        assert coverage_points(0.10) == 1
        assert coverage_points(0.10 + 1e-9) == 1
        assert coverage_points(0.30) == 2
        assert coverage_points(0.30 + 1e-9) == 2
        assert coverage_points(1.0) == 2

    def test_action_matches_severity(self):
        for label, tier, cov in [
            ("patches", "high", 0.0),
            ("crazing", "low", 0.9),
            ("rolled-in_scale", "medium", 0.2),
        ]:
            a = assess_severity(SeverityInput(label, tier, cov))
            assert a.recommended_action
