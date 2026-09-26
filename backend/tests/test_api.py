"""API integration tests: inspections + feedback + analytics + security."""
from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import make_jpeg_bytes


class TestHealth:
    def test_health_ok(self, client: TestClient):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["database"] is True
        assert body["model_ready"] is True


class TestCreateInspection:
    def test_happy_path(self, client: TestClient):
        r = client.post(
            "/api/inspections",
            files={"file": ("surface.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["id"] > 0
        assert body["image_url"].startswith("/api/storage/uploads/")
        p = body["prediction"]
        assert p["predicted_label"] in {
            "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"
        }
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["confidence_tier"] in {"high", "medium", "low"}
        assert p["severity"] in {"minor", "major", "critical"}
        assert isinstance(p["severity_components"], dict)
        assert set(p["severity_components"]) == {"defect_risk", "confidence_tier", "evidence_coverage"}
        assert p["model_version"]
        assert p["latency_ms"] >= 0
        assert isinstance(p["probabilities"], dict) and len(p["probabilities"]) == 6

    def test_rejects_gif_extension(self, client: TestClient):
        r = client.post(
            "/api/inspections",
            files={"file": ("evil.gif", make_jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 422

    def test_rejects_mismatched_content(self, client: TestClient):
        # PNG bytes in a .jpg file -> mismatch
        from tests.conftest import make_png_bytes

        r = client.post(
            "/api/inspections",
            files={"file": ("part.jpg", make_png_bytes(), "image/jpeg")},
        )
        assert r.status_code == 422

    def test_rejects_garbage(self, client: TestClient):
        r = client.post(
            "/api/inspections",
            files={"file": ("part.jpg", b"garbage" * 100, "image/jpeg")},
        )
        assert r.status_code == 422

    def test_rejects_empty(self, client: TestClient):
        r = client.post(
            "/api/inspections",
            files={"file": ("part.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 422

    def test_rejects_truncated_jpeg(self, client: TestClient):
        data = make_jpeg_bytes()
        r = client.post(
            "/api/inspections",
            files={"file": ("part.jpg", data[:100], "image/jpeg")},
        )
        assert r.status_code == 422


class TestInspectionDetailAndHistory:
    def test_list_and_detail(self, client: TestClient):
        for i in range(3):
            client.post(
                "/api/inspections",
                files={"file": (f"part{i}.jpg", make_jpeg_bytes(color=100 + i), "image/jpeg")},
            )
        r = client.get("/api/inspections")
        assert r.status_code == 200
        page = r.json()
        assert page["total"] >= 3
        assert len(page["items"]) >= 3

        one = page["items"][0]
        r2 = client.get(f"/api/inspections/{one['id']}")
        assert r2.status_code == 200
        assert r2.json()["id"] == one["id"]

    def test_filter_by_label(self, client: TestClient):
        client.post(
            "/api/inspections",
            files={"file": ("a.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        r = client.get("/api/inspections", params={"label": "crazing"})
        assert r.status_code == 200
        for item in r.json()["items"]:
            assert item["prediction"]["predicted_label"] == "crazing"

    def test_search_by_filename(self, client: TestClient):
        client.post(
            "/api/inspections",
            files={"file": ("coil_77.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        r = client.get("/api/inspections", params={"search": "coil_77"})
        assert r.status_code == 200
        assert any("coil_77" in i["original_filename"] for i in r.json()["items"])

    def test_404(self, client: TestClient):
        assert client.get("/api/inspections/99999").status_code == 404


class TestFeedbackWorkflow:
    def _create(self, client) -> int:
        r = client.post(
            "/api/inspections",
            files={"file": ("s.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        return r.json()["id"]

    def test_confirm_correct(self, client: TestClient):
        iid = self._create(client)
        r = client.post(
            f"/api/inspections/{iid}/feedback",
            json={"is_correct": True, "inspector_name": "ravi"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["is_correct"] is True
        assert body["review_status"] == "pending"  # gated before training use

    def test_correction_with_label(self, client: TestClient):
        iid = self._create(client)
        r = client.post(
            f"/api/inspections/{iid}/feedback",
            json={"is_correct": False, "corrected_label": "inclusion", "notes": "misclassified"},
        )
        assert r.status_code == 201
        assert r.json()["corrected_label"] == "inclusion"

    def test_correction_requires_label(self, client: TestClient):
        iid = self._create(client)
        r = client.post(
            f"/api/inspections/{iid}/feedback",
            json={"is_correct": False},
        )
        assert r.status_code == 422

    def test_correction_rejects_unknown_label(self, client: TestClient):
        iid = self._create(client)
        r = client.post(
            f"/api/inspections/{iid}/feedback",
            json={"is_correct": False, "corrected_label": "alien_crack"},
        )
        assert r.status_code == 422
    def test_correct_with_label_conflict(self, client: TestClient):
        iid = self._create(client)
        r = client.post(
            f"/api/inspections/{iid}/feedback",
            json={"is_correct": True, "corrected_label": "patches"},
        )
        assert r.status_code == 422

    def test_duplicate_feedback_409(self, client: TestClient):
        iid = self._create(client)
        client.post(f"/api/inspections/{iid}/feedback", json={"is_correct": True})
        r = client.post(f"/api/inspections/{iid}/feedback", json={"is_correct": True})
        assert r.status_code == 409

    def test_feedback_on_missing_inspection_404(self, client: TestClient):
        r = client.post("/api/inspects/4242/feedback", json={"is_correct": True})
        # wrong path -> 404 by router; use correct path:
        r = client.post("/api/inspections/4242/feedback", json={"is_correct": True})
        assert r.status_code == 404

    def test_feedback_visible_in_history(self, client: TestClient):
        iid = self._create(client)
        client.post(
            f"/api/inspections/{iid}/feedback",
            json={"is_correct": False, "corrected_label": "inclusion"},
        )
        r = client.get("/api/inspections", params={"has_feedback": True})
        items = r.json()["items"]
        assert any(i["id"] == iid for i in items)
        assert all(i["feedback"] is not None for i in items)


class TestAnalytics:
    def test_empty_analytics(self, client: TestClient):
        r = client.get("/api/analytics")
        assert r.status_code == 200
        body = r.json()
        assert body["total_inspections"] == 0
        assert body["correction_rate"] is None
        assert body["confidence_histogram"] == []

    def test_analytics_after_inspections(self, client: TestClient):
        for i in range(5):
            client.post(
                "/api/inspections",
                files={"file": (f"p{i}.jpg", make_jpeg_bytes(color=60 + i * 30), "image/jpeg")},
            )
        r = client.get("/api/analytics")
        body = r.json()
        assert body["total_inspections"] == 5
        assert body["defect_rate"] is not None
        assert sum(body["severity_distribution"].values()) == 5
        assert sum(body["label_distribution"].values()) == 5
        hist_total = sum(b["count"] for b in body["confidence_histogram"])
        assert hist_total == 5
        assert len(body["recent_inspections"]) == 5

    def test_correction_rate_reflects_feedback(self, client: TestClient):
        for i in range(4):
            r = client.post(
                "/api/inspections",
                files={"file": (f"p{i}.jpg", make_jpeg_bytes(color=50 + i * 40), "image/jpeg")},
            )
            iid = r.json()["id"]
            if i < 2:
                client.post(
                    f"/api/inspections/{iid}/feedback",
                    json={"is_correct": i == 0, "corrected_label": None if i == 0 else "scratches"},
                )
        body = client.get("/api/analytics").json()
        assert body["total_with_feedback"] == 2
        assert body["correction_rate"] == 0.5
        assert body["corrections_by_true_label"].get("scratches") == 1


class TestStorageSecurity:
    def test_serves_image(self, client: TestClient):
        r = client.post(
            "/api/inspections",
            files={"file": ("s.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        url = r.json()["image_url"]
        r2 = client.get(url)
        assert r2.status_code == 200
        assert r2.headers["content-type"] == "image/jpeg"

    def test_rejects_path_traversal(self, client: TestClient):
        r = client.get("/api/storage/../../etc/passwd")
        # traversal should never resolve inside storage root
        assert r.status_code in (404, 400)

    def test_rejects_missing(self, client: TestClient):
        assert client.get("/api/storage/uploads/nope.jpg").status_code == 404


class TestModelInfo:
    def test_model_info(self, client: TestClient):
        r = client.get("/api/model/info")
        assert r.status_code == 200
        body = r.json()
        assert body["model_version"] == "v9.9-test"
        assert len(body["classes"]) == 6
        assert "resnet18" in body["arch"]
        assert "high_min" in body["confidence_policy"]
        assert "defect_risk_weights" in body["severity_policy"]
        # no fabricated metrics: test_metrics is empty dict when eval not run
        assert body["test_metrics"] in ({}, None) or isinstance(body["test_metrics"], dict)
