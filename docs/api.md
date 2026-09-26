# API Reference (summary)

Interactive docs: `http://localhost:8000/api/docs` (OpenAPI/Swagger generated).

| Method | Path | Purpose | Status codes |
|---|---|---|---|
| POST | `/api/inspections` | Upload image (multipart `file`) → full inspection | 201, 413, 422, 503 |
| GET | `/api/inspections` | Paginated history; filters: `label`, `severity`, `tier`, `has_feedback`, `search`, `page`, `page_size` | 200 |
| GET | `/api/inspections/{id}` | Single inspection with prediction + feedback | 200, 404 |
| POST | `/api/inspections/{id}/feedback` | Inspector verdict `{is_correct, corrected_label?, notes?, inspector_name?}` | 201, 404, 409, 422 |
| GET | `/api/analytics` | Aggregates computed from stored rows only | 200 |
| GET | `/api/model/info` | Active model card (classes, metrics, policies) | 200 |
| GET | `/api/health` | DB + model readiness | 200 |
| GET | `/api/storage/{relpath}` | Serve stored original/overlay images | 200, 404 |

## Response shapes (abridged)

```jsonc
// GET /api/inspections/42
{
  "id": 42, "created_at": "2026-09-02T10:00:00Z",
  "original_filename": "coil_a12.jpg", "image_width": 200, "image_height": 200,
  "image_url": "/api/storage/uploads/<hash>.jpg",
  "overlay_url": "/api/storage/results/<hash>_overlay.jpg",
  "prediction": {
    "model_version": "v1.0.0",
    "predicted_label": "inclusion", "predicted_index": 1,
    "confidence": 0.93, "confidence_tier": "high",
    "probabilities": {"crazing": 0.01, "inclusion": 0.93, "...": 0},
    "needs_review": false,
    "evidence_coverage": 0.14,
    "evidence_bbox": {"x": 40, "y": 60, "w": 90, "h": 70},
    "severity": "major", "severity_score": 4,
    "severity_components": {"defect_risk": 3, "confidence_tier": 0, "evidence_coverage": 1},
    "recommended_action": "Quarantine part; schedule rework or regrade.",
    "latency_ms": 38.2
  },
  "feedback": null
}
```

## Error contract

All errors share one envelope:

```json
{ "error": "InvalidImageError", "detail": "file content does not match its extension" }
```

- `413 ImageTooLargeError` — upload exceeds `VI_MAX_UPLOAD_MB` or pixel limit
- `422 InvalidImageError / InvalidFeedbackError` — rejected upload/feedback
- `409 FeedbackAlreadyExistsError` — one verdict per inspection
- `404 InspectionNotFoundError` / storage miss
- `503 ModelNotReadyError` — checkpoint missing (run training first)
- `500 InternalServerError` — unexpected (logged with traceback)

## Security notes

- Upload validation: size cap → extension allow-list → magic bytes →
  extension/format agreement → PIL `verify()` → decode + **re-encode** to a
  canonical grayscale JPEG (defangs polyglot/pixel-bomb-ish files; dimension
  cap blocks decompression bombs).
- Stored filenames are server-generated (`<sha-prefix>-<uuid>.jpg`);
  original names are kept only as display metadata (truncated, never used in
  paths) → path traversal impossible by construction.
- Storage route resolves and requires the final path to remain inside the
  storage root.
- Secrets/env: everything flows through `VI_*` variables; `.env` is gitignored;
  `.env.example` documents every knob. No credentials in code.
