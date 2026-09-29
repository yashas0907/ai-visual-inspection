"""User-POV end-to-end feature test against the RUNNING stack.

Simulates a real inspector's session: uploads images of every defect type,
reviews results, submits corrections/confirmations, browses history with
filters, checks analytics, reads the model card, tries abuse cases, and
measures real response times for each interaction.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path

BASE = os.environ.get("POV_BASE", "http://127.0.0.1:8001")
DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "test"

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    (PASS if cond else FAIL).append(name)
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))


def req(method: str, path: str, data=None, headers=None, files=None):
    """Minimal HTTP client incl. multipart (no external deps)."""
    url = BASE + path
    if files:
        boundary = uuid.uuid4().hex
        fname, fpath = files
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode() + Path(fpath).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        headers = {**{"Content-Type": f"multipart/form-data; boundary={boundary}"}, **{"Content-Length": str(len(body))}}
    elif data is not None:
        body = json.dumps(data).encode()
        headers = {"Content-Type": "application/json", **{"Content-Length": str(len(body))}}
    else:
        body = None
    r = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read()
            ms = (time.perf_counter() - t0) * 1000
            ctype = resp.headers.get("Content-Type", "")
            if "image" in ctype or not raw.strip().startswith((b"{", b"[")):
                return resp.status, {"binary": len(raw)}, ms
            return resp.status, json.loads(raw), ms
    except urllib.error.HTTPError as e:
        ms = (time.perf_counter() - t0) * 1000
        raw = e.read() or b"{}"
        try:
            return e.code, json.loads(raw), ms
        except Exception:
            return e.code, {}, ms


# ---------------------------------------------------------------- 1. Health
s, b, _ = req("GET", "/api/health")
check("health endpoint ok", s == 200 and b["status"] == "ok", f"model={b.get('model_version')}")

# ------------------------------------------------- 2. Inspect all 6 classes
per_class: dict[str, dict] = {}
upload_times = []
for cls in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]:
    img = next(iter(sorted(DATA.joinpath(cls).glob("*.jpg"))))
    s, b, ms = req("POST", "/api/inspections", files=(f"{cls}_test.jpg", img))
    upload_times.append(ms)
    ok = s == 201 and b["prediction"]["predicted_label"] == cls
    check(f"upload+inspect '{cls}'", ok, f"{ms:.0f}ms, conf={b['prediction']['confidence']:.3f}, sev={b['prediction']['severity']}")
    per_class[cls] = b

avg_upload = statistics.mean(upload_times)
check("upload latency reasonable (<8s incl. Grad-CAM)", avg_upload < 8000, f"avg {avg_upload:.0f}ms across 6 uploads")

# ------------------------------------------------------- 3. Result contents
r = next(iter(per_class.values()))
p = r["prediction"]
check("result has image URL", r["image_url"].startswith("/api/storage/"))
check("result has overlay URL", r["overlay_url"] and r["overlay_url"].startswith("/api/storage/results/"))
check("probabilities complete (6 classes)", len(p["probabilities"]) == 6)
check("probabilities sum ~1.0", abs(sum(p["probabilities"].values()) - 1.0) < 0.02)
check("evidence bbox present", isinstance(p["evidence_bbox"], dict))
check("severity components breakdown", set(p["severity_components"]) == {"defect_risk", "confidence_tier", "evidence_coverage"})
check("recommended action present", len(p["recommended_action"]) > 10)
check("model version stamped", p["model_version"] == "v1.0.0")

# serve original + overlay through the storage route
s1, _, ms1 = req("GET", r["image_url"])
s2, _, ms2 = req("GET", r["overlay_url"])
check("original image served", s1 == 200, f"{ms1:.0f}ms")
check("explainability overlay served", s2 == 200, f"{ms2:.0f}ms")

# ------------------------------------------------------------ 4. Feedback
# 4a: confirm correct
iid = per_class["crazing"]["id"]
s, b, ms = req("POST", f"/api/inspections/{iid}/feedback", {"is_correct": True, "inspector_name": "QA-1"})
check("feedback: confirm correct", s == 201 and b["is_correct"] is True, f"{ms:.0f}ms")
# 4b: correct a prediction
iid2 = per_class["inclusion"]["id"]
s, b, ms = req("POST", f"/api/inspections/{iid2}/feedback", {"is_correct": False, "corrected_label": "rolled-in_scale", "notes": "looks like scale streaks", "inspector_name": "QA-2"})
check("feedback: correction with label", s == 201 and b["corrected_label"] == "rolled-in_scale", f"{ms:.0f}ms")
# 4c: duplicate rejected
s, b, _ = req("POST", f"/api/inspections/{iid2}/feedback", {"is_correct": True})
check("feedback: duplicate blocked (409)", s == 409)
# 4d: invalid correction label
s, b, _ = req("POST", f"/api/inspections/{per_class['patches']['id']}/feedback", {"is_correct": False, "corrected_label": "alien"})
check("feedback: unknown label rejected (422)", s == 422)

# ------------------------------------------------------------ 5. History
s, b, ms = req("GET", "/api/inspections?page=1&page_size=10")
check("history list", s == 200 and b["total"] >= 6, f"{ms:.0f}ms, total={b['total']}")
s, b, _ = req("GET", "/api/inspections?label=scratches")
check("history filter by label", all(i["prediction"]["predicted_label"] == "scratches" for i in b["items"]))
s, b, _ = req("GET", "/api/inspections?severity=" + per_class["crazing"]["prediction"]["severity"])
check("history filter by severity", all(i["prediction"]["severity"] == per_class["crazing"]["prediction"]["severity"] for i in b["items"]))
s, b, _ = req("GET", "/api/inspections?has_feedback=true")
check("history filter by feedback presence", all(i["feedback"] is not None for i in b["items"]))
s, b, _ = req("GET", f"/api/inspections/{iid}")
check("inspection detail with feedback", s == 200 and b["feedback"]["is_correct"] is True)

# ------------------------------------------------------------ 6. Analytics
s, b, ms = req("GET", "/api/analytics")
check("analytics reflects real rows", s == 200 and b["total_inspections"] >= 6, f"{ms:.0f}ms")
check("analytics: severity distribution sums", sum(b["severity_distribution"].values()) == b["total_inspections"])
check("analytics: label distribution sums", sum(b["label_distribution"].values()) == b["total_inspections"])
check("analytics: confidence histogram sums", sum(x["count"] for x in b["confidence_histogram"]) == b["total_inspections"])
check("analytics: correction rate computed", b["correction_rate"] is not None, f"rate={b['correction_rate']}")
check("analytics: corrections by true label", b["corrections_by_true_label"].get("rolled-in_scale", 0) >= 1)
check("analytics: mean latency reported", b["mean_latency_ms"] is not None, f"{b['mean_latency_ms']}ms")

# ------------------------------------------------------------ 7. Model info
s, b, ms = req("GET", "/api/model/info")
check("model card served", s == 200, f"{ms:.0f}ms")
check("model card: genuine metrics", b["test_metrics"].get("accuracy") == 1.0)
check("model card: confidence policy", b["confidence_policy"]["high_min"] == 0.8)
check("model card: severity rubric exposed", "defect_risk_weights" in b["severity_policy"])

# ------------------------------------------------------------ 8. Abuse cases
s, b, _ = req("POST", "/api/inspections", files=("evil.jpg", DATA / "crazing" / next(iter(sorted(DATA.joinpath('crazing').glob('*.jpg')))).name))
# ^ actually valid (same image twice is fine — just tests dedup isn't blocking legit re-inspection)
check("re-inspection of same surface allowed", s == 201)

# real abuse: text file named .jpg
tmp = Path(__file__).parent / "evil.txt.jpg"
tmp.write_text("not an image")
s, b, _ = req("POST", "/api/inspections", files=("evil.txt.jpg", tmp))
check("fake image rejected (422)", s == 422, b.get("detail", "")[:60])
tmp.unlink()

# traversal
s, _, _ = req("GET", "/api/storage/..%2F..%2F..%2FWindows%2Fwin.ini")
check("path traversal blocked", s in (400, 404), f"status={s}")
s, _, _ = req("GET", f"/api/storage/{'a'*300}.jpg")
check("missing image 404", s in (400, 404, 414), f"status={s}")
s, b, _ = req("GET", "/api/inspections/99999")
check("missing inspection 404", s == 404)

# ---------------------------------------------------------------- summary
print()
print(f"{'='*60}")
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print(" -", f)
    sys.exit(1)
print("ALL FEATURE CHECKS PASSED")
