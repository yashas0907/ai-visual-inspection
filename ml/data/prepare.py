"""Dataset acquisition and split creation.

Downloads the NEU surface defect classification data (1440 images, 6 classes)
from the verified Hugging Face mirror and writes an ``ImageFolder``-style
layout under ``data/processed`` with deterministic train/val/test splits.

The mirror is not the official NEU-DET distribution channel (which lives on
IEEE DataPort behind an account). The upstream data originates from the
published NEU Surface Defects Database. See docs/dataset.md for provenance.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from ml.config import DataConfig, set_global_seed

SPLIT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_PARQUET = Path(__file__).resolve().parents[2] / "data" / "raw" / "neu_det_caption.parquet"
MANIFEST = SPLIT_DIR / "manifest.json"

USER_AGENT = "ai-visual-inspection/1.0 (portfolio project; educational use)"


def _download_parquet(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:  # noqa: S310 - fixed https URL
        shutil.copyfileobj(resp, fh)


def _extract_images(parquet_path: Path, tmp_dir: Path) -> list[dict[str, Any]]:
    """Read the parquet file and write each image to tmp_dir/<class>/<filename>.

    Uses pyarrow directly to avoid a hard dependency on ``datasets``.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet_path, columns=["image", "filename", "label_str"])
    records: list[dict[str, Any]] = []
    images_col = table.column("image").to_pylist()
    filenames = table.column("filename").to_pylist()
    labels = table.column("label_str").to_pylist()
    for img_struct, name, label in zip(images_col, filenames, labels):
        raw = img_struct["bytes"]
        digest = hashlib.sha1(raw).hexdigest()[:10]
        class_dir = tmp_dir / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{Path(name).stem}_{digest}.jpg"
        with Image.open(io.BytesIO(raw)) as im:
            im.convert("L").save(out_path, "JPEG", quality=95)
        records.append(
            {
                "path": str(out_path.relative_to(tmp_dir)).replace("\\", "/"),
                "label": str(label),
                "sha1": digest,
                "orig_filename": name,
            }
        )
    return records


def _train_val_test_split(records: list[dict[str, Any]], cfg: DataConfig) -> dict[str, list[dict[str, Any]]]:
    """Deterministic class-stratified split (train/val/test)."""
    import random as _random

    rng = _random.Random(cfg.seed)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_class.setdefault(rec["label"], []).append(rec)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for label, items in sorted(by_class.items()):
        items = sorted(items, key=lambda r: r["sha1"])  # order independent of file system
        rng.shuffle(items)
        n = len(items)
        n_test = round(n * cfg.test_split)
        n_val = round(n * cfg.val_split)
        splits["test"].extend(items[:n_test])
        splits["val"].extend(items[n_test : n_test + n_val])
        splits["train"].extend(items[n_test + n_val :])
    return splits


def prepare_dataset(cfg: DataConfig, force: bool = False) -> Path:
    """Download + materialize the dataset splits. Returns the processed dir."""
    if MANIFEST.exists() and not force:
        return SPLIT_DIR

    set_global_seed(cfg.seed)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = SPLIT_DIR.parent / "_tmp_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    _download_parquet(cfg.source_url, RAW_PARQUET)
    records = _extract_images(RAW_PARQUET, tmp_dir)
    splits = _train_val_test_split(records, cfg)

    # Move files into final split layout.
    for split, items in splits.items():
        for rec in items:
            src = tmp_dir / rec["path"]
            dst = SPLIT_DIR / split / rec["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    shutil.rmtree(tmp_dir)
    counts = {
        split: {label: sum(1 for r in items if r["label"] == label) for label in sorted({r["label"] for r in items})}
        for split, items in splits.items()
    }
    manifest = {
        "dataset_name": cfg.dataset_name,
        "source_url": cfg.source_url,
        "classes": list(cfg.classes),
        "total_images": len(records),
        "split_counts": counts,
        "seed": cfg.seed,
        "image_size_note": "originals are 200x200 grayscale; training resizes to "
        f"{cfg.image_size}x{cfg.image_size}",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (SPLIT_DIR / "classes.txt").write_text("\n".join(cfg.classes) + "\n", encoding="utf-8")
    return SPLIT_DIR


if __name__ == "__main__":
    import argparse

    from ml.config import load_config

    parser = argparse.ArgumentParser(description="Prepare the NEU defect dataset splits")
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "configs" / "train_v1.yaml")
    parser.add_argument("--force", action="store_true", help="Re-extract even if manifest exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = prepare_dataset(cfg.data, force=args.force)
    print(f"Dataset ready at {out}")
    print((out / "manifest.json").read_text(encoding="utf-8"))
