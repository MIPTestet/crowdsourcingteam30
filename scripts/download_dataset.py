#!/usr/bin/env python3
"""
Deterministic dataset downloader organized 1-folder-per-TSV split.

Reads TSV pools under yandex_tasks/{bbox,review}/ (plus instruction_examples),
downloads each split's CC0 images into its own directory under data/images/:
  - bbox_main/             (400 images from yandex_tasks/bbox/tasks_main.tsv)
  - bbox_train/            (9 images from yandex_tasks/bbox/tasks_train.tsv)
  - bbox_exam/             (12 images from yandex_tasks/bbox/tasks_exam.tsv)
  - bbox_honeypots/        (20 images from yandex_tasks/bbox/tasks_honeypots.tsv)
  - review_train/          (6 images from yandex_tasks/review/tasks_train.tsv)
  - review_exam/           (8 images from yandex_tasks/review/tasks_exam.tsv)
  - review_honeypots/      (8 images from yandex_tasks/review/tasks_honeypots.tsv)
  - instruction_examples/  (7 images from data/manifests/instruction_examples.jsonl)

When frozen manifests (data/manifests/*.jsonl) are present, verifies exact SHA-256
and pixel dimensions in addition to PIL image decode verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = ROOT_DIR / "data" / "manifests"
YANDEX_DIR = ROOT_DIR / "yandex_tasks"
IMAGES_DIR = ROOT_DIR / "data" / "images"
HEADERS = {"User-Agent": "SHAD-Butterfly-Biodiversity-Pilot/1.0"}

FALLBACK_INSTRUCTION_ITEMS = [
    {"photo_id": 727216039, "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/727216039/large.jpg"},
    {"photo_id": 703704800, "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/703704800/large.jpg"},
    {"photo_id": 704530635, "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/704530635/large.jpg"},
    {"photo_id": 760795175, "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/760795175/large.jpg"},
    {"photo_id": 58865565, "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/58865565/large.jpg"},
    {"photo_id": 763099857, "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/763099857/large.jpg"},
    {"photo_id": 762809662, "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/762809662/large.jpg"},
]


def load_metadata_index() -> dict[int, dict[str, Any]]:
    """Load optional sha256/dimension metadata from manifests when available."""
    index: dict[int, dict[str, Any]] = {}
    if not MANIFESTS_DIR.exists():
        return index
    for name in ("main_400.jsonl", "control_bbox.jsonl", "control_review.jsonl", "instruction_examples.jsonl"):
        p = MANIFESTS_DIR / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            index[int(rec["photo_id"])] = rec
    return index


def load_tsv_items(tsv_path: Path, meta_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = int(row["INPUT:photo_id"])
            url = row["INPUT:image_url"].strip()
            meta = meta_index.get(pid, {})
            items.append(
                {
                    "photo_id": pid,
                    "image_url": url,
                    "sha256": meta.get("sha256"),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                }
            )
    return items


def load_jsonl_items(jsonl_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        items.append(
            {
                "photo_id": int(rec["photo_id"]),
                "image_url": rec["image_url"].strip(),
                "sha256": rec.get("sha256"),
                "width": rec.get("width"),
                "height": rec.get("height"),
            }
        )
    return items


def _check_image_bytes(data: bytes, item: dict[str, Any]) -> None:
    pid = item["photo_id"]
    expected_sha = item.get("sha256")
    if expected_sha:
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch for photo_id={pid}: got {actual_sha}, expected {expected_sha}")

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        w, h = im.size
        exp_w, exp_h = item.get("width"), item.get("height")
        if exp_w is not None and exp_h is not None and (w, h) != (int(exp_w), int(exp_h)):
            raise ValueError(f"Dimension mismatch for photo_id={pid}: got {(w, h)}, expected {(exp_w, exp_h)}")


def download_one(item: dict[str, Any], target_dir: Path) -> tuple[int, str]:
    pid = int(item["photo_id"])
    url = item["image_url"]
    ext = url.split(".")[-1].split("?")[0].lower()
    if ext not in {"jpg", "jpeg", "png"}:
        ext = "jpg"
    out_path = target_dir / f"{pid}.{ext}"

    if out_path.exists():
        data = out_path.read_bytes()
        expected_sha = item.get("sha256")
        if expected_sha:
            if hashlib.sha256(data).hexdigest() == expected_sha:
                return pid, "cached"
        else:
            try:
                _check_image_bytes(data, item)
                return pid, "cached"
            except Exception:
                pass

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.content
    _check_image_bytes(data, item)
    out_path.write_bytes(data)
    return pid, "downloaded"


def sync_items(label: str, items: list[dict[str, Any]], target_dir: Path, workers: int) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SYNC] {label}: {len(items)} images -> {target_dir}")

    cached = 0
    downloaded = 0
    errors: list[tuple[int, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, item, target_dir): item["photo_id"] for item in items}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                _, status = fut.result()
                if status == "cached":
                    cached += 1
                else:
                    downloaded += 1
            except Exception as exc:
                errors.append((pid, str(exc)))

    print(f"[DONE] {label}: downloaded={downloaded}, cached={cached}, errors={len(errors)}")
    if errors:
        for pid, msg in errors[:10]:
            print(f"  [ERROR] photo_id={pid}: {msg}")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download iNaturalist CC0 butterfly dataset into per-TSV folders.")
    parser.add_argument(
        "--split",
        choices=["all", "main", "control", "bbox", "review", "instruction"],
        default="all",
        help="Which subset(s) to download (default: all).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=IMAGES_DIR)
    args = parser.parse_args()

    meta_index = load_metadata_index()

    tsv_jobs: list[tuple[str, str, Path, Path]] = [
        ("bbox_main", "bbox", YANDEX_DIR / "bbox" / "tasks_main.tsv", args.out_dir / "bbox_main"),
        ("bbox_train", "bbox", YANDEX_DIR / "bbox" / "tasks_train.tsv", args.out_dir / "bbox_train"),
        ("bbox_exam", "bbox", YANDEX_DIR / "bbox" / "tasks_exam.tsv", args.out_dir / "bbox_exam"),
        ("bbox_honeypots", "bbox", YANDEX_DIR / "bbox" / "tasks_honeypots.tsv", args.out_dir / "bbox_honeypots"),
        ("review_train", "review", YANDEX_DIR / "review" / "tasks_train.tsv", args.out_dir / "review_train"),
        ("review_exam", "review", YANDEX_DIR / "review" / "tasks_exam.tsv", args.out_dir / "review_exam"),
        ("review_honeypots", "review", YANDEX_DIR / "review" / "tasks_honeypots.tsv", args.out_dir / "review_honeypots"),
    ]

    for label, stage, tsv_path, target_dir in tsv_jobs:
        is_main = label == "bbox_main"
        selected = (
            args.split == "all"
            or (args.split == "main" and is_main)
            or (args.split == "control" and not is_main)
            or (args.split == "bbox" and stage == "bbox")
            or (args.split == "review" and stage == "review")
        )
        if not selected:
            continue
        if not tsv_path.exists():
            print(f"[SKIP] TSV not found: {tsv_path}")
            continue
        items = load_tsv_items(tsv_path, meta_index)
        sync_items(f"{stage}/{tsv_path.name}", items, target_dir, args.workers)

    if args.split in {"all", "instruction"}:
        instr_path = MANIFESTS_DIR / "instruction_examples.jsonl"
        instr_items = load_jsonl_items(instr_path) if instr_path.exists() else FALLBACK_INSTRUCTION_ITEMS
        sync_items("instruction_examples", instr_items, args.out_dir / "instruction_examples", args.workers)


if __name__ == "__main__":
    main()
