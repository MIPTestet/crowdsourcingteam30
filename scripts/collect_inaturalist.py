#!/usr/bin/env python3
"""
Step 1: Collect CC0 candidate images from the iNaturalist API across 7 stratified groups,
verify HTTP 200 direct S3 URLs, resolution, file size, and EXIF orientation,
deduplicate by photo_id, observation_id, and perceptual dHash,
and render numbered contact sheets for visual inspection.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

API_BASE = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "SHAD-Butterfly-Biodiversity-Pilot/1.0"}

ROOT_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_DIR = ROOT_DIR / "data" / "candidates"
CONTACT_SHEETS_DIR = ROOT_DIR / "previews" / "contact_sheets"

MIN_MAX_SIDE = 640
MIN_SHORT_SIDE = 420
MAX_FILE_BYTES = 2_500_000
MAX_DHASH_DIST = 6
MAX_PER_OBSERVER_PER_GROUP = 2
MAX_PER_SPECIES_PER_GROUP = 8

TAXON_IDS = {
    "Papilionoidea": 47224,
    "Lepidoptera": 47157,
    "Lycaenidae": 47923,
    "Hesperiidae": 47653,
    "Sphingidae": 47213,
    "Noctuidae": 48670,
    "Geometridae": 49530,
    "Odonata": 47792,
    "Syrphidae": 49995,
    "Anthophila": 630955,
    "Vespidae": 52747,
    "Angiospermae": 47125,
}


def compute_dhash(img: Image.Image) -> int:
    """Compute 64-bit difference hash (dHash) on RGB image."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for val in diff.flatten():
        bits = (bits << 1) | int(bool(val))
    return bits


def hamming_dist(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lookup_taxon_id(session: requests.Session, name: str) -> int:
    if name in TAXON_IDS:
        return TAXON_IDS[name]
    time.sleep(1.0)
    resp = session.get(f"{API_BASE}/taxa", params={"q": name, "per_page": 5}, timeout=20)
    resp.raise_for_status()
    for t in resp.json().get("results", []):
        if t.get("name", "").lower() == name.lower():
            TAXON_IDS[name] = t["id"]
            return t["id"]
    raise ValueError(f"Taxon not found: {name}")


def build_candidate_url(photo: dict[str, Any]) -> tuple[str, str]:
    raw_url = photo.get("url", "")
    pid = photo["id"]
    ext = raw_url.split(".")[-1].split("?")[0].lower() or "jpg"
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        ext = "jpg"
    if "/square." in raw_url:
        large_url = raw_url.replace("/square.", "/large.")
        orig_url = raw_url.replace("/square.", "/original.")
    else:
        large_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/large.{ext}"
        orig_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/original.{ext}"
    return large_url, orig_url


def fetch_observations(
    session: requests.Session,
    params: dict[str, Any],
    start_page: int = 1,
    max_pages: int = 2,
    per_page: int = 200,
) -> list[dict[str, Any]]:
    results = []
    for offset in range(max_pages):
        page = start_page + offset
        q = {
            "photos": "true",
            "photo_license": "cc0",
            "per_page": per_page,
            "page": page,
            **params,
        }
        time.sleep(1.05)
        resp = session.get(f"{API_BASE}/observations", params=q, timeout=30)
        if resp.status_code == 429:
            print("[WARN] 429 rate limit; sleeping 15s...")
            time.sleep(15)
            resp = session.get(f"{API_BASE}/observations", params=q, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("results", [])
        if not batch:
            break
        results.extend(batch)
        if len(batch) < per_page:
            break
    return results


def validate_and_download_photo(url: str) -> tuple[bytes, Image.Image, int, int, int | None] | None:
    """Download image bytes and validate HTTP 200, size, PIL decode, resolution, and EXIF."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        if resp.status_code != 200:
            return None
        data = resp.content
        if len(data) == 0 or len(data) > MAX_FILE_BYTES:
            return None
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            exif = im.getexif()
            orientation = exif.get(274) if exif else None
            if orientation not in (None, 1):
                return None
            w, h = im.size
            if max(w, h) < MIN_MAX_SIDE or min(w, h) < MIN_SHORT_SIDE:
                return None
            rgb = im.convert("RGB")
        return data, rgb, w, h, orientation
    except Exception:
        return None


def try_fetch_obs_photo(obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str, bytes, Image.Image, int, int, int | None] | None:
    for photo in obs.get("photos") or []:
        if photo.get("license_code") != "cc0":
            continue
        large_url, orig_url = build_candidate_url(photo)
        res = validate_and_download_photo(large_url)
        chosen_url, chosen_variant = large_url, "large"
        if res is None:
            res = validate_and_download_photo(orig_url)
            chosen_url, chosen_variant = orig_url, "original"
        if res is not None:
            data, rgb, w, h, orientation = res
            return obs, photo, chosen_url, chosen_variant, data, rgb, w, h, orientation
    return None


def collect_group(
    session: requests.Session,
    group_name: str,
    target_count: int,
    query_specs: list[dict[str, Any]],
    seen_photo_ids: set[int],
    seen_obs_ids: set[int],
    seen_hashes: list[int],
    exclude_taxon_ids: set[int] | None = None,
    max_per_species: int = MAX_PER_SPECIES_PER_GROUP,
) -> list[dict[str, Any]]:
    group_dir = CANDIDATES_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    collected: list[dict[str, Any]] = []
    observer_counts: Counter[str] = Counter()
    species_counts: Counter[int] = Counter()

    for raw_spec in query_specs:
        if len(collected) >= target_count:
            break
        spec = dict(raw_spec)
        max_pages = spec.pop("_max_pages", 2)
        start_page = spec.pop("_start_page", 1)
        sub_take = spec.pop("_take", target_count)
        sub_tag = spec.pop("_tag", group_name)

        print(f"[{group_name}:{sub_tag}] Querying {spec} (pages={start_page}..{start_page+max_pages-1}, take={sub_take})...")
        observations = fetch_observations(session, spec, start_page=start_page, max_pages=max_pages)
        print(f"[{group_name}:{sub_tag}]   Fetched {len(observations)} raw observations")

        # Pre-filter metadata cheaply before concurrent S3 downloads
        eligible_obs: list[dict[str, Any]] = []
        pre_obs_cnt: Counter[str] = Counter()
        pre_sp_cnt: Counter[int] = Counter()
        for obs in observations:
            obs_id = obs.get("id")
            if not obs_id or obs_id in seen_obs_ids:
                continue
            taxon = obs.get("taxon") or {}
            taxon_id = taxon.get("id")
            ancestors = set(taxon.get("ancestor_ids") or [])
            if exclude_taxon_ids and (taxon_id in exclude_taxon_ids or (ancestors & exclude_taxon_ids)):
                continue
            user = obs.get("user") or {}
            user_login = str(user.get("login") or user.get("id") or "unknown")
            if observer_counts[user_login] + pre_obs_cnt[user_login] >= MAX_PER_OBSERVER_PER_GROUP:
                continue
            if taxon_id and species_counts[taxon_id] + pre_sp_cnt[taxon_id] >= max_per_species:
                continue
            cc0_photos = [p for p in (obs.get("photos") or []) if p.get("license_code") == "cc0" and p.get("id") not in seen_photo_ids]
            if not cc0_photos:
                continue
            eligible_obs.append(obs)
            pre_obs_cnt[user_login] += 1
            if taxon_id:
                pre_sp_cnt[taxon_id] += 1
            if len(eligible_obs) >= min(len(observations), sub_take * 2 + 20):
                break

        sub_collected = 0
        batch_size = 24
        for start in range(0, len(eligible_obs), batch_size):
            if len(collected) >= target_count or sub_collected >= sub_take:
                break
            chunk = eligible_obs[start : start + batch_size]
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(try_fetch_obs_photo, obs) for obs in chunk]
                for fut in as_completed(futures):
                    if len(collected) >= target_count or sub_collected >= sub_take:
                        break
                    res = fut.result()
                    if res is None:
                        continue
                    obs, photo, chosen_url, chosen_variant, data, rgb, w, h, orientation = res
                    pid = photo["id"]
                    obs_id = obs["id"]
                    if pid in seen_photo_ids or obs_id in seen_obs_ids:
                        continue

                    dhash = compute_dhash(rgb)
                    if any(hamming_dist(dhash, prev) <= MAX_DHASH_DIST for prev in seen_hashes):
                        continue

                    taxon = obs.get("taxon") or {}
                    taxon_id = taxon.get("id")
                    ancestors = list(taxon.get("ancestor_ids") or [])
                    user = obs.get("user") or {}
                    user_login = str(user.get("login") or user.get("id") or "unknown")

                    ext = chosen_url.split(".")[-1].split("?")[0].lower()
                    if ext not in {"jpg", "jpeg", "png"}:
                        ext = "jpg"
                    local_rel = f"data/candidates/{group_name}/{pid}.{ext}"
                    (ROOT_DIR / local_rel).write_bytes(data)

                    record = {
                        "photo_id": pid,
                        "observation_id": obs_id,
                        "observation_url": f"https://www.inaturalist.org/observations/{obs_id}",
                        "image_url": chosen_url,
                        "url_variant": chosen_variant,
                        "license_code": "cc0",
                        "attribution": photo.get("attribution") or "",
                        "observer_login": user_login,
                        "taxon_id": taxon_id,
                        "taxon_name": taxon.get("name") or "",
                        "common_name": taxon.get("preferred_common_name") or "",
                        "ancestor_ids": ancestors,
                        "quality_grade": obs.get("quality_grade") or "",
                        "group_candidate": group_name,
                        "sub_tag": sub_tag,
                        "width": w,
                        "height": h,
                        "file_bytes": len(data),
                        "exif_orientation": orientation,
                        "sha256": sha256_bytes(data),
                        "dhash_hex": f"{dhash:016x}",
                        "local_rel_path": local_rel,
                        "description": (obs.get("description") or "")[:200],
                    }
                    collected.append(record)
                    sub_collected += 1
                    seen_photo_ids.add(pid)
                    seen_obs_ids.add(obs_id)
                    seen_hashes.append(dhash)
                    observer_counts[user_login] += 1
                    if taxon_id:
                        species_counts[taxon_id] += 1

        print(f"[{group_name}:{sub_tag}]   Added {sub_collected} (group total {len(collected)}/{target_count})")

    print(f"[{group_name}] FINISHED {len(collected)} / {target_count} unique verified CC0 candidates.")
    return collected


def render_contact_sheets(records: list[dict[str, Any]], group_name: str, per_sheet: int = 20) -> None:
    CONTACT_SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    cols, rows = 5, 4
    cell_w, cell_h = 340, 270
    pad = 8
    header_h = 26

    font = ImageFont.load_default()
    num_sheets = math.ceil(len(records) / per_sheet)
    for s_idx in range(num_sheets):
        chunk = records[s_idx * per_sheet : (s_idx + 1) * per_sheet]
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 24, 28))
        draw = ImageDraw.Draw(sheet)

        for i, rec in enumerate(chunk):
            r = i // cols
            c = i % cols
            x0, y0 = c * cell_w, r * cell_h
            img_path = ROOT_DIR / rec["local_rel_path"]
            try:
                with Image.open(img_path) as im:
                    im_rgb = im.convert("RGB")
                    im_rgb.thumbnail((cell_w - 2 * pad, cell_h - header_h - 2 * pad), Image.Resampling.LANCZOS)
                    ix = x0 + (cell_w - im_rgb.width) // 2
                    iy = y0 + header_h + (cell_h - header_h - im_rgb.height) // 2
                    sheet.paste(im_rgb, (ix, iy))
            except Exception:
                pass
            global_idx = s_idx * per_sheet + i
            label = f"#{global_idx:03d} {rec['photo_id']} {rec['taxon_name'][:17]}"
            draw.text((x0 + pad, y0 + 5), label, fill=(235, 240, 245), font=font)
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(60, 65, 75))

        out_path = CONTACT_SHEETS_DIR / f"{group_name}_sheet_{s_idx + 1:02d}.jpg"
        sheet.save(out_path, quality=88)


def main() -> None:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    saturniidae_id = lookup_taxon_id(session, "Saturniidae")
    erebidae_id = lookup_taxon_id(session, "Erebidae")
    neuroptera_id = lookup_taxon_id(session, "Neuroptera")
    cetoniinae_id = lookup_taxon_id(session, "Cetoniinae")

    seen_photo_ids: set[int] = set()
    seen_obs_ids: set[int] = set()
    seen_hashes: list[int] = []
    all_records: list[dict[str, Any]] = []

    group_configs = [
        (
            "butterfly_standard",
            180,
            [
                {"_tag": "papilionoidea_votes", "taxon_id": TAXON_IDS["Papilionoidea"], "term_id": 1, "term_value_id": 2, "quality_grade": "research", "order_by": "votes", "_start_page": 1, "_max_pages": 2, "_take": 120},
                {"_tag": "papilionoidea_recent", "taxon_id": TAXON_IDS["Papilionoidea"], "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_start_page": 3, "_max_pages": 2, "_take": 60},
            ],
            None,
            MAX_PER_SPECIES_PER_GROUP,
        ),
        (
            "moth_adult",
            110,
            [
                {"_tag": "sphingidae", "taxon_id": TAXON_IDS["Sphingidae"], "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 22},
                {"_tag": "noctuidae", "taxon_id": TAXON_IDS["Noctuidae"], "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 22},
                {"_tag": "saturniidae", "taxon_id": saturniidae_id, "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 22},
                {"_tag": "erebidae", "taxon_id": erebidae_id, "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 22},
                {"_tag": "other_moths", "taxon_id": TAXON_IDS["Lepidoptera"], "without_taxon_id": TAXON_IDS["Papilionoidea"], "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 30},
            ],
            {TAXON_IDS["Papilionoidea"]},
            6,
        ),
        (
            "multi_or_occluded",
            120,
            [
                {"_tag": "mating", "taxon_id": TAXON_IDS["Papilionoidea"], "q": "mating", "quality_grade": "research", "_max_pages": 2, "_take": 45},
                {"_tag": "puddling", "taxon_id": TAXON_IDS["Papilionoidea"], "q": "puddling", "quality_grade": "research", "_max_pages": 1, "_take": 35},
                {"_tag": "pair", "taxon_id": TAXON_IDS["Papilionoidea"], "q": "pair", "quality_grade": "research", "_max_pages": 1, "_take": 25},
                {"_tag": "two", "taxon_id": TAXON_IDS["Papilionoidea"], "q": "two", "quality_grade": "research", "_max_pages": 1, "_take": 25},
                {"_tag": "occluded_flowers", "taxon_id": TAXON_IDS["Papilionoidea"], "q": "hidden", "quality_grade": "research", "_max_pages": 1, "_take": 25},
            ],
            None,
            10,
        ),
        (
            "small_or_camouflaged",
            120,
            [
                {"_tag": "lycaenidae", "taxon_id": TAXON_IDS["Lycaenidae"], "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 35},
                {"_tag": "hesperiidae", "taxon_id": TAXON_IDS["Hesperiidae"], "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 30},
                {"_tag": "geometridae", "taxon_id": TAXON_IDS["Geometridae"], "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 30},
                {"_tag": "camouflage", "taxon_id": TAXON_IDS["Lepidoptera"], "q": "camouflage", "quality_grade": "research", "_max_pages": 1, "_take": 25},
                {"_tag": "tiny", "taxon_id": TAXON_IDS["Papilionoidea"], "q": "tiny", "quality_grade": "research", "_max_pages": 1, "_take": 25},
            ],
            None,
            8,
        ),
        (
            "hard_neg_lookalike",
            90,
            [
                {"_tag": "odonata", "taxon_id": TAXON_IDS["Odonata"], "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 20},
                {"_tag": "syrphidae", "taxon_id": TAXON_IDS["Syrphidae"], "quality_grade": "research", "_max_pages": 1, "_take": 18},
                {"_tag": "bees", "taxon_id": TAXON_IDS["Anthophila"], "quality_grade": "research", "_max_pages": 1, "_take": 18},
                {"_tag": "wasps", "taxon_id": TAXON_IDS["Vespidae"], "quality_grade": "research", "_max_pages": 1, "_take": 14},
                {"_tag": "neuroptera", "taxon_id": neuroptera_id, "term_id": 1, "term_value_id": 2, "quality_grade": "research", "_max_pages": 1, "_take": 15},
                {"_tag": "cetoniinae", "taxon_id": cetoniinae_id, "quality_grade": "research", "_max_pages": 1, "_take": 15},
            ],
            {TAXON_IDS["Lepidoptera"]},
            6,
        ),
        (
            "hard_neg_immature",
            55,
            [
                {"_tag": "larva", "taxon_id": TAXON_IDS["Lepidoptera"], "term_id": 1, "term_value_id": 6, "quality_grade": "research", "_max_pages": 1, "_take": 35},
                {"_tag": "pupa", "taxon_id": TAXON_IDS["Lepidoptera"], "term_id": 1, "term_value_id": 4, "quality_grade": "research", "_max_pages": 1, "_take": 25},
            ],
            None,
            5,
        ),
        (
            "empty_habitat",
            65,
            [
                {"_tag": "blossom", "taxon_id": TAXON_IDS["Angiospermae"], "quality_grade": "research", "q": "blossom", "_max_pages": 1, "_take": 25},
                {"_tag": "flower", "taxon_id": TAXON_IDS["Angiospermae"], "quality_grade": "research", "q": "flower", "_max_pages": 1, "_take": 25},
                {"_tag": "meadow", "taxon_id": TAXON_IDS["Angiospermae"], "quality_grade": "research", "q": "meadow", "_max_pages": 1, "_take": 25},
            ],
            {TAXON_IDS["Lepidoptera"]},
            5,
        ),
    ]

    for g_name, target, specs, excl, max_sp in group_configs:
        recs = collect_group(
            session=session,
            group_name=g_name,
            target_count=target,
            query_specs=specs,
            seen_photo_ids=seen_photo_ids,
            seen_obs_ids=seen_obs_ids,
            seen_hashes=seen_hashes,
            exclude_taxon_ids=excl,
            max_per_species=max_sp,
        )
        all_records.extend(recs)
        render_contact_sheets(recs, g_name)

    out_jsonl = CANDIDATES_DIR / "candidates.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDONE! Saved {len(all_records)} total verified CC0 candidates to {out_jsonl}")


if __name__ == "__main__":
    main()
