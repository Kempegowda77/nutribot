"""
USDA FoodData Central (FDC) Dataset Loader
============================================
Downloads and parses the official USDA FoodData Central dataset.
Source: https://fdc.nal.usda.gov/download-datasets.html

Dataset: Foundation Foods + SR Legacy (free, public domain)
Output: data/processed/usda_foods.csv

Run:
    python data/sources/usda_fdc_loader.py
"""

import io
import os
import sys
import time
import zipfile
from pathlib import Path
import csv
import urllib.request

# -- Project root on path -----------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# -- Constants ----------------------------------------------------------------
# USDA FDC public download — SR Legacy (Standard Reference, ~28MB zip)
# License: Public Domain (U.S. Government work)
USDA_SR_LEGACY_URL = (
    "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip"
)
USDA_FOUNDATION_URL = (
    "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2024-10-31.zip"
)

RAW_DIR     = ROOT / "data" / "sources" / "raw"
OUT_DIR     = ROOT / "data" / "processed"
OUT_FILE    = OUT_DIR / "usda_foods.csv"

# Nutrient IDs we care about
# Key = nutrient 'id' column in nutrient.csv (= nutrient_id in food_nutrient.csv)
# These are stable USDA nutrient IDs across all FDC dataset releases
NUTRIENT_MAP = {
    "1008": "calories_kcal",   # Energy (kcal)
    "1003": "protein_g",       # Protein
    "1005": "carb_g",          # Carbohydrate
    "1004": "fat_g",           # Total lipid (fat)
    "1079": "fiber_g",         # Fiber, total dietary
    "2000": "sugar_g",         # Sugars, total
    "1093": "sodium_mg",       # Sodium
    "1087": "calcium_mg",      # Calcium
    "1089": "iron_mg",         # Iron
    "1162": "vitamin_c_mg",    # Vitamin C
    # Also accept nutrient_nbr variants (208=energy, 203=protein, etc.)
    "208":  "calories_kcal",
    "203":  "protein_g",
    "205":  "carb_g",
    "204":  "fat_g",
    "291":  "fiber_g",
    "269":  "sugar_g",
    "307":  "sodium_mg",
    "301":  "calcium_mg",
    "303":  "iron_mg",
    "401":  "vitamin_c_mg",
}

# -- Download helper -----------------------------------------------------------

def download_file(url: str, dest: Path, label: str = "") -> Path:
    """Download a URL to dest with a simple progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [cache] {label} already downloaded -> {dest.name}")
        return dest

    print(f"  [download] {label}")
    print(f"    URL: {url}")

    headers = {"User-Agent": "NutriChat-Research/1.0 (academic use)"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 65536
            with open(dest, "wb") as f:
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    f.write(data)
                    downloaded += len(data)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r    {pct:.1f}% ({downloaded // 1024} KB / {total // 1024} KB)", end="")
        print(f"\n  [OK] Downloaded -> {dest}")
    except Exception as e:
        print(f"\n  [ERR] Download failed: {e}")
        print("  Trying alternative approach with requests library...")
        try:
            import requests
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk_data in r.iter_content(chunk_size=65536):
                        f.write(chunk_data)
                        downloaded += len(chunk_data)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r    {pct:.1f}%", end="")
            print(f"\n  [OK] Downloaded -> {dest}")
        except Exception as e2:
            if dest.exists():
                dest.unlink()
            raise RuntimeError(f"Download failed: {e2}") from e2
    return dest


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """Extract a zip archive, return extraction folder."""
    extract_to.mkdir(parents=True, exist_ok=True)
    folder_name = zip_path.stem
    out_folder = extract_to / folder_name
    if out_folder.exists():
        print(f"  [cache] Already extracted -> {out_folder.name}/")
        return out_folder
    print(f"  [extract] {zip_path.name} -> {out_folder.name}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_folder)
    print(f"  [OK] Extracted")
    return out_folder


# -- SR Legacy parser ----------------------------------------------------------

def parse_sr_legacy(folder: Path) -> list[dict]:
    """
    Parse USDA SR Legacy CSV files into a flat list of food records.
    Files: food.csv, food_nutrient.csv, nutrient.csv
    """
    print(f"\n  [parse] SR Legacy from {folder}")

    # Find files (handle nested dirs)
    food_csv      = _find_file(folder, "food.csv")
    nutrient_csv  = _find_file(folder, "food_nutrient.csv")
    nutr_def_csv  = _find_file(folder, "nutrient.csv")

    if not food_csv or not nutrient_csv or not nutr_def_csv:
        # Try foundation food naming
        food_csv      = _find_file(folder, "food.csv")
        nutrient_csv  = _find_file(folder, "food_nutrient.csv")
        nutr_def_csv  = _find_file(folder, "nutrient.csv")

    print(f"    food.csv:         {food_csv}")
    print(f"    food_nutrient.csv:{nutrient_csv}")
    print(f"    nutrient.csv:     {nutr_def_csv}")

    # Build nutrient_id -> nutrient_nbr map  (id col -> number col)
    nutr_id_to_nbr: dict[str, str] = {}
    if nutr_def_csv:
        with open(nutr_def_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = row.get("id", "").strip()
                nbr = row.get("nutrient_nbr", "").strip()
                nutr_id_to_nbr[nid] = nbr

    # Load food name -> fdc_id
    foods: dict[str, dict] = {}  # fdc_id -> {name, food_category_id}
    if food_csv:
        with open(food_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fdc_id = row.get("fdc_id", "").strip()
                name   = row.get("description", "").strip()
                if fdc_id and name:
                    foods[fdc_id] = {
                        "name":     name,
                        "category": row.get("food_category_id", ""),
                        "source":   "USDA_SR_Legacy",
                    }

    print(f"    Loaded {len(foods):,} food entries")

    # Load nutrients per food
    records_map: dict[str, dict] = {}
    if nutrient_csv:
        with open(nutrient_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fdc_id    = row.get("fdc_id", "").strip()
                nutr_id   = row.get("nutrient_id", "").strip()
                amount    = row.get("amount", "0").strip()
                nutr_nbr  = nutr_id_to_nbr.get(nutr_id, "")

                # Try direct match by nutrient_id first (e.g. "1008" -> calories)
                # Then by nutrient_nbr (e.g. "208" -> calories)
                col = NUTRIENT_MAP.get(nutr_id) or NUTRIENT_MAP.get(nutr_nbr)
                if not col:
                    continue
                if fdc_id not in foods:
                    continue

                if fdc_id not in records_map:
                    records_map[fdc_id] = {**foods[fdc_id]}

                try:
                    records_map[fdc_id][col] = float(amount)
                except ValueError:
                    records_map[fdc_id][col] = 0.0

    # Build final list
    records = []
    for fdc_id, rec in records_map.items():
        if rec.get("calories_kcal", 0) <= 0:
            continue  # skip entries with no calorie data
        records.append(rec)

    print(f"    [OK] Parsed {len(records):,} food records with calorie data")
    return records


def _find_file(folder: Path, filename: str) -> Path | None:
    """Recursively find a file in a folder."""
    for p in folder.rglob(filename):
        return p
    return None


# -- Save to CSV ---------------------------------------------------------------

COLUMNS = ["name", "source", "category", "calories_kcal",
           "protein_g", "carb_g", "fat_g", "fiber_g",
           "sugar_g", "sodium_mg", "calcium_mg", "iron_mg", "vitamin_c_mg"]


def save_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            # Fill missing nutrient columns with 0.0
            row = {col: rec.get(col, 0.0) for col in COLUMNS}
            row["name"]     = rec.get("name", "")
            row["source"]   = rec.get("source", "USDA")
            row["category"] = rec.get("category", "")
            writer.writerow(row)
    print(f"\n  [OK] Saved {len(records):,} records -> {path}")


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  USDA FoodData Central Loader")
    print("  Downloading real nutrition data for NutriChat AI")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Download SR Legacy ZIP
    sr_zip = RAW_DIR / "usda_sr_legacy.zip"
    print("\n[1/3] Downloading USDA SR Legacy dataset...")
    try:
        download_file(USDA_SR_LEGACY_URL, sr_zip, "USDA SR Legacy (~28MB)")
        sr_folder = extract_zip(sr_zip, RAW_DIR)
        sr_records = parse_sr_legacy(sr_folder)
    except Exception as e:
        print(f"  [WARN]  SR Legacy download failed: {e}")
        print("  Continuing with Foundation Foods only...")
        sr_records = []

    # -- Download Foundation Foods ZIP
    fn_zip = RAW_DIR / "usda_foundation.zip"
    print("\n[2/3] Downloading USDA Foundation Foods dataset...")
    try:
        download_file(USDA_FOUNDATION_URL, fn_zip, "USDA Foundation Foods (~15MB)")
        fn_folder = extract_zip(fn_zip, RAW_DIR)
        fn_records = parse_sr_legacy(fn_folder)  # same schema
        for r in fn_records:
            r["source"] = "USDA_Foundation"
    except Exception as e:
        print(f"  [WARN]  Foundation Foods download failed: {e}")
        fn_records = []

    # -- Merge and deduplicate
    print("\n[3/3] Merging datasets...")
    all_records = sr_records + fn_records
    print(f"  Total before dedup: {len(all_records):,}")

    # Deduplicate by name (case-insensitive)
    seen: set[str] = set()
    deduped = []
    for rec in all_records:
        key = rec["name"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(rec)

    print(f"  After dedup: {len(deduped):,}")

    if not deduped:
        print("\n  [ERR] No data downloaded. Check your internet connection.")
        print("  You can manually download from: https://fdc.nal.usda.gov/download-datasets.html")
        sys.exit(1)

    save_records(deduped, OUT_FILE)

    print("\n" + "=" * 60)
    print(f"  [OK] USDA Dataset ready: {OUT_FILE}")
    print(f"  Total foods: {len(deduped):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
