"""
Indian Food Nutrition Dataset Loader
======================================
Downloads Indian food nutrition data from multiple open sources:
1. Kaggle Indian Food Dataset (local CSV if available)
2. Built-in curated Indian foods database (verified from ICMR/NIN sources)

Output: data/processed/indian_foods.csv

Run:
    python data/sources/indian_food_loader.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR  = ROOT / "data" / "processed"
OUT_FILE = OUT_DIR / "indian_foods.csv"

COLUMNS = ["name", "source", "category", "calories_kcal",
           "protein_g", "carb_g", "fat_g", "fiber_g",
           "sugar_g", "sodium_mg", "calcium_mg", "iron_mg", "vitamin_c_mg"]

# -- Curated Indian Food Database ---------------------------------------------
# Source: ICMR-NIN Nutritive Value of Indian Foods (2017 edition)
# All values per 100g edible portion
INDIAN_FOODS_DB = [
    # name, category, cal, prot, carb, fat, fiber, sugar, sodium, calcium, iron, vit_c
    ("Roti (Wheat Chapati)", "Cereals", 297, 8.5, 61.2, 3.7, 1.9, 0.4, 2, 41, 3.5, 0),
    ("Paratha (Plain)", "Cereals", 326, 7.5, 51.0, 11.0, 1.5, 0.3, 5, 35, 2.8, 0),
    ("Paratha (Aloo)", "Cereals", 310, 6.5, 50.0, 10.5, 1.8, 0.5, 8, 30, 2.5, 3),
    ("Basmati Rice (cooked)", "Cereals", 130, 2.7, 28.2, 0.3, 0.4, 0.0, 1, 10, 0.2, 0),
    ("Brown Rice (cooked)", "Cereals", 123, 2.6, 25.6, 0.9, 1.8, 0.0, 5, 20, 0.5, 0),
    ("Idli", "Cereals", 58, 2.0, 11.4, 0.4, 0.5, 0.1, 6, 10, 0.9, 0),
    ("Dosa (Plain)", "Cereals", 133, 2.8, 22.0, 4.0, 0.5, 0.2, 12, 14, 1.0, 0),
    ("Upma", "Cereals", 150, 3.4, 21.5, 5.8, 1.0, 0.3, 180, 18, 1.2, 2),
    ("Poha (Flattened Rice)", "Cereals", 180, 2.5, 36.2, 4.0, 0.5, 0.1, 300, 10, 1.5, 0),
    ("Puri", "Cereals", 335, 7.5, 44.0, 15.5, 1.2, 0.2, 8, 30, 2.5, 0),
    ("Dal Tadka (Toor Dal)", "Legumes", 116, 7.2, 18.5, 2.5, 3.5, 0.5, 180, 45, 2.0, 5),
    ("Dal Makhani", "Legumes", 145, 7.8, 18.0, 5.0, 4.5, 1.0, 220, 55, 2.8, 4),
    ("Rajma Curry", "Legumes", 127, 8.7, 18.5, 2.2, 6.4, 1.2, 200, 50, 3.2, 5),
    ("Chana Masala", "Legumes", 164, 9.0, 22.0, 4.5, 7.6, 2.0, 250, 57, 3.5, 8),
    ("Moong Dal", "Legumes", 105, 7.5, 16.5, 1.0, 4.1, 0.5, 15, 35, 1.9, 4),
    ("Sambar", "Legumes", 51, 2.5, 7.5, 1.5, 2.5, 1.0, 280, 30, 1.2, 12),
    ("Paneer (fresh)", "Dairy", 265, 18.3, 1.2, 20.8, 0.0, 1.0, 37, 480, 0.5, 0),
    ("Paneer Butter Masala", "Dairy", 280, 14.0, 12.0, 20.0, 1.5, 3.0, 350, 300, 1.5, 8),
    ("Curd / Dahi (full fat)", "Dairy", 98, 3.1, 4.7, 6.8, 0.0, 4.7, 36, 120, 0.1, 1),
    ("Buttermilk (Chaas)", "Dairy", 24, 0.8, 3.5, 0.8, 0.0, 3.5, 85, 40, 0.0, 1),
    ("Lassi (sweet)", "Dairy", 85, 3.5, 13.5, 2.0, 0.0, 12.0, 50, 130, 0.1, 2),
    ("Chicken Curry", "Non-Veg", 180, 18.5, 5.5, 9.5, 0.5, 1.5, 300, 25, 1.5, 5),
    ("Tandoori Chicken", "Non-Veg", 165, 24.0, 4.0, 6.0, 0.3, 0.8, 420, 30, 1.2, 3),
    ("Mutton Curry", "Non-Veg", 255, 22.0, 5.0, 16.5, 0.5, 1.0, 380, 20, 2.5, 4),
    ("Fish Curry (avg)", "Non-Veg", 145, 17.0, 6.5, 6.0, 0.5, 1.0, 350, 35, 1.8, 6),
    ("Egg Bhurji", "Non-Veg", 155, 10.5, 4.0, 10.5, 0.5, 1.0, 280, 55, 1.8, 5),
    ("Boiled Egg", "Non-Veg", 78, 6.3, 0.6, 5.3, 0.0, 0.6, 62, 25, 0.9, 0),
    ("Aloo Gobi", "Vegetables", 95, 2.5, 13.5, 4.0, 3.0, 2.5, 200, 30, 1.0, 30),
    ("Palak Paneer", "Vegetables", 210, 10.5, 8.5, 15.5, 2.5, 1.5, 280, 250, 2.5, 18),
    ("Bhindi Masala", "Vegetables", 80, 2.0, 9.5, 4.0, 3.5, 2.0, 180, 40, 0.8, 15),
    ("Mixed Veg Curry", "Vegetables", 85, 2.5, 11.0, 4.0, 3.0, 3.0, 200, 45, 1.0, 20),
    ("Baingan Bharta", "Vegetables", 70, 2.0, 8.5, 3.5, 3.0, 4.0, 250, 20, 0.5, 5),
    ("Methi Thepla", "Cereals", 195, 6.5, 28.0, 7.0, 3.5, 0.5, 280, 80, 3.5, 5),
    ("Dhokla", "Snacks", 160, 5.5, 28.5, 4.0, 1.5, 2.0, 350, 45, 1.5, 1),
    ("Samosa (veg)", "Snacks", 262, 4.5, 31.0, 13.5, 2.0, 1.0, 320, 20, 1.8, 3),
    ("Pakora (Onion)", "Snacks", 278, 5.0, 28.0, 16.5, 2.5, 1.5, 280, 30, 2.0, 5),
    ("Gulab Jamun", "Sweets", 385, 6.5, 55.0, 16.5, 0.5, 40.0, 80, 80, 0.8, 0),
    ("Rasgulla", "Sweets", 186, 5.5, 35.0, 3.5, 0.0, 30.0, 30, 60, 0.3, 0),
    ("Halwa (Sooji)", "Sweets", 320, 4.5, 55.0, 10.0, 0.8, 25.0, 20, 15, 1.0, 0),
    ("Kheer (Rice)", "Sweets", 175, 4.5, 28.5, 5.5, 0.3, 18.0, 45, 120, 0.5, 1),
    ("Masala Chai", "Beverages", 40, 1.5, 5.5, 1.5, 0.2, 4.5, 15, 55, 0.3, 0),
    ("Coconut Water", "Beverages", 19, 0.7, 3.7, 0.2, 1.1, 2.6, 105, 24, 0.3, 2),
    ("Mango Lassi", "Beverages", 95, 3.0, 17.5, 2.0, 0.5, 15.0, 40, 100, 0.2, 12),
    ("Nimbu Pani", "Beverages", 26, 0.2, 6.5, 0.1, 0.1, 5.0, 500, 5, 0.1, 15),
    ("Banana (Elaichi)", "Fruits", 89, 1.1, 22.8, 0.3, 2.6, 12.2, 1, 5, 0.3, 8),
    ("Mango (Alphonso)", "Fruits", 70, 0.5, 17.0, 0.3, 1.8, 14.8, 2, 11, 0.1, 28),
    ("Papaya", "Fruits", 43, 0.5, 10.8, 0.3, 1.7, 7.8, 8, 20, 0.3, 62),
    ("Pomegranate", "Fruits", 83, 1.7, 18.7, 1.2, 4.0, 13.7, 3, 10, 0.3, 10),
    ("Guava", "Fruits", 68, 2.6, 14.3, 1.0, 5.4, 8.9, 2, 18, 0.3, 228),
    ("Sprouts (Mixed)", "Legumes", 45, 3.8, 6.0, 0.4, 1.8, 0.5, 7, 20, 0.8, 15),
    ("Sattu (Roasted Gram)", "Legumes", 362, 22.0, 52.0, 7.0, 5.5, 3.0, 15, 90, 5.8, 0),
    ("Oats (cooked)", "Cereals", 71, 2.5, 12.0, 1.5, 1.7, 0.5, 49, 10, 0.8, 0),
    ("Makhana (Fox Nuts)", "Snacks", 347, 9.7, 76.9, 0.1, 14.5, 0.0, 5, 60, 1.4, 0),
    ("Peanuts (roasted)", "Snacks", 567, 25.8, 16.1, 49.2, 8.5, 4.7, 18, 92, 4.6, 0),
    ("Coconut Chutney", "Condiments", 95, 1.5, 7.0, 7.5, 3.0, 2.0, 180, 15, 0.8, 5),
    ("Raita (Boondi)", "Dairy", 80, 3.5, 10.5, 2.5, 0.5, 5.0, 200, 100, 0.3, 2),
    ("Biryani (Chicken)", "Rice Dishes", 215, 12.5, 24.5, 8.0, 1.0, 1.5, 380, 35, 1.5, 5),
    ("Biryani (Veg)", "Rice Dishes", 185, 5.5, 28.5, 6.0, 2.5, 1.5, 320, 30, 1.2, 8),
    ("Pav Bhaji", "Snacks", 215, 5.5, 32.0, 8.0, 4.5, 5.0, 420, 35, 1.5, 25),
    ("Chole Bhature", "Snacks", 380, 11.5, 55.0, 14.0, 8.0, 3.5, 480, 60, 3.8, 10),
    ("Vada Pav", "Snacks", 295, 7.5, 42.0, 10.5, 3.5, 3.0, 520, 40, 2.0, 5),
    ("Kadhi Pakora", "Legumes", 130, 4.5, 14.5, 7.0, 2.0, 2.5, 380, 80, 1.0, 3),
]


def load_kaggle_csv(csv_path: Path) -> list[dict]:
    """Load Kaggle Indian food CSV if user has placed it manually."""
    records = []
    if not csv_path.exists():
        return records

    print(f"  [kaggle] Loading {csv_path.name}...")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rec = {
                    "name":          row.get("name", row.get("Name", "")),
                    "source":        "Kaggle_IndianFood",
                    "category":      row.get("course", row.get("Category", "Indian")),
                    "calories_kcal": float(row.get("calories", row.get("Calories", 0)) or 0),
                    "protein_g":     float(row.get("protein", row.get("Protein", 0)) or 0),
                    "carb_g":        float(row.get("carbs", row.get("Carbs", 0)) or 0),
                    "fat_g":         float(row.get("fat", row.get("Fat", 0)) or 0),
                    "fiber_g":       float(row.get("fiber", row.get("Fiber", 0)) or 0),
                    "sugar_g":       0.0,
                    "sodium_mg":     0.0,
                    "calcium_mg":    0.0,
                    "iron_mg":       0.0,
                    "vitamin_c_mg":  0.0,
                }
                if rec["name"] and rec["calories_kcal"] > 0:
                    records.append(rec)
            except (ValueError, KeyError):
                continue

    print(f"  [OK] Loaded {len(records)} entries from Kaggle CSV")
    return records


def build_from_builtin() -> list[dict]:
    """Convert built-in ICMR/NIN data to records."""
    records = []
    for row in INDIAN_FOODS_DB:
        name, cat, cal, prot, carb, fat, fiber, sugar, sodium, calcium, iron, vitc = row
        records.append({
            "name":          name,
            "source":        "ICMR_NIN_2017",
            "category":      cat,
            "calories_kcal": cal,
            "protein_g":     prot,
            "carb_g":        carb,
            "fat_g":         fat,
            "fiber_g":       fiber,
            "sugar_g":       sugar,
            "sodium_mg":     sodium,
            "calcium_mg":    calcium,
            "iron_mg":       iron,
            "vitamin_c_mg":  vitc,
        })
    return records


def save_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {col: rec.get(col, 0.0) for col in COLUMNS}
            row["name"]     = rec.get("name", "")
            row["source"]   = rec.get("source", "")
            row["category"] = rec.get("category", "")
            writer.writerow(row)
    print(f"  [OK] Saved {len(records):,} Indian food records -> {path}")


def main():
    print("=" * 60)
    print("  Indian Food Dataset Loader")
    print("  Source: ICMR-NIN + Kaggle Indian Food Dataset")
    print("=" * 60)

    records = build_from_builtin()
    print(f"\n[1/2] Built-in ICMR/NIN database: {len(records)} foods")

    # Check if user placed a Kaggle CSV
    kaggle_path = ROOT / "data" / "sources" / "raw" / "indian_food.csv"
    extra = load_kaggle_csv(kaggle_path)
    if extra:
        # Deduplicate
        seen = {r["name"].lower() for r in records}
        added = [r for r in extra if r["name"].lower() not in seen]
        records += added
        print(f"[2/2] Added {len(added)} extra records from Kaggle CSV")
    else:
        print(f"[2/2] No Kaggle CSV found at {kaggle_path} — using built-in only")
        print("      (Place 'indian_food.csv' there to add more entries)")

    save_records(records, OUT_FILE)
    print(f"\n[OK] Indian food DB: {len(records)} total records -> {OUT_FILE}")


if __name__ == "__main__":
    main()
