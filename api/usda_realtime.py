"""
Real-Time USDA FoodData Central API Client
===========================================
Fetches live nutrition data from the official USDA FDC REST API.
Free API key: https://fdc.nal.usda.gov/api-key-signup.html

Usage:
    from api.usda_realtime import USDAClient
    client = USDAClient()
    result = client.search("apple")
"""

import os
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

# Free USDA API key (sign up at https://fdc.nal.usda.gov/api-key-signup.html)
# Set as env var: USDA_API_KEY=your_key
# Default demo key works for low-volume requests
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
DEMO_API_KEY  = "DEMO_KEY"   # Works for ~30 req/hour; get free key for production

# Simple in-memory cache to avoid redundant API calls
_cache: dict[str, dict] = {}


class USDAClient:
    """
    Lightweight USDA FoodData Central API client.
    No external dependencies — uses only stdlib urllib.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("USDA_API_KEY", DEMO_API_KEY)
        self._request_times: list[float] = []

    def _get(self, endpoint: str, params: dict) -> dict:
        params["api_key"] = self.api_key
        query = urllib.parse.urlencode(params)
        url   = f"{USDA_BASE_URL}/{endpoint}?{query}"

        # Simple rate limiting (30 req/min for DEMO_KEY)
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= 28:
            sleep_time = 61 - (now - self._request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        req = urllib.request.Request(
            url, headers={"User-Agent": "NutriChat-Research/1.0 (academic)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self._request_times.append(time.time())
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search USDA for a food item. Returns list of nutrition dicts.

        Returns:
            [{"name": str, "calories": float, "protein_g": float,
              "carb_g": float, "fat_g": float, "fiber_g": float,
              "fdc_id": int, "data_type": str}, ...]
        """
        cache_key = f"search:{query.lower().strip()}:{max_results}"
        if cache_key in _cache:
            return _cache[cache_key]

        resp = self._get("foods/search", {
            "query":     query,
            "pageSize":  max_results,
            "dataType":  "Foundation,SR Legacy,Branded",
        })

        if "error" in resp or "foods" not in resp:
            return []

        results = []
        for food in resp.get("foods", []):
            nutrients = {n["nutrientNumber"]: n.get("value", 0)
                         for n in food.get("foodNutrients", [])}
            results.append({
                "fdc_id":     food.get("fdcId"),
                "name":       food.get("description", ""),
                "data_type":  food.get("dataType", ""),
                "calories":   float(nutrients.get("208", nutrients.get("1008", 0))),
                "protein_g":  float(nutrients.get("203", nutrients.get("1003", 0))),
                "carb_g":     float(nutrients.get("205", nutrients.get("1005", 0))),
                "fat_g":      float(nutrients.get("204", nutrients.get("1004", 0))),
                "fiber_g":    float(nutrients.get("291", nutrients.get("1079", 0))),
                "sugar_g":    float(nutrients.get("269", nutrients.get("2000", 0))),
                "sodium_mg":  float(nutrients.get("307", nutrients.get("1093", 0))),
            })

        _cache[cache_key] = results
        return results

    def get_by_fdc_id(self, fdc_id: int) -> dict:
        """Get detailed nutrition for a specific FDC food ID."""
        cache_key = f"fdc:{fdc_id}"
        if cache_key in _cache:
            return _cache[cache_key]

        resp = self._get(f"food/{fdc_id}", {"format": "abridged"})
        if "error" in resp:
            return {}

        nutrients = {n.get("nutrient", {}).get("number", ""): n.get("amount", 0)
                     for n in resp.get("foodNutrients", [])}

        result = {
            "fdc_id":     fdc_id,
            "name":       resp.get("description", ""),
            "data_type":  resp.get("dataType", ""),
            "calories":   float(nutrients.get("208", nutrients.get("1008", 0))),
            "protein_g":  float(nutrients.get("203", nutrients.get("1003", 0))),
            "carb_g":     float(nutrients.get("205", nutrients.get("1005", 0))),
            "fat_g":      float(nutrients.get("204", nutrients.get("1004", 0))),
            "fiber_g":    float(nutrients.get("291", nutrients.get("1079", 0))),
            "sugar_g":    float(nutrients.get("269", nutrients.get("2000", 0))),
            "sodium_mg":  float(nutrients.get("307", nutrients.get("1093", 0))),
        }
        _cache[cache_key] = result
        return result

    def lookup_food(self, food_name: str) -> Optional[dict]:
        """
        Best-match lookup: returns the most relevant food record or None.
        Used during real-time inference to ground responses.
        """
        results = self.search(food_name, max_results=3)
        if not results:
            return None
        # Prefer Foundation > SR Legacy > Branded
        priority = {"Foundation": 0, "SR Legacy": 1, "Survey (FNDDS)": 2, "Branded": 3}
        results.sort(key=lambda r: priority.get(r.get("data_type", ""), 99))
        return results[0]


# -- Local fallback lookup (from unified_food_db.csv) -------------------------

class LocalFoodDB:
    """
    Fast local lookup from unified_food_db.csv.
    Used as fallback when USDA API is unavailable.
    """
    def __init__(self):
        self._db: dict[str, dict] = {}
        db_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "unified_food_db.csv"
        if db_path.exists():
            import csv
            with open(db_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    name = row.get("name", "").strip()
                    if name:
                        self._db[name.lower()] = {
                            "name":      name,
                            "calories":  float(row.get("calories_kcal", 0) or 0),
                            "protein_g": float(row.get("protein_g", 0) or 0),
                            "carb_g":    float(row.get("carb_g", 0) or 0),
                            "fat_g":     float(row.get("fat_g", 0) or 0),
                            "fiber_g":   float(row.get("fiber_g", 0) or 0),
                            "source":    row.get("source", "local"),
                        }

    def lookup(self, food_name: str) -> Optional[dict]:
        """Fuzzy key lookup — exact match first, then prefix search."""
        key = food_name.lower().strip()
        if key in self._db:
            return self._db[key]
        # Prefix / substring match
        for k, v in self._db.items():
            if key in k or k in key:
                return v
        return None

    def __len__(self):
        return len(self._db)


# -- Combined resolver ---------------------------------------------------------

class NutritionResolver:
    """
    Resolves food nutrition data:
    1. Try local DB (fast, no API call)
    2. Fallback to USDA live API
    """
    def __init__(self, prefer_local: bool = True):
        self.local_db   = LocalFoodDB()
        self.usda_client = USDAClient()
        self.prefer_local = prefer_local
        print(f"[NutritionResolver] Local DB: {len(self.local_db)} foods")

    def resolve(self, food_name: str) -> Optional[dict]:
        if self.prefer_local:
            local = self.local_db.lookup(food_name)
            if local:
                return {**local, "lookup_source": "local_db"}

        # Live USDA API
        api_result = self.usda_client.lookup_food(food_name)
        if api_result:
            return {**api_result, "lookup_source": "usda_api"}

        # Last resort: local without prefer
        return self.local_db.lookup(food_name)


if __name__ == "__main__":
    print("Testing USDA Real-Time API + Local DB...")
    resolver = NutritionResolver()

    test_foods = ["apple", "roti", "chicken breast", "paneer", "banana"]
    for food in test_foods:
        result = resolver.resolve(food)
        if result:
            print(f"\n  {food}:")
            print(f"    Calories: {result.get('calories', 'N/A')} kcal/100g")
            print(f"    Protein:  {result.get('protein_g', 'N/A')} g")
            print(f"    Source:   {result.get('lookup_source', 'N/A')}")
        else:
            print(f"\n  {food}: NOT FOUND")
