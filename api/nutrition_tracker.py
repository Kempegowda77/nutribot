"""
 NutriChat AI — Meal Tracker & Nutrition Lookup
Logs meals, orders, and tracks daily calorie intake per user.
"""
import csv, json, time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
LOGS_FILE   = ROOT / "data" / "meal_logs.json"
ORDERS_FILE = ROOT / "data" / "orders.json"
MENU_FILE   = ROOT / "data" / "restaurant_menu.json"
UNIFIED_DB  = ROOT / "data" / "processed" / "unified_food_db.csv"

DAILY_GOAL = 2000  # default kcal


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _today() -> str:
    return date.today().isoformat()


# ── Menu ─────────────────────────────────────────────────────────────────────

def get_menu() -> List[Dict]:
    if not MENU_FILE.exists():
        return []
    with open(MENU_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_menu_item(item_id: str) -> Optional[Dict]:
    for item in get_menu():
        if item["id"] == item_id:
            return item
    return None


# ── Nutrition DB Lookup ───────────────────────────────────────────────────────

def _load_unified_db() -> List[Dict]:
    """Load unified_food_db.csv into memory."""
    if not UNIFIED_DB.exists():
        return []
    records = []
    with open(UNIFIED_DB, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

_DB_CACHE: Optional[List[Dict]] = None

def _get_db() -> List[Dict]:
    global _DB_CACHE
    if _DB_CACHE is None:
        _DB_CACHE = _load_unified_db()
    return _DB_CACHE


def lookup_food(query: str) -> Optional[Dict]:
    """Find best matching food in unified DB (case-insensitive partial match)."""
    q = query.lower().strip()
    db = _get_db()
    # exact match first
    for r in db:
        if r.get("name", "").lower() == q:
            return _format_db_record(r)
    # partial match
    for r in db:
        if q in r.get("name", "").lower():
            return _format_db_record(r)
    # also search menu
    for item in get_menu():
        if q in item["name"].lower():
            return {
                "name": item["name"],
                "calories_kcal": item["calories_kcal"],
                "protein_g": item.get("protein_g", 0),
                "carb_g": item.get("carb_g", 0),
                "fat_g": item.get("fat_g", 0),
                "fiber_g": item.get("fiber_g", 0),
                "sodium_mg": item.get("sodium_mg", 0),
                "source": "Restaurant Menu",
            }
    return None


def _format_db_record(r: Dict) -> Dict:
    def _f(v): 
        try: return round(float(v or 0), 2)
        except: return 0.0
    return {
        "name": r.get("name", ""),
        "calories_kcal": _f(r.get("calories_kcal")),
        "protein_g":  _f(r.get("protein_g")),
        "carb_g":     _f(r.get("carb_g")),
        "fat_g":      _f(r.get("fat_g")),
        "fiber_g":    _f(r.get("fiber_g")),
        "sugar_g":    _f(r.get("sugar_g")),
        "sodium_mg":  _f(r.get("sodium_mg")),
        "calcium_mg": _f(r.get("calcium_mg")),
        "iron_mg":    _f(r.get("iron_mg")),
        "vitamin_c_mg": _f(r.get("vitamin_c_mg")),
        "source": r.get("source", ""),
    }


# ── Meal Logging ──────────────────────────────────────────────────────────────

def log_meal(username: str, food_name: str, calories: float,
             protein: float, carbs: float, fat: float,
             qty: float = 1.0, meal_type: str = "meal",
             source: str = "order") -> Dict:
    logs = _load(LOGS_FILE)
    today = _today()
    key = f"{username}:{today}"
    if key not in logs:
        logs[key] = {"username": username, "date": today, "meals": [], "total": _empty_total()}

    entry = {
        "food_name": food_name,
        "qty": qty,
        "calories": round(calories * qty, 1),
        "protein_g": round(protein * qty, 2),
        "carb_g": round(carbs * qty, 2),
        "fat_g": round(fat * qty, 2),
        "meal_type": meal_type,
        "source": source,
        "timestamp": time.time(),
        "time_str": datetime.now().strftime("%I:%M %p"),
    }
    logs[key]["meals"].append(entry)

    # Update running totals
    t = logs[key]["total"]
    t["calories"]  = round(t["calories"]  + entry["calories"], 1)
    t["protein_g"] = round(t["protein_g"] + entry["protein_g"], 2)
    t["carb_g"]    = round(t["carb_g"]    + entry["carb_g"], 2)
    t["fat_g"]     = round(t["fat_g"]     + entry["fat_g"], 2)

    _save(LOGS_FILE, logs)
    return entry


def _empty_total() -> Dict:
    return {"calories": 0.0, "protein_g": 0.0, "carb_g": 0.0, "fat_g": 0.0}


def get_daily_log(username: str, day: Optional[str] = None) -> Dict:
    logs = _load(LOGS_FILE)
    today = day or _today()
    key = f"{username}:{today}"
    return logs.get(key, {"username": username, "date": today, "meals": [], "total": _empty_total()})


def get_daily_summary(username: str, daily_goal: int = DAILY_GOAL) -> Dict:
    log = get_daily_log(username)
    total = log["total"]
    cals = total["calories"]
    remaining = max(0, daily_goal - cals)
    pct = min(100, round(cals / daily_goal * 100, 1)) if daily_goal > 0 else 0
    return {
        "date": log["date"],
        "meals_count": len(log["meals"]),
        "total_calories": cals,
        "daily_goal": daily_goal,
        "remaining_calories": remaining,
        "percent_done": pct,
        "protein_g": total["protein_g"],
        "carb_g":    total["carb_g"],
        "fat_g":     total["fat_g"],
        "meals": log["meals"],
    }


# ── Orders ────────────────────────────────────────────────────────────────────

def place_order(username: str, items: List[Dict]) -> Dict:
    """
    items: [{"item_id": "R001", "qty": 2}, ...]
    Returns order dict with total price and nutrition.
    """
    menu_map = {it["id"]: it for it in get_menu()}
    order_items = []
    total_price = 0.0
    total_cals  = 0.0

    for it in items:
        menu_item = menu_map.get(it["item_id"])
        if not menu_item:
            continue
        qty = max(1, int(it.get("qty", 1)))
        item_price = menu_item["price"] * qty
        item_cals  = menu_item["calories_kcal"] * qty

        order_items.append({
            "item_id":   menu_item["id"],
            "name":      menu_item["name"],
            "qty":       qty,
            "unit_price": menu_item["price"],
            "subtotal":  item_price,
            "calories":  round(item_cals, 1),
            "protein_g": round(menu_item.get("protein_g", 0) * qty, 2),
            "carb_g":    round(menu_item.get("carb_g",    0) * qty, 2),
            "fat_g":     round(menu_item.get("fat_g",     0) * qty, 2),
            "emoji":     menu_item.get("emoji", "🍽️"),
        })
        total_price += item_price
        total_cals  += item_cals

        # Auto-log each ordered item as a meal
        log_meal(
            username=username,
            food_name=menu_item["name"],
            calories=menu_item["calories_kcal"],
            protein=menu_item.get("protein_g", 0),
            carbs=menu_item.get("carb_g", 0),
            fat=menu_item.get("fat_g", 0),
            qty=qty,
            meal_type="ordered",
            source="order",
        )

    order_id = f"ORD{int(time.time())}"
    order = {
        "order_id":    order_id,
        "username":    username,
        "items":       order_items,
        "total_price": round(total_price, 2),
        "total_calories": round(total_cals, 1),
        "status":      "confirmed",
        "timestamp":   time.time(),
        "time_str":    datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }

    orders = _load(ORDERS_FILE)
    if username not in orders:
        orders[username] = []
    orders[username].append(order)
    _save(ORDERS_FILE, orders)
    return order


def get_order_history(username: str, limit: int = 10) -> List[Dict]:
    orders = _load(ORDERS_FILE)
    user_orders = orders.get(username, [])
    return list(reversed(user_orders))[:limit]


# ── Meal Suggestion ───────────────────────────────────────────────────────────

def suggest_next_meal(username: str, daily_goal: int = DAILY_GOAL, goal: str = "general fitness") -> Dict:
    summary = get_daily_summary(username, daily_goal)
    remaining = summary["remaining_calories"]
    total_protein = summary["protein_g"]
    total_carbs   = summary["carb_g"]

    menu = get_menu()

    if remaining <= 0:
        return {
            "message": "🎯 You've reached your daily calorie goal! Consider a light walk or just water.",
            "suggestions": [],
            "reason": "Goal reached",
        }

    # Filter suitable items
    candidates = []
    for item in menu:
        if item["calories_kcal"] <= remaining * 1.1:
            score = 0
            # Score based on goal
            if "weight loss" in goal or "diabetes" in goal:
                score += max(0, 10 - item["calories_kcal"] / 30)
                score += item.get("fiber_g", 0) * 2
            elif "muscle" in goal or "protein" in goal:
                score += item.get("protein_g", 0) * 3
            elif "heart" in goal:
                score += item.get("fiber_g", 0) * 2
                score -= item.get("fat_g", 0) * 0.5
            else:
                score += item.get("protein_g", 0) * 2
                score += item.get("fiber_g", 0)
                score -= item["calories_kcal"] / 100

            # Bonus: if protein intake is low, prefer protein-rich
            if total_protein < 50:
                score += item.get("protein_g", 0) * 2
            # Bonus: if carbs are high, prefer low-carb
            if total_carbs > daily_goal * 0.4:
                score -= item.get("carb_g", 0) * 0.5

            candidates.append((score, item))

    candidates.sort(key=lambda x: -x[0])
    top = [c[1] for c in candidates[:4]]

    current_time = datetime.now().hour
    if current_time < 10:
        meal_time = "breakfast"
    elif current_time < 14:
        meal_time = "lunch"
    elif current_time < 17:
        meal_time = "snack"
    else:
        meal_time = "dinner"

    message = (
        f"🍽️ You have {remaining:.0f} kcal remaining for the day. "
        f"Here are my top {meal_time} suggestions based on your goal ({goal}):"
    )

    return {
        "message": message,
        "suggestions": top[:4],
        "remaining_calories": remaining,
        "meal_time": meal_time,
        "reason": f"Goal: {goal}, Remaining: {remaining:.0f} kcal",
    }
