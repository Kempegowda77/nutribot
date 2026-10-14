"""
Synthetic Hinglish food chatbot dataset generator.
Produces 800 training + 200 validation samples covering 5 intents.
Run: python data/generate_dataset.py
 """
import json
import random
from pathlib import Path

random.seed(42)

# ── Food library ─────────────────────────────────────────────────────────────
# Each food: [cal_per_serving, carb, protein, fat, fiber, sugar, sodium, vitamin, mineral]
FOODS = {
    "roti":     [70,  0.72, 0.11, 0.05, 0.22, 0.00, 0.10, 0.05, 0.10],
    "chapati":  [70,  0.72, 0.11, 0.05, 0.22, 0.00, 0.10, 0.05, 0.10],
    "rice":     [130, 0.88, 0.07, 0.02, 0.05, 0.00, 0.02, 0.03, 0.05],
    "dal":      [116, 0.40, 0.70, 0.10, 0.50, 0.05, 0.15, 0.10, 0.30],
    "paneer":   [265, 0.05, 0.70, 0.60, 0.00, 0.05, 0.10, 0.10, 0.30],
    "chicken":  [165, 0.00, 0.90, 0.30, 0.00, 0.00, 0.15, 0.20, 0.20],
    "egg":      [70,  0.05, 0.80, 0.50, 0.00, 0.05, 0.15, 0.30, 0.20],
    "milk":     [61,  0.20, 0.40, 0.30, 0.00, 0.30, 0.10, 0.10, 0.40],
    "banana":   [89,  0.80, 0.05, 0.02, 0.20, 0.50, 0.01, 0.30, 0.10],
    "apple":    [52,  0.70, 0.02, 0.01, 0.30, 0.40, 0.01, 0.20, 0.05],
    "oats":     [150, 0.60, 0.40, 0.20, 0.60, 0.10, 0.05, 0.10, 0.20],
    "idli":     [39,  0.70, 0.10, 0.02, 0.10, 0.00, 0.10, 0.05, 0.10],
    "dosa":     [120, 0.60, 0.10, 0.20, 0.10, 0.00, 0.15, 0.05, 0.10],
    "poha":     [180, 0.80, 0.10, 0.10, 0.10, 0.05, 0.10, 0.05, 0.10],
    "paratha":  [160, 0.60, 0.10, 0.30, 0.10, 0.00, 0.15, 0.05, 0.10],
    "bread":    [80,  0.70, 0.10, 0.10, 0.10, 0.05, 0.20, 0.05, 0.10],
    "curd":     [98,  0.10, 0.30, 0.20, 0.00, 0.20, 0.10, 0.10, 0.30],
    "fish":     [140, 0.00, 0.80, 0.30, 0.00, 0.00, 0.20, 0.30, 0.30],
    "rajma":    [127, 0.50, 0.60, 0.10, 0.60, 0.05, 0.10, 0.10, 0.30],
    "chana":    [164, 0.50, 0.60, 0.20, 0.50, 0.05, 0.10, 0.10, 0.30],
    "salad":    [20,  0.10, 0.05, 0.02, 0.40, 0.10, 0.05, 0.40, 0.20],
    "yogurt":   [59,  0.10, 0.30, 0.10, 0.00, 0.20, 0.10, 0.10, 0.30],
    "mutton":   [250, 0.00, 0.70, 0.70, 0.00, 0.00, 0.20, 0.20, 0.20],
    "upma":     [150, 0.70, 0.10, 0.10, 0.10, 0.00, 0.20, 0.05, 0.10],
    "sprouts":  [90,  0.40, 0.60, 0.05, 0.40, 0.05, 0.05, 0.20, 0.30],
}
FOOD_NAMES = list(FOODS.keys())

MEALS = ["breakfast", "lunch", "dinner", "snack", "pre-workout meal", "post-workout meal"]
GOALS = ["weight loss", "muscle gain", "diabetes control", "heart health", "energy boost", "keto diet"]
QUANTITIES = ["1", "2", "3", "ek", "do", "teen", "half", "1 cup", "100g", "150g", "200g", "50g"]

# ── Template sets per intent ──────────────────────────────────────────────────
TEMPLATES = {
    0: [  # calorie_query
        "aaj {qty} {food} khaya kitna calorie hua",
        "{qty} {food} mein kitna calorie hota hai",
        "how many calories in {qty} {food}",
        "{food} ke {qty} serving mein calorie batao",
        "kal maine {qty} {food} khayi kitna calorie tha",
        "{qty} {food} ka calorie count kya hai",
        "mujhe batao {food} mein kitni calories hoti hain",
        "{food} khaya to kitna calorie consume kiya",
        "{qty} {food} kitni calories deta hai",
        "approximate calorie {qty} {food} ka kya hai",
        "calorie count karo {qty} {food} ka",
        "agar {qty} {food} khau to kitna calorie hoga",
        "how much calories does {qty} {food} have",
        "{food} mein calories kya hoti hain roughly",
        "aaj ka calorie intake check karo {qty} {food} se",
    ],
    1: [  # meal_suggestion
        "low carb {meal} suggest karo",
        "healthy {meal} mein kya option hai",
        "{meal} mein kya healthy khau",
        "quick {meal} suggest karo jo healthy ho",
        "protein rich {meal} kya ho sakta hai",
        "{meal} ke liye light food batao",
        "aaj {meal} mein kya banau jo nutritious ho",
        "suggest me a healthy {meal} option",
        "what should I eat for {meal}",
        "{meal} ke liye best Indian food kya hai",
        "healthy Indian {meal} ideas batao",
        "{meal} mein kya khaun low calorie",
        "vegetarian {meal} suggestions chahiye",
        "high protein {meal} kya ho sakta hai",
        "keto {meal} options India mein",
    ],
    2: [  # diet_advice
        "{goal} ke liye kya khana chahiye",
        "{goal} diet tips batao",
        "kya khane se {goal} mein help milegi",
        "{goal} ke liye best Indian foods kaunse hain",
        "{goal} ke liye diet plan chahiye",
        "how to eat for {goal}",
        "{goal} mein konsa food avoid karein",
        "Indian diet for {goal}",
        "{goal} ke liye nutrients ki zaroorat kya hai",
        "kya khane se {goal} achieve hoga",
        "meal plan for {goal}",
        "{goal} diet mein kya important hai",
        "beginner diet guide for {goal}",
        "{goal} ke liye kitna protein lena chahiye",
        "best superfoods for {goal}",
    ],
    3: [  # nutrition_info
        "{food} mein kitna protein hai",
        "{food} ka nutrition value kya hai",
        "{food} mein kaunsa vitamin hota hai",
        "is {food} mein iron hai",
        "{food} mein carbs kitna hai",
        "{food} ka fat content kya hai",
        "{food} mein kaunse minerals hote hain",
        "nutritional information of {food}",
        "{food} ke kya health benefits hain",
        "{food} mein fiber kitna hai",
        "{food} ki nutrition profile batao",
        "what nutrients are in {food}",
        "{food} calcium ka source hai",
        "{food} ka glycemic index kya hai",
        "{food} mein omega 3 hai kya",
    ],
    4: [  # portion_control
        "ek din mein kitna {food} khana chahiye",
        "{food} ka healthy portion size kya hai",
        "kitna {food} zyada ho jata hai",
        "recommended {food} serving size kya hai",
        "{food} ki kitni matra sahi hai",
        "how much {food} should I eat daily",
        "{food} ka safe amount kya hai per day",
        "portion control {food} ke liye tips",
        "kya {qty} {food} bahut zyada hai",
        "{food} overeat se kya hoga",
        "daily limit of {food} kya hona chahiye",
        "max kitna {food} khana theek hai",
        "{food} ke liye portion guide chahiye",
        "balanced diet mein {food} kitna hona chahiye",
        "kitna {food} ek meal mein sahi hai",
    ],
}

# Regression targets per intent (realistic ranges)
REG_RANGES = {
    0: (30, 500),    # actual calories in food
    1: (250, 700),   # target meal calories
    2: (1200, 2500), # daily calorie target
    3: (2, 50),      # grams of nutrient
    4: (50, 300),    # grams per serving
}


def noise(vec):
    """Add small random noise to nutrition vector."""
    return [min(1.0, max(0.0, v + random.uniform(-0.05, 0.05))) for v in vec]


def make_sample(intent_id: int) -> dict:
    food = random.choice(FOOD_NAMES)
    food_data = FOODS[food]
    cal_per_serving = food_data[0]
    base_nutr = food_data[1:]  # 8-dim

    template = random.choice(TEMPLATES[intent_id])
    qty = random.choice(QUANTITIES)
    meal = random.choice(MEALS)
    goal = random.choice(GOALS)

    text = template.format(food=food, qty=qty, meal=meal, goal=goal)

    lo, hi = REG_RANGES[intent_id]
    if intent_id == 0:
        qty_num = 1 if qty in ("ek", "half", "1 cup", "100g", "150g", "200g", "50g", "1") else \
                  2 if qty in ("do", "2") else 3
        reg = round(cal_per_serving * qty_num + random.uniform(-10, 10), 1)
        reg = max(lo, min(hi, reg))
    else:
        reg = round(random.uniform(lo, hi), 1)

    budget_offset = random.uniform(50, 200)
    calorie_budget = round(reg + budget_offset, 1) if random.random() > 0.2 else None

    return {
        "text": text,
        "intent": intent_id,
        "regression_target": reg,
        "nutrition_vector": [round(v, 3) for v in noise(base_nutr)],
        "calorie_budget": calorie_budget,
        "language": "hinglish",
    }


def generate(n_per_intent: int, split: str) -> list:
    samples = []
    for intent_id in range(5):
        for _ in range(n_per_intent):
            samples.append(make_sample(intent_id))
    random.shuffle(samples)
    print(f"  [{split}] {len(samples)} samples generated")
    return samples


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    train_data = generate(160, "train")  # 160 × 5 = 800
    val_data   = generate(40,  "val")    # 40  × 5 = 200

    train_path = out_dir / "train.json"
    val_path   = out_dir / "val.json"

    train_path.write_text(json.dumps(train_data, indent=2, ensure_ascii=False), encoding="utf-8")
    val_path.write_text(json.dumps(val_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ Saved: {train_path}")
    print(f"✅ Saved: {val_path}")
    print("Intent distribution:")
    for i in range(5):
        n = sum(1 for s in train_data if s["intent"] == i)
        print(f"  Intent {i}: {n} train samples")
