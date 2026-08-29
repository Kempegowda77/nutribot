# -*- coding: utf-8 -*-
"""
NutriChat AI - Comprehensive Chatbot Test Suite
===============================================
Tests every intent, edge case, Hinglish input, emotional tone,
misspellings, and fallback behaviour.

Run:
    python test_chatbot.py
    python test_chatbot.py --verbose
    python test_chatbot.py --category greeting calorie_query
    python test_chatbot.py --smoke "calories in paneer"
"""

import sys
import io
import os
import time
import argparse
from pathlib import Path

# Force UTF-8 stdout so emojis in responses don't crash on Windows cp1252
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# Bootstrap project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ANSI colour helpers (strip=True avoids colorama writing raw ANSI on Windows)
try:
    import colorama
    colorama.init(strip=False, convert=True)
    GREEN  = "\033[92m"; RED    = "\033[91m"
    YELLOW = "\033[93m"; CYAN   = "\033[96m"
    BOLD   = "\033[1m";  RESET  = "\033[0m"
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""

# ── Dummy user config ────────────────────────────────────────────────────────
TEST_USER     = "test_user"
TEST_NAME     = "Tester"
TEST_GOAL     = 2000
TEST_FIT_GOAL = "general fitness"

# ── Import engine ────────────────────────────────────────────────────────────
print(f"{CYAN}{BOLD}Loading NutriChat AI engine...{RESET}")
try:
    from api.chatbot_engine import process_message
    print(f"{GREEN}[OK] Engine imported successfully{RESET}\n")
except Exception as exc:
    print(f"{RED}[FAIL] Could not import engine: {exc}{RESET}")
    sys.exit(1)


# =============================================================================
# Test-case builder
# =============================================================================

def tc(category, text, expected_intent=None, must_contain=None,
       must_not_contain=None, description=""):
    return {
        "category":        category,
        "text":            text,
        "expected_intent": expected_intent,
        "must_contain":    must_contain or [],
        "must_not_contain": must_not_contain or [],
        "description":     description or text,
    }


TEST_CASES = [

    # ── GREETING ─────────────────────────────────────────────────────────────
    tc("greeting", "hi",               "greeting", ["NutriChat", "help"]),
    tc("greeting", "hello",            "greeting", ["NutriChat"]),
    tc("greeting", "hey",              "greeting", ["NutriChat"]),
    tc("greeting", "hii",              "greeting", ["NutriChat"]),
    tc("greeting", "namaste",          "greeting", ["NutriChat"]),
    tc("greeting", "namaskar",         "greeting", ["NutriChat"]),
    tc("greeting", "good morning",     "greeting", ["NutriChat"]),
    tc("greeting", "good evening",     "greeting", ["NutriChat"]),
    tc("greeting", "good afternoon",   "greeting", ["NutriChat"]),
    tc("greeting", "Hello!",           "greeting", ["NutriChat"]),

    # ── THANKS ───────────────────────────────────────────────────────────────
    tc("thanks", "thank you",          "thanks",   ["help"]),    # 'Happy to help!' always present
    tc("thanks", "thanks",             "thanks",   []),
    tc("thanks", "shukriya",           "thanks",   []),
    tc("thanks", "dhanyawad",          "thanks",   []),
    tc("thanks", "great job",          "thanks",   []),
    tc("thanks", "awesome, thanks!",   "thanks",   []),
    tc("thanks", "perfect!",           "thanks",   []),

    # ── BOT INFO ─────────────────────────────────────────────────────────────
    tc("bot_info", "who are you",               "bot_info", ["NutriChat", "pipeline"]),
    tc("bot_info", "how do you work",           "bot_info", ["NLTK", "DistilBERT"]),
    tc("bot_info", "what can you do",           "bot_info", ["calorie", "nutrition"]),
    tc("bot_info", "tell me about yourself",    "bot_info", ["NutriChat"]),
    tc("bot_info", "what is nutrichat",         "bot_info", ["NutriChat"]),
    tc("bot_info", "what are your features",    "bot_info", ["calorie"]),

    # ── CALORIE QUERY ─────────────────────────────────────────────────────────
    tc("calorie_query", "calories in paneer",              "calorie_query", ["kcal"]),
    tc("calorie_query", "how many calories in rice",       "calorie_query", ["kcal"]),
    tc("calorie_query", "calorie in 1 roti",               "calorie_query", ["kcal"]),
    tc("calorie_query", "how many calories does dal have", "calorie_query", ["kcal"]),
    tc("calorie_query", "energy in chicken",               "calorie_query", ["kcal"]),
    tc("calorie_query", "kitna calorie hai paneer mein",   "calorie_query", ["kcal"]),
    tc("calorie_query", "egg calorie",                     "calorie_query", ["kcal"]),
    tc("calorie_query", "banana calories",                 "calorie_query", ["kcal"]),
    tc("calorie_query", "calories in oats",                "calorie_query", ["kcal"]),
    tc("calorie_query", "how many kcal in milk",           "calorie_query", ["kcal"]),
    tc("calorie_query", "calorie count for biryani",       "calorie_query", ["kcal"]),
    # Hinglish
    tc("calorie_query", "anda me kitne calorie hain",      "calorie_query", ["kcal"],
       description="Hinglish: egg calories"),
    tc("calorie_query", "chawal me calorie",               "calorie_query", ["kcal"],
       description="Hinglish: rice calories"),
    tc("calorie_query", "doodh calorie",                   "calorie_query", ["kcal"],
       description="Hinglish: milk calories"),

    # ── NUTRITION INFO ────────────────────────────────────────────────────────
    tc("nutrition_info", "protein in chicken",                    "nutrition_info", ["protein", "g"]),
    tc("nutrition_info", "nutrition info about rice",             "nutrition_info", ["protein"]),
    tc("nutrition_info", "what is the protein content of paneer", "nutrition_info", ["protein"]),
    tc("nutrition_info", "iron in spinach",                       "nutrition_info", ["spinach"]),
    tc("nutrition_info", "carbs in banana",                       "nutrition_info", ["carb"]),
    tc("nutrition_info", "fat in ghee",                           "nutrition_info", ["fat"]),
    tc("nutrition_info", "fiber in oats",                         "nutrition_info", ["fiber"]),
    tc("nutrition_info", "vitamin c in orange",                   "nutrition_info", []),
    tc("nutrition_info", "calcium in milk",                       "nutrition_info", []),
    tc("nutrition_info", "nutrition details of dal",              "nutrition_info", ["protein"]),
    tc("nutrition_info", "macros of egg",                         "nutrition_info", ["protein"]),
    # Hinglish
    tc("nutrition_info", "machli mein protein kitna hai",  "nutrition_info", ["protein"],
       description="Hinglish: fish protein"),
    tc("nutrition_info", "dahi ka nutrition",              "nutrition_info", ["protein"],
       description="Hinglish: curd nutrition"),

    # ── PORTION CONTROL ───────────────────────────────────────────────────────
    tc("portion_control", "how much rice should I eat",       "portion_control", ["portion"]),
    tc("portion_control", "how much dal should I eat daily",  "portion_control", ["portion"]),
    tc("portion_control", "portion size of paneer",           "portion_control", ["portion"]),
    tc("portion_control", "recommended intake of roti",       "portion_control", ["portion"]),
    tc("portion_control", "serving size of chicken",          "portion_control", ["portion"]),
    tc("portion_control", "kitna rice khana chahiye",         "portion_control", ["portion"],
       description="Hinglish portion"),
    tc("portion_control", "daily limit of sugar",             "portion_control", ["portion"]),
    tc("portion_control", "how many rotis per day",           "portion_control", ["portion"]),

    # ── DIET ADVICE ───────────────────────────────────────────────────────────
    tc("diet_advice", "diet tips for weight loss",        "diet_advice", ["weight", "calorie"]),
    tc("diet_advice", "how to lose weight through diet",  "diet_advice", ["weight"]),
    tc("diet_advice", "diet for muscle gain",             "diet_advice", ["protein", "muscle"]),
    tc("diet_advice", "diet plan for diabetes",           "diet_advice", ["diabetes", "sugar"]),
    tc("diet_advice", "heart healthy diet",               "diet_advice", ["heart"]),
    tc("diet_advice", "keto diet guide",                  "diet_advice", ["keto", "carb"]),
    tc("diet_advice", "vegan diet tips",                  "diet_advice", ["vegan", "protein"]),
    tc("diet_advice", "diet for thyroid",                 "diet_advice", ["thyroid"]),
    tc("diet_advice", "diet for pcos",                    "diet_advice", ["PCOS"]),
    tc("diet_advice", "healthy eating tips",              "diet_advice", ["vegetable"]),
    tc("diet_advice", "what to eat for cholesterol",      "diet_advice", ["cholesterol"],
       description="Cholesterol diet query (what to eat for X)"),
    tc("diet_advice", "blood pressure diet",              "diet_advice", ["sodium"]),
    tc("diet_advice", "plant based diet",                 "diet_advice", ["vegan", "protein"]),

    # ── MEAL SUGGESTION ───────────────────────────────────────────────────────
    tc("meal_suggestion", "suggest a healthy lunch",          "meal_suggestion", ["kcal"]),
    tc("meal_suggestion", "recommend a meal for dinner",      "meal_suggestion", ["kcal"]),
    tc("meal_suggestion", "what should I eat for breakfast",  "meal_suggestion", ["kcal"]),
    tc("meal_suggestion", "kya khau aaj",                     "meal_suggestion", ["kcal"],
       description="Hinglish: what to eat today"),
    tc("meal_suggestion", "healthy food options",             "meal_suggestion", ["kcal"]),
    tc("meal_suggestion", "food suggestions for weight loss", None,              ["weight"],
       description="Ambiguous: diet_advice or meal_suggestion both valid"),
    tc("meal_suggestion", "meal plan for today",              "meal_suggestion", ["kcal"]),
    tc("meal_suggestion", "next meal idea",                   "meal_suggestion", ["kcal"]),

    # ── TODAY INTAKE ──────────────────────────────────────────────────────────
    tc("today_intake", "what have I eaten today",      "today_intake", ["calorie", "goal"]),
    tc("today_intake", "today's calorie intake",       "today_intake", ["calorie"]),
    tc("today_intake", "show my daily intake",         "today_intake", ["calorie"]),
    tc("today_intake", "how much have I eaten",        "today_intake", ["calorie"]),
    tc("today_intake", "aaj ka khana track karo",      "today_intake", ["calorie"],
       description="Hinglish: track today food"),
    tc("today_intake", "mera calorie today",           "today_intake", ["calorie"],
       description="Hinglish: my calorie today"),
    tc("today_intake", "my intake log",                "today_intake", ["calorie"]),

    # ── ORDER INTENT ──────────────────────────────────────────────────────────
    tc("order_intent", "I want to order food",         "order_intent", ["cart", "menu"]),
    tc("order_intent", "show me the menu",             "order_intent", ["menu"]),
    tc("order_intent", "add to cart",                  "order_intent", ["cart"]),
    tc("order_intent", "place order",                  "order_intent", ["cart", "menu"]),
    tc("order_intent", "mujhe dal chahiye",             "order_intent", ["cart"],
       description="Hinglish: I want dal"),
    tc("order_intent", "buy paneer butter masala",      "order_intent", ["cart"]),

    # ── EDGE CASES ────────────────────────────────────────────────────────────
    tc("edge_case", "hello, I want calories in paneer",   None, ["kcal"],
       description="Greeting + calorie hybrid"),
    tc("edge_case", "how many protein in chicken please", "nutrition_info", ["protein"],
       description="Slightly ungrammatical"),
    tc("edge_case", "PANEER CALORIES",                    "calorie_query",  ["kcal"],
       description="ALL CAPS input"),
    tc("edge_case", "calories  in   rice",                "calorie_query",  ["kcal"],
       description="Extra whitespace"),
    tc("edge_case", "what's the calorie in panneer",      "calorie_query",  [],
       description="Misspelled food name"),
    tc("edge_case", "give diet tips",                     "diet_advice",    [],
       description="Short diet query"),
    tc("edge_case", "I'm trying to lose fat",             "diet_advice",    ["weight"],
       description="Fat-loss phrasing"),

    # ── FALLBACK / OUT-OF-DOMAIN ──────────────────────────────────────────────
    tc("fallback", "abcdefgh random input",           None, [],
       description="Complete gibberish"),
    tc("fallback", "what is the capital of France",   None, [],
       description="Out-of-domain"),
    tc("fallback", "play a song",                     None, [],
       description="Irrelevant request"),
    tc("fallback", "tell me a joke",                  None, [],
       description="Out-of-domain"),
    tc("fallback", "123456",                          None, [],
       description="Numbers only"),
    tc("fallback", "   ",                             None, [],
       description="Whitespace-only (no crash)"),

    # ── SENTIMENT / EMOTIONAL TONE ────────────────────────────────────────────
    tc("sentiment", "I love paneer! calories please!",          "calorie_query",  ["kcal"],
       description="Positive sentiment"),
    tc("sentiment", "I hate dieting but need weight loss tips", "diet_advice",    ["weight"],
       description="Negative sentiment"),
    tc("sentiment", "not sure what to eat today",               "meal_suggestion", ["kcal"],
       description="Uncertain tone"),
    tc("sentiment", "please suggest something healthy",         "meal_suggestion", ["kcal"],
       description="Polite request"),
    tc("sentiment", "urgent! I need to know rice calories",     "calorie_query",  ["kcal"],
       description="Urgent tone"),

    # ── HINGLISH COMPREHENSIVE ────────────────────────────────────────────────
    tc("hinglish", "anda me kitna protein hai",      "nutrition_info", ["protein"],
       description="Egg protein Hinglish"),
    tc("hinglish", "chawal kitna calorie",           "calorie_query",  ["kcal"],
       description="Rice calorie Hinglish"),
    tc("hinglish", "dahi ka protein content",        "nutrition_info", ["protein"],
       description="Curd protein Hinglish"),
    tc("hinglish", "mujhe weight loss diet chahiye", "diet_advice",    ["weight"],
       description="Weight loss diet Hinglish"),
    tc("hinglish", "aaj ka khana log karo",         "today_intake",    ["calorie"],
       description="Log today food Hinglish"),
    tc("hinglish", "kya khau healthy",              "meal_suggestion", ["kcal"],
       description="What to eat healthy Hinglish"),
    tc("hinglish", "aloo calorie",                  "calorie_query",   ["kcal"],
       description="Potato calorie Hinglish"),
    tc("hinglish", "machli protein",                "nutrition_info",  ["protein"],
       description="Fish protein Hinglish"),
]


# =============================================================================
# Test runner
# =============================================================================

def run_single(case: dict) -> dict:
    text = case["text"].strip()

    # Whitespace-only: just ensure no crash
    if not text:
        try:
            result = process_message(" ", TEST_USER, TEST_NAME, TEST_GOAL, TEST_FIT_GOAL)
            return {"pass": True, "errors": [], "intent": result.get("intent",""),
                    "response": result.get("response",""), "elapsed_ms": 0}
        except Exception as exc:
            return {"pass": False, "errors": [str(exc)], "intent": "",
                    "response": "", "elapsed_ms": 0}

    try:
        t0 = time.perf_counter()
        result = process_message(text, TEST_USER, TEST_NAME, TEST_GOAL, TEST_FIT_GOAL)
        elapsed = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        return {"pass": False, "errors": [f"EXCEPTION: {exc}"],
                "intent": "", "response": "", "elapsed_ms": 0}

    response = result.get("response", "")
    intent   = result.get("intent", "")
    errors   = []

    if case["expected_intent"] and intent != case["expected_intent"]:
        errors.append(
            f"Intent mismatch: got '{intent}', expected '{case['expected_intent']}'"
        )
    for kw in case["must_contain"]:
        if kw.lower() not in response.lower():
            errors.append(f"Missing keyword in response: '{kw}'")
    for kw in case["must_not_contain"]:
        if kw.lower() in response.lower():
            errors.append(f"Forbidden keyword found: '{kw}'")
    if not response.strip():
        errors.append("Empty response")

    return {
        "pass":       len(errors) == 0,
        "errors":     errors,
        "intent":     intent,
        "response":   response,
        "result":     result,
        "elapsed_ms": elapsed,
    }


def run_all(categories=None, verbose=False):
    cases = TEST_CASES
    if categories:
        cases = [c for c in TEST_CASES if c["category"] in categories]

    by_cat = {}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c)

    total = passed = failed = 0
    total_ms = 0.0
    all_results = []

    for cat, cat_cases in by_cat.items():
        print(f"\n{'='*70}")
        print(f"  CATEGORY: {cat.upper()}  ({len(cat_cases)} tests)")
        print(f"{'='*70}")

        cat_pass = cat_fail = 0
        for case in cat_cases:
            res = run_single(case)
            total += 1
            total_ms += res.get("elapsed_ms", 0)

            if res["pass"]:
                passed += 1; cat_pass += 1
                status = f"{GREEN}[PASS]{RESET}"
            else:
                failed += 1; cat_fail += 1
                status = f"{RED}[FAIL]{RESET}"

            label   = (case["description"] or case["text"])[:54]
            elapsed = res.get("elapsed_ms", 0)
            intent  = res.get("intent", "--")

            print(f"  {status}  [{elapsed:5.0f}ms]  [{intent:<18}]  {label}")

            if not res["pass"]:
                if res.get("errors"):
                    for err in res["errors"]:
                        print(f"           {YELLOW}-> {err}{RESET}")

            if verbose and res.get("response"):
                short = res["response"][:110].replace("\n", " ")
                print(f"           {CYAN}Response: {short}...{RESET}")

            all_results.append({"case": case, **res})

        col = GREEN if cat_fail == 0 else (YELLOW if cat_pass > 0 else RED)
        print(f"\n  {col}Category: {cat_pass}/{len(cat_cases)} passed{RESET}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    pct   = round(passed / total * 100, 1) if total else 0
    p_col = GREEN if pct >= 90 else (YELLOW if pct >= 70 else RED)

    print(f"  Total tests  : {total}")
    print(f"  {GREEN}Passed       : {passed}{RESET}")
    print(f"  {RED}Failed       : {failed}{RESET}")
    print(f"  {p_col}Pass rate    : {pct}%{RESET}")
    if total:
        print(f"  Avg latency  : {total_ms/total:.0f} ms")
    print(f"{'='*70}")

    # ── Intent coverage ───────────────────────────────────────────────────────
    EXPECTED = {"greeting","thanks","bot_info","calorie_query","nutrition_info",
                "portion_control","diet_advice","meal_suggestion","today_intake","order_intent"}
    seen    = {r.get("intent") for r in all_results if r.get("intent")}
    missing = EXPECTED - seen
    if missing:
        print(f"\n{YELLOW}[WARN] Intents never triggered: {missing}{RESET}")
    else:
        print(f"\n{GREEN}[OK] All expected intents were triggered.{RESET}")

    # ── Failed detail ─────────────────────────────────────────────────────────
    failures = [r for r in all_results if not r["pass"]]
    if failures:
        print(f"\n{RED}{BOLD}--- FAILED TEST DETAIL ---{RESET}")
        for r in failures:
            c = r["case"]
            print(f"\n  Input    : {c['text']}")
            print(f"  Category : {c['category']}")
            print(f"  Expected : intent={c['expected_intent']}, contains={c['must_contain']}")
            print(f"  Got      : intent={r.get('intent','--')}")
            for err in r.get("errors", []):
                print(f"  Issue    : {err}")
            if r.get("response"):
                short = r["response"][:200].replace("\n", " ")
                print(f"  Response : {short}")

    print()
    return passed, failed, total


# =============================================================================
# Smoke test helper
# =============================================================================

def smoke_test(message: str):
    print(f"\n{BOLD}SMOKE TEST{RESET}")
    print(f"Input    : {message}")
    try:
        result = process_message(message, TEST_USER, TEST_NAME, TEST_GOAL, TEST_FIT_GOAL)
        print(f"Intent   : {result['intent']}")
        print(f"Source   : {result.get('ml_source','--')}")
        print(f"Conf     : {result.get('confidence', 0):.3f}")
        print(f"Entities : {result.get('nlp_meta', {}).get('food_entities', [])}")
        print(f"Sentiment: {result.get('nlp_meta', {}).get('sentiment', '--')}")
        print(f"\nResponse :\n{result['response']}\n")
    except Exception as exc:
        print(f"{RED}Error: {exc}{RESET}")


# =============================================================================
# Entry point
# =============================================================================

ALL_CATEGORIES = [
    "greeting","thanks","bot_info","calorie_query","nutrition_info",
    "portion_control","diet_advice","meal_suggestion","today_intake",
    "order_intent","edge_case","fallback","sentiment","hinglish",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutriChat AI Test Suite")
    parser.add_argument("--verbose",  "-v", action="store_true",
                        help="Print truncated response for every test")
    parser.add_argument("--category", "-c", nargs="+", choices=ALL_CATEGORIES,
                        help="Run only specific category/categories")
    parser.add_argument("--smoke",    "-s", metavar="MSG",
                        help="Run a single quick test (no full suite)")
    args = parser.parse_args()

    if args.smoke:
        smoke_test(args.smoke)
        sys.exit(0)

    passed, failed, total = run_all(categories=args.category, verbose=args.verbose)
    sys.exit(0 if failed == 0 else 1)
