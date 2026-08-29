"""
NutriChat AI — Chatbot Engine (NLTK + DistilBERT Hybrid)
=========================================================
IEEE-publishable multi-stage NLP pipeline:

  Stage 1: NLTK Preprocessing
           Tokenize → POS Tag → Lemmatize → Stopword removal
  Stage 2: Food Named Entity Recognition (NLTK + custom food lexicon)
  Stage 3: VADER Sentiment Analysis (tone-adaptive responses)
  Stage 4: DistilBERT Intent Classification (trained model)
  Stage 5: NLTK Natural Language Generation (varied, grounded responses)

All processing is offline. No external API required.

Intent IDs (from training):
  0 → calorie_query
  1 → meal_suggestion
  2 → diet_advice
  3 → nutrition_info
  4 → portion_control
  + keyword-only: greeting, thanks, today_intake, order_intent
"""

import random
import re
import logging
from typing import Dict, List, Optional

from api.nutrition_tracker import (
    lookup_food, get_daily_summary, suggest_next_meal, get_menu
)
from api.ml_inference import predict_intent
from api.nltk_engine import (
    run_nlp_pipeline,
    generate_calorie_response,
    generate_nutrition_response,
    generate_portion_response,
    generate_not_found_response,
    generate_missing_food_response,
)

logger = logging.getLogger("nutrichat.chatbot")


# ── Food lookup helper (uses NLTK entities + fallback) ────────────────────────

def _lookup_with_nltk(food_entities: List[str], raw_text: str) -> Optional[Dict]:
    """
    Try NLTK-extracted entities first, then fall back to regex extraction.
    """
    # Try NLTK entities (ordered by confidence)
    for food in food_entities:
        result = lookup_food(food)
        if result:
            return result
        # Try sub-phrases
        parts = food.split()
        for length in range(len(parts) - 1, 0, -1):
            result = lookup_food(" ".join(parts[:length]))
            if result:
                return result

    # Regex fallback: "X in/of <food>"
    m = re.search(r'\b(?:in|of)\s+([a-z][a-z\s]{1,30})', raw_text.lower())
    if m:
        result = lookup_food(m.group(1).strip())
        if result:
            return result

    return None


# ── Intent-specific response generators ──────────────────────────────────────

def _calorie_response(nlp: Dict, text: str, calorie_estimate: float) -> str:
    food_data = _lookup_with_nltk(nlp["food_entities"], text)
    if not food_data:
        if nlp["food_entities"]:
            return generate_not_found_response(nlp["food_entities"][0], nlp["food_entities"])
        return generate_missing_food_response("calorie_query")
    return generate_calorie_response(food_data, food_data["name"], nlp["sentiment"])


def _nutrition_response(nlp: Dict, text: str) -> str:
    food_data = _lookup_with_nltk(nlp["food_entities"], text)
    if not food_data:
        if nlp["food_entities"]:
            return generate_not_found_response(nlp["food_entities"][0], nlp["food_entities"])
        return generate_missing_food_response("nutrition_info")
    return generate_nutrition_response(food_data, food_data["name"], nlp["sentiment"])


def _portion_response(nlp: Dict, text: str) -> str:
    food_data = _lookup_with_nltk(nlp["food_entities"], text)
    if not food_data:
        if nlp["food_entities"]:
            return generate_not_found_response(nlp["food_entities"][0], nlp["food_entities"])
        return generate_missing_food_response("portion_control")
    return generate_portion_response(food_data, nlp["sentiment"])


def _diet_advice_response(text: str, nlp: Dict) -> str:
    t = text.lower()
    sentiment = nlp["sentiment"]
    opener = random.choice(["Here's personalized advice for you! ",
                            "Great focus on your health! ", ""])
    if "weight loss" in t or "slim" in t or "fat loss" in t:
        return (
            f"{opener}⚖️ **Weight Loss Diet Tips:**\n\n"
            "- Eat high-fiber foods: dal, rajma, sprouts, vegetables 🥦\n"
            "- Prefer protein: paneer, eggs, chicken, fish 🍗\n"
            "- Avoid: refined sugar, maida, fried snacks ❌\n"
            "- Stay hydrated — 8+ glasses of water 💧\n"
            "- Small, frequent meals (5–6/day) over 3 large ones\n"
            "- Safe calorie deficit: 300–500 kcal/day\n\n"
            "*💡 Try Idli, Sprout Salad, or Dal from our menu!*"
        )
    elif "muscle" in t or "bulking" in t or "gain" in t:
        return (
            f"{opener}💪 **Muscle Gain Diet Tips:**\n\n"
            "- Protein target: **1.6–2.2g per kg body weight** daily\n"
            "- Best sources: Chicken, Fish, Paneer, Eggs, Rajma 🍗\n"
            "- Eat carbs pre/post workout for energy 🌾\n"
            "- Don't skip breakfast — eggs or oats 🥚\n"
            "- Calorie surplus: 200–300 kcal/day for lean gains\n\n"
            "*💡 Try Tandoori Chicken or Paneer Butter Masala!*"
        )
    elif "diabetes" in t or "blood sugar" in t:
        return (
            f"{opener}🩺 **Diabetes Management Diet:**\n\n"
            "- Low-glycemic foods: brown rice, oats, dal, vegetables 🥗\n"
            "- Avoid: white rice, maida, sugary drinks ❌\n"
            "- Eat every 3–4 hours to stabilize blood sugar\n"
            "- High fiber slows glucose absorption (sprouts, rajma)\n\n"
            "*💡 Try: Oats Bowl, Sprout Salad, or Moong Dal!*\n"
            "*⚠️ Always consult your doctor for a personalized plan.*"
        )
    elif "heart" in t or "cholesterol" in t or "blood pressure" in t:
        return (
            f"{opener}❤️ **Heart Health Diet Tips:**\n\n"
            "- Omega-3 rich foods: Fish, walnuts, flaxseeds 🐟\n"
            "- Fiber-rich foods reduce cholesterol: oats, dal, rajma\n"
            "- Limit sodium: avoid pickles, papad, processed foods 🧂\n"
            "- Avoid trans fats: vanaspati, deep fried foods ❌\n\n"
            "*💡 Try: Fish Curry, Oats Bowl, or Coconut Water!*"
        )
    elif "keto" in t:
        return (
            f"{opener}🥩 **Keto Diet Guide:**\n\n"
            "- Very low carb (<50g/day), high fat, moderate protein\n"
            "- Eat: Eggs, Paneer, Chicken, Fish, Coconut oil, Ghee 🥚\n"
            "- Avoid: Rice, Roti, Dal, Fruits, Sugar ❌\n"
            "- Good Indian keto: Egg Bhurji, Paneer, Tandoori Chicken\n\n"
            "*💡 Try: Boiled Eggs or Tandoori Chicken from our menu!*"
        )
    elif "vegan" in t or "plant" in t:
        return (
            f"{opener}🌱 **Vegan Diet Tips:**\n\n"
            "- Protein: Dal, Rajma, Chana, Tofu, Sprouts 💪\n"
            "- B12 supplement is essential on a vegan diet\n"
            "- Iron: Spinach, Rajma, Sesame + Vitamin C for absorption\n"
            "- Calcium: Ragi, Sesame, Fortified plant milk 🦴\n\n"
            "*💡 Try: Dal Tadka, Sprout Salad, or Rajma!*"
        )
    elif "thyroid" in t or "pcos" in t:
        return (
            f"{opener}🏥 **Thyroid/PCOS Diet Tips:**\n\n"
            "- Avoid excess soy and raw cruciferous veggies (thyroid)\n"
            "- Low-GI foods help with PCOS: oats, dal, vegetables\n"
            "- Anti-inflammatory: turmeric, ginger, berries\n"
            "- Regular small meals stabilize hormones\n\n"
            "*⚠️ Consult a doctor for a personalized plan.*"
        )
    else:
        return (
            f"{opener}🌟 **General Healthy Eating Tips:**\n\n"
            "- Eat a rainbow: all colored vegetables 🌈\n"
            "- Plate rule: 50% vegetables | 25% protein | 25% carbs\n"
            "- Limit processed and packaged foods\n"
            "- Cook at home using minimal oil\n"
            "- Drink 2–3 liters of water daily 💧\n"
            "- Sleep 7–8 hours — affects metabolism! 😴\n\n"
            "*Tell me your specific goal (weight loss, muscle gain, diabetes, "
            "heart health, keto) for personalized advice!*"
        )


def _meal_suggestion_response(username: str, daily_goal: int, goal: str, nlp: Dict) -> str:
    result = suggest_next_meal(username, daily_goal, goal)
    sentiment = nlp["sentiment"]
    opener = random.choice(_OPENERS_MEAL[sentiment["label"]])
    msg = opener + result["message"] + "\n\n"
    if result["suggestions"]:
        for item in result["suggestions"]:
            msg += (
                f"{item.get('emoji', '🍽️')} **{item['name']}** — "
                f"{item['calories_kcal']} kcal | ₹{item['price']}\n"
            )
        msg += "\n*Type 'order [food name]' to add to your cart!*"
    return msg

_OPENERS_MEAL = {
    "positive": ["Great timing! Here are some meal ideas for you:\n\n",
                 "You're on a roll! Here's what I'd suggest:\n\n"],
    "neutral":  ["Based on your daily goal, here are suggestions:\n\n",
                 "Here are some healthy options for your next meal:\n\n"],
    "negative": ["Let's get you back on track! Try these:\n\n",
                 "No worries — here are some good options:\n\n"],
}


def _today_intake_response(username: str, daily_goal: int) -> str:
    summary = get_daily_summary(username, daily_goal)
    cals    = summary["total_calories"]
    remain  = summary["remaining_calories"]
    pct     = summary["percent_done"]
    meals   = summary["meals_count"]

    bar_filled = int(pct / 10)
    bar = "🟩" * bar_filled + "⬜" * (10 - bar_filled)

    response = (
        f"📊 **Today's Nutrition Summary**\n\n"
        f"{bar} {pct}%\n\n"
        f"- Calories consumed: **{cals:.0f} kcal**\n"
        f"- Daily goal: {daily_goal} kcal\n"
        f"- Remaining: **{remain:.0f} kcal**\n"
        f"- Meals logged: {meals}\n\n"
        f"**Macros today:**\n"
        f"💪 Protein: {summary['protein_g']:.1f}g\n"
        f"🌾 Carbs: {summary['carb_g']:.1f}g\n"
        f"🧈 Fat: {summary['fat_g']:.1f}g\n"
    )
    if pct >= 100:
        response += "\n✅ Daily goal reached! Great job 🎉"
    elif pct >= 75:
        response += f"\n💡 Almost there! {remain:.0f} kcal left."
    elif pct < 30 and meals == 0:
        response += "\n⚠️ You haven't logged any meals yet today!"
    return response


def _greeting_response(display_name: str) -> str:
    greetings = ["Namaste", "Hello", "Hey", "Hi"]
    return (
        f"👋 {random.choice(greetings)}, **{display_name}**! I'm NutriChat AI 🌿\n\n"
        f"I can help you with:\n"
        f"- 🔥 Calorie lookup — *'calories in paneer'*\n"
        f"- 🧪 Nutrition info — *'protein in chicken'*\n"
        f"- 🍽️ Meal suggestions — *'suggest healthy lunch'*\n"
        f"- 💊 Diet advice — *'diet tips for weight loss'*\n"
        f"- 📊 Daily intake — *'what have I eaten today'*\n"
        f"- 📏 Portion guide — *'how much rice should I eat'*\n\n"
        f"What would you like to know? 😊"
    )


def _thanks_response(nlp: Dict) -> str:
    if nlp["sentiment"]["label"] == "positive":
        return random.choice([
            "Happy to help! Keep up the great work — stay healthy! 💪",
            "Happy to help! Small consistent choices lead to big results 🌟",
            "Happy to help anytime! Eating right is the best investment you can make 🥗",
        ])
    return random.choice([
        "Happy to help! Stay healthy 🌿",
        "Glad I could help! Keep tracking your nutrition — healthy habits matter 📊",
        "Happy to help anytime! Keep up the healthy lifestyle 💪",
    ])


def _bot_info_response() -> str:
    return (
        "🤖 **About NutriChat AI**\n\n"
        "I'm a hybrid NLP + Deep Learning nutrition assistant. Here's how I work:\n\n"
        "**Pipeline:**\n"
        "1. 🔤 *NLTK Preprocessing* — I tokenize, lemmatize & POS-tag your message\n"
        "2. 🥗 *Food NER* — I extract food names using a 500+ item Indian food lexicon\n"
        "3. 😊 *VADER Sentiment* — I detect your tone to adapt my response\n"
        "4. 🧠 *DistilBERT Classifier* — A trained transformer classifies your intent\n"
        "5. 📊 *USDA/ICMR-NIN Database* — I look up real nutrition data (no guessing!)\n\n"
        "**I can answer:**\n"
        "- 🔥 Calories in any food — *'calories in paneer'*\n"
        "- 🧪 Full nutrition info — *'nutrition info about rice'*\n"
        "- 📏 Portion guidance — *'how much dal should I eat'*\n"
        "- 💊 Diet advice — *'diet tips for weight loss'*\n"
        "- 🍽️ Meal suggestions — *'suggest a healthy lunch'*\n"
        "- 📊 Daily tracking — *'what have I eaten today'*\n\n"
        "*Built for IEEE research — fully offline, no external APIs!* 🌿"
    )


def _general_response(text: str, nlp: Dict) -> str:
    """Fallback for unrecognized / out-of-domain queries."""
    nouns = [tok for tok, tag in nlp["pos_tags"] if tag in ("NN", "NNS", "NNP")]
    noun_hint = f" about *{', '.join(nouns[:3])}*" if nouns else ""
    return (
        f"🤖 I'm sorry, I **didn't train on this part**{noun_hint}.\n\n"
        f"My training covers only **nutrition & food topics**. I can help you with:\n"
        f"- 🔥 *'calories in paneer'* — calorie lookup\n"
        f"- 🧪 *'protein in chicken'* — nutrition info\n"
        f"- 📏 *'how much rice should I eat'* — portion guide\n"
        f"- 💊 *'diet tips for weight loss'* — diet advice\n"
        f"- 🍽️ *'suggest a healthy lunch'* — meal ideas\n"
        f"- 📊 *'what have I eaten today'* — daily tracker\n\n"
        f"Please rephrase your question around food or nutrition! 🌿"
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def process_message(
    text: str,
    username: str,
    display_name: str,
    daily_goal: int = 2000,
    goal: str = "general fitness",
) -> Dict:
    """
    Hybrid NLP pipeline:
      1. NLTK preprocessing (tokenize, POS, lemmatize)
      2. NLTK food NER + VADER sentiment
      3. DistilBERT intent classification
      4. NLTK NLG response generation using real nutrition DB
    """
    # ── Stage 1–3: NLTK pipeline ─────────────────────────────────────────────
    nlp = run_nlp_pipeline(text)

    logger.debug(
        f"[NLTK] entities={nlp['food_entities']} "
        f"sentiment={nlp['sentiment']['label']} "
        f"lemmas={nlp['lemmas'][:5]}"
    )

    # ── Stage 4: DistilBERT intent classification ─────────────────────────────
    prediction    = predict_intent(text)
    intent        = prediction["intent"]
    source        = prediction["source"]
    confidence    = prediction.get("confidence", 0.0)
    calorie_est   = prediction.get("calorie_estimate", 0.0)

    logger.debug(
        f"[DistilBERT] intent={intent} conf={confidence:.2f} src={source}"
    )

    # ── Stage 5: NLTK NLG response ───────────────────────────────────────────
    if intent == "greeting":
        response = _greeting_response(display_name)
    elif intent == "thanks":
        response = _thanks_response(nlp)
    elif intent == "bot_info":
        response = _bot_info_response()
    elif intent == "calorie_query":
        response = _calorie_response(nlp, text, calorie_est)
    elif intent == "nutrition_info":
        response = _nutrition_response(nlp, text)
    elif intent == "diet_advice":
        response = _diet_advice_response(text, nlp)
    elif intent == "portion_control":
        response = _portion_response(nlp, text)
    elif intent == "meal_suggestion":
        response = _meal_suggestion_response(username, daily_goal, goal, nlp)
    elif intent == "today_intake":
        response = _today_intake_response(username, daily_goal)
    elif intent == "order_intent":
        response = (
            "🛒 Ready to order! Browse the menu on the right and click "
            "**'Add to Cart'** on any item. Then click **Checkout** to place your order!"
        )
    else:
        # Out-of-domain or unrecognized query
        logger.info(f"[Fallback] Unrecognized intent for: '{text[:60]}'")
        response = _general_response(text, nlp)

    return {
        "intent":           intent,
        "response":         response,
        "ml_source":        f"distilbert+nltk ({source})",
        "confidence":       round(confidence, 3),
        "calorie_estimate": round(calorie_est, 1),
        "nlp_meta": {
            "food_entities": nlp["food_entities"],
            "sentiment":     nlp["sentiment"]["label"],
            "lemmas":        nlp["lemmas"][:8],
        },
    }
