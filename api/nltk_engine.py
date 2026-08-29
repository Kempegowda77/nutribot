"""
NutriChat AI — NLTK NLP Engine
================================
IEEE-publishable NLP pipeline using NLTK:

  Stage 1 — Preprocessing  : Tokenize → POS Tag → Lemmatize → Stopword removal
  Stage 2 — Food NER       : Noun extraction + food lexicon matching + WordNet expansion
  Stage 3 — Sentiment      : VADER polarity detection for tone adjustment
  Stage 4 — NLG            : Linguistically varied, intent-aware response generation

All processing is offline — no external API required.
"""

import random
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("nutrichat.nltk")

# ── NLTK bootstrap (download required corpora once) ────────────────────────────

def _bootstrap_nltk():
    import nltk
    packages = [
        "punkt", "punkt_tab", "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "wordnet", "stopwords", "vader_lexicon", "omw-1.4",
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

_bootstrap_nltk()

# ── Lazy imports ───────────────────────────────────────────────────────────────

def _get_nltk_tools():
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
    from nltk.corpus import stopwords, wordnet
    from nltk.sentiment import SentimentIntensityAnalyzer
    return word_tokenize, WordNetLemmatizer(), pos_tag, stopwords, wordnet, SentimentIntensityAnalyzer()

# ── Indian food lexicon (ICMR-NIN aligned) ────────────────────────────────────

FOOD_LEXICON = {
    # Cereals & grains
    "rice", "wheat", "roti", "chapati", "paratha", "naan", "dosa", "idli",
    "upma", "poha", "oats", "bread", "maida", "atta", "semolina", "suji",
    "jowar", "bajra", "ragi", "corn", "maize", "barley",
    # Pulses & legumes
    "dal", "lentil", "rajma", "chana", "moong", "masoor", "urad", "toor",
    "chickpea", "pea", "beans", "soybean", "tofu", "sprout",
    # Vegetables
    "spinach", "palak", "broccoli", "carrot", "tomato", "potato", "onion",
    "cauliflower", "gobi", "capsicum", "bhindi", "okra", "eggplant", "brinjal",
    "pumpkin", "cucumber", "beetroot", "mushroom", "peas", "methi",
    # Fruits
    "banana", "apple", "mango", "orange", "papaya", "guava", "grapes",
    "watermelon", "pomegranate", "lemon", "coconut", "date", "fig",
    # Dairy & protein
    "milk", "curd", "yogurt", "paneer", "cheese", "butter", "ghee", "cream",
    "egg", "chicken", "mutton", "fish", "salmon", "tuna", "prawn", "shrimp",
    # Snacks & dishes
    "samosa", "pakora", "vada", "biryani", "pulao", "curry", "sabzi",
    "khichdi", "halwa", "ladoo", "barfi", "kheer", "raita",
    # Nuts & seeds
    "almond", "cashew", "walnut", "peanut", "sesame", "flaxseed",
    # Oils & fats
    "oil", "groundnut", "mustard", "sunflower", "olive",
    # Beverages
    "tea", "coffee", "juice", "lassi", "chaas", "buttermilk",
}

# Hinglish food aliases → canonical name
HINGLISH_FOOD_MAP = {
    "anda": "egg", "murgi": "chicken", "machli": "fish", "chawal": "rice",
    "doodh": "milk", "dahi": "curd", "makhan": "butter", "cheeni": "sugar",
    "namak": "salt", "haldi": "turmeric", "mirch": "chili", "pyaz": "onion",
    "lahsun": "garlic", "adrak": "ginger", "nimbu": "lemon", "aloo": "potato",
    "tamatar": "tomato", "palak": "spinach", "gobhi": "cauliflower",
    "matar": "peas", "rajma": "kidney beans", "chana": "chickpea",
    "moong": "moong dal", "urad": "urad dal", "arhar": "toor dal",
    "gehun": "wheat", "bajra": "pearl millet", "jowar": "sorghum",
    "makki": "corn", "ragi": "finger millet", "sabudana": "sago",
}

# ── Stage 1: Preprocessing ─────────────────────────────────────────────────────

def preprocess(text: str) -> Dict:
    """
    Full NLTK preprocessing pipeline.
    Returns tokens, lemmas, POS tags, and cleaned text.
    """
    word_tokenize, lemmatizer, pos_tag, stopwords, wordnet, _ = _get_nltk_tools()

    # Normalize Hinglish → English
    t = text.lower().strip()
    for hindi, english in HINGLISH_FOOD_MAP.items():
        t = re.sub(r'\b' + hindi + r'\b', english, t)

    # Tokenize
    tokens = word_tokenize(t)

    # POS tag
    pos_tags = pos_tag(tokens)

    # Lemmatize with POS hint
    def _get_wn_pos(tag):
        if tag.startswith('J'): return wordnet.ADJ
        if tag.startswith('V'): return wordnet.VERB
        if tag.startswith('R'): return wordnet.ADV
        return wordnet.NOUN

    stop_words = set(stopwords.words('english'))
    lemmas = [
        lemmatizer.lemmatize(tok, _get_wn_pos(tag))
        for tok, tag in pos_tags
        if tok.isalpha() and tok not in stop_words
    ]

    return {
        "original": text,
        "normalized": t,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "lemmas": lemmas,
    }


# ── Stage 2: Food Named Entity Recognition ─────────────────────────────────────

def extract_food_entities(preprocessed: Dict) -> List[str]:
    """
    Extract food entities using:
      - POS tag filtering (nouns NN/NNS/NNP)
      - Food lexicon matching
      - WordNet food hypernym check
    Returns ranked list of candidate food names.
    """
    _, _, _, _, wordnet, _ = _get_nltk_tools()

    pos_tags = preprocessed["pos_tags"]
    lemmas   = preprocessed["lemmas"]

    # Extract noun phrases
    noun_candidates = []
    for i, (tok, tag) in enumerate(pos_tags):
        if tag in ("NN", "NNS", "NNP", "NNPS"):
            # Try bigram first (e.g., "dal makhani", "palak paneer")
            if i + 1 < len(pos_tags):
                bigram = tok + " " + pos_tags[i + 1][0]
                if any(b in FOOD_LEXICON for b in bigram.split()):
                    noun_candidates.append(bigram)
            noun_candidates.append(tok)

    # Score candidates: lexicon match > WordNet food synset
    found_foods = []
    seen = set()

    for candidate in noun_candidates:
        c = candidate.lower()
        if c in seen:
            continue
        seen.add(c)
        words = c.split()
        # Direct lexicon hit
        if any(w in FOOD_LEXICON for w in words):
            found_foods.insert(0, c)  # highest priority
            continue
        # WordNet food check
        for syn in wordnet.synsets(words[0], pos=wordnet.NOUN):
            hypernyms = [h.name() for h in syn.closure(lambda s: s.hypernyms())]
            if any("food" in h or "nutriment" in h or "dish" in h for h in hypernyms):
                found_foods.append(c)
                break

    # Also check lemmas directly
    for lemma in lemmas:
        if lemma in FOOD_LEXICON and lemma not in seen:
            found_foods.append(lemma)
            seen.add(lemma)

    return found_foods[:3]  # top 3 candidates


# ── Stage 3: Sentiment Analysis ────────────────────────────────────────────────

def analyze_sentiment(text: str) -> Dict:
    """
    VADER sentiment analysis — adjusts response tone.
    Returns: {"label": "positive"|"neutral"|"negative", "score": float}
    """
    try:
        _, _, _, _, _, sia = _get_nltk_tools()
        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "score": compound, "raw": scores}
    except Exception:
        return {"label": "neutral", "score": 0.0, "raw": {}}


# ── Stage 4: Natural Language Generation ───────────────────────────────────────

# Varied sentence openers indexed by sentiment
_OPENERS = {
    "positive": [
        "Great question! ", "Love that you asked! ", "Absolutely! ",
        "Sure! ", "Happy to help! ",
    ],
    "neutral": [
        "Here's what I found: ", "Let me break that down for you. ",
        "Good to know! ", "Here's the info: ", "",
    ],
    "negative": [
        "No worries, I've got you covered! ",
        "Let me help with that. ", "I understand your concern. ",
    ],
}

_CLOSERS = [
    "\n\n💡 *Tip: Track this in your daily log for better results!*",
    "\n\n🎯 *Want a full meal plan around this? Just ask!*",
    "\n\n📊 *Type 'what have I eaten today' to see your daily summary.*",
    "\n\n🌿 *Pair this with regular hydration for best results!*",
    "",
]


def generate_calorie_response(food_data: Dict, food_name: str, sentiment: Dict) -> str:
    opener = random.choice(_OPENERS[sentiment["label"]])
    closer = random.choice(_CLOSERS)
    n = food_data
    return (
        f"{opener}🔥 **{n['name']}** — Nutritional Breakdown (per 100g)\n\n"
        f"| Nutrient | Amount |\n"
        f"|---|---|\n"
        f"| Calories | **{n['calories_kcal']} kcal** |\n"
        f"| Protein | {n['protein_g']}g |\n"
        f"| Carbohydrates | {n['carb_g']}g |\n"
        f"| Fat | {n['fat_g']}g |\n"
        f"| Fiber | {n['fiber_g']}g |\n"
        f"| Sodium | {n.get('sodium_mg', 'N/A')}mg |\n\n"
        f"*Source: {n.get('source', 'USDA/ICMR-NIN Database')}*"
        f"{closer}"
    )


def generate_nutrition_response(food_data: Dict, food_name: str, sentiment: Dict) -> str:
    opener = random.choice(_OPENERS[sentiment["label"]])
    n = food_data
    lines = [f"{opener}🧪 **Full Nutrition Profile — {n['name']}** (per 100g)\n"]
    macros = [
        ("💪 Protein", n["protein_g"], "g"),
        ("🌾 Carbohydrates", n["carb_g"], "g"),
        ("🧈 Fat", n["fat_g"], "g"),
        ("🌿 Fiber", n["fiber_g"], "g"),
    ]
    for label, val, unit in macros:
        lines.append(f"- {label}: **{val}{unit}**")
    if n.get("calcium_mg"):
        lines.append(f"- 🦴 Calcium: **{n['calcium_mg']}mg**")
    if n.get("iron_mg"):
        lines.append(f"- ⚙️ Iron: **{n['iron_mg']}mg**")
    if n.get("vitamin_c_mg"):
        lines.append(f"- 🍊 Vitamin C: **{n['vitamin_c_mg']}mg**")
    if n.get("sodium_mg"):
        lines.append(f"- 🧂 Sodium: **{n['sodium_mg']}mg**")
    lines.append(f"\n*Source: {n.get('source', 'USDA/ICMR-NIN')}*")
    return "\n".join(lines)


def generate_portion_response(food_data: Dict, sentiment: Dict) -> str:
    opener = random.choice(_OPENERS[sentiment["label"]])
    n = food_data
    cals = n["calories_kcal"]
    if cals < 50:
        portion = "200–300g — very low calorie, eat freely 🟢"
    elif cals < 120:
        portion = "150–200g — light and nutritious 🟢"
    elif cals < 200:
        portion = "100–150g — moderate portion 🟡"
    elif cals < 300:
        portion = "80–100g — calorie-dense, watch intake 🟡"
    else:
        portion = "50–80g — high calorie, limit serving 🔴"
    return (
        f"{opener}📏 **Portion Guide — {n['name']}**\n\n"
        f"- Calories per 100g: **{cals} kcal**\n"
        f"- Recommended serving: **{portion}**\n"
        f"- Protein per 100g: {n['protein_g']}g\n\n"
        f"**Standard daily limits:**\n"
        f"- Grains/Rice: 200–300g cooked\n"
        f"- Dal/Legumes: 100–150g cooked\n"
        f"- Vegetables: 300–500g (unlimited)\n"
        f"- Dairy: 200–300ml milk / 100g paneer\n"
        f"- Non-veg: 100–150g per meal"
    )


def generate_not_found_response(food_name: str, foods_found: List[str]) -> str:
    suggestions = ", ".join(f"*{f}*" for f in ["roti", "dal", "paneer", "rice", "chicken", "egg"])
    return (
        f"I couldn't find **{food_name}** in my USDA/ICMR-NIN database. "
        f"This may be a regional name or spelling variation.\n\n"
        f"Try searching for: {suggestions}\n\n"
        f"*I cover 500+ Indian and international foods.*"
    )


def generate_missing_food_response(intent: str) -> str:
    examples = {
        "calorie_query":   "e.g., *'calories in paneer'* or *'how many calories in 2 rotis'*",
        "nutrition_info":  "e.g., *'protein in chicken'* or *'nutrition of dal makhani'*",
        "portion_control": "e.g., *'how much rice should I eat'* or *'portion size of paneer'*",
    }
    return (
        f"Please mention a food item! "
        f"{examples.get(intent, 'e.g., *calories in paneer*')} 🍽️"
    )


# ── Full NLP pipeline (public API) ─────────────────────────────────────────────

def run_nlp_pipeline(text: str) -> Dict:
    """
    Run the complete NLTK pipeline on user input.
    Returns preprocessed data, food entities, and sentiment.
    Used by chatbot_engine.py before generating a response.
    """
    preprocessed = preprocess(text)
    food_entities = extract_food_entities(preprocessed)
    sentiment     = analyze_sentiment(text)

    return {
        "preprocessed":   preprocessed,
        "food_entities":  food_entities,
        "sentiment":      sentiment,
        "lemmas":         preprocessed["lemmas"],
        "pos_tags":       preprocessed["pos_tags"],
    }
