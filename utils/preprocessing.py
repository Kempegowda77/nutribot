import re
from typing import Dict, List, Set


DEFAULT_STOPWORDS: Set[str] = {
    "hai",
    "hain",
    "ki",
    "ka",
    "ke",
    "ko",
    "me",
    "mein",
    "mai",
    "mera",
    "meri",
    "please",
    "pls",
    "kr",
    "karo",
    "karo",
}


FOOD_LEXICON: Set[str] = {
    "roti",
    "chapati",
    "rice",
    "dal",
    "paneer",
    "chicken",
    "egg",
    "eggs",
    "milk",
    "banana",
    "apple",
    "oats",
    "salad",
    "idli",
    "dosa",
    "upma",
    "poha",
    "paratha",
    "bread",
    "curd",
    "yogurt",
    "fish",
    "mutton",
    "rajma",
    "chana",
}


def normalize_code_mixed_text(text: str) -> str:
    """
    Light normalization for Hinglish/English user queries.
    """
    out = text.lower().strip()
    out = re.sub(r"http\S+|www\.\S+", " ", out)
    out = re.sub(r"[^a-z0-9\s]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def remove_stopwords(text: str, stopwords: Set[str] | None = None) -> str:
    stop = stopwords or DEFAULT_STOPWORDS
    tokens = [tok for tok in text.split() if tok not in stop]
    return " ".join(tokens)


def extract_food_entities(text: str, food_lexicon: Set[str] | None = None) -> List[str]:
    lexicon = food_lexicon or FOOD_LEXICON
    normalized = normalize_code_mixed_text(text)
    tokens = normalized.split()
    entities = [tok for tok in tokens if tok in lexicon]

    # Preserve ordr while de-duplicating.
    seen: Set[str] = set()
    ordered_entities: List[str] = []
    for ent in entities:
        if ent not in seen:
            seen.add(ent)
            ordered_entities.append(ent)
    return ordered_entities


def preprocess_query(text: str) -> Dict[str, object]:
    normalized = normalize_code_mixed_text(text)
    cleaned = remove_stopwords(normalized)
    entities = extract_food_entities(cleaned)
    return {
        "normalized_text": normalized,
        "cleaned_text": cleaned,
        "food_entities": entities,
    }

