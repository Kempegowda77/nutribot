"""
NutriChat AI — Real-Time ML Inference Engine
=============================================
Loads the trained NutriChatModel + distilBERT backbone checkpoint and
provides live intent classification + calorie regression for every message.

Intent map (matches training):
  0 → calorie_query
  1 → meal_suggestion
  2 → diet_advice
  3 → nutrition_info
  4 → portion_control

Falls back to keyword matching if the model is not loaded.
"""

import json
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("nutrichat.ml")

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = ROOT / "models" / "nutrichat_muril_checkpoint.pt"
MODEL_CONFIG_PATH = ROOT / "models" / "model_config.json"

# Training used distilbert-base-multilingual-cased by default
DEFAULT_BACKBONE = "distilbert-base-multilingual-cased"

INTENT_LABELS = {
    0: "calorie_query",
    1: "meal_suggestion",
    2: "diet_advice",
    3: "nutrition_info",
    4: "portion_control",
}

# ── Extra intents handled purely by keyword (not in training set) ──────────────
EXTRA_INTENT_PATTERNS = {
    "greeting": [
        r"^(hi|hello|hey|hii+|namaste|namaskar|good morning|good evening|good afternoon)[\s!.]*$"
    ],
    "thanks": [
        r"\b(thank(?:s|you)?|shukriya|dhanyawad|great job|awesome|perfect|well done|nice one)\b",
        r"^thanks?[\s!.]*$",
    ],
    "today_intake": [
        r"\b(aaj.*khaya|today.*eat(?:en)|daily.*intake|how much.*ate|mera.*calorie"
        r"|my.*intake|aaj ka|track|log)\b",
        r"what have i eaten",
        r"how much have i eaten",
        r"what.*eaten.*today",
        r"show.*daily.*intake",
        r"today.s\s+calorie\s+intake",
        r"today.s.*intake",
    ],
    "order_intent": [
        r"\b(order|buy|add to cart|place order|mujhe chahiye|dena|khilao|show.*menu)\b",
        r"mujhe\s+\w+\s+chahiye",
    ],
    # Meta questions about the bot itself
    "bot_info": [
        r"\b(how.*bot.*work|bot.*work|how.*you.*work|what.*you.*do|what can you do"
        r"|how.*this.*work|about.*bot|tell.*about.*yourself|who are you"
        r"|what.*nutrichat|how.*nutrichat|features|capabilities)\b"
    ],
}

MEMORY_SLOTS = 4
MAX_LENGTH = 128
NUTRITION_DIM = 8


class NutriChatMLEngine:
    """
    Singleton inference engine.
    Call `get_engine()` to retrieve the shared instance.
    """

    _instance: Optional["NutriChatMLEngine"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._ready = False
        self._model = None
        self._extractor = None
        self._tokenizer = None
        self._device = "cpu"
        self._backbone_name = DEFAULT_BACKBONE
        self._embed_dim = 128
        self._load_thread = threading.Thread(target=self._load, daemon=True)
        self._load_thread.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self, text: str) -> Dict:
        """
        Returns:
          {
            "intent": str,          # e.g. "calorie_query"
            "intent_id": int,
            "confidence": float,    # 0-1, from softmax
            "calorie_estimate": float,
            "source": "ml" | "keyword" | "fallback"
          }
        """
        # ① Check extra intents first (greeting/thanks/order/today)
        extra = self._check_extra_intents(text)
        if extra:
            return {
                "intent": extra,
                "intent_id": -1,
                "confidence": 1.0,
                "calorie_estimate": 0.0,
                "source": "keyword",
            }

        # ② Use ML model if ready
        if self._ready:
            return self._ml_predict(text)

        # ③ Keyword fallback
        return self._keyword_predict(text)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load(self):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # ---- determine backbone from checkpoint --------------------------
            backbone_name = DEFAULT_BACKBONE
            if CHECKPOINT_PATH.exists():
                try:
                    ckpt_probe = torch.load(
                        str(CHECKPOINT_PATH), map_location="cpu", weights_only=False
                    )
                    backbone_name = ckpt_probe.get("backbone", DEFAULT_BACKBONE)
                    del ckpt_probe
                except Exception:
                    pass

            logger.info(f"[ML] Loading backbone: {backbone_name}")
            tokenizer = AutoTokenizer.from_pretrained(backbone_name)
            backbone_model = AutoModel.from_pretrained(backbone_name)

            # ---- load model config ------------------------------------------
            with open(MODEL_CONFIG_PATH) as f:
                cfg = json.load(f)
            embed_dim = cfg["embed_dim"]

            # ---- build architecture (same as training) -----------------------
            sys.path.insert(0, str(ROOT))
            from models.nutrichat_model import NutriChatModel

            class BackboneEmbeddingExtractor(torch.nn.Module):
                def __init__(self, backbone, embed_dim):
                    super().__init__()
                    self.backbone = backbone
                    self.proj = torch.nn.Linear(768, embed_dim)
                    self.dropout = torch.nn.Dropout(0.1)

                def forward(self, input_ids, attention_mask):
                    out = self.backbone(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    hidden = out.last_hidden_state
                    return self.proj(self.dropout(hidden))

            extractor = BackboneEmbeddingExtractor(backbone_model, embed_dim)
            model = NutriChatModel(**cfg)

            # ---- load weights -----------------------------------------------
            if CHECKPOINT_PATH.exists():
                logger.info(f"[ML] Loading checkpoint: {CHECKPOINT_PATH}")
                ckpt = torch.load(
                    str(CHECKPOINT_PATH), map_location="cpu", weights_only=False
                )
                model.load_state_dict(ckpt["model_state"], strict=False)
                extractor.load_state_dict(ckpt["extractor_state"], strict=False)
                val_acc = ckpt.get("val_accuracy", "?")
                logger.info(f"[ML] Checkpoint loaded (val_acc={val_acc})")
            else:
                logger.warning("[ML] No checkpoint found — using untrained weights")

            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            model.to(device).eval()
            extractor.to(device).eval()

            self._model = model
            self._extractor = extractor
            self._tokenizer = tokenizer
            self._device = device
            self._embed_dim = embed_dim
            self._backbone_name = backbone_name
            self._ready = True
            logger.info(f"[ML] ✅ Engine ready on {device}")

        except Exception as e:
            logger.error(f"[ML] Failed to load model: {e}", exc_info=True)
            self._ready = False

    # ── Inference ──────────────────────────────────────────────────────────────

    def _ml_predict(self, text: str) -> Dict:
        import torch

        enc = self._tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            token_emb = self._extractor(input_ids, attention_mask)
            B = input_ids.size(0)
            memory = torch.zeros(
                B, MEMORY_SLOTS, self._embed_dim, device=self._device
            )
            nutr_vec = torch.zeros(B, NUTRITION_DIM, device=self._device)
            outputs = self._model(
                token_embeddings=token_emb,
                nutrition_vector=nutr_vec,
                memory_embeddings=memory,
                attention_mask=attention_mask,
            )

        probs = outputs["probs"][0].cpu().tolist()
        intent_id = int(outputs["logits"][0].argmax().item())
        confidence = probs[intent_id]
        calorie_est = float(outputs["regression"][0].squeeze().item())

        intent_label = INTENT_LABELS.get(intent_id, "general")

        # Always run keyword check and let it override if it has a strong match
        kw_result = self._keyword_predict(text)

        # Keyword wins if ML is not confident
        if confidence < 0.70:
            if kw_result["intent"] != "general":
                return {**kw_result, "ml_confidence": confidence}

        # Keyword wins if it strongly matches nutrition/calorie/portion
        # but ML predicted something unrelated (meal_suggestion / diet_advice)
        PRECISION_INTENTS = ("nutrition_info", "calorie_query", "portion_control")
        if kw_result["intent"] in PRECISION_INTENTS \
                and intent_label not in PRECISION_INTENTS:
            return {**kw_result, "ml_confidence": confidence}

        return {
            "intent": intent_label,
            "intent_id": intent_id,
            "confidence": confidence,
            "calorie_estimate": max(0.0, calorie_est),
            "source": "ml",
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_extra_intents(text: str) -> Optional[str]:
        t = text.lower().strip()
        for intent, patterns in EXTRA_INTENT_PATTERNS.items():
            for p in patterns:
                if re.search(p, t, re.IGNORECASE):
                    return intent
        return None

    @staticmethod
    def _keyword_predict(text: str) -> Dict:
        """Keyword fallback — used when model is loading or low-confidence."""
        t = text.lower()
        patterns = [
            # ── TODAY INTAKE (check first — 'today calorie intake' must go here, not calorie_query) ──
            ("today_intake", [
                r"today.s\s+(?:calorie|food|meal|intake)",
                r"what.*(?:i|i've).*eaten.*today",
                r"how much.*(?:i|i've).*eaten",
            ]),
            # ── Portion (check BEFORE nutrition to avoid 'daily limit' going to nutrition_info) ──
            ("portion_control", [
                r"\bportion\b",
                r"\bserving size\b",
                r"how much.*eat",
                r"how many.*(?:roti|chapati|rice|dal|rotis).*(?:day|daily|should)",
                r"(?:roti|chapati|rice|dal|egg)s?\s+(?:per|a)\s+day",
                r"kitna\s+(?:rice|dal|roti|paneer|khana)\s+khana",
                r"kitni matra",
                r"recommended.*(?:intake|serving|portion)",
                r"daily\s+limit\s+of\s+(?!calories)",
            ]),
            # ── Meal suggestion (check BEFORE diet_advice for 'what should I eat') ──────────────
            ("meal_suggestion", [
                r"\bsuggest\b(?!.*diet)",   # 'suggest' but NOT 'suggest diet'
                r"\brecommend\b(?!.*diet)",
                r"kya khau",
                r"meal plan", r"next meal",
                r"healthy.*option",
                r"what\s+(?:should\s+i|to|can\s+i)\s+eat(?!.*(?:for|with)\s+(?:weight|diabetes|heart|muscle|cholesterol|thyroid|pcos|hypertension))",
                r"food\s+suggest(?!.*weight\s+loss|.*diet)",
            ]),
            # ── Nutrition info ────────────────────────────────────────────────
            ("nutrition_info",  [
                r"nutrition(?:\s+info|\s+of|\s+in|\s+about|\s+detail|\s+fact)?",
                r"(info|information|detail|fact)\s*(about|on|of)\s*\w",
                r"\b(protein|carbs?|carbohydrate|fiber|fibre|vitamin"
                r"|iron|calcium|nutriti|sodium|mineral|omega|glycemic"
                r"|macros?|micro|zinc|potassium|phosphorus|magnesium)\b",
                r"\bfat\s+(in|of|content)\b",
            ]),
            # ── Calorie ───────────────────────────────────────────────────────
            ("calorie_query",   [
                r"calori", r"kcal",
                r"how many cal", r"kitna calorie",
                r"energy\s*(in|of|content)",
            ]),
            # ── Diet advice ───────────────────────────────────────────────────
            ("diet_advice",     [
                r"\bdiet\b", r"weight loss", r"\blose\s+(?:weight|fat)\b",
                r"muscle gain", r"\bdiabetes\b",
                r"heart health", r"\bketo\b", r"\bvegan\b", r"vegetarian",
                r"\bfitness\b", r"\bthyroid\b", r"\bpcos\b",
                r"blood pressure", r"\bcholesterol\b", r"healthy eating",
                r"what.*(?:eat|food).*(?:for|with)\s+(?:cholesterol|diabetes|heart|thyroid|pcos|hypertension)",
            ]),
            # ── Meal suggestion — must come AFTER diet_advice ────────────────────
            ("meal_suggestion", [
                r"\bsuggest\b(?!.*diet)",
                r"\brecommend\b(?!.*diet)",
                r"kya khau",
                r"meal plan", r"next meal",
                r"healthy.*option",
                r"what\s+(?:should\s+i|to|can\s+i)\s+eat(?!\s+for\s+(?:weight|diabetes|heart|muscle|cholesterol|thyroid|pcos))",
                r"food\s+suggest(?!.*weight\s+loss|.*diet)",
            ]),
        ]
        for intent, pats in patterns:
            for p in pats:
                if re.search(p, t, re.IGNORECASE):
                    return {"intent": intent, "intent_id": -1,
                            "confidence": 0.85, "calorie_estimate": 0.0,
                            "source": "keyword"}
        return {"intent": "general", "intent_id": -1,
                "confidence": 0.5, "calorie_estimate": 0.0,
                "source": "fallback"}


# ── Singleton accessor ─────────────────────────────────────────────────────────

_engine: Optional[NutriChatMLEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> NutriChatMLEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                logger.info("[ML] Initializing ML inference engine...")
                _engine = NutriChatMLEngine()
    return _engine


def predict_intent(text: str) -> Dict:
    """Convenience function — calls the singleton engine."""
    return get_engine().predict(text)
