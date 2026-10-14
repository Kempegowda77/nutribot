"""
NutriChat AI — Gemini LLM Brain
===============================
Routes chatbot responses through Google Gemini (gemini-1.5-flash).
Nutrition data from the local DB is injected as context so the AI
answers with accurate, grounded information instead of hallucinating.

Setup:
  1. pip install google-generativeai
  2. Set GEMINI_API_KEY in a .env file at the project root
     OR as an environment variable.

If the API key is missing or the call fails, falls back to the
original template-based responses.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nutrichat.gemini")

# ── Load .env if present ───────────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        # Parse manually if python-dotenv not installed
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NutriChat AI, an expert nutrition and diet assistant specializing in Indian food and healthy eating. You are knowledgeable, friendly, warm, and conversational — like a trusted dietitian friend.

Your personality:
- Speak naturally and conversationally, never robotic
- Use emojis appropriately to make responses engaging (but don't overdo it)
- Be concise yet thorough — get to the point quickly
- Mix Hindi/Hinglish naturally when the user does (e.g., "acha", "bilkul", "haan")
- Always be encouraging and positive about health journeys
- For medical conditions (diabetes, heart, thyroid), always recommend consulting a doctor

Your expertise covers:
- Indian nutrition: dal, roti, rice, paneer, sabzi, dosas, idlis, biryani, etc.
- USDA & ICMR-NIN nutritional data
- Calorie counting, macros, micronutrients
- Diet plans: weight loss, muscle gain, diabetes, heart health, keto, vegan, PCOS
- Portion control and meal timing
- Meal suggestions tailored to Indian palates

Rules:
- If nutrition data is provided in the context, use those EXACT numbers — do NOT make up calories or macros
- If no data is found, say so honestly and suggest similar foods
- Keep responses under 200 words unless the user asks for detailed info
- Never recommend dangerous diets or extreme restrictions
- Always end with an actionable tip or question to keep the conversation going
"""


def _get_client():
    """Lazy-load the Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
    except ImportError:
        logger.warning("[Gemini] google-generativeai not installed. Run: pip install google-generativeai")
        return None
    except Exception as e:
        logger.error(f"[Gemini] Failed to init client: {e}")
        return None


# Singleton client
_client = None
_client_checked = False


def get_gemini_client():
    global _client, _client_checked
    if not _client_checked:
        _client = _get_client()
        _client_checked = True
    return _client


def is_available() -> bool:
    """Check if Gemini is configured and ready."""
    return get_gemini_client() is not None


def ask_gemini(
    user_message: str,
    nutrition_context: Optional[str] = None,
    user_context: Optional[str] = None,
    chat_history: Optional[list] = None,
) -> Optional[str]:
    """
    Send a message to Gemini and return the response text.

    Args:
        user_message:      The raw user message.
        nutrition_context: Formatted nutrition data string (from local DB).
        user_context:      User profile info (name, goal, daily intake).
        chat_history:      List of prior {"role": ..., "parts": [...]} dicts.

    Returns:
        The AI response string, or None if unavailable/failed.
    """
    client = get_gemini_client()
    if not client:
        return None

    # Build the enriched prompt with context
    parts = []

    if user_context:
        parts.append(f"[User Context]\n{user_context}")

    if nutrition_context:
        parts.append(f"[Nutrition Data from Database]\n{nutrition_context}")

    parts.append(f"[User Message]\n{user_message}")

    full_prompt = "\n\n".join(parts)

    try:
        # Use chat session if history provided
        if chat_history:
            chat = client.start_chat(history=chat_history)
            response = chat.send_message(full_prompt)
        else:
            response = client.generate_content(full_prompt)

        return response.text.strip()

    except Exception as e:
        logger.error(f"[Gemini] API call failed: {e}")
        return None
