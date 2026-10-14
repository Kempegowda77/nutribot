"""
 NutriChat AI — Auth Module
File-based user registration and login.
"""
import hashlib, json, secrets, time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
USERS_FILE = ROOT / "data" / "users.json"
TOKENS_FILE = ROOT / "data" / "tokens.json"
TOKEN_TTL = 86400 * 7  # 7 days


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


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_user(username: str, email: str, password: str, daily_calorie_goal: int = 2000) -> Dict:
    users = _load(USERS_FILE)
    uname = username.strip().lower()
    if uname in users:
        return {"success": False, "error": "Username already exists."}
    for u in users.values():
        if u.get("email", "").lower() == email.strip().lower():
            return {"success": False, "error": "Email already registered."}
    users[uname] = {
        "username": username.strip(),
        "email": email.strip().lower(),
        "password_hash": _hash(password),
        "daily_calorie_goal": daily_calorie_goal,
        "created_at": time.time(),
        "goal": "general fitness",
        "age": None,
        "weight_kg": None,
    }
    _save(USERS_FILE, users)
    return {"success": True, "username": username.strip()}


def login_user(username: str, password: str) -> Dict:
    users = _load(USERS_FILE)
    uname = username.strip().lower()
    if uname not in users:
        return {"success": False, "error": "Invalid username or password."}
    user = users[uname]
    if user["password_hash"] != _hash(password):
        return {"success": False, "error": "Invalid username or password."}
    token = secrets.token_hex(32)
    tokens = _load(TOKENS_FILE)
    tokens[token] = {"username": uname, "expires_at": time.time() + TOKEN_TTL}
    _save(TOKENS_FILE, tokens)
    return {
        "success": True, "token": token,
        "username": user["username"], "email": user["email"],
        "daily_calorie_goal": user["daily_calorie_goal"],
        "goal": user.get("goal", "general fitness"),
    }


def validate_token(token: str) -> Optional[Dict]:
    tokens = _load(TOKENS_FILE)
    entry = tokens.get(token)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        del tokens[token]
        _save(TOKENS_FILE, tokens)
        return None
    users = _load(USERS_FILE)
    uname = entry["username"]
    user = users.get(uname)
    if not user:
        return None
    return {
        "username": uname,
        "display_name": user["username"],
        "email": user["email"],
        "daily_calorie_goal": user["daily_calorie_goal"],
        "goal": user.get("goal", "general fitness"),
        "age": user.get("age"),
        "weight_kg": user.get("weight_kg"),
    }


def update_user_profile(token: str, updates: Dict) -> Dict:
    user_info = validate_token(token)
    if not user_info:
        return {"success": False, "error": "Invalid or expired token."}
    users = _load(USERS_FILE)
    uname = user_info["username"]
    for k, v in updates.items():
        if k in {"daily_calorie_goal", "goal", "age", "weight_kg"}:
            users[uname][k] = v
    _save(USERS_FILE, users)
    return {"success": True}


def logout_user(token: str) -> Dict:
    tokens = _load(TOKENS_FILE)
    if token in tokens:
        del tokens[token]
        _save(TOKENS_FILE, tokens)
    return {"success": True}
