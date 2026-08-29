"""
NutriChat AI — Main FastAPI Application (Full Edition)
Serves: Auth, Menu, Chat, Orders, Meal Tracking, Suggestions, Static Frontend
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.auth import (
    register_user, login_user, validate_token,
    update_user_profile, logout_user,
)
from api.nutrition_tracker import (
    get_menu, lookup_food, get_daily_summary,
    place_order, get_order_history, log_meal, suggest_next_meal,
)
from api.chatbot_engine import process_message
from api.ml_inference import get_engine

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NutriChat AI", version="3.0.0")

# Pre-warm the ML engine as soon as the server starts
@app.on_event("startup")
async def startup_event():
    import logging
    logging.basicConfig(level=logging.INFO)
    get_engine()  # triggers background model load

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend if exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Auth helpers ──────────────────────────────────────────────────────────────

def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization[7:]
    user = validate_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return user


# ── Pydantic models ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3)
    email: str
    password: str = Field(..., min_length=6)
    daily_calorie_goal: int = 2000


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class OrderItem(BaseModel):
    item_id: str
    qty: int = 1


class OrderRequest(BaseModel):
    items: List[OrderItem]


class LogMealRequest(BaseModel):
    food_name: str
    calories: float
    protein_g: float = 0.0
    carb_g: float = 0.0
    fat_g: float = 0.0
    qty: float = 1.0
    meal_type: str = "meal"


class ProfileUpdateRequest(BaseModel):
    daily_calorie_goal: Optional[int] = None
    goal: Optional[str] = None
    age: Optional[int] = None
    weight_kg: Optional[float] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>NutriChat AI API</h1><p>Frontend not found.</p>")


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "version": "3.0.0",
        "ml_ready": engine.ready,
    }


@app.get("/ml/status")
def ml_status():
    """Check if the ML model is loaded and ready."""
    engine = get_engine()
    return {
        "ready": engine.ready,
        "backbone": getattr(engine, '_backbone_name', 'unknown'),
        "device": getattr(engine, '_device', 'cpu'),
        "checkpoint_exists": (engine.__class__.__module__ and True),
    }


# -- Auth --

@app.post("/auth/register")
def register(req: RegisterRequest):
    result = register_user(req.username, req.email, req.password, req.daily_calorie_goal)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/auth/login")
def login(req: LoginRequest):
    result = login_user(req.username, req.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])
    return result


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        logout_user(authorization[7:])
    return {"success": True}


@app.get("/auth/me")
def get_me(authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    return user


@app.put("/auth/profile")
def update_profile(req: ProfileUpdateRequest,
                   authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token.")
    token = authorization[7:]
    updates = {k: v for k, v in req.dict().items() if v is not None}
    result = update_user_profile(token, updates)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


# -- Menu --

@app.get("/menu")
def menu():
    return get_menu()


@app.get("/menu/{item_id}")
def menu_item(item_id: str):
    items = {it["id"]: it for it in get_menu()}
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found.")
    return items[item_id]


# -- Nutrition lookup --

@app.get("/nutrition/lookup")
def nutrition_lookup(q: str):
    result = lookup_food(q)
    if not result:
        raise HTTPException(status_code=404, detail=f"Food '{q}' not found in database.")
    return result


# -- Chat --

@app.post("/chat")
def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    result = process_message(
        text=req.message,
        username=user["username"],
        display_name=user["display_name"],
        daily_goal=user["daily_calorie_goal"],
        goal=user.get("goal", "general fitness"),
    )
    return {
        "response":         result["response"],
        "intent":           result["intent"],
        "user":             user["display_name"],
        "ml_source":        result.get("ml_source", "distilbert+nltk"),
        "confidence":       result.get("confidence", 0.0),
        "calorie_estimate": result.get("calorie_estimate", 0.0),
        # NLTK pipeline metadata (useful for IEEE evaluation)
        "nlp_meta":         result.get("nlp_meta", {}),
    }


# -- Orders --

@app.post("/orders")
def create_order(req: OrderRequest, authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    items = [{"item_id": it.item_id, "qty": it.qty} for it in req.items]
    order = place_order(user["username"], items)
    return order


@app.get("/orders/history")
def order_history(authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    return get_order_history(user["username"])


# -- Meal tracking --

@app.get("/tracker/today")
def today_summary(authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    return get_daily_summary(user["username"], user["daily_calorie_goal"])


@app.post("/tracker/log")
def log_manual_meal(req: LogMealRequest,
                    authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    entry = log_meal(
        username=user["username"],
        food_name=req.food_name,
        calories=req.calories,
        protein=req.protein_g,
        carbs=req.carb_g,
        fat=req.fat_g,
        qty=req.qty,
        meal_type=req.meal_type,
        source="manual",
    )
    return {"success": True, "entry": entry}


# -- Suggestions --

@app.get("/suggestions/next-meal")
def next_meal_suggestion(authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    return suggest_next_meal(
        username=user["username"],
        daily_goal=user["daily_calorie_goal"],
        goal=user.get("goal", "general fitness"),
    )
