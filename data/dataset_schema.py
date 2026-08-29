import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.preprocessing import preprocess_query


@dataclass
class NutriChatRecord:
    text: str
    intent: int
    regression_target: float
    nutrition_vector: List[float] = field(default_factory=list)
    calorie_budget: Optional[float] = None
    session_id: str = "default"
    timestamp: str = ""
    language: str = "hinglish"
    food_entities: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NutriChatRecord":
        if "text" not in data or "intent" not in data or "regression_target" not in data:
            raise ValueError("Record must include text, intent, and regression_target.")

        record = NutriChatRecord(
            text=str(data["text"]),
            intent=int(data["intent"]),
            regression_target=float(data["regression_target"]),
            nutrition_vector=[float(x) for x in data.get("nutrition_vector", [])],
            calorie_budget=(
                float(data["calorie_budget"]) if data.get("calorie_budget") is not None else None
            ),
            session_id=str(data.get("session_id", "default")),
            timestamp=str(data.get("timestamp", "")),
            language=str(data.get("language", "hinglish")),
            food_entities=[str(x) for x in data.get("food_entities", [])],
        )
        if not record.food_entities:
            record.food_entities = list(preprocess_query(record.text)["food_entities"])
        return record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "intent": self.intent,
            "regression_target": self.regression_target,
            "nutrition_vector": self.nutrition_vector,
            "calorie_budget": self.calorie_budget,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "language": self.language,
            "food_entities": self.food_entities,
        }


def load_nutrichat_dataset(path: str | Path) -> List[NutriChatRecord]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset root must be a list of records.")
    return [NutriChatRecord.from_dict(item) for item in data]


def validate_nutrichat_dataset(path: str | Path) -> Dict[str, int]:
    records = load_nutrichat_dataset(path)
    num_with_budget = sum(1 for r in records if r.calorie_budget is not None)
    num_hinglish = sum(1 for r in records if r.language.lower() == "hinglish")
    return {
        "num_records": len(records),
        "num_with_budget": num_with_budget,
        "num_hinglish": num_hinglish,
    }

