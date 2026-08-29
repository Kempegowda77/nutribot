from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel


class HFMultiTaskBaseline(nn.Module):
    """
    HuggingFace encoder baseline with dual heads.
    Supports model_name such as:
    - bert-base-uncased
    - distilbert-base-uncased
    """

    def __init__(self, model_name: str, num_classes: int, regression_dim: int = 1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.regressor = nn.Linear(hidden_size, regression_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        regression = self.regressor(pooled)
        return {"logits": logits, "regression": regression, "probs": torch.softmax(logits, dim=-1)}


class RuleBasedChatbotBaseline:
    """
    Lightweight heuristic baseline.
    - intent by keyword rules
    - nutrition/regression by simple numeric extraction
    """

    def __init__(self, intent_labels: List[str]) -> None:
        self.intent_labels = intent_labels

    def predict(self, text: str) -> Dict[str, float | int]:
        t = text.lower()
        if any(k in t for k in ["suggest", "recommend", "dinner", "lunch", "breakfast"]):
            intent = 1
        elif any(k in t for k in ["calorie", "protein", "carb", "fat"]):
            intent = 0
        else:
            intent = 0

        tokens = t.replace(",", " ").split()
        numbers = []
        for token in tokens:
            try:
                numbers.append(float(token))
            except ValueError:
                continue
        regression = float(sum(numbers)) if numbers else 200.0
        confidence = 0.55

        return {
            "intent_id": int(intent),
            "confidence": confidence,
            "regression": regression,
        }

