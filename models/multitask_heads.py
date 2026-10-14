from typing import Dict

import torch
import torch.nn as nn


class MultiTaskHeads(nn.Module):
    """
    Two -head output module:
    - Classification head -> logits for softmax-based classification
    - Regression head -> linear output

    Input:
    - features: [B, D]

    Outputs:
    - logits: [B, num_classes]
    - probs: [B, num_classes] (softmax over logits)
    - regression: [B, regression_dim]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        regression_dim: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")
        if regression_dim <= 0:
            raise ValueError("regression_dim must be greater than 0.")

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.regressor = nn.Linear(hidden_dim, regression_dim)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.dim() != 2:
            raise ValueError(
                f"features must be 2D [B, D], got shape {tuple(features.shape)}"
            )

        shared_features = self.shared(features)

        logits = self.classifier(shared_features)
        probs = torch.softmax(logits, dim=-1)
        regression = self.regressor(shared_features)

        return {
            "logits": logits,
            "probs": probs,
            "regression": regression,
        }

