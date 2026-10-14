from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintPenalty(nn.Module):
    """
    Penalizes model predictionsthat exceed a constraint vector.

    Typical use:
    - predictions: predicted calories/macros (or scalar calorie output)
    - constraints: budget/upper-bound values for each prediction dimension

    Penalty is zero when prediction <= constraint and positive otherwise.
    """

    def __init__(
        self,
        penalty_weight: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        squared: bool = True,
    ) -> None:
        super().__init__()
        self.penalty_weight = penalty_weight
        self.reduction = reduction
        self.squared = squared

        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'.")
        if self.penalty_weight < 0:
            raise ValueError("penalty_weight must be >= 0.")

    def forward(
        self,
        predictions: torch.Tensor,
        constraints: torch.Tensor,
    ) -> torch.Tensor:
        if predictions.shape != constraints.shape:
            raise ValueError(
                "predictions and constraints must have identical shape. "
                f"Got {tuple(predictions.shape)} and {tuple(constraints.shape)}."
            )

        # Only penalize exceedance over the constraint.
        exceedance = F.relu(predictions - constraints)
        penalty = exceedance.pow(2) if self.squared else exceedance
        penalty = penalty * self.penalty_weight

        if self.reduction == "mean":
            return penalty.mean()
        if self.reduction == "sum":
            return penalty.sum()
        return penalty

