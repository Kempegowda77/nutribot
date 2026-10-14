from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    print_metrics: bool = True,
) -> Dict[str, float]:
    """ .
    Evaluate model on:
    - Accuracy (classification)
    - MAE (regression)
    - Constraint violation rate

    Expected batch keys:
    - token_embeddings, nutrition_vector, memory_embeddings
    - intent_label, regression_target
    Optional:
    - attention_mask, memory_mask, constraint_vector
    """
    model = model.to(device)
    model.eval()

    correct = 0
    total_samples = 0

    mae_sum = 0.0
    mae_count = 0

    violation_count = 0
    violation_total = 0

    for batch in dataloader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        outputs = model(
            token_embeddings=batch["token_embeddings"],
            nutrition_vector=batch["nutrition_vector"],
            memory_embeddings=batch["memory_embeddings"],
            attention_mask=batch.get("attention_mask"),
            memory_mask=batch.get("memory_mask"),
            constraint_vector=batch.get("constraint_vector"),
        )

        # Classification accuracy
        logits = outputs["logits"]
        intent_targets = batch["intent_label"].long()
        preds = logits.argmax(dim=-1)
        correct += int((preds == intent_targets).sum().item())
        total_samples += int(intent_targets.numel())

        # Regression MAE
        regression_pred = outputs["regression"].float()
        regression_target = batch["regression_target"].float()
        if regression_target.dim() == 1:
            regression_target = regression_target.unsqueeze(-1)
        regression_target = regression_target.view_as(regression_pred)

        abs_err = torch.abs(regression_pred - regression_target)
        mae_sum += float(abs_err.sum().item())
        mae_count += int(abs_err.numel())

        # Constraint violation rate: fraction of prediction elements that exceed budget.
        constraint_vector = batch.get("constraint_vector")
        if constraint_vector is not None:
            constraints = constraint_vector.float()
            if constraints.dim() == 1:
                constraints = constraints.unsqueeze(-1)
            if constraints.size(1) == 1 and regression_pred.size(1) > 1:
                constraints = constraints.expand(-1, regression_pred.size(1))
            constraints = constraints.view_as(regression_pred)

            violations = regression_pred > constraints
            violation_count += int(violations.sum().item())
            violation_total += int(violations.numel())

    accuracy = (correct / total_samples) if total_samples > 0 else 0.0
    mae = (mae_sum / mae_count) if mae_count > 0 else 0.0
    constraint_violation_rate = (
        (violation_count / violation_total) if violation_total > 0 else 0.0
    )

    metrics = {
        "accuracy": accuracy,
        "mae": mae,
        "constraint_violation_rate": constraint_violation_rate,
    }

    if print_metrics:
        print(
            "Evaluation | "
            f"accuracy={metrics['accuracy']:.4f} "
            f"mae={metrics['mae']:.4f} "
            f"constraint_violation_rate={metrics['constraint_violation_rate']:.4f}"
        )

    return metrics

