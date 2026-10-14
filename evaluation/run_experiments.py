from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.nutrichat_model import NutriChatModel

Variant = Literal["baseline", "memory ", "constraint"]


@dataclass
class ExperimentConfig:
    variant: Variant
    epochs: int = 3
    lr: float = 1e-3
    classification_weight: float = 1.0
    regression_weight: float = 1.0


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _prepare_inputs_for_variant(
    batch: Dict[str, torch.Tensor],
    embed_dim: int,
    variant: Variant,
) -> Dict[str, Optional[torch.Tensor]]:
    token_embeddings = batch["token_embeddings"]
    nutrition_vector = batch["nutrition_vector"]
    attention_mask = batch.get("attention_mask")

    if variant == "baseline":
        bsz = token_embeddings.size(0)
        # Disable memory effect by using a single all-zero memory vector.
        memory_embeddings = torch.zeros(
            (bsz, 1, embed_dim),
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        memory_mask = torch.ones((bsz, 1), device=token_embeddings.device, dtype=torch.long)
        constraint_vector = None
    elif variant == "memory":
        memory_embeddings = batch["memory_embeddings"]
        memory_mask = batch.get("memory_mask")
        constraint_vector = None
    else:
        memory_embeddings = batch["memory_embeddings"]
        memory_mask = batch.get("memory_mask")
        constraint_vector = batch.get("constraint_vector")

    return {
        "token_embeddings": token_embeddings,
        "nutrition_vector": nutrition_vector,
        "memory_embeddings": memory_embeddings,
        "attention_mask": attention_mask,
        "memory_mask": memory_mask,
        "constraint_vector": constraint_vector,
    }


def _compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    cls_w: float,
    reg_w: float,
) -> Dict[str, torch.Tensor]:
    logits = outputs["logits"]
    reg_pred = outputs["regression"]
    cls_target = batch["intent_label"].long()
    reg_target = batch["regression_target"].float()

    if reg_target.dim() == 1:
        reg_target = reg_target.unsqueeze(-1)
    reg_target = reg_target.view_as(reg_pred)

    cls_loss = cls_criterion(logits, cls_target)
    reg_loss = reg_criterion(reg_pred, reg_target)
    total = cls_w * cls_loss + reg_w * reg_loss

    constraint_loss = torch.tensor(0.0, device=logits.device)
    if "constraint_penalty" in outputs:
        constraint_loss = outputs["constraint_penalty"]
        total = total + constraint_loss

    return {
        "total": total,
        "cls": cls_loss,
        "reg": reg_loss,
        "constraint": constraint_loss,
    }


def _train_one_epoch(
    model: NutriChatModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    variant: Variant,
    cls_w: float,
    reg_w: float,
) -> Dict[str, float]:
    model.train()
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    totals = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "constraint": 0.0}

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        model_inputs = _prepare_inputs_for_variant(batch, model.embed_dim, variant)
        outputs = model(**model_inputs)
        losses = _compute_multitask_loss(outputs, batch, cls_criterion, reg_criterion, cls_w, reg_w)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        totals["loss"] += float(losses["total"].item())
        totals["cls"] += float(losses["cls"].item())
        totals["reg"] += float(losses["reg"].item())
        totals["constraint"] += float(losses["constraint"].item())

    denom = max(1, len(dataloader))
    return {k: v / denom for k, v in totals.items()}


@torch.no_grad()
def _evaluate_variant(
    model: NutriChatModel,
    dataloader: DataLoader,
    device: str,
    variant: Variant,
) -> Dict[str, float]:
    model.eval()

    correct = 0
    total_samples = 0
    mae_sum = 0.0
    mae_count = 0
    violations = 0
    violation_total = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        model_inputs = _prepare_inputs_for_variant(batch, model.embed_dim, variant)
        outputs = model(**model_inputs)

        logits = outputs["logits"]
        pred_class = logits.argmax(dim=-1)
        targets = batch["intent_label"].long()
        correct += int((pred_class == targets).sum().item())
        total_samples += int(targets.numel())

        reg_pred = outputs["regression"].float()
        reg_target = batch["regression_target"].float()
        if reg_target.dim() == 1:
            reg_target = reg_target.unsqueeze(-1)
        reg_target = reg_target.view_as(reg_pred)

        err = torch.abs(reg_pred - reg_target)
        mae_sum += float(err.sum().item())
        mae_count += int(err.numel())

        constraint_vector = model_inputs.get("constraint_vector")
        if constraint_vector is not None:
            constraints = constraint_vector.float()
            if constraints.dim() == 1:
                constraints = constraints.unsqueeze(-1)
            if constraints.size(1) == 1 and reg_pred.size(1) > 1:
                constraints = constraints.expand(-1, reg_pred.size(1))
            constraints = constraints.view_as(reg_pred)

            batch_violations = reg_pred > constraints
            violations += int(batch_violations.sum().item())
            violation_total += int(batch_violations.numel())

    return {
        "accuracy": (correct / total_samples) if total_samples else 0.0,
        "mae": (mae_sum / mae_count) if mae_count else 0.0,
        "constraint_violation_rate": (violations / violation_total) if violation_total else 0.0,
    }


def _format_results_table(rows: List[Dict[str, float | str]]) -> str:
    headers = ["variant", "accuracy", "mae", "constraint_violation_rate"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    f"{float(row['accuracy']):.4f}",
                    f"{float(row['mae']):.4f}",
                    f"{float(row['constraint_violation_rate']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run_experiments(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_init_kwargs: Dict[str, int | float],
    experiment_configs: Optional[Iterable[ExperimentConfig]] = None,
    device: str = "cpu",
) -> List[Dict[str, float | str]]:
    """
    Runs three model variants and prints a results table:
    - baseline: no effective memory, no constraints
    - memory: uses memory embeddings, no constraints
    - constraint: uses memory + constraint penalty path
    """
    if experiment_configs is None:
        experiment_configs = [
            ExperimentConfig(variant="baseline"),
            ExperimentConfig(variant="memory"),
            ExperimentConfig(variant="constraint"),
        ]

    results: List[Dict[str, float | str]] = []

    for cfg in experiment_configs:
        model = NutriChatModel(**model_init_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        for epoch in range(cfg.epochs):
            train_stats = _train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                variant=cfg.variant,
                cls_w=cfg.classification_weight,
                reg_w=cfg.regression_weight,
            )
            print(
                f"[{cfg.variant}] epoch={epoch + 1}/{cfg.epochs} "
                f"loss={train_stats['loss']:.4f} "
                f"cls={train_stats['cls']:.4f} "
                f"reg={train_stats['reg']:.4f} "
                f"constraint={train_stats['constraint']:.4f}"
            )

        metrics = _evaluate_variant(
            model=model,
            dataloader=val_loader,
            device=device,
            variant=cfg.variant,
        )
        row: Dict[str, float | str] = {"variant": cfg.variant}
        row.update(metrics)
        results.append(row)

    print("\nExperiment Results")
    print(_format_results_table(results))
    return results


if __name__ == "__main__":
    print(
        "Import and call run_experiments(train_loader, val_loader, model_init_kwargs, ...)\n"
        "Expected batch keys: token_embeddings, nutrition_vector, memory_embeddings, "
        "intent_label, regression_target, and optional masks/constraint_vector."
    )

