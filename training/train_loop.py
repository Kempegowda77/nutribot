from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MultiTaskTrainer:
    """
    Training loop for multi-task learning:
    - classification loss from model logits
    - regression loss from model regression output
    - optional constraint penalty from model outputs

    Expected batch key:
    - token_embeddings: [B, S, E]
    - nutrition_vector: [B, N] or [B, S, N]
    - memory_embeddings: [B, M, E]
    - intent_label: [B]
    - regression_target: [B] or [B, R]
    Optional:
    - attention_mask: [B, S]
    - memory_mask: [B, M]
    - constraint_vector: [B] or [B, 1] or [B, R]
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        classification_criterion: Optional[nn.Module] = None,
        regression_criterion: Optional[nn.Module] = None,
        classification_weight: float = 1.0,
        regression_weight: float = 1.0,
        device: str = "cpu",
        log_every: int = 10,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.classification_criterion = classification_criterion or nn.CrossEntropyLoss()
        self.regression_criterion = regression_criterion or nn.MSELoss()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.device = device
        self.log_every = max(1, log_every)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        regression_pred = outputs["regression"]
        intent_targets = batch["intent_label"].long()
        regression_targets = batch["regression_target"].float()

        if regression_targets.dim() == 1:
            regression_targets = regression_targets.unsqueeze(-1)
        regression_targets = regression_targets.view_as(regression_pred)

        cls_loss = self.classification_criterion(logits, intent_targets)
        reg_loss = self.regression_criterion(regression_pred, regression_targets)

        total = (
            self.classification_weight * cls_loss
            + self.regression_weight * reg_loss
        )

        constraint_loss = torch.tensor(0.0, device=logits.device)
        if "constraint_penalty" in outputs:
            constraint_loss = outputs["constraint_penalty"]
            total = total + constraint_loss

        return {
            "total_loss": total,
            "classification_loss": cls_loss,
            "regression_loss": reg_loss,
            "constraint_loss": constraint_loss,
        }

    def train_one_epoch(self, dataloader: DataLoader, epoch_index: int = 0) -> Dict[str, float]:
        self.model.train()
        running = {
            "total_loss": 0.0,
            "classification_loss": 0.0,
            "regression_loss": 0.0,
            "constraint_loss": 0.0,
        }

        for step, batch in enumerate(dataloader, start=1):
            batch = self._to_device(batch)

            outputs = self.model(
                token_embeddings=batch["token_embeddings"],
                nutrition_vector=batch["nutrition_vector"],
                memory_embeddings=batch["memory_embeddings"],
                attention_mask=batch.get("attention_mask"),
                memory_mask=batch.get("memory_mask"),
                constraint_vector=batch.get("constraint_vector"),
            )

            loss_dict = self._compute_losses(outputs, batch)
            total_loss = loss_dict["total_loss"]

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            for k in running:
                running[k] += float(loss_dict[k].item())

            if step % self.log_every == 0:
                print(
                    f"[Epoch {epoch_index + 1} | Step {step}/{len(dataloader)}] "
                    f"total={loss_dict['total_loss'].item():.4f} "
                    f"cls={loss_dict['classification_loss'].item():.4f} "
                    f"reg={loss_dict['regression_loss'].item():.4f} "
                    f"constraint={loss_dict['constraint_loss'].item():.4f}"
                )

        num_steps = max(1, len(dataloader))
        epoch_metrics = {k: v / num_steps for k, v in running.items()}
        print(
            f"[Epoch {epoch_index + 1} Completed] "
            f"total={epoch_metrics['total_loss']:.4f} "
            f"cls={epoch_metrics['classification_loss']:.4f} "
            f"reg={epoch_metrics['regression_loss']:.4f} "
            f"constraint={epoch_metrics['constraint_loss']:.4f}"
        )
        return epoch_metrics

    def fit(self, train_loader: DataLoader, num_epochs: int) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for epoch in range(num_epochs):
            metrics = self.train_one_epoch(train_loader, epoch_index=epoch)
            history.append(metrics)
        return history

