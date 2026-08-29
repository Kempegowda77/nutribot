from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.baselines import HFMultiTaskBaseline, RuleBasedChatbotBaseline


def _compute_metrics(
    pred_intents: List[int],
    true_intents: List[int],
    pred_reg: List[float],
    true_reg: List[float],
    constraints: List[float] | None = None,
) -> Dict[str, float]:
    total = max(1, len(true_intents))
    accuracy = sum(int(p == t) for p, t in zip(pred_intents, true_intents)) / total
    mae = sum(abs(p - t) for p, t in zip(pred_reg, true_reg)) / max(1, len(true_reg))

    cvr = 0.0
    if constraints:
        violations = sum(int(p > c) for p, c in zip(pred_reg, constraints))
        cvr = violations / max(1, len(constraints))
    return {"accuracy": accuracy, "mae": mae, "constraint_violation_rate": cvr}


@torch.no_grad()
def evaluate_hf_baseline(
    model_name: str,
    dataloader: DataLoader,
    num_classes: int,
    device: str = "cpu",
) -> Dict[str, float]:
    model = HFMultiTaskBaseline(model_name=model_name, num_classes=num_classes).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    losses: List[float] = []
    pred_intents: List[int] = []
    true_intents: List[int] = []
    pred_reg: List[float] = []
    true_reg: List[float] = []
    constraints: List[float] = []

    for batch in dataloader:
        text_batch = batch["raw_text"] if "raw_text" in batch else [""] * len(batch["intent_label"])
        encoded = tokenizer(
            text_batch,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        intent_target = batch["intent_label"].long().to(device)
        reg_target = batch["regression_target"].float().to(device).view(-1, 1)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_loss = criterion_cls(out["logits"], intent_target)
        reg_loss = criterion_reg(out["regression"], reg_target)
        losses.append(float((cls_loss + reg_loss).item()))

        pred_intents.extend(out["logits"].argmax(dim=-1).cpu().tolist())
        true_intents.extend(intent_target.cpu().tolist())
        pred_reg.extend(out["regression"].view(-1).cpu().tolist())
        true_reg.extend(reg_target.view(-1).cpu().tolist())

        if "constraint_vector" in batch:
            c = batch["constraint_vector"]
            if c.dim() > 1:
                c = c[:, 0]
            constraints.extend(c.float().cpu().tolist())

    metrics = _compute_metrics(
        pred_intents=pred_intents,
        true_intents=true_intents,
        pred_reg=pred_reg,
        true_reg=true_reg,
        constraints=constraints if constraints else None,
    )
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    return metrics


def evaluate_rule_based(
    dataloader: DataLoader,
    intent_labels: List[str],
) -> Dict[str, float]:
    model = RuleBasedChatbotBaseline(intent_labels=intent_labels)
    pred_intents: List[int] = []
    true_intents: List[int] = []
    pred_reg: List[float] = []
    true_reg: List[float] = []
    constraints: List[float] = []

    for batch in dataloader:
        text_batch = batch["raw_text"] if "raw_text" in batch else [""] * len(batch["intent_label"])
        intents = batch["intent_label"].tolist()
        regs = batch["regression_target"].float().tolist()

        for i, text in enumerate(text_batch):
            out = model.predict(text)
            pred_intents.append(int(out["intent_id"]))
            true_intents.append(int(intents[i]))
            pred_reg.append(float(out["regression"]))
            true_reg.append(float(regs[i]))

        if "constraint_vector" in batch:
            c = batch["constraint_vector"]
            if c.dim() > 1:
                c = c[:, 0]
            constraints.extend(c.float().tolist())

    return _compute_metrics(
        pred_intents=pred_intents,
        true_intents=true_intents,
        pred_reg=pred_reg,
        true_reg=true_reg,
        constraints=constraints if constraints else None,
    )


def run_baseline_suite(
    dataloader: DataLoader,
    num_classes: int,
    intent_labels: List[str],
    device: str = "cpu",
) -> List[Dict[str, float | str]]:
    bert_metrics = evaluate_hf_baseline(
        model_name="bert-base-uncased",
        dataloader=dataloader,
        num_classes=num_classes,
        device=device,
    )
    distil_metrics = evaluate_hf_baseline(
        model_name="distilbert-base-uncased",
        dataloader=dataloader,
        num_classes=num_classes,
        device=device,
    )
    rule_metrics = evaluate_rule_based(
        dataloader=dataloader,
        intent_labels=intent_labels,
    )

    rows: List[Dict[str, float | str]] = [
        {"model": "BERT", **bert_metrics},
        {"model": "DistilBERT", **distil_metrics},
        {"model": "Rule-based", **rule_metrics},
    ]

    headers = ["model", "accuracy", "mae", "constraint_violation_rate"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            f"| {row['model']} | {float(row['accuracy']):.4f} | "
            f"{float(row['mae']):.4f} | {float(row['constraint_violation_rate']):.4f} |"
        )
    return rows

