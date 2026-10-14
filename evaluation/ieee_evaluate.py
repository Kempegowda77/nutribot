"""
IEEE-Standard Evaluation Suite for NutriChat AI
=================================================
Computes all metrics needed for an IEEE paper:
  - Intent Classification: Accuracy, F1 (macro), Precision, Recall per class
  - Calorie Regression: MAE, RMSE, R² score
  - System Latency: p50, p95 response time (ms)

Run:
    python evaluation/ieee_evaluate.py
"""

import json
import sys
import time
import math
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from data.data_loader import create_intent_regression_dataloader
from models.nutrichat_model import NutriChatModel

CHECKPOINT_PATH = ROOT / "models" / "nutrichat_muril_checkpoint.pt"
MODEL_CONFIG    = ROOT / "models" / "model_config.json"
VAL_JSON        = ROOT / "data" / "val.json"

INTENT_NAMES = {
    0: "calorie_query",
    1: "meal_suggestion",
    2: "diet_advice",
    3: "nutrition_info",
    4: "portion_control",
}

MURIL_NAME    = "google/muril-base-cased"
MURIL_DIM     = 768
MEMORY_SLOTS  = 4
MAX_LENGTH    = 128
NUTRITION_DIM = 8


# -- Metric helpers ------------------------------------------------------------

def compute_f1_per_class(
    y_true: List[int], y_pred: List[int], n_classes: int
) -> Dict[str, float]:
    """Compute precision, recall, F1 per class and macro average."""
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    results = {}
    f1s = []
    for c in range(n_classes):
        prec  = tp[c] / max(1, tp[c] + fp[c])
        rec   = tp[c] / max(1, tp[c] + fn[c])
        f1    = 2 * prec * rec / max(1e-9, prec + rec)
        results[INTENT_NAMES[c]] = {"precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)

    results["macro_f1"] = sum(f1s) / len(f1s)
    results["accuracy"] = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
    return results


def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """MAE, RMSE, R² score."""
    n    = len(y_true)
    mae  = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse  = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    mean_t  = sum(y_true) / n
    ss_tot  = sum((t - mean_t) ** 2 for t in y_true)
    ss_res  = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    r2      = 1 - ss_res / max(1e-9, ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """p50, p95, p99 latency stats."""
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "p50_ms":  s[int(0.50 * n)],
        "p95_ms":  s[int(0.95 * n)],
        "p99_ms":  s[int(0.99 * n)],
        "mean_ms": sum(s) / n,
        "max_ms":  s[-1],
    }


# -- Load model ----------------------------------------------------------------

def load_model(cfg: dict, device: str):
    """Load trained checkpoint."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run: python training/run_training.py"
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    backbone_name = ckpt.get("backbone", MURIL_NAME)

    print(f"  Backbone: {backbone_name}")
    print(f"  Trained epoch: {ckpt.get('epoch', '?')}")
    print(f"  Checkpoint val_acc: {ckpt.get('val_accuracy', '?'):.3f}")

    tokenizer   = AutoTokenizer.from_pretrained(backbone_name)
    muril_model = AutoModel.from_pretrained(backbone_name)

    from training.run_training import MuRILEmbeddingExtractor
    extractor = MuRILEmbeddingExtractor(muril_model, cfg["embed_dim"], finetune=False)
    extractor.load_state_dict(ckpt["extractor_state"], strict=False)
    extractor = extractor.to(device).eval()

    model = NutriChatModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    return tokenizer, extractor, model


# -- Main evaluation -----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  NutriChat AI — IEEE Evaluation Suite")
    print(f"  Device: {device}")
    print("=" * 60)

    with open(MODEL_CONFIG) as f:
        cfg = json.load(f)
    embed_dim     = cfg["embed_dim"]
    nutrition_dim = cfg.get("nutrition_dim", NUTRITION_DIM)

    print("\n[1/3] Loading model checkpoint...")
    tokenizer, extractor, model = load_model(cfg, device)

    print("\n[2/3] Loading validation dataset...")
    val_loader_base = create_intent_regression_dataloader(
        VAL_JSON, tokenizer, batch_size=32, max_length=MAX_LENGTH, shuffle=False
    )

    def collate_fn(batch):
        input_ids         = torch.stack([b["input_ids"]        for b in batch])
        attention_mask    = torch.stack([b["attention_mask"]    for b in batch])
        intent_label      = torch.stack([b["intent_label"]      for b in batch])
        regression_target = torch.stack([b["regression_target"] for b in batch])
        nutrition_vector  = (
            torch.stack([b["nutrition_vector"] for b in batch])
            if "nutrition_vector" in batch[0]
            else torch.zeros(len(batch), nutrition_dim)
        )
        return {
            "input_ids":          input_ids,
            "attention_mask":     attention_mask,
            "intent_label":       intent_label,
            "regression_target":  regression_target,
            "nutrition_vector":   nutrition_vector,
        }

    val_loader = DataLoader(val_loader_base.dataset, batch_size=32,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"  Val samples: {len(val_loader.dataset)}")

    print("\n[3/3] Running evaluation...")
    y_true_intent, y_pred_intent = [], []
    y_true_reg,    y_pred_reg    = [], []
    latencies_ms                 = []

    with torch.no_grad():
        for batch in val_loader:
            B              = batch["input_ids"].size(0)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_label   = batch["intent_label"].to(device)
            reg_target     = batch["regression_target"].to(device)
            nutr_vec       = batch["nutrition_vector"].to(device)
            memory         = torch.zeros(B, MEMORY_SLOTS, embed_dim, device=device)

            t0        = time.perf_counter()
            token_emb = extractor(input_ids, attention_mask)
            outputs   = model(
                token_embeddings=token_emb,
                nutrition_vector=nutr_vec,
                memory_embeddings=memory,
                attention_mask=attention_mask,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000 / B)

            preds    = outputs["logits"].argmax(-1)
            reg_pred = outputs["regression"].squeeze(-1)

            y_true_intent.extend(intent_label.cpu().tolist())
            y_pred_intent.extend(preds.cpu().tolist())
            y_true_reg.extend(reg_target.cpu().tolist())
            y_pred_reg.extend(reg_pred.cpu().tolist())

    # -- Compute metrics
    intent_metrics = compute_f1_per_class(y_true_intent, y_pred_intent, n_classes=5)
    reg_metrics    = compute_regression_metrics(y_true_reg, y_pred_reg)
    latency_stats  = compute_latency_stats(latencies_ms)

    # -- Print IEEE-style results table
    print("\n" + "=" * 60)
    print("  RESULTS (IEEE Paper Table)")
    print("=" * 60)

    print("\n  -- Intent Classification --")
    print(f"  {'Intent':<22} {'Precision':>10} {'Recall':>9} {'F1':>8}")
    print("  " + "-" * 52)
    for k, v in intent_metrics.items():
        if k in ("macro_f1", "accuracy"):
            continue
        print(f"  {k:<22} {v['precision']:>10.4f} {v['recall']:>9.4f} {v['f1']:>8.4f}")
    print("  " + "-" * 52)
    print(f"  {'Macro F1':<22} {intent_metrics['macro_f1']:>10.4f}")
    print(f"  {'Accuracy':<22} {intent_metrics['accuracy']:>10.4f}")

    print("\n  -- Calorie Regression --")
    print(f" first commit  MAE:  {reg_metrics['mae']:.2f} kcal")
    print(f"  RMSE: {reg_metrics['rmse']:.2f} kcal")
    print(f"  R²:   {reg_metrics['r2']:.4f}")

    print("\n  -- System Latency --")
    print(f"  p50:  {latency_stats['p50_ms']:.2f} ms/sample")
    print(f"  p95:  {latency_stats['p95_ms']:.2f} ms/sample")
    print(f"  Mean: {latency_stats['mean_ms']:.2f} ms/sample")

    # -- Save results JSON
    results = {
        "intent_classification": intent_metrics,
        "calorie_regression":    reg_metrics,
        "latency":               latency_stats,
        "num_val_samples":       len(y_true_intent),
    }
    out_path = ROOT / "evaluation" / "ieee_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [OK] Full results saved -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
