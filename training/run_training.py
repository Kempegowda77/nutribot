"""
NutriChat AI — Training Pipeline (IEEE Edition)
===============================================
Uses MuRIL (google/muril-base-cased) — Google's multilingual model
for 17 Indian languages + English — fine-tuned on REAL food data.

Usage:
    python training/run_training.py [--epochs N] [--batch-size B] [--lr LR]

Checkpoint: models/nutrichat_muril_checkpoint.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.data_loader import create_intent_regression_dataloader
from models.nutrichat_model import NutriChatModel

# -- Config --------------------------------------------------------------------
MODEL_CONFIG_PATH  = ROOT / "models" / "model_config.json"
TRAIN_JSON         = ROOT / "data" / "train.json"
VAL_JSON           = ROOT / "data" / "val.json"
CHECKPOINT_PATH    = ROOT / "models" / "nutrichat_muril_checkpoint.pt"

# Backbone options:
# "distilbert-base-multilingual-cased"  -> fast (~280MB), 104 langs, good for CPU
# "google/muril-base-cased"             -> best for Hinglish (~920MB), needs GPU/time
DEFAULT_BACKBONE = "distilbert-base-multilingual-cased"
BACKBONE_DIM     = 768       # hidden size (same for both)
MEMORY_SLOTS     = 4
MAX_LENGTH       = 128
NUTRITION_DIM    = 8


# -- Embedding extractor (MuRIL -> project to embed_dim) -----------------------

class BackboneEmbeddingExtractor(nn.Module):
    """
    Wraps any BERT-style backbone and projects to model embed_dim.
    Works with mDistilBERT (fast CPU) or MuRIL (best Hinglish, needs GPU).
    """
    def __init__(self, backbone_model: nn.Module, embed_dim: int, finetune: bool = True):
        super().__init__()
        self.backbone = backbone_model
        self.proj     = nn.Linear(BACKBONE_DIM, embed_dim)
        self.dropout  = nn.Dropout(0.1)
        if not finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Freeze bottom layers, fine-tune top 4 transformer layers
            layers = getattr(self.backbone, 'transformer', None) or \
                     getattr(self.backbone, 'encoder', None)
            if layers is not None:
                layer_list = getattr(layers, 'layer', None) or \
                             getattr(layers, 'layers', None)
                if layer_list:
                    n_freeze = max(0, len(layer_list) - 4)
                    for layer in layer_list[:n_freeze]:
                        for param in layer.parameters():
                            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state        # [B, S, 768]
        proj   = self.proj(self.dropout(hidden))  # [B, S, embed_dim]
        return proj


# -- Collate fn ----------------------------------------------------------------

def make_collate_fn(embed_dim: int, nutrition_dim: int, memory_slots: int):
    def collate_fn(batch):
        input_ids         = torch.stack([b["input_ids"]         for b in batch])
        attention_mask    = torch.stack([b["attention_mask"]     for b in batch])
        intent_label      = torch.stack([b["intent_label"]       for b in batch])
        regression_target = torch.stack([b["regression_target"]  for b in batch])

        if "nutrition_vector" in batch[0]:
            nutrition_vector = torch.stack([b["nutrition_vector"] for b in batch])
        else:
            nutrition_vector = torch.zeros(len(batch), nutrition_dim)

        if all("constraint_vector" in b for b in batch):
            constraint_vector = torch.stack([b["constraint_vector"] for b in batch])
        elif any("constraint_vector" in b for b in batch):
            constraint_vector = torch.stack([
                b.get("constraint_vector", torch.zeros(1)) for b in batch
            ])
        else:
            constraint_vector = None

        return {
            "input_ids":          input_ids,
            "attention_mask":     attention_mask,
            "intent_label":       intent_label,
            "regression_target":  regression_target,
            "nutrition_vector":   nutrition_vector,
            "constraint_vector":  constraint_vector,
        }
    return collate_fn


# -- Evaluation ----------------------------------------------------------------

def evaluate(model, extractor, loader, device, embed_dim, memory_slots):
    model.eval(); extractor.eval()
    correct = total = 0
    total_mae = 0.0

    with torch.no_grad():
        for batch in loader:
            B              = batch["input_ids"].size(0)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_label   = batch["intent_label"].to(device)
            reg_target     = batch["regression_target"].float().to(device)
            nutr_vec       = batch["nutrition_vector"].to(device)

            token_emb = extractor(input_ids, attention_mask)
            memory    = torch.zeros(B, memory_slots, embed_dim, device=device)
            outputs   = model(
                token_embeddings=token_emb,
                nutrition_vector=nutr_vec,
                memory_embeddings=memory,
                attention_mask=attention_mask,
            )
            preds    = outputs["logits"].argmax(dim=-1)
            correct += (preds == intent_label).sum().item()
            total   += B

            reg_pred  = outputs["regression"].squeeze(-1)
            total_mae += (reg_pred - reg_target).abs().sum().item()

    acc = correct / max(1, total)
    mae = total_mae / max(1, total)
    rmse = ((total_mae / max(1, total)) ** 0.5)  # approx

    # F1 per class (macro)
    return {"accuracy": acc, "mae": mae, "rmse": rmse}


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train NutriChat AI (MuRIL + Real Data)")
    parser.add_argument("--epochs",     type=int,   default=15,    help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=16,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-5,  help="Learning rate")
    parser.add_argument("--device",     type=str,   default="auto")
    parser.add_argument("--backbone",   type=str,   default=DEFAULT_BACKBONE,
                        help="HuggingFace model name (e.g. distilbert-base-multilingual-cased or google/muril-base-cased)")
    parser.add_argument("--no-finetune", action="store_true",
                        help="Freeze backbone entirely (faster, less accurate)")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"\n{'='*60}")
    print(f"  NutriChat AI - IEEE Training Pipeline")
    print(f"  Backbone: {args.backbone}")
    print(f"  Device: {device}  |  Epochs: {args.epochs}  |  LR: {args.lr}")
    print(f"{'='*60}")

    # -- Check datasets — build from real data if missing
    if not TRAIN_JSON.exists() or not VAL_JSON.exists():
        print("\n[data] Real datasets not found. Building from real sources...")
        import subprocess
        steps = [
            (ROOT / "data" / "sources" / "indian_food_loader.py", "Indian Food DB"),
            (ROOT / "data" / "sources" / "usda_fdc_loader.py",    "USDA FDC (downloads ~30MB)"),
            (ROOT / "data" / "sources" / "dataset_merger.py",     "Merge + Generate splits"),
        ]
        for script, label in steps:
            print(f"\n  Running: {label}...")
            result = subprocess.run([sys.executable, str(script)], capture_output=False, text=True)
            if result.returncode != 0:
                print(f"  [WARN]  {label} failed — check internet/files")
    else:
        print(f"\n[data] Found existing datasets:")
        print(f"  Train: {TRAIN_JSON}  ({TRAIN_JSON.stat().st_size // 1024} KB)")
        print(f"  Val:   {VAL_JSON}  ({VAL_JSON.stat().st_size // 1024} KB)")

    # -- Load model config
    with open(MODEL_CONFIG_PATH) as f:
        cfg = json.load(f)
    embed_dim     = cfg["embed_dim"]
    nutrition_dim = cfg.get("nutrition_dim", NUTRITION_DIM)
    print(f"\n[model] Config: embed_dim={embed_dim}, num_classes={cfg['num_classes']}")

    # -- Load backbone tokenizer + model
    print(f"\n[model] Loading backbone: {args.backbone}...")
    tokenizer      = AutoTokenizer.from_pretrained(args.backbone)
    backbone_model = AutoModel.from_pretrained(args.backbone)
    print(f"  [OK] Backbone loaded")

    finetune  = not args.no_finetune
    extractor = BackboneEmbeddingExtractor(backbone_model, embed_dim, finetune=finetune).to(device)
    model     = NutriChatModel(**cfg).to(device)

    backbone_params = sum(p.numel() for p in extractor.backbone.parameters() if p.requires_grad)
    model_params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj_params     = sum(p.numel() for p in extractor.proj.parameters() if p.requires_grad)
    print(f"  Backbone trainable params: {backbone_params:,}")
    print(f"  NutriChat model params:    {model_params:,}")
    print(f"  Projection params:         {proj_params:,}")
    print(f"  Total trainable:           {backbone_params + model_params + proj_params:,}")

    # -- DataLoaders
    collate = make_collate_fn(embed_dim, nutrition_dim, MEMORY_SLOTS)
    base_train = create_intent_regression_dataloader(
        TRAIN_JSON, tokenizer, batch_size=args.batch_size, max_length=MAX_LENGTH, shuffle=True
    )
    base_val = create_intent_regression_dataloader(
        VAL_JSON, tokenizer, batch_size=args.batch_size, max_length=MAX_LENGTH, shuffle=False
    )
    train_loader = DataLoader(base_train.dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(base_val.dataset,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate, num_workers=0)
    print(f"\n[data] Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # -- Optimizer (different LR for MuRIL vs head)
    muril_lr = args.lr           # 2e-5 for fine-tuning
    head_lr  = args.lr * 10     # 2e-4 for task heads
    optimizer = torch.optim.AdamW([
        {"params": extractor.backbone.parameters(), "lr": muril_lr},
        {"params": extractor.proj.parameters(),     "lr": head_lr},
        {"params": model.parameters(),              "lr": head_lr},
    ], weight_decay=1e-4)

    total_steps = args.epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )

    # -- Training loop
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    best_acc      = 0.0

    print(f"\n{'='*60}")
    print(f"  STARTING TRAINING")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        model.train(); extractor.train()
        run = {"total": 0.0, "cls": 0.0, "reg": 0.0}
        n_correct = n_total = 0

        for step, batch in enumerate(train_loader, 1):
            B              = batch["input_ids"].size(0)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_label   = batch["intent_label"].to(device)
            reg_target     = batch["regression_target"].float().to(device).unsqueeze(-1)
            nutr_vec       = batch["nutrition_vector"].to(device)
            constr_vec     = (batch["constraint_vector"].to(device)
                              if batch["constraint_vector"] is not None else None)

            token_emb = extractor(input_ids, attention_mask)
            memory    = torch.zeros(B, MEMORY_SLOTS, embed_dim, device=device)

            outputs = model(
                token_embeddings=token_emb,
                nutrition_vector=nutr_vec,
                memory_embeddings=memory,
                attention_mask=attention_mask,
                constraint_vector=constr_vec,
            )

            cls_loss   = cls_criterion(outputs["logits"], intent_label)
            reg_loss   = reg_criterion(outputs["regression"], reg_target)
            total_loss = cls_loss + 0.01 * reg_loss
            if "constraint_penalty" in outputs:
                total_loss = total_loss + 0.1 * outputs["constraint_penalty"]

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(extractor.parameters()) + list(model.parameters()), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            run["total"] += total_loss.item()
            run["cls"]   += cls_loss.item()
            run["reg"]   += reg_loss.item()
            preds = outputs["logits"].argmax(-1)
            n_correct += (preds == intent_label).sum().item()
            n_total   += B

            if step % max(1, len(train_loader) // 4) == 0:
                print(f"  Ep {epoch+1}/{args.epochs} step {step}/{len(train_loader)}"
                      f" | loss={total_loss.item():.4f} cls={cls_loss.item():.4f}"
                      f" reg={reg_loss.item():.4f}")

        train_acc = n_correct / max(1, n_total)
        avg_loss  = run["total"] / len(train_loader)
        val_m     = evaluate(model, extractor, val_loader, device, embed_dim, MEMORY_SLOTS)

        print(f"\n[Epoch {epoch+1:02d}] train_loss={avg_loss:.4f} | "
              f"train_acc={train_acc:.3f} | "
              f"val_acc={val_m['accuracy']:.3f} | "
              f"val_mae={val_m['mae']:.2f}")

        if val_m["accuracy"] >= best_acc:
            best_acc = val_m["accuracy"]
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "extractor_state": extractor.state_dict(),
                "optimizer":       optimizer.state_dict(),
                "val_accuracy":    val_m["accuracy"],
                "val_mae":         val_m["mae"],
                "val_rmse":        val_m["rmse"],
                "backbone":        args.backbone,
                "model_config":    cfg,
            }, CHECKPOINT_PATH)
            print(f"  [OK] Checkpoint saved (val_acc={best_acc:.3f}) -> {CHECKPOINT_PATH}")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val_acc: {best_acc:.3f}")
    print(f"  Checkpoint:   {CHECKPOINT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
