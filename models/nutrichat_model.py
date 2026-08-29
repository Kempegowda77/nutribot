from typing import Dict, Optional

import torch
import torch.nn as nn

from models.constraint_penalty import ConstraintPenalty
from models.custom_transformer_encoder import CustomEmbeddingTransformerEncoder
from models.memory_attention import QueryMemoryAttention
from models.multitask_heads import MultiTaskHeads
from models.nutrition_fusion import NutritionFusion


class NutriChatModel(nn.Module):
    """
    End-to-end model that combines:
    - Nutrition fusion over token embeddings
    - Transformer encoder over fused embeddings
    - Query-memory attention
    - Dual heads (classification + regression)
    - Optional constraint penalty on regression outputs
    """

    def __init__(
        self,
        embed_dim: int,
        nutrition_dim: int,
        num_classes: int,
        regression_dim: int = 1,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 2048,
        transformer_layers: int = 4,
        transformer_dropout: float = 0.1,
        attn_dim: int = 128,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        penalty_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.regression_dim = regression_dim

        self.nutrition_fusion = NutritionFusion(
            embed_dim=embed_dim,
            nutrition_dim=nutrition_dim,
            dropout=dropout,
            use_layer_norm=True,
        )

        self.encoder = CustomEmbeddingTransformerEncoder(
            embed_dim=embed_dim,
            num_heads=transformer_heads,
            ff_dim=transformer_ff_dim,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
            activation="gelu",
            norm_first=True,
        )

        self.memory_attention = QueryMemoryAttention(
            embed_dim=embed_dim,
            attn_dim=attn_dim,
            dropout=dropout,
        )

        self.query_context_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = MultiTaskHeads(
            input_dim=embed_dim,
            num_classes=num_classes,
            regression_dim=regression_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

        self.constraint_penalty = ConstraintPenalty(
            penalty_weight=penalty_weight,
            reduction="mean",
            squared=True,
        )

    @staticmethod
    def _masked_mean_pool(sequence: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # sequence: [B, S, E]
        if mask is None:
            return sequence.mean(dim=1)

        if mask.dim() != 2:
            raise ValueError("attention_mask must be [B, S] when provided.")
        if mask.size(0) != sequence.size(0) or mask.size(1) != sequence.size(1):
            raise ValueError("attention_mask shape must match first two dims of sequence.")

        mask_f = mask.float().unsqueeze(-1)  # [B, S, 1]
        summed = (sequence * mask_f).sum(dim=1)  # [B, E]
        denom = mask_f.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        return summed / denom

    def _align_constraints(
        self,
        constraints: torch.Tensor,
        regression: torch.Tensor,
    ) -> torch.Tensor:
        """
        Align constraint tensor to regression output shape [B, regression_dim].
        Supports [B], [B, 1], or [B, regression_dim].
        """
        if constraints.dim() == 1:
            aligned = constraints.unsqueeze(-1)
        elif constraints.dim() == 2:
            aligned = constraints
        else:
            raise ValueError("constraint_vector must be [B], [B, 1], or [B, R].")

        if aligned.size(0) != regression.size(0):
            raise ValueError("Batch size mismatch between constraint_vector and regression.")

        if aligned.size(1) == 1 and regression.size(1) > 1:
            aligned = aligned.expand(-1, regression.size(1))
        elif aligned.size(1) != regression.size(1):
            raise ValueError(
                "constraint_vector second dimension must be 1 or match regression_dim."
            )

        return aligned

    def forward(
        self,
        token_embeddings: torch.Tensor,
        nutrition_vector: torch.Tensor,
        memory_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        constraint_vector: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if memory_embeddings.dim() != 3:
            raise ValueError(
                f"memory_embeddings must be [B, M, E], got {tuple(memory_embeddings.shape)}"
            )
        if memory_embeddings.size(2) != self.embed_dim:
            raise ValueError(
                f"memory_embeddings last dim must be embed_dim={self.embed_dim}"
            )

        fused_embeddings = self.nutrition_fusion(token_embeddings, nutrition_vector)
        encoded_tokens = self.encoder(fused_embeddings, attention_mask=attention_mask)

        query_embedding = self._masked_mean_pool(encoded_tokens, attention_mask)

        memory_out = self.memory_attention(
            query=query_embedding,
            memory=memory_embeddings,
            memory_mask=memory_mask,
        )
        context = memory_out["context"]
        attn_weights = memory_out["attn_weights"]

        combined = torch.cat([query_embedding, context], dim=-1)
        model_features = self.query_context_proj(combined)

        head_outputs = self.heads(model_features)
        regression = head_outputs["regression"]

        outputs: Dict[str, torch.Tensor] = {
            "fused_embeddings": fused_embeddings,
            "encoded_tokens": encoded_tokens,
            "query_embedding": query_embedding,
            "context": context,
            "attn_weights": attn_weights,
            "logits": head_outputs["logits"],
            "probs": head_outputs["probs"],
            "regression": regression,
        }

        if constraint_vector is not None:
            aligned_constraints = self._align_constraints(constraint_vector, regression)
            outputs["constraint_penalty"] = self.constraint_penalty(
                predictions=regression,
                constraints=aligned_constraints,
            )

        return outputs

