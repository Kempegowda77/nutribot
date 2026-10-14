from typing import Optional

import torch
import torch.nn as nn


class NutritionFusion(nn.Module):
    """
    Fuses token embeddings with nutrition features and projects back to
    the origina l embedding size.

    Inputs:
    - token_embeddings: [batch, seq_len, embed_dim]
    - nutrition_vector: [batch, nutrition_dim] or [batch, seq_len, nutrition_dim]

    Output:
    - fused_embeddings: [batch, seq_len, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int,
        nutrition_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.nutrition_dim = nutrition_dim

        self.proj = nn.Linear(embed_dim + nutrition_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm: Optional[nn.Module] = nn.LayerNorm(embed_dim) if use_layer_norm else None

    def forward(
        self,
        token_embeddings: torch.Tensor,
        nutrition_vector: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.dim() != 3:
            raise ValueError(
                f"token_embeddings must be 3D [B, S, E], got shape {tuple(token_embeddings.shape)}"
            )
        if token_embeddings.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected embedding dim {self.embed_dim}, got {token_embeddings.size(-1)}"
            )

        batch_size, seq_len, _ = token_embeddings.shape

        if nutrition_vector.dim() == 2:
            # [B, N] -> [B, S, N]
            if nutrition_vector.size(0) != batch_size:
                raise ValueError(
                    "Batch size mismatch between token_embeddings and nutrition_vector."
                )
            if nutrition_vector.size(1) != self.nutrition_dim:
                raise ValueError(
                    f"Expected nutrition dim {self.nutrition_dim}, got {nutrition_vector.size(1)}"
                )
            nutrition_expanded = nutrition_vector.unsqueeze(1).expand(-1, seq_len, -1)

        elif nutrition_vector.dim() == 3:
            # [B, S, N]
            if nutrition_vector.size(0) != batch_size or nutrition_vector.size(1) != seq_len:
                raise ValueError(
                    "nutrition_vector with 3D input must have shape [B, S, N] matching token_embeddings."
                )
            if nutrition_vector.size(2) != self.nutrition_dim:
                raise ValueError(
                    f"Expected nutrition dim {self.nutrition_dim}, got {nutrition_vector.size(2)}"
                )
            nutrition_expanded = nutrition_vector
        else:
            raise ValueError(
                f"nutrition_vector must be 2D [B, N] or 3D [B, S, N], got shape {tuple(nutrition_vector.shape)}"
            )

        concat_features = torch.cat([token_embeddings, nutrition_expanded], dim=-1)
        fused = self.proj(concat_features)
        fused = self.dropout(fused)

        if self.norm is not None:
            fused = self.norm(fused)

        return fused

