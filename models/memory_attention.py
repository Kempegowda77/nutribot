from typing import Dict

import torch
import torch.nn as nn


class QueryMemoryAttention(nn.Module):
    """
     Attention from current query embedding to memory embeddings.

    Inputs:
    - query: [B, E]
    - memory: [B, M, E]
    - memory_mask (optional): [B, M] where 1=valid and 0=padding

    Outputs:
    - context: [B, E]
    - attn_weights: [B, M]
    """

    def __init__(self, embed_dim: int, attn_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim

        self.query_proj = nn.Linear(embed_dim, attn_dim)
        self.memory_proj = nn.Linear(embed_dim, attn_dim)
        self.score_proj = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if query.dim() != 2:
            raise ValueError(f"query must be [B, E], got {tuple(query.shape)}")
        if memory.dim() != 3:
            raise ValueError(f"memory must be [B, M, E], got {tuple(memory.shape)}")
        if query.size(0) != memory.size(0):
            raise ValueError("Batch size mismatch between query and memory.")
        if query.size(1) != self.embed_dim or memory.size(2) != self.embed_dim:
            raise ValueError(
                f"Embedding dim mismatch. Expected {self.embed_dim} for query and memory."
            )

        # Project query and memory into shared attention space.
        q = self.query_proj(query).unsqueeze(1)  # [B, 1, A]
        m = self.memory_proj(memory)             # [B, M, A]

        # Additive attention scores.
        scores = self.score_proj(torch.tanh(q + m)).squeeze(-1)  # [B, M]

        if memory_mask is not None:
            if memory_mask.dim() != 2:
                raise ValueError("memory_mask must be [B, M] when provided.")
            if memory_mask.shape != scores.shape:
                raise ValueError("memory_mask shape must match [B, M].")
            scores = scores.masked_fill(memory_mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # [B, M]
        attn_weights = self.dropout(attn_weights)

        # Weighted sum over memory vectors.
        context = torch.bmm(attn_weights.unsqueeze(1), memory).squeeze(1)  # [B, E]

        return {"context": context, "attn_weights": attn_weights}

