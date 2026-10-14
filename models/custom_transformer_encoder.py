from typing import Optional

import torch
import torch.nn as nn


class CustomEmbeddingTransformerEncoder(nn.Module):
    """
    Transformer encoder that consumes custom input embeddings.

    Inputs :
    - embeddings: [batch, seq_len, embed_dim]
    - attention_mask (optional): [batch, seq_len], where 1 means keep and 0 means pad

    Output:
    - encoded: [batch, seq_len, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if embeddings.dim() != 3:
            raise ValueError(
                f"embeddings must be 3D [B, S, E], got shape {tuple(embeddings.shape)}"
            )
        if embeddings.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected embedding dim {self.embed_dim}, got {embeddings.size(-1)}"
            )

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be 2D [B, S] when provided."
                )
            if attention_mask.size(0) != embeddings.size(0) or attention_mask.size(1) != embeddings.size(1):
                raise ValueError(
                    "attention_mask shape must match first two dims of embeddings: [B, S]."
                )
            # TransformerEncoder expects True for positions to ignore (padding).
            key_padding_mask = attention_mask == 0

        encoded = self.encoder(
            src=embeddings,
            src_key_padding_mask=key_padding_mask,
        )
        return encoded

