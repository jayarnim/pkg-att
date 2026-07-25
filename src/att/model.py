import torch
import torch.nn as nn
from .layers.broadcast import BroadCast
from .layers.score import build as build_score
from .layers.simplex import SoftmaxProjection


class AttentionMechanism(nn.Module):
    def __init__(
        self, 
        score: str, 
        dim: int,
        beta: int,
        dropout: float=None,
    ):
        super().__init__()
        self.broadcast = BroadCast()
        self.score = build_score(
            name=score,
            dim=dim,
            dropout=dropout,
        )
        self.simplex = SoftmaxProjection(
            beta=beta,
        )

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor=None,
    ) -> torch.Tensor:
        """
        Dimension:
        ---
        - q: (B,D) or (B,Nq,D)
        - k: (Nk,D) or (B,Nk,D)
        - v: (Nk,D) or (B,Nk,D)
        - mask: (Nk,) or (B,Nk) or (B,Nq,Nk)
        """
        
        # BROADCASTING ==========
        # q: -> (B,1,D) or (B,Nq,D)
        # k: -> (1,Nk,D) or (B,Nk,D)
        # v: -> (1,Nk,D) or (B,Nk,D)
        # mask: -> (1,1,Nk) or (B,1,Nk) or (B,Nq,Nk)
        q, k, v, mask = self.broadcast(q=q, k=k, v=v, mask=mask)

        # ATTENTION SCORES ==========
        # (B,Nq,Nk)
        scores = self.score(q=q, k=k)

        # MASKING ==========
        # (B,Nq,Nk)
        if mask is not None:
            scores = scores.masked_fill(mask=mask, value=float("-inf"))
        
        # WEIGHTS ==========
        # (B,Nq,Nk)
        weights = self.simplex(scores)

        # WEIGHTED SUM ==========
        # (B,Nq,Nk) x (B,Nk,D) -> (B,Nq,D)
        # (B,1,Nk) x (B,Nk,D) -> (B,1,D) -> (B,D)
        return (weights @ v).squeeze(1)