"""
CLEAR: Contrastive Learning for Enhanced scRNA-seq Representation
MoCo-based contrastive learning for single-cell embeddings
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, Callable
import warnings 
from .base_model import BaseModel


class TwoViewAugmenter(nn.Module):
    """Generate two augmented views via feature masking, Gaussian noise, and scale jitter"""
    def __init__(
        self,
        feature_mask_prob: float = 0.2,
        gaussian_noise_std: float = 0.1,
        scale_jitter: float = 0.0,
    ):
        super().__init__()
        self.feature_mask_prob = float(feature_mask_prob)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.scale_jitter = float(scale_jitter)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        v = x

        if self.feature_mask_prob > 0:
            mask = (torch.rand_like(v) < self.feature_mask_prob)
            v = v.masked_fill(mask, 0.0)

        if self.gaussian_noise_std > 0:
            v = v + torch.randn_like(v) * self.gaussian_noise_std

        if self.scale_jitter > 0:
            s = (1.0 + (torch.rand(v.size(0), 1, device=v.device) * 2 - 1.0) * self.scale_jitter)
            v = v * s

        return v

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._augment(x), self._augment(x)


class CLEARMLPEncoder(nn.Module):
    """Simple MLP encoder: input -> hidden (ReLU) -> output"""
    def __init__(self, input_dim: int, hidden_dim: int = 1024, output_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoCoHead(nn.Module):
    """
    Momentum Contrast head: query encoder + momentum key encoder + queue
    
    Note: No reconstruction; learns embeddings via InfoNCE loss
    """
    def __init__(
        self,
        input_dim: int,
        dim: int = 128,
        hidden_dim: int = 1024,
        queue_size: int = 1024,
        momentum: float = 0.999,
        temperature: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        self.encoder_q = CLEARMLPEncoder(input_dim, hidden_dim=hidden_dim, output_dim=dim, dropout=dropout)
        self.encoder_k = CLEARMLPEncoder(input_dim, hidden_dim=hidden_dim, output_dim=dim, dropout=dropout)

        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        if self.queue_size % batch_size != 0:
            return

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, labels) for InfoNCE loss"""
        q = F.normalize(self.encoder_q(view1), dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(view2), dim=1)

        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k)
        return logits, labels

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Inference using momentum encoder"""
        self.encoder_k.eval()
        return F.normalize(self.encoder_k(x), dim=1)


class CLEARModel(BaseModel):
    """
    CLEAR: MoCo-based contrastive learning for scRNA-seq
    
    Supports batch formats:
    - Env loader: (x_norm, x_raw) -> views generated from x_norm
    - Contrastive loader: (view1, view2) -> used directly
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: Optional[list] = None,
        hidden_dim: int = 1024,
        queue_size: int = 1024,
        momentum: float = 0.999,
        temperature: float = 0.2,
        dropout: float = 0.0,
        model_name: str = "CLEAR",
        view_augmenter: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        feature_mask_prob: float = 0.2,
        gaussian_noise_std: float = 0.1,
        scale_jitter: float = 0.0,
    ):
        """
        Args:
            input_dim: Number of genes
            latent_dim: Embedding dimension
            hidden_dim: MLP hidden dimension
            queue_size: Negative sample queue size
            momentum: Momentum for key encoder update
            temperature: Temperature for InfoNCE loss
            dropout: Dropout rate
            view_augmenter: Custom augmenter, defaults to TwoViewAugmenter
            feature_mask_prob: Feature masking probability
            gaussian_noise_std: Gaussian noise std
            scale_jitter: Scale jitter range
        """
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims or [], model_name=model_name)
        
        self.moco = MoCoHead(
            input_dim=input_dim,
            dim=latent_dim,
            hidden_dim=hidden_dim,
            queue_size=queue_size,
            momentum=momentum,
            temperature=temperature,
            dropout=dropout,
        )
        self.criterion = nn.CrossEntropyLoss()

        if view_augmenter is None:
            self.view_augmenter = TwoViewAugmenter(
                feature_mask_prob=feature_mask_prob,
                gaussian_noise_std=gaussian_noise_std,
                scale_jitter=scale_jitter,
            )
        else:
            self.view_augmenter = view_augmenter

    def _prepare_batch(self, batch_data: Any, device: str):
        """Handle (x_norm, x_raw) or explicit (view1, view2)"""
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
            x = batch_data[0].to(device).float()

            if len(batch_data) >= 2 and torch.is_tensor(batch_data[1]):
                b1 = batch_data[1].to(device).float()
                if b1.shape == x.shape and b1 is not batch_data[0]:
                    # Heuristic: treat as x_raw if non-negative with large values
                    if torch.all(b1 >= 0) and (b1.max() > 5.0):
                        v1, v2 = self.view_augmenter(x)
                        return v1, {"view2": v2}
                    return x, {"view2": b1}

            v1, v2 = self.view_augmenter(x)
            return v1, {"view2": v2}

        raise ValueError("CLEARModel expects tensor batch or (x_norm, x_raw)/(view1, view2)")

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to normalized embedding using momentum encoder"""
        return self.moco.infer(x)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Not implemented: contrastive model has no decoder"""
        raise NotImplementedError("CLEARModel is contrastive-only; no decode()")

    def forward(self, x: torch.Tensor, view2: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass: compute logits and labels for InfoNCE"""
        logits, labels = self.moco(x, view2)
        return {"logits": logits, "labels": labels}

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE loss"""
        loss = self.criterion(outputs["logits"], outputs["labels"])
        return {"total_loss": loss, "recon_loss": loss}

    @torch.no_grad()
    def extract_latent(self, data_loader, device: str = "cuda", return_reconstructions: bool = False):
        """Extract embeddings (ignores return_reconstructions)"""
        if return_reconstructions:
            warnings.warn("CLEARModel has no decoder; return_reconstructions ignored")
        
        self.eval()
        self.to(device)

        zs = []
        for batch in data_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            z = self.encode(x)
            zs.append(z.detach().cpu().numpy())
        
        return {"latent": np.concatenate(zs, axis=0)}


def create_clear_model(input_dim: int, latent_dim: int = 128, **kwargs) -> CLEARModel:
    """
    Create CLEAR model
    
    Examples:
        >>> model = create_clear_model(2000, latent_dim=128, queue_size=2048)
    """
    return CLEARModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)