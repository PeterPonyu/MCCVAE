"""
scDeepCluster: Autoencoder with ZINB reconstruction and DEC-style clustering
Multi-stage training: AE pretrain → KMeans init → joint clustering refinement
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from sklearn.cluster import KMeans
from scipy.sparse import issparse
from .base_model import BaseModel


class ZINBLoss(nn.Module):
    """Zero-Inflated Negative Binomial loss"""
    def forward(self, x, mean, disp, pi, scale_factor=None):
        eps = 1e-8
        if scale_factor is not None:
            if scale_factor.dim() == 1:
                scale_factor = scale_factor.unsqueeze(-1)
            mean = mean * scale_factor

        mean = mean.clamp(min=eps, max=1e6)
        disp = disp.clamp(min=eps, max=1e6)
        pi = pi.clamp(min=eps, max=1 - eps)

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log1p(mean / (disp + eps)) + x * (torch.log(disp + eps) - torch.log(mean + eps))
        nb = t1 + t2

        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
        non_zero_case = -torch.log(1.0 - pi + eps) + nb
        out = torch.where(x < eps, zero_case, non_zero_case)
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out.mean()


class scDeepClusterAE(nn.Module):
    """Autoencoder with ZINB decoder (pi, disp, mean)"""
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Optional[list] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 64]
        dims = [input_dim] + hidden_dims + [latent_dim]

        self.encoder = nn.Sequential(
            nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dims[1], dims[2]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dims[2], dims[3]),
        )

        self.pi = nn.Sequential(
            nn.Linear(latent_dim, dims[2]), nn.ReLU(),
            nn.Linear(dims[2], dims[1]), nn.ReLU(),
            nn.Linear(dims[1], dims[0]), nn.Sigmoid(),
        )
        self.disp = nn.Sequential(
            nn.Linear(latent_dim, dims[2]), nn.ReLU(),
            nn.Linear(dims[2], dims[1]), nn.ReLU(),
            nn.Linear(dims[1], dims[0]), nn.Softplus(),
        )
        self.mean = nn.Sequential(
            nn.Linear(latent_dim, dims[2]), nn.ReLU(),
            nn.Linear(dims[2], dims[1]), nn.ReLU(),
            nn.Linear(dims[1], dims[0]), nn.Softplus(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.pi(z), self.disp(z), self.mean(z)


class ClusteringLayer(nn.Module):
    """DEC-style clustering layer with Student's t-distribution"""
    def __init__(self, n_clusters: int, n_features: int, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.empty(n_clusters, n_features))
        nn.init.xavier_uniform_(self.clusters)

    def forward(self, x):
        q = 1.0 / (1.0 + (torch.sum((x.unsqueeze(1) - self.clusters) ** 2, dim=2) / self.alpha))
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q


class scDeepClusterModel(BaseModel):
    """
    scDeepCluster: ZINB autoencoder + DEC clustering
    
    Training phases:
    1. Pretrain: AE with ZINB loss only
    2. Init: KMeans on latent space
    3. Finetune: Joint optimization (ZINB + clustering KL)
    
    Batch format: (x, raw_x, size_factors) or (x, size_factors)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        n_clusters: int = 10,
        hidden_dims: Optional[list] = None,
        alpha: float = 1.0,
        model_name: str = "scDeepCluster",
    ):
        """
        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            n_clusters: Number of clusters
            hidden_dims: Encoder/decoder hidden dimensions
            alpha: Student's t-distribution parameter
        """
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims or [], model_name=model_name)
        self.n_clusters = n_clusters
        self.alpha = alpha

        self.ae = scDeepClusterAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
        self.cluster = ClusteringLayer(n_clusters=n_clusters, n_features=latent_dim, alpha=alpha)
        self.zinb = ZINBLoss()
        self.is_trained = False

    def _prepare_batch(self, batch_data: Any, device: str):
        """Extract x, raw counts, and size factors from batch"""
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) == 3:
                x, raw, sf = batch_data
                return x.to(device).float(), {"raw": raw.to(device).float(), "sf": sf.to(device).float()}

            if len(batch_data) == 2:
                x0, x1 = batch_data
                x0 = x0.to(device).float()
                x1 = x1.to(device)

                if torch.is_tensor(x1) and torch.is_floating_point(x1) and x1.shape == x0.shape:
                    raw = x1.float()
                    sf = raw.sum(dim=1).clamp_min(1.0)
                    sf = sf / sf.mean().clamp_min(1e-6)
                    return x0, {"raw": raw, "sf": sf}

                sf = x1.float().to(device)
                return x0, {"raw": x0, "sf": sf}

        x = batch_data.to(device).float()
        return x, {"raw": x, "sf": None}

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to latent space"""
        z, _, _, _ = self.ae(x)
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode to mean expression"""
        return self.ae.mean(z)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass returning latent, ZINB params, and cluster assignments"""
        z, pi, disp, mean = self.ae(x)
        q = self.cluster(z)
        return {"latent": z, "pi": pi, "disp": disp, "mean": mean, "q": q}

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], raw=None, sf=None, p_target=None, **kwargs):
        """Compute loss: ZINB reconstruction + clustering KL (if p_target provided)"""
        raw = x if raw is None else raw
        zinb_loss = self.zinb(raw, outputs["mean"], outputs["disp"], outputs["pi"], sf)

        kl = torch.tensor(0.0, device=x.device)
        if p_target is not None:
            kl = F.kl_div(outputs["q"].log(), p_target, reduction="batchmean")

        total = zinb_loss + kl
        return {"total_loss": total, "recon_loss": zinb_loss, "cluster_kl": kl}

    @staticmethod
    def _target_distribution(q: torch.Tensor) -> torch.Tensor:
        """Compute target distribution p from soft assignments q"""
        weight = q ** 2 / q.sum(0).clamp_min(1e-12)
        return (weight.t() / weight.sum(1).clamp_min(1e-12)).t()

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 500,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 10,
        verbose: int = 1,
        pretrain_epochs: int = 200,
        tol: float = 1e-3,
        **kwargs,
    ):
        """
        Three-stage training:
        1. Pretrain AE with ZINB loss
        2. Initialize clusters with KMeans
        3. Finetune with clustering KL
        """
        self.to(device)
        opt = torch.optim.Adam(list(self.ae.parameters()), lr=lr)

        # Stage 1: Pretrain AE
        for ep in range(pretrain_epochs):
            self.train()
            tot = 0.0
            n = 0
            for batch in train_loader:
                x, bk = self._prepare_batch(batch, device)
                out = self.forward(x, **bk)
                loss = self.compute_loss(x, out, **bk)["recon_loss"]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ae.parameters(), max_norm=1.0)
                opt.step()
                tot += float(loss.item())
                n += 1
            if verbose >= 1 and (ep + 1) % 50 == 0:
                print(f"Pretrain {ep+1:3d}/{pretrain_epochs} | Loss: {tot/max(n,1):.4f}")

        # Stage 2: Initialize clusters with KMeans
        self.eval()
        z_all = []
        with torch.no_grad():
            for batch in train_loader:
                x, bk = self._prepare_batch(batch, device)
                z_all.append(self.encode(x).cpu().numpy())
        z_all = np.concatenate(z_all, axis=0)
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42).fit(z_all)
        self.cluster.clusters.data = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)

        # Stage 3: Finetune with clustering
        opt = torch.optim.Adam(list(self.ae.parameters()) + list(self.cluster.parameters()), lr=lr)
        y_last = km.labels_

        finetune_epochs = max(0, epochs - pretrain_epochs)
        for ep in range(finetune_epochs):
            self.train()

            q_full = []
            with torch.no_grad():
                for batch in train_loader:
                    x, bk = self._prepare_batch(batch, device)
                    q_full.append(self.forward(x, **bk)["q"])
            q_full = torch.cat(q_full, dim=0)
            p_full = self._target_distribution(q_full).detach()

            tot = 0.0
            n = 0
            offset = 0
            for batch in train_loader:
                x, bk = self._prepare_batch(batch, device)
                bsz = x.size(0)
                p_batch = p_full[offset:offset + bsz]
                offset += bsz

                out = self.forward(x, **bk)
                loss_dict = self.compute_loss(x, out, **bk, p_target=p_batch)
                loss = loss_dict["total_loss"]

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                opt.step()

                tot += float(loss.item())
                n += 1

            if (ep + 1) % 10 == 0:
                self.eval()
                y_pred = []
                with torch.no_grad():
                    for batch in train_loader:
                        x, bk = self._prepare_batch(batch, device)
                        y_pred.append(self.forward(x, **bk)["q"].argmax(1).cpu().numpy())
                y_pred = np.concatenate(y_pred, axis=0)
                delta = float(np.mean(y_pred != y_last))
                y_last = y_pred
                if verbose >= 1:
                    print(f"Finetune {ep+1:3d}/{finetune_epochs} | Loss: {tot/max(n,1):.4f} | Delta: {delta:.4f}")
                if delta < tol:
                    break

        self.is_trained = True
        return {"train_loss": []}


def create_scdeepcluster_model(input_dim: int, latent_dim: int = 32, n_clusters: int = 10, **kwargs) -> scDeepClusterModel:
    """
    Create scDeepCluster model
    
    Example:
        >>> model = create_scdeepcluster_model(2000, latent_dim=32, n_clusters=10)
    """
    return scDeepClusterModel(input_dim=input_dim, latent_dim=latent_dim, n_clusters=n_clusters, **kwargs)