"""
scGNN: Graph VAE for single-cell data with kNN graph construction
Reconstructs cell-cell adjacency via inner product decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import warnings
import numpy as np
from .base_model import BaseModel


def build_knn_graph(x: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Build symmetric kNN adjacency from cosine similarity"""
    with torch.no_grad():
        x_norm = F.normalize(x, p=2, dim=1)
        sim = torch.matmul(x_norm, x_norm.t())
        _, idx = torch.topk(sim, k=k + 1, dim=-1)
        idx = idx[:, 1:]  # remove self

        n = x.size(0)
        adj = torch.zeros((n, n), device=x.device)
        adj.scatter_(1, idx, 1.0)
        adj = torch.maximum(adj, adj.t())
        adj.fill_diagonal_(1.0)
    return adj


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}"""
    deg = adj.sum(1)
    deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
    D = torch.diag(deg_inv_sqrt)
    return D @ adj @ D


class GraphConvolution(nn.Module):
    """Single graph convolution layer"""
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.dropout = dropout
        self.act = act
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        support = x @ self.weight
        out = adj @ support
        return self.act(out)


class InnerProductDecoder(nn.Module):
    """Decode adjacency via z @ z.T"""
    def forward(self, z):
        return torch.sigmoid(z @ z.t())


class GCNVAE(nn.Module):
    """GCN-based VAE core"""
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim, dropout, act=F.relu)
        self.gc_mu = GraphConvolution(hidden_dim, latent_dim, dropout, act=lambda x: x)
        self.gc_logvar = GraphConvolution(hidden_dim, latent_dim, dropout, act=lambda x: x)
        self.decoder = InnerProductDecoder()

    def encode(self, x, adj):
        h = self.gc1(x, adj)
        return self.gc_mu(h, adj), self.gc_logvar(h, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_recon = self.decoder(z)
        return adj_recon, mu, logvar, z


class scGNNModel(BaseModel):
    """
    scGNN: Graph VAE for single-cell data
    
    Features:
    - Automatic kNN graph construction from features
    - Graph convolution encoder
    - Inner product decoder for adjacency reconstruction
    - VAE loss: adjacency BCE + KL divergence
    
    Note: decode() returns adjacency matrix, not gene expression
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        k_neighbors: int = 10,
        model_name: str = "scGNN",
    ):
        """
        Args:
            input_dim: Gene dimension
            latent_dim: Latent dimension
            hidden_dim: GCN hidden dimension
            dropout: Dropout rate
            k_neighbors: Number of neighbors for kNN graph
        """
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[hidden_dim],
            model_name=model_name,
        )

        self.k_neighbors = k_neighbors
        self.vae = GCNVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    def _prepare_batch(self, batch_data: Any, device: str):
        x_norm, _ = batch_data
        return x_norm.to(device), {}

    def encode(self, x: torch.Tensor, **_) -> torch.Tensor:
        """Encode to latent space [B, latent_dim]"""
        adj = normalize_adj(build_knn_graph(x, self.k_neighbors))
        mu, _ = self.vae.encode(x, adj)
        return mu

    def decode(self, z: torch.Tensor, **_) -> torch.Tensor:
        """Decode latent to adjacency matrix [B, B]"""
        return z @ z.t()

    def forward(self, x: torch.Tensor, **_) -> Dict[str, torch.Tensor]:
        """Forward pass with graph construction"""
        adj = normalize_adj(build_knn_graph(x, self.k_neighbors))
        adj_recon, mu, logvar, z = self.vae(x, adj)
        return {
            "adj_recon": adj_recon,
            "adj_true": adj,
            "mu": mu,
            "logvar": logvar,
            "latent": z,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        **_,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss: adjacency BCE + KL divergence"""

        adj_true = outputs["adj_true"]
        adj_pred = outputs["adj_recon"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        n = x.size(0)

        recon_loss = F.binary_cross_entropy(adj_pred, adj_true, reduction="mean")

        kl_loss = -0.5 / n * torch.mean(
            torch.sum(1 + 2 * logvar - mu.pow(2) - torch.exp(logvar).pow(2), dim=1)
        )

        total = recon_loss + kl_loss
        return {
            "total_loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
    
    @torch.no_grad()
    def extract_latent(self, data_loader, device: str = "cuda", return_reconstructions: bool = False):
        """Extract latent representations"""
        if return_reconstructions:
            warnings.warn("scGNN decode() returns adjacency, not gene reconstruction; ignoring return_reconstructions.")

        self.eval()
        self.to(device)

        latents = []
        for batch in data_loader:
            x, _ = self._prepare_batch(batch, device)
            z = self.encode(x)
            latents.append(z.detach().cpu().numpy())

        return {"latent": np.concatenate(latents, axis=0)}


def create_scgnn_model(input_dim: int, latent_dim: int = 16, **kwargs) -> scGNNModel:
    """
    Create scGNN model
    
    Example:
        >>> model = create_scgnn_model(2000, latent_dim=16, k_neighbors=10)
    """
    return scGNNModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)