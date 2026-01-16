"""
scSimGCL: Simple Graph Contrastive Learning for single-cell data
Features adaptive graph construction and noise-based augmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import warnings
from typing import Dict, Optional, Any, Tuple
from .base_model import BaseModel

try:
    from torch_geometric.nn.conv import GCNConv
except ImportError:
    GCNConv = None


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Create N identical copies of a module"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GraphConstructor(nn.Module):
    """Adaptive graph constructor with attention-based mechanism"""
    def __init__(self, input_dim: int, n_heads: int, phi: float, dropout: float = 0.0):
        super().__init__()
        assert input_dim % n_heads == 0
        self.d_k = input_dim // n_heads
        self.n_heads = n_heads
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.n_heads), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(n_heads, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Construct adjacency matrix from query and key"""
        query, key = [
            l(x).view(query.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        attns = self._attention(query.squeeze(2), key.squeeze(2))
        adj = torch.where(
            attns >= self.phi,
            torch.ones_like(attns),
            torch.zeros_like(attns),
        )
        return adj

    def _attention(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute attention scores"""
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


def data_augmentation(
    x: torch.Tensor,
    adj: torch.Tensor,
    prob_feature: float,
    prob_edge: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply feature masking and edge dropout augmentation"""
    device = x.device
    batch_size, input_dim = x.shape

    # Feature masking
    tensor_p = torch.ones((batch_size, input_dim), device=device) * (1 - prob_feature)
    mask_feature = torch.bernoulli(tensor_p)

    # Edge dropout
    tensor_p = torch.ones((batch_size, batch_size), device=device) * (1 - prob_edge)
    mask_edge = torch.bernoulli(tensor_p)

    return mask_feature * x, mask_edge * adj


def sim(z1: torch.Tensor, z2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Compute similarity between two embeddings"""
    if normalize:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
    return torch.mm(z1, z2.T)


def contrastive_loss(
    z: torch.Tensor,
    z_aug: torch.Tensor,
    adj: torch.Tensor,
    tau: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute contrastive loss with neighbor-aware positives"""
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, normalize))
    inter_view_sim = f(sim(z, z_aug, normalize))

    positive = (
        inter_view_sim.diag()
        + (intra_view_sim * adj).sum(1)
        + (inter_view_sim * adj).sum(1)
    )

    denominator = intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag()
    loss = positive / (denominator + 1e-8)

    adj_count = torch.sum(adj, 1) * 2 + 1
    loss = torch.log(loss + 1e-8) / adj_count

    return -torch.mean(loss, 0)


def final_contrastive_loss(
    alpha1: float,
    alpha2: float,
    z: torch.Tensor,
    z_aug: torch.Tensor,
    adj: torch.Tensor,
    adj_aug: torch.Tensor,
    tau: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Bidirectional contrastive loss"""
    loss = alpha1 * contrastive_loss(z, z_aug, adj, tau, normalize)
    loss += alpha2 * contrastive_loss(z_aug, z, adj_aug, tau, normalize)
    return loss


class scSimGCLCore(nn.Module):
    """Core scSimGCL model"""
    def __init__(
        self,
        input_dim: int,
        gcn_dim: int,
        mlp_dim: int,
        graph_heads: int,
        phi: float,
        prob_feature: float,
        prob_edge: float,
        tau: float,
        alpha: float,
        beta: float,
        dropout: float,
    ):
        super().__init__()
        self.prob_feature = prob_feature
        self.prob_edge = prob_edge
        self.tau = tau
        self.alpha = alpha
        self.beta = beta

        self.graph_constructor = GraphConstructor(input_dim, graph_heads, phi, dropout=0)

        if GCNConv is None:
            # Fallback simple GCN
            self.gcn = nn.Sequential(
                nn.Linear(input_dim, gcn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self._use_pyg = False
        else:
            self.gcn = GCNConv(input_dim, gcn_dim)
            self._use_pyg = True

        self.w_imp = nn.Linear(gcn_dim, input_dim)
        self.mlp = nn.Linear(gcn_dim, mlp_dim)
        self.dropout = nn.Dropout(p=dropout)

    def _gcn_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GCN forward with PyG or fallback"""
        if self._use_pyg:
            return self.gcn(x, edge_index)
        else:
            return self.gcn(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        x = self.dropout(x)

        # Construct graph
        adj = self.graph_constructor(x, x)
        adj = adj - torch.diag_embed(adj.diag())
        edge_index = torch.nonzero(adj == 1).T

        # Augmentation
        x_aug, adj_aug = data_augmentation(x, adj, self.prob_feature, self.prob_edge)
        edge_index_aug = torch.nonzero(adj_aug == 1).T

        # GCN encoding
        z = self._gcn_forward(x, edge_index)
        z_aug = self._gcn_forward(x_aug, edge_index_aug)

        # Imputation/reconstruction
        x_imp = self.w_imp(z)

        # MLP projection for contrastive learning
        z_mlp = self.mlp(z)
        z_mlp_aug = self.mlp(z_aug)

        # Contrastive loss
        loss_cl = final_contrastive_loss(
            self.alpha, self.beta, z_mlp, z_mlp_aug, adj, adj_aug, self.tau, normalize=True
        )

        return z, x_imp, loss_cl

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without augmentation"""
        adj = self.graph_constructor(x, x)
        adj = adj - torch.diag_embed(adj.diag())
        edge_index = torch.nonzero(adj == 1).T
        return self._gcn_forward(x, edge_index)


class scSimGCLModel(BaseModel):
    """
    scSimGCL: Simple Graph Contrastive Learning for single-cell data
    
    Features:
    - Adaptive graph construction via attention mechanism
    - Feature masking and edge dropout augmentation
    - GCN encoder with contrastive learning
    - MAE reconstruction loss for imputation
    - Simple and efficient architecture
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 277,
        mlp_dim: int = 118,
        graph_heads: int = 5,
        phi: float = 0.45,
        prob_feature: float = 0.1,
        prob_edge: float = 0.5,
        tau: float = 0.8,
        alpha: float = 0.55,
        beta: float = 0.4,
        lambda_cl: float = 0.887,
        dropout: float = 0.4,
        model_name: str = "scSimGCL",
    ):
        """
        Args:
            input_dim: Gene dimension
            latent_dim: GCN hidden dimension
            mlp_dim: MLP projection dimension
            graph_heads: Number of attention heads for graph construction
            phi: Threshold for edge creation
            prob_feature: Feature masking probability
            prob_edge: Edge dropout probability
            tau: Temperature for contrastive loss
            alpha: Weight for z→z_aug contrastive loss
            beta: Weight for z_aug→z contrastive loss
            lambda_cl: Weight for contrastive loss vs reconstruction
            dropout: Dropout rate
        """
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[mlp_dim],
            model_name=model_name,
        )

        self.mlp_dim = mlp_dim
        self.lambda_cl = lambda_cl

        self.model = scSimGCLCore(
            input_dim=input_dim,
            gcn_dim=latent_dim,
            mlp_dim=mlp_dim,
            graph_heads=graph_heads,
            phi=phi,
            prob_feature=prob_feature,
            prob_edge=prob_edge,
            tau=tau,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
        )

    def _prepare_batch(self, batch_data: Any, device: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            y = batch_data[1] if len(batch_data) > 1 else None
            return x, {"y": y}
        return batch_data.to(device).float(), {}

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to latent space"""
        return self.model.encode(x)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latent to reconstruction"""
        return self.model.w_imp(z)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        z, x_imp, loss_cl = self.model(x)
        return {
            "latent": z,
            "reconstruction": x_imp,
            "loss_cl": loss_cl,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss: MAE reconstruction + contrastive"""
        x_imp = outputs["reconstruction"]
        loss_cl = outputs["loss_cl"]

        # Masked MAE loss (only on non-zero entries)
        mask = (x != 0).float()
        loss_mae = F.l1_loss(mask * x_imp, mask * x)

        total_loss = loss_mae + self.lambda_cl * loss_cl

        return {
            "total_loss": total_loss,
            "recon_loss": loss_mae,
            "cl_loss": loss_cl,
        }

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 200,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 25,
        verbose: int = 1,
        **kwargs,
    ) -> Dict[str, list]:
        """
        Train scSimGCL model
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = {
            "train_loss": [],
            "val_loss": [],
            "recon_loss": [],
            "cl_loss": [],
        }

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.train()
            train_losses = []
            recon_losses = []
            cl_losses = []

            for batch_data in train_loader:
                x, _ = self._prepare_batch(batch_data, device)

                optimizer.zero_grad()
                outputs = self.forward(x)
                losses = self.compute_loss(x, outputs)

                loss = losses["total_loss"]
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                recon_losses.append(losses["recon_loss"].item())
                cl_losses.append(losses["cl_loss"].item())

            avg_train = np.mean(train_losses)
            history["train_loss"].append(avg_train)
            history["recon_loss"].append(np.mean(recon_losses))
            history["cl_loss"].append(np.mean(cl_losses))

            # Validation
            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_data in val_loader:
                        x, _ = self._prepare_batch(batch_data, device)
                        outputs = self.forward(x)
                        losses = self.compute_loss(x, outputs)
                        val_losses.append(losses["total_loss"].item())

                avg_val = np.mean(val_losses)
                history["val_loss"].append(avg_val)
                current_loss = avg_val

                # Early stopping only when validation is available
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"\n✓ Early stopping at epoch {epoch+1}")
                    break
            else:
                # No early stopping when val_loader is None, just save best train model
                if save_path and avg_train < best_loss:
                    best_loss = avg_train
                    self.save_model(save_path)

            if verbose >= 1 and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train:.4f}"
                if val_loader is not None:
                    msg += f" | Val Loss: {avg_val:.4f}"
                msg += f" | Recon: {np.mean(recon_losses):.4f} | CL: {np.mean(cl_losses):.4f}"
                print(msg)

        if verbose >= 1:
            print("\n✓ Training finished!")
        return history

    @torch.no_grad()
    def extract_latent(
        self,
        data_loader,
        device: str = "cuda",
        return_reconstructions: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Extract latent representations"""
        self.eval()
        self.to(device)

        latents = []
        reconstructions = [] if return_reconstructions else None

        for batch_data in data_loader:
            x, _ = self._prepare_batch(batch_data, device)
            z = self.encode(x)
            latents.append(z.cpu().numpy())

            if return_reconstructions:
                recon = self.decode(z)
                reconstructions.append(recon.cpu().numpy())

        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(reconstructions, axis=0)
        return result


def create_scsimgcl_model(input_dim: int, latent_dim: int = 277, **kwargs) -> scSimGCLModel:
    """
    Create scSimGCL model
    
    Example:
        >>> model = create_scsimgcl_model(2000, latent_dim=277, phi=0.45)
    """
    return scSimGCLModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
