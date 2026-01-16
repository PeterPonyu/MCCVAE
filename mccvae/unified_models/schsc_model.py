"""
scHSC: Hard Sample Contrastive Learning for single-cell data
Uses pseudo-labeling and hard sample mining for improved contrastive learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Dict, Optional, Any, Tuple
from .base_model import BaseModel

try:
    from sklearn.neighbors import kneighbors_graph
    from sklearn.decomposition import PCA
except ImportError:
    kneighbors_graph = None
    PCA = None

try:
    import scanpy as sc
except ImportError:
    sc = None


class MeanAct(nn.Module):
    """Mean activation for ZINB"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """Dispersion activation for ZINB"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class ZINBLoss(nn.Module):
    """Zero-Inflated Negative Binomial Loss"""
    def __init__(self, pi: torch.Tensor, disp: torch.Tensor, scale_factor: torch.Tensor = None):
        super().__init__()
        self.pi = pi
        self.disp = disp
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if self.scale_factor is not None:
            scale_factor = self.scale_factor[:, None]
            mean = mean * scale_factor

        t1 = torch.lgamma(self.disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + self.disp + eps)
        t2 = (self.disp + x) * torch.log(1.0 + (mean / (self.disp + eps))) + (
            x * (torch.log(self.disp + eps) - torch.log(mean + eps))
        )
        nb_final = self._nan2inf(t1 + t2)
        nb_case = nb_final - torch.log(1.0 - self.pi + eps)

        zero_nb = torch.pow(self.disp / (self.disp + mean + eps), self.disp)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)

        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        return torch.mean(result)

    @staticmethod
    def _nan2inf(x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class HSCEncoder(nn.Module):
    """Dual-view encoder with attribute and structure branches"""
    def __init__(self, input_dim: int, dataset_size: int, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        # Attribute encoder (2 views)
        self.AE1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )
        self.AE2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )
        # Structure encoder (2 views)
        self.SE1 = nn.Sequential(
            nn.Linear(dataset_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )
        self.SE2 = nn.Sequential(
            nn.Linear(dataset_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )
        # ZINB decoder
        self.dec_mean = nn.Sequential(nn.Linear(hidden_dim, input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(hidden_dim, input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Z1 = F.normalize(self.AE1(x), dim=1, p=2)
        Z2 = F.normalize(self.AE2(x), dim=1, p=2)
        E1 = F.normalize(self.SE1(adj), dim=1, p=2)
        E2 = F.normalize(self.SE2(adj), dim=1, p=2)

        z_mean = (Z1 + Z2) / 2
        _mean = self.dec_mean(z_mean)
        _disp = self.dec_disp(z_mean)
        _pi = self.dec_pi(z_mean)

        return Z1, Z2, E1, E2, _mean, _disp, _pi

    def forward_full(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Z1 = F.normalize(self.AE1(x), dim=1, p=2)
        Z2 = F.normalize(self.AE2(x), dim=1, p=2)
        return Z1, Z2


def get_knn_adj(data: np.ndarray, k: int = 15, pca_dim: int = 50) -> np.ndarray:
    """Build kNN adjacency matrix"""
    if kneighbors_graph is None or PCA is None:
        raise ImportError("scikit-learn required for kNN graph construction")

    if pca_dim and pca_dim < data.shape[1]:
        pca = PCA(n_components=pca_dim)
        data = pca.fit_transform(data)

    adj = kneighbors_graph(data, k, mode="connectivity", metric="cosine", include_self=False)
    return adj.toarray()


def laplacian_filtering(adj: np.ndarray, x: np.ndarray, t: int = 1) -> np.ndarray:
    """Apply Laplacian smoothing filter"""
    # Normalized adjacency
    adj = adj + np.eye(adj.shape[0])
    deg = np.diag(adj.sum(axis=1))
    deg_inv_sqrt = np.diag(np.power(np.diag(deg) + 1e-8, -0.5))
    norm_adj = deg_inv_sqrt @ adj @ deg_inv_sqrt

    # Apply filter t times
    x_filtered = x.copy()
    for _ in range(t):
        x_filtered = norm_adj @ x_filtered

    return x_filtered


class scHSCModel(BaseModel):
    """
    scHSC: Hard Sample Contrastive Learning for single-cell data
    
    Features:
    - Dual-view encoding: attribute + structure encoders
    - Hard sample mining with pseudo-labeling
    - Adaptive weight adjustment based on clustering quality
    - Comprehensive similarity from both attribute and structure views
    - ZINB reconstruction loss
    - Laplacian filtering preprocessing
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        n_clusters: int = 8,
        dropout: float = 0.5,
        alpha: float = 0.5,
        beta: float = 1.0,
        tau: float = 0.9,
        model_name: str = "scHSC",
    ):
        """
        Args:
            input_dim: Gene dimension
            latent_dim: Latent/hidden dimension
            n_clusters: Target number of clusters
            dropout: Dropout rate
            alpha: Weight for attribute embedding in comprehensive similarity
            beta: Parameter for pseudo matrix shape
            tau: Threshold for high-confidence sample selection
        """
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[256],
            model_name=model_name,
        )

        self.n_clusters = n_clusters
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self._dataset_size = None
        self._encoder = None

    def _init_encoder(self, dataset_size: int):
        """Initialize encoder with dataset size"""
        self._dataset_size = dataset_size
        self._encoder = HSCEncoder(
            input_dim=self.input_dim,
            dataset_size=dataset_size,
            hidden_dim=self.latent_dim,
            dropout=self.dropout,
        )

    def _prepare_batch(self, batch_data: Any, device: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if isinstance(batch_data, dict):
            x = batch_data["laplacian_filtered"].to(device).float()
            adj = batch_data["adjacency_matrix"].to(device).float()
            raw = batch_data.get("raw_count")
            sf = batch_data.get("size_factor")
            return x, {
                "adj": adj,
                "raw": raw.to(device).float() if raw is not None else None,
                "size_factor": sf.to(device).float() if sf is not None else None,
            }
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            return x, {}
        return batch_data.to(device).float(), {}

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to latent space (average of two views)"""
        if self._encoder is None:
            raise RuntimeError("Encoder not initialized. Call fit() first or provide dataset_size.")
        Z1, Z2 = self._encoder.forward_full(x)
        return (Z1 + Z2) / 2

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode is handled internally via ZINB; not directly available"""
        raise NotImplementedError("scHSC uses ZINB decoder integrated in forward pass")

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        if self._encoder is None:
            raise RuntimeError("Encoder not initialized")
        if adj is None:
            raise ValueError("adj (adjacency matrix) required for forward pass")

        Z1, Z2, E1, E2, _mean, _disp, _pi = self._encoder(x, adj)
        z = (Z1 + Z2) / 2
        return {
            "latent": z,
            "Z1": Z1,
            "Z2": Z2,
            "E1": E1,
            "E2": E2,
            "mean": _mean,
            "disp": _disp,
            "pi": _pi,
        }

    def compute_loss(
        self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Use fit() for scHSC training with hard sample mining")

    def _comprehensive_similarity(
        self,
        Z1: torch.Tensor,
        Z2: torch.Tensor,
        E1: torch.Tensor,
        E2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate comprehensive similarity matrices"""
        alpha = self.alpha
        ZE_11 = alpha * Z1 @ Z1.T + (1 - alpha) * E1 @ E1.T
        ZE_12 = alpha * Z1 @ Z2.T + (1 - alpha) * E1 @ E2.T
        ZE_22 = alpha * Z2 @ Z2.T + (1 - alpha) * E2 @ E2.T
        return ZE_11, ZE_12, ZE_22

    def _hard_sample_infoNCE(
        self,
        ZE_11: torch.Tensor,
        ZE_12: torch.Tensor,
        ZE_22: torch.Tensor,
        pos_neg_weight_11: torch.Tensor,
        pos_neg_weight_12: torch.Tensor,
        pos_neg_weight_22: torch.Tensor,
        pos_weight: torch.Tensor,
        mask_11: torch.Tensor,
        mask_12: torch.Tensor,
    ) -> torch.Tensor:
        """Hard sample InfoNCE loss"""
        node_num = ZE_11.size(0)

        pos = torch.exp(torch.diag(ZE_12) * pos_weight)
        pos = torch.cat([pos, pos], dim=0)

        pos_neg_11 = mask_11 * torch.exp(ZE_11 * pos_neg_weight_11)
        pos_neg_12 = mask_12 * torch.exp(ZE_12 * pos_neg_weight_12)
        pos_neg_22 = mask_11 * torch.exp(ZE_22 * pos_neg_weight_22)

        neg = torch.cat(
            [
                torch.sum(pos_neg_11, dim=1) + torch.sum(pos_neg_12, dim=1) - pos[:node_num],
                torch.sum(pos_neg_22, dim=1) + torch.sum(pos_neg_12, dim=0) - pos[node_num:],
            ],
            dim=0,
        )

        infoNCE = (-torch.log(pos / (pos + neg + 1e-8))).sum() / (2 * node_num)
        return infoNCE

    def _square_euclid_distance(self, Z: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean distance"""
        ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
        CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
        ZC = Z @ center.T
        return ZZ + CC - 2 * ZC

    def _high_confidence(
        self, Z: torch.Tensor, center: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Select high confidence samples"""
        distance_norm = torch.min(
            F.softmax(self._square_euclid_distance(Z, center), dim=1), dim=1
        ).values
        value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - self.tau)))
        index = torch.where(
            distance_norm <= value[-1],
            torch.ones_like(distance_norm),
            torch.zeros_like(distance_norm),
        )
        H = torch.nonzero(index).reshape(-1)
        H_mat = np.ix_(H.cpu().numpy(), H.cpu().numpy())
        return H, H_mat

    def _pseudo_matrix(
        self,
        P: torch.Tensor,
        ZE_11: torch.Tensor,
        ZE_12: torch.Tensor,
        ZE_22: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pseudo matrix for weight updates"""
        device = P.device
        Q = (P == P.unsqueeze(1)).float().to(device)

        ZE_max = torch.cat([ZE_11, ZE_12, ZE_22]).max()
        ZE_min = torch.cat([ZE_11, ZE_12, ZE_22]).min()

        ZE_11_norm = (ZE_11 - ZE_min) / (ZE_max - ZE_min + 1e-8)
        ZE_12_norm = (ZE_12 - ZE_min) / (ZE_max - ZE_min + 1e-8)
        ZE_22_norm = (ZE_22 - ZE_min) / (ZE_max - ZE_min + 1e-8)

        M_mat_11 = torch.abs(Q - ZE_11_norm) ** self.beta
        M_mat_12 = torch.abs(Q - ZE_12_norm) ** self.beta
        M_mat_22 = torch.abs(Q - ZE_22_norm) ** self.beta

        M_1 = torch.diag(M_mat_12)
        return M_1, M_mat_11, M_mat_12, M_mat_22

    def _phi_clustering(
        self, feature: torch.Tensor, target_clusters: int, use_louvain: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster features and return labels and centers"""
        if sc is None:
            # Fallback to KMeans
            from sklearn.cluster import KMeans
            feature_np = feature.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=target_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(feature_np)
            centers = kmeans.cluster_centers_
            return labels.astype(np.float32), centers.astype(np.float32)

        feature_np = feature.detach().cpu().numpy()
        adata = sc.AnnData(feature_np)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")

        # Binary search for resolution
        min_res, max_res = 0.0, 5.0
        for _ in range(20):
            mid_res = (min_res + max_res) / 2
            if use_louvain:
                sc.tl.louvain(adata, resolution=mid_res)
                n_found = len(adata.obs["louvain"].unique())
            else:
                sc.tl.leiden(adata, resolution=mid_res)
                n_found = len(adata.obs["leiden"].unique())

            if n_found == target_clusters:
                break
            elif n_found < target_clusters:
                min_res = mid_res
            else:
                max_res = mid_res

        labels = adata.obs["louvain" if use_louvain else "leiden"].astype(int).values
        centers = np.vstack(
            [np.mean(feature_np[labels == c], axis=0) for c in np.unique(labels)]
        )
        return labels.astype(np.float32), centers.astype(np.float32)

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        lr: float = 1e-5,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 1,
        verbose: int = 1,
        use_louvain: bool = True,
        use_leiden: bool = False,
        wzinb: float = 0,
        **kwargs,
    ) -> Dict[str, list]:
        """
        Train scHSC model with hard sample contrastive learning
        
        Args:
            use_louvain: Use Louvain clustering
            use_leiden: Use Leiden clustering
            wzinb: Weight for ZINB loss (0 for auto)
        """
        # Get full data for preprocessing
        full_data = []
        for batch in train_loader:
            if isinstance(batch, dict):
                full_data.append(batch["laplacian_filtered"])
            elif isinstance(batch, (list, tuple)):
                full_data.append(batch[0])
            else:
                full_data.append(batch)
        full_x = torch.cat(full_data, dim=0).numpy()
        dataset_size = full_x.shape[0]

        # Build adjacency matrix
        adj_matrix = get_knn_adj(full_x, k=18)
        # Apply Laplacian filtering
        x_filtered = laplacian_filtering(adj_matrix, full_x, t=1)

        # Initialize encoder
        self._init_encoder(dataset_size)
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Convert to tensors
        x_tensor = torch.tensor(x_filtered, dtype=torch.float32, device=device)
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

        # Initialize weights
        batch_size = min(512, dataset_size)
        pos_weight = torch.ones(batch_size, device=device)
        pos_neg_weight_11 = torch.ones((batch_size, batch_size), device=device)
        pos_neg_weight_12 = torch.ones((batch_size, batch_size), device=device)
        pos_neg_weight_22 = torch.ones((batch_size, batch_size), device=device)

        mask_11 = (torch.ones((batch_size, batch_size)) - torch.eye(batch_size)).to(device)
        mask_12 = torch.ones((batch_size, batch_size), device=device)

        history = {"train_loss": []}
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.train()

            # Sample batch
            perm = torch.randperm(dataset_size)[:batch_size]
            x_batch = x_tensor[perm]
            # Use full adjacency rows for structure encoder (not subgraph)
            adj_batch = adj_tensor[perm]  # [batch_size, dataset_size]

            # Forward
            Z1, Z2, E1, E2, _mean, _disp, _pi = self._encoder(x_batch, adj_batch)

            # Comprehensive similarity
            ZE_11, ZE_12, ZE_22 = self._comprehensive_similarity(Z1, Z2, E1, E2)

            # Contrastive loss
            contrastive_loss = self._hard_sample_infoNCE(
                ZE_11, ZE_12, ZE_22,
                pos_neg_weight_11, pos_neg_weight_12, pos_neg_weight_22,
                pos_weight, mask_11, mask_12
            )

            # ZINB loss (using filtered data as target)
            zinb_loss_fn = ZINBLoss(pi=_pi, disp=_disp)
            zinb_loss = zinb_loss_fn(x_batch, _mean)

            # Auto weight
            if wzinb > 0:
                w_zinb = wzinb
            else:
                w_zinb = (contrastive_loss / (zinb_loss + 1e-8)).detach()

            loss = contrastive_loss + w_zinb * zinb_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            history["train_loss"].append(loss_val)

            # Update weights periodically
            if (epoch + 1) % 3 == 0:
                self.eval()
                with torch.no_grad():
                    # Recompute with same batch (adj_batch already has full rows)
                    Z1, Z2, E1, E2, _, _, _ = self._encoder(x_batch, adj_batch)
                    ZE_11, ZE_12, ZE_22 = self._comprehensive_similarity(Z1, Z2, E1, E2)
                    Z = (Z1 + Z2) / 2

                    # Clustering
                    P, center = self._phi_clustering(Z, self.n_clusters, use_louvain=use_louvain)
                    P = torch.tensor(P, device=device)
                    center = torch.tensor(center, device=device)

                    # High confidence samples
                    H, H_mat = self._high_confidence(Z, center)

                    # Pseudo matrix
                    M_1, M_mat_11, M_mat_12, M_mat_22 = self._pseudo_matrix(P, ZE_11, ZE_12, ZE_22)

                    # Update weights
                    pos_weight[H] = M_1[H].data
                    pos_neg_weight_11[H_mat] = M_mat_11[H_mat].data
                    pos_neg_weight_12[H_mat] = M_mat_12[H_mat].data
                    pos_neg_weight_22[H_mat] = M_mat_22[H_mat].data

            # Compute validation loss if val_loader provided
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    # Sample validation batch
                    val_perm = torch.randperm(dataset_size)[:batch_size]
                    val_x_batch = x_tensor[val_perm]
                    val_adj_batch = adj_tensor[val_perm]
                    vZ1, vZ2, vE1, vE2, v_mean, v_disp, v_pi = self._encoder(val_x_batch, val_adj_batch)
                    vZE_11, vZE_12, vZE_22 = self._comprehensive_similarity(vZ1, vZ2, vE1, vE2)
                    val_contrastive = self._hard_sample_infoNCE(
                        vZE_11, vZE_12, vZE_22,
                        pos_neg_weight_11, pos_neg_weight_12, pos_neg_weight_22,
                        pos_weight, mask_11, mask_12
                    )
                    val_zinb_fn = ZINBLoss(pi=v_pi, disp=v_disp)
                    val_zinb = val_zinb_fn(val_x_batch, v_mean)
                    val_loss = float((val_contrastive + w_zinb * val_zinb).item())
                current_loss = val_loss

                # Early stopping check only when validation is available
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
                if save_path and loss_val < best_loss:
                    best_loss = loss_val
                    self.save_model(save_path)

            if verbose >= 1 and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss_val:.6f}"
                if val_loader is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
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
        if return_reconstructions:
            warnings.warn("scHSC reconstruction via ZINB; returning mean as reconstruction")

        self.eval()
        self.to(device)

        # Get full data
        full_data = []
        for batch in data_loader:
            if isinstance(batch, dict):
                full_data.append(batch["laplacian_filtered"])
            elif isinstance(batch, (list, tuple)):
                full_data.append(batch[0])
            else:
                full_data.append(batch)
        full_x = torch.cat(full_data, dim=0)

        # Compute latent
        Z1, Z2 = self._encoder.forward_full(full_x.to(device).float())
        z = ((Z1 + Z2) / 2).cpu().numpy()

        return {"latent": z}


def create_schsc_model(input_dim: int, latent_dim: int = 32, **kwargs) -> scHSCModel:
    """
    Create scHSC model
    
    Example:
        >>> model = create_schsc_model(2000, latent_dim=32, n_clusters=8)
    """
    return scHSCModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
