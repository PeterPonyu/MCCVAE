"""
scGPCL: Graph Prototypical Contrastive Learning for single-cell data
Combines instance-wise and prototype-based contrastive learning with ZINB reconstruction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Dict, Optional, List, Any, Tuple
from .base_model import BaseModel

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


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
    def forward(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        disp: torch.Tensor,
        pi: torch.Tensor,
        scale_factor: torch.Tensor = None,
        ridge_lambda: float = 0.0,
    ) -> torch.Tensor:
        eps = 1e-10
        if scale_factor is not None:
            scale_factor = scale_factor[:, None]
            mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
            x * (torch.log(disp + eps) - torch.log(mean + eps))
        )
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result = result + ridge

        return torch.mean(result)


class MLPEncoder(nn.Module):
    """MLP-based encoder for scGPCL"""
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers[:-1])  # Remove last dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MLPDecoder(nn.Module):
    """MLP-based decoder for scGPCL"""
    def __init__(self, latent_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers[:-1])  # Remove last dropout

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class scGPCLCore(nn.Module):
    """Core scGPCL model"""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        n_clusters: int,
        tau: float = 0.25,
        alpha: float = 1.0,
        granularity: List[float] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.granularity = granularity or [1.0, 1.5, 2.0]
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = MLPEncoder(input_dim, hidden_dims, dropout)

        # Decoder
        decoder_dims = hidden_dims[::-1]
        if len(decoder_dims) > 1:
            self.decoder = MLPDecoder(latent_dim, decoder_dims[:-1], dropout)
            hidden_out = decoder_dims[-2] if len(decoder_dims) > 1 else latent_dim
        else:
            self.decoder = nn.Identity()
            hidden_out = latent_dim

        # ZINB layers
        self.mean_layer = nn.Sequential(nn.Linear(hidden_out, input_dim), MeanAct())
        self.disp_layer = nn.Sequential(nn.Linear(hidden_out, input_dim), DispAct())
        self.pi_layer = nn.Sequential(nn.Linear(hidden_out, input_dim), nn.Sigmoid())

        # Cluster centers for fine-tuning
        self.mu = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.n_cluster_list = [n_clusters]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.decoder(z)
        mean = self.mean_layer(hidden)
        disp = self.disp_layer(hidden)
        pi = self.pi_layer(hidden)
        return mean, disp, pi

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rep = self.encode(x)
        mean, disp, pi = self.decode(rep)
        return mean, disp, pi, rep

    def ins_contrastive_loss(self, rep1: torch.Tensor, rep2: torch.Tensor) -> torch.Tensor:
        """Instance-wise contrastive loss"""
        batch_size = rep1.size(0)
        device = rep1.device

        pos_mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        neg_mask = 1 - pos_mask

        rep1 = F.normalize(rep1, dim=1)
        rep2 = F.normalize(rep2, dim=1)
        contrast_feature = torch.cat([rep1, rep2], dim=0)
        anchor_feature = contrast_feature

        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, anchor_feature.T), self.tau
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        pos_mask = pos_mask.repeat(2, 2)
        neg_mask = neg_mask.repeat(2, 2)

        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(batch_size * 2, device=device).view(-1, 1),
            0,
        )
        pos_mask = pos_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        exp_logits = exp_logits * (neg_mask + pos_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()

        return loss

    def proto_loss(
        self,
        rep: torch.Tensor,
        cluster_assignments: List[np.ndarray],
        centroids_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Prototypical contrastive loss"""
        if not cluster_assignments or not centroids_list:
            return torch.tensor(0.0, device=rep.device)

        def cos_sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            return torch.matmul(z1, z2.T)

        f = lambda x: torch.exp(x / self.tau)
        loss_proto = torch.tensor(0.0, device=rep.device)

        for assignment, centroids in zip(cluster_assignments, centroids_list):
            n_clusters = centroids.size(0)
            pos_prototypes = centroids[assignment]
            pos_score = f(cos_sim(rep, pos_prototypes)).diag()

            # Negative prototypes
            neg_scores = []
            for i, proto_id in enumerate(assignment):
                neg_ids = [j for j in range(n_clusters) if j != proto_id]
                if neg_ids:
                    neg_proto = centroids[neg_ids]
                    neg_score = f(cos_sim(rep[i:i+1], neg_proto)).sum()
                    neg_scores.append(neg_score)
            
            if neg_scores:
                neg_score_tensor = torch.stack(neg_scores)
                loss = -torch.log(pos_score / (neg_score_tensor + pos_score + 1e-8))
                loss_proto = loss_proto + loss.mean()

        return loss_proto / max(len(cluster_assignments), 1)

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """Soft cluster assignment"""
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """Target distribution for clustering"""
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def cluster_loss(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """KL divergence clustering loss"""
        return torch.mean(torch.sum(p * torch.log(p / (q + 1e-6) + 1e-8), dim=-1))

    def infer_num_proto(self) -> None:
        """Infer number of prototypes at different granularities"""
        if isinstance(self.granularity, str):
            self.granularity = eval(self.granularity)
        
        if len(self.granularity) == 1:
            self.n_cluster_list = [int(self.n_clusters * self.granularity[0])]
        elif len(self.granularity) == 2:
            c1 = int(self.n_clusters * self.granularity[0])
            c2 = int(max(c1 + 1, np.ceil(self.n_clusters * self.granularity[1])))
            self.n_cluster_list = [c1, c2]
        else:
            c1 = int(np.floor(self.n_clusters * self.granularity[0]))
            c2 = int(max(c1 + 1, self.n_clusters * self.granularity[1]))
            c3 = int(max(c2 + 1, np.ceil(self.n_clusters * self.granularity[2])))
            self.n_cluster_list = [c1, c2, c3]


class scGPCLModel(BaseModel):
    """
    scGPCL: Graph Prototypical Contrastive Learning for single-cell data
    
    Features:
    - Instance-wise contrastive learning
    - Multi-granularity prototypical contrastive learning
    - ZINB reconstruction loss
    - Two-stage training: pre-training + fine-tuning
    - Feature dropout augmentation
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: List[int] = None,
        n_clusters: int = 10,
        tau: float = 0.25,
        alpha: float = 1.0,
        granularity: List[float] = None,
        dropout: float = 0.1,
        feature_drop_rate: float = 0.2,
        model_name: str = "scGPCL",
    ):
        """
        Args:
            input_dim: Gene dimension
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions
            n_clusters: Number of clusters
            tau: Temperature for contrastive loss
            alpha: Alpha parameter for soft assignment
            granularity: Granularity levels for prototypes [1.0, 1.5, 2.0]
            dropout: Dropout rate
            feature_drop_rate: Feature dropout rate for augmentation
        """
        hidden_dims = hidden_dims or [256, latent_dim]
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            model_name=model_name,
        )

        self.n_clusters = n_clusters
        self.feature_drop_rate = feature_drop_rate
        self.granularity = granularity or [1.0, 1.5, 2.0]

        self.model = scGPCLCore(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            n_clusters=n_clusters,
            tau=tau,
            alpha=alpha,
            granularity=self.granularity,
            dropout=dropout,
        )
        self.recon_loss_fn = ZINBLoss()

    def _prepare_batch(self, batch_data: Any, device: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            batch_kwargs = {}
            if len(batch_data) >= 2:
                batch_kwargs["x_raw"] = batch_data[1].to(device).float() if torch.is_tensor(batch_data[1]) else None
            return x, batch_kwargs
        return batch_data.to(device).float(), {}

    def _feature_dropout(self, x: torch.Tensor, drop_rate: float) -> torch.Tensor:
        """Apply feature dropout augmentation"""
        if drop_rate <= 0 or not self.training:
            return x
        mask = torch.bernoulli(torch.ones_like(x) * (1 - drop_rate))
        return x * mask

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to latent space"""
        return self.model.encode(x)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latent to ZINB parameters (returns mean)"""
        mean, _, _ = self.model.decode(z)
        return mean

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        mean, disp, pi, rep = self.model(x)
        return {
            "latent": rep,
            "mean": mean,
            "disp": disp,
            "pi": pi,
            "reconstruction": mean,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        x_raw: torch.Tensor = None,
        size_factors: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute ZINB reconstruction loss"""
        mean = outputs["mean"]
        disp = outputs["disp"]
        pi = outputs["pi"]

        target = x_raw if x_raw is not None else x
        recon_loss = self.recon_loss_fn(target, mean, disp, pi, size_factors)

        return {
            "total_loss": recon_loss,
            "recon_loss": recon_loss,
        }

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 25,
        verbose: int = 1,
        warmup_epochs: int = 20,
        lam1: float = 1.0,
        lam2: float = 1.0,
        fine_tune_epochs: int = 50,
        fine_tune_lr: float = 1e-4,
        lam3: float = 1.0,
        **kwargs,
    ) -> Dict[str, list]:
        """
        Two-stage training: pre-training + fine-tuning
        
        Args:
            warmup_epochs: Epochs before enabling prototypical loss
            lam1: Weight for instance-wise contrastive loss
            lam2: Weight for prototypical loss
            fine_tune_epochs: Fine-tuning epochs
            fine_tune_lr: Fine-tuning learning rate
            lam3: Weight for clustering loss in fine-tuning
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        history = {"train_loss": [], "val_loss": [], "recon_loss": [], "ins_loss": [], "proto_loss": []}

        best_loss = float("inf")
        patience_counter = 0

        if verbose >= 1:
            print("=== Pre-training Phase ===")

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            total_recon = 0
            total_ins = 0
            total_proto = 0
            n_batches = 0

            for batch_data in train_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                batch_size = x.size(0)

                # Two augmented views
                x1 = self._feature_dropout(x, self.feature_drop_rate)
                x2 = self._feature_dropout(x, self.feature_drop_rate)

                optimizer.zero_grad()

                # Forward for both views
                mean1, disp1, pi1, rep1 = self.model(x1)
                mean2, disp2, pi2, rep2 = self.model(x2)

                # Reconstruction loss
                x_raw = batch_kwargs.get("x_raw", x)
                sf = batch_kwargs.get("size_factors")
                recon_loss = (
                    self.recon_loss_fn(x_raw, mean1, disp1, pi1, sf)
                    + self.recon_loss_fn(x_raw, mean2, disp2, pi2, sf)
                ) / 2

                # Instance-wise contrastive loss
                ins_loss = self.model.ins_contrastive_loss(rep1, rep2)

                # Prototypical loss (after warmup)
                if epoch >= warmup_epochs:
                    if epoch == warmup_epochs:
                        self.model.infer_num_proto()

                    rep_np = rep2.detach().cpu().numpy()
                    cluster_assignments = []
                    centroids_list = []

                    for n_cluster in self.model.n_cluster_list:
                        if KMeans is not None:
                            kmeans = KMeans(n_clusters=n_cluster, n_init=10, random_state=42)
                            y_pred = kmeans.fit_predict(rep_np)
                            centroids = []
                            for i in np.unique(y_pred):
                                c = rep_np[y_pred == i]
                                centroid = np.mean(c, axis=0)
                                centroids.append(torch.tensor(centroid, device=device))
                            centroids = torch.stack(centroids, dim=0)
                            cluster_assignments.append(y_pred)
                            centroids_list.append(centroids)

                    proto_loss = self.model.proto_loss(rep1, cluster_assignments, centroids_list)
                else:
                    proto_loss = torch.tensor(0.0, device=device)

                loss = recon_loss + lam1 * ins_loss + lam2 * proto_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_ins += ins_loss.item()
                total_proto += float(proto_loss.item())
                n_batches += 1

            avg_loss = total_loss / n_batches
            history["train_loss"].append(avg_loss)
            history["recon_loss"].append(total_recon / n_batches)
            history["ins_loss"].append(total_ins / n_batches)
            history["proto_loss"].append(total_proto / n_batches)

            # Validation loss for early stopping
            if val_loader is not None:
                self.eval()
                val_loss_total = 0
                val_n_batches = 0
                with torch.no_grad():
                    for batch_data in val_loader:
                        x, batch_kwargs = self._prepare_batch(batch_data, device)
                        x1 = self._feature_dropout(x, self.feature_drop_rate)
                        x2 = self._feature_dropout(x, self.feature_drop_rate)
                        mean1, disp1, pi1, rep1 = self.model(x1)
                        mean2, disp2, pi2, rep2 = self.model(x2)
                        x_raw = batch_kwargs.get("x_raw", x)
                        sf = batch_kwargs.get("size_factors")
                        recon_loss = (
                            self.recon_loss_fn(x_raw, mean1, disp1, pi1, sf)
                            + self.recon_loss_fn(x_raw, mean2, disp2, pi2, sf)
                        ) / 2
                        ins_loss = self.model.ins_contrastive_loss(rep1, rep2)
                        val_loss_total += (recon_loss + lam1 * ins_loss).item()
                        val_n_batches += 1
                val_loss = val_loss_total / val_n_batches
                history["val_loss"].append(val_loss)
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
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # No early stopping when val_loader is None, just save best train model
                if save_path and avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_model(save_path)

            if verbose >= 1 and (epoch + 1) % 10 == 0:
                loss_str = f"Loss: {avg_loss:.4f}"
                if val_loader is not None:
                    loss_str += f" | Val: {val_loss:.4f}"
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"{loss_str} | Recon: {total_recon/n_batches:.4f} | "
                    f"Ins: {total_ins/n_batches:.4f} | Proto: {total_proto/n_batches:.4f}"
                )

        # Fine-tuning phase
        if fine_tune_epochs > 0 and KMeans is not None:
            if verbose >= 1:
                print("\n=== Fine-tuning Phase ===")

            optimizer = torch.optim.Adadelta(self.parameters(), lr=fine_tune_lr, rho=0.95)

            # Initialize cluster centers
            self.eval()
            all_reps = []
            for batch_data in train_loader:
                x, _ = self._prepare_batch(batch_data, device)
                with torch.no_grad():
                    rep = self.model.encode(x)
                all_reps.append(rep.cpu().numpy())
            all_reps = np.concatenate(all_reps, axis=0)

            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
            kmeans.fit(all_reps)
            self.model.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device))

            for epoch in range(fine_tune_epochs):
                self.train()
                total_loss = 0
                n_batches = 0

                # Compute target distribution
                self.eval()
                all_reps = []
                for batch_data in train_loader:
                    x, _ = self._prepare_batch(batch_data, device)
                    with torch.no_grad():
                        rep = self.model.encode(x)
                    all_reps.append(rep)
                all_reps = torch.cat(all_reps, dim=0)
                q = self.model.soft_assign(all_reps)
                p = self.model.target_distribution(q).detach()

                self.train()
                idx = 0
                for batch_data in train_loader:
                    x, batch_kwargs = self._prepare_batch(batch_data, device)
                    batch_size = x.size(0)
                    x1 = self._feature_dropout(x, self.feature_drop_rate)

                    optimizer.zero_grad()
                    mean, disp, pi, rep = self.model(x1)

                    # Reconstruction
                    x_raw = batch_kwargs.get("x_raw", x)
                    sf = batch_kwargs.get("size_factors")
                    recon_loss = self.recon_loss_fn(x_raw, mean, disp, pi, sf)

                    # Clustering loss
                    q_batch = self.model.soft_assign(rep)
                    p_batch = p[idx : idx + batch_size]
                    cluster_loss = self.model.cluster_loss(p_batch, q_batch)

                    loss = recon_loss + lam3 * cluster_loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1
                    idx += batch_size

                if verbose >= 1 and (epoch + 1) % 10 == 0:
                    print(f"Fine-tune Epoch {epoch+1:3d}/{fine_tune_epochs} | Loss: {total_loss/n_batches:.4f}")

        if save_path:
            self.save_model(save_path)

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
            outputs = self.forward(x)
            latents.append(outputs["latent"].cpu().numpy())
            if return_reconstructions:
                reconstructions.append(outputs["reconstruction"].cpu().numpy())

        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(reconstructions, axis=0)
        return result


def create_scgpcl_model(input_dim: int, latent_dim: int = 64, **kwargs) -> scGPCLModel:
    """
    Create scGPCL model
    
    Example:
        >>> model = create_scgpcl_model(2000, latent_dim=64, n_clusters=10)
    """
    return scGPCLModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
