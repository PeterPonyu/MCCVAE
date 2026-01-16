"""
scDAC: Single-cell Deep AutoEncoder with Clustering via DPMM
Combines autoencoder reconstruction with Dirichlet Process Mixture Model regularization
"""
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from sklearn.mixture import BayesianGaussianMixture
from scipy.sparse import issparse
from .base_model import BaseModel


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "mish":
        return nn.Mish()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


class _Layer1D(nn.Module):
    """Normalization + Activation + Dropout layer"""
    def __init__(self, dim: int, norm: Optional[str] = None, act: Optional[str] = None, drop: float = 0.0):
        super().__init__()
        layers = []
        if norm == "bn":
            layers.append(nn.BatchNorm1d(dim))
        elif norm == "ln":
            layers.append(nn.LayerNorm(dim))
        if act:
            layers.append(_act(act))
        if drop and drop > 0:
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    """Configurable MLP with flexible normalization, activation, and dropout"""
    def __init__(
        self,
        features: list,
        hid_act: str = "mish",
        out_act: Optional[str] = None,
        norm: Optional[str] = None,
        hid_norm: Optional[str] = None,
        drop: float = 0.0,
        hid_drop: float = 0.0,
    ):
        super().__init__()
        layers = []
        for i in range(1, len(features)):
            is_last = i == len(features) - 1
            cur_norm = norm if is_last else hid_norm
            cur_act = out_act if is_last else hid_act
            cur_drop = drop if is_last else hid_drop

            layers.append(nn.Linear(features[i - 1], features[i]))
            if cur_norm or cur_act or cur_drop:
                layers.append(_Layer1D(features[i], norm=cur_norm, act=cur_act, drop=cur_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class scDACEAutoEncoder(nn.Module):
    """Standard autoencoder backbone"""
    def __init__(self, input_dim: int, encoder_dims: list, latent_dim: int, decoder_dims: list, norm: str, drop: float):
        super().__init__()
        self.encoder = MLP([input_dim] + encoder_dims + [latent_dim], norm=norm, hid_norm=norm, hid_drop=drop, out_act=None)
        self.decoder = MLP([latent_dim] + decoder_dims + [input_dim], norm=norm, hid_norm=norm, hid_drop=drop, out_act=None)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class scDACModel(BaseModel):
    """
    scDAC: Autoencoder with DPMM (Dirichlet Process Mixture Model) regularization
    
    Features:
    - Two-phase training: warmup + DPMM-regularized
    - Periodic DPMM refitting on latent space
    - Custom fit() method for DPMM integration
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        encoder_dims: Optional[list] = None,
        decoder_dims: Optional[list] = None,
        norm_type: str = "bn",
        dropout_rate: float = 0.1,
        dpmm_warmup_ratio: float = 0.6,
        dpmm_loss_weight: float = 1.0,
        n_components: int = 50,
        model_name: str = "scDAC",
    ):
        """
        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            encoder_dims: Encoder hidden dimensions
            decoder_dims: Decoder hidden dimensions
            norm_type: 'bn' or 'ln'
            dropout_rate: Dropout rate
            dpmm_warmup_ratio: Fraction of epochs before DPMM starts (0.6 = first 60% warmup)
            dpmm_loss_weight: Weight for DPMM loss
            n_components: Number of DPMM components
        """
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=encoder_dims or [], model_name=model_name)
        self.ae = scDACEAutoEncoder(
            input_dim=input_dim,
            encoder_dims=encoder_dims or [256, 128],
            latent_dim=latent_dim,
            decoder_dims=decoder_dims or [128, 256],
            norm=norm_type,
            drop=dropout_rate,
        )
        self.dpmm_warmup_ratio = dpmm_warmup_ratio
        self.dpmm_loss_weight = dpmm_loss_weight
        self.n_components = n_components
        self.dpmm_params = None
        self.recon_loss_fn = nn.MSELoss()

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to latent space"""
        _, z = self.ae(x)
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode from latent space"""
        return self.ae.decoder(z)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        x_hat, z = self.ae(x)
        return {"reconstruction": x_hat, "latent": z}

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Compute loss: reconstruction + DPMM regularization"""
        recon = self.recon_loss_fn(outputs["reconstruction"], x)

        dpmm = torch.tensor(0.0, device=x.device)
        if self.dpmm_params is not None:
            dpmm = self._dpmm_loss(outputs["latent"])

        total = recon + self.dpmm_loss_weight * dpmm
        return {"total_loss": total, "recon_loss": recon, "dpmm_loss": dpmm}

    def _update_dpmm_params(self, bgm: BayesianGaussianMixture, device: torch.device):
        """Extract DPMM parameters from fitted sklearn model"""
        self.dpmm_params = {
            "means": torch.as_tensor(bgm.means_, dtype=torch.float32, device=device),
            "weight_concentration": torch.as_tensor(bgm.weight_concentration_, dtype=torch.float32, device=device),
            "precisions_cholesky": torch.as_tensor(bgm.precisions_cholesky_, dtype=torch.float32, device=device),
        }

    def _dpmm_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute DPMM log-likelihood loss"""
        dp = self.dpmm_params
        n_features = z.size(1)

        digamma_sum = torch.special.digamma(dp["weight_concentration"][0] + dp["weight_concentration"][1])
        digamma_a = torch.special.digamma(dp["weight_concentration"][0])
        digamma_b = torch.special.digamma(dp["weight_concentration"][1])

        log_weights_b = torch.cat(
            [torch.zeros(1, device=z.device), torch.cumsum(digamma_b - digamma_sum, dim=0)[:-1]]
        )
        log_weights = digamma_a - digamma_sum + log_weights_b

        log_det = torch.sum(dp["precisions_cholesky"].clamp_min(1e-12).log(), dim=1)
        precisions = dp["precisions_cholesky"] ** 2

        log_prob = (
            torch.sum((dp["means"] ** 2 * precisions), dim=1)
            - 2.0 * torch.mm(z, (dp["means"] * precisions).T)
            + torch.mm(z ** 2, precisions.T)
        )
        log_gauss_pre = -0.5 * (n_features * math.log(math.pi * 2.0) + log_prob) + log_det
        log_likelihood = torch.logsumexp(log_gauss_pre + log_weights, dim=1)
        return -log_likelihood.mean()

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 500,
        lr: float = 1e-4,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 10,
        verbose: int = 1,
        **kwargs,
    ):
        """
        Custom fit with DPMM refitting
        
        Training phases:
        1. Warmup: Train autoencoder without DPMM (first dpmm_warmup_ratio epochs)
        2. DPMM: Refit DPMM on latent space, train with DPMM loss
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        dpmm_warmup_epochs = int(epochs * self.dpmm_warmup_ratio)

        for epoch in range(epochs):
            self.train()

            if epoch >= dpmm_warmup_epochs:
                z_all = []
                with torch.no_grad():
                    for batch in train_loader:
                        x, _ = self._prepare_batch(batch, device)
                        z = self.encode(x)
                        z_all.append(z.detach().cpu().numpy())
                z_all = np.concatenate(z_all, axis=0)
                try:
                    bgm = BayesianGaussianMixture(
                        n_components=self.n_components,
                        weight_concentration_prior=1e-10,
                        mean_precision_prior=80,
                        covariance_type="diag",
                        init_params="kmeans",
                        max_iter=1000,
                        warm_start=True,
                    ).fit(z_all)
                    self._update_dpmm_params(bgm, device=torch.device(device))
                except Exception:
                    pass

            total = 0.0
            n_batches = 0
            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs, **kwargs)
                loss_dict = self.compute_loss(x, out, **batch_kwargs, **kwargs)
                loss = loss_dict["total_loss"]
                loss.backward()
                optimizer.step()
                total += float(loss.item())
                n_batches += 1

            if verbose >= 1:
                print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {total/max(n_batches,1):.4f}")

        return {"train_loss": []}


def create_scdac_model(input_dim: int, latent_dim: int = 32, **kwargs) -> scDACModel:
    """
    Create scDAC model
    
    Example:
        >>> model = create_scdac_model(2000, latent_dim=32, dpmm_warmup_ratio=0.6)
    """
    return scDACModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)