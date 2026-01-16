"""
scDHMap: Hyperbolic VAE with ZINB reconstruction and t-SNE repulsion
Latent space on Lorentz hyperboloid, with Poincaré ball interface
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances
from .base_model import BaseModel

eps = 1e-6


def lorentz2poincare(x: torch.Tensor) -> torch.Tensor:
    """Lorentz [B, d+1] -> Poincaré [B, d]"""
    return x[..., 1:] / (x[..., 0:1] + 1.0)


def poincare2lorentz(x: torch.Tensor) -> torch.Tensor:
    """Poincaré [B, d] -> Lorentz [B, d+1]"""
    x2 = torch.sum(x * x, dim=-1, keepdim=True).clamp_max(1 - 1e-6)
    denom = (1.0 - x2).clamp_min(1e-6)
    x0 = (1.0 + x2) / denom
    xi = 2.0 * x / denom
    return torch.cat([x0, xi], dim=-1)


def lorentz_distance_mat(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pairwise hyperbolic distances in Lorentz model"""
    minkowski_prod = -torch.matmul(u[..., 0:1], v[..., 0:1].transpose(-1, -2)) + torch.matmul(u[..., 1:], v[..., 1:].transpose(-1, -2))
    clamped = (-minkowski_prod).clamp(min=1.0 + eps, max=1e6)
    return torch.acosh(clamped)


def _binary_search_perplexity(dists: np.ndarray, perplexity: float, tol: float = 1e-5) -> np.ndarray:
    """Binary search for Gaussian kernel bandwidth matching target perplexity"""
    low = -np.inf
    high = np.inf
    beta = 1.0
    for _ in range(50):
        p = np.exp(-dists * beta)
        sum_p = np.sum(p)
        if sum_p == 0:
            p = np.ones(len(dists))
            sum_p = len(dists)
        entropy = np.log(sum_p) + beta * np.sum(dists * p) / sum_p
        h_diff = entropy - np.log(perplexity)
        if np.abs(h_diff) < tol:
            break
        if h_diff > 0:
            low = beta
            beta = beta * 2 if high == np.inf else (beta + high) / 2
        else:
            high = beta
            beta = beta / 2 if low == -np.inf else (beta + low) / 2
    p /= np.sum(p)
    return p


def compute_gaussian_perplexity(dists: np.ndarray, perplexity: float) -> np.ndarray:
    """Compute t-SNE affinities"""
    n = dists.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        dist_i = np.delete(dists[i], i)
        p_i = _binary_search_perplexity(dist_i, perplexity)
        P[i] = np.insert(p_i, i, 0.0)
    P = P / np.sum(P, axis=1, keepdims=True)
    return P


class HyperbolicWrappedNorm(Distribution):
    """Wrapped normal distribution on Lorentz hyperboloid"""
    arg_constraints = {"loc": constraints.real_vector, "scale": constraints.positive}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args=None):
        self.loc = loc
        self.scale = scale.clamp(min=1e-6, max=10.0)
        self.dim = loc.shape[-1] - 1
        self.device = loc.device
        super().__init__(batch_shape=self.loc.shape[:-1], event_shape=self.loc.shape[-1:], validate_args=validate_args)

    def sample(self):
        shape = self._extended_shape(torch.Size())
        v = torch.randn(shape[:-1] + (self.dim,), device=self.device, dtype=self.loc.dtype) * self.scale
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps).clamp(max=3.0)
        exp_v = torch.cat([torch.cosh(v_norm), torch.sinh(v_norm) * v / v_norm], dim=-1)
        return self._mobius_mat_vec(self.loc, exp_v)

    def rsample(self, sample_shape=torch.Size()):
        return self.sample()

    def _mobius_mat_vec(self, y, z):
        y0, y_ = y[..., 0:1], y[..., 1:]
        z0, z_ = z[..., 0:1], z[..., 1:]
        yz = torch.sum(y_ * z_, dim=-1, keepdim=True)
        A = -y0 * z0 - yz
        C = -y0 * z_ - z0 * y_
        return torch.cat([A, C], dim=-1)

    def log_prob(self, value):
        minkowski_prod = value[..., 0:1] * self.loc[..., 0:1] - torch.sum(value[..., 1:] * self.loc[..., 1:], dim=-1, keepdim=True)
        minkowski_prod = minkowski_prod.clamp(min=1.0 + eps, max=1e6)
        dist = torch.acosh(minkowski_prod).clamp(max=10.0)
        log_prob_val = -dist ** 2 / (2 * self.scale ** 2) - 0.5 * self.dim * torch.log(2 * math.pi * self.scale ** 2)
        return log_prob_val.clamp(min=-100, max=100)


class ZINBLoss(nn.Module):
    """Zero-Inflated Negative Binomial loss"""
    def forward(self, x, mean, disp, pi, scale_factor=None):
        eps_ = 1e-10
        if scale_factor is not None:
            scale_factor = scale_factor[:, None]
            mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps_) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps_)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps_))) + (x * (torch.log(disp + eps_) - torch.log(mean + eps_)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps_)
        zero_nb = torch.pow(disp / (disp + mean + eps_), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps_)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        return torch.mean(torch.sum(result, dim=1))


class MeanAct(nn.Module):
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class scDHMapCore(nn.Module):
    """Core hyperbolic VAE with ZINB decoder"""
    def __init__(self, input_dim, encode_layers, decode_layers, z_dim, dropout, device):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.device = device

        self.encoder = self._mlp([input_dim] + encode_layers, dropout)
        self.decoder = self._mlp([z_dim + 1] + decode_layers, dropout)

        self.enc_mu = nn.Linear(encode_layers[-1], z_dim)
        self.enc_sigma = nn.Sequential(nn.Linear(encode_layers[-1], z_dim), nn.Softplus())

        self.dec_mean = nn.Sequential(nn.Linear(decode_layers[-1], input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(decode_layers[-1], input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(decode_layers[-1], input_dim), nn.Sigmoid())

        self.zinb = ZINBLoss().to(device)

    def _mlp(self, layers, dropout):
        net = []
        for i in range(1, len(layers)):
            net += [nn.Linear(layers[i - 1], layers[i]), nn.BatchNorm1d(layers[i]), nn.ELU()]
            if dropout and dropout > 0:
                net += [nn.Dropout(dropout)]
        return nn.Sequential(*net)

    def _polar_project(self, x):
        """Project to Lorentz hyperboloid via polar coordinates"""
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp_min(eps)
        x_unit = x / x_norm
        x_norm = x_norm.clamp(0, 5.0)
        z = torch.cat((torch.cosh(x_norm), torch.sinh(x_norm) * x_unit), dim=1)
        return z

    def ae_forward(self, x):
        h = self.encoder(x)
        tmp = self.enc_mu(h).clamp(min=-5.0, max=5.0)
        z_mu = self._polar_project(tmp)

        z_sigma_sq = self.enc_sigma(h).clamp(min=1e-6, max=2.0)
        q_z = HyperbolicWrappedNorm(z_mu, z_sigma_sq)
        z = q_z.sample()

        dec_h = self.decoder(z)
        mean = self.dec_mean(dec_h)
        disp = self.dec_disp(dec_h)
        pi = self.dec_pi(dec_h)
        return q_z, z, z_mu, mean, disp, pi

    def tsne_repel(self, z_mu, p, gamma: float):
        """t-SNE repulsion term in hyperbolic space"""
        n = z_mu.size(0)
        dist = lorentz_distance_mat(z_mu, z_mu)
        num = 1.0 / gamma / (1.0 + (dist / gamma) ** 2)
        num = num.clamp(min=1e-12, max=1.0)

        p = (p / p.sum(dim=1, keepdim=True).clamp_min(eps)).clamp(min=eps, max=1.0)
        attraction = -torch.sum(p * torch.log(num).clamp(min=-50, max=50))

        den = (torch.sum(num, dim=1) - 1.0).clamp_min(eps)
        repellant = torch.sum(torch.log(den).clamp(min=-50, max=50))

        return ((repellant + attraction) / n).clamp(min=-1000, max=1000)

    def kld_loss(self, q_z, z):
        """KL divergence from prior (origin on hyperboloid)"""
        loc = torch.cat([torch.ones(z.shape[0], 1, device=self.device), torch.zeros(z.shape[0], self.z_dim, device=self.device)], dim=-1)
        p_z = HyperbolicWrappedNorm(loc, torch.ones(z.shape[0], self.z_dim, device=self.device))
        kl = (q_z.log_prob(z) - p_z.log_prob(z)).clamp(min=-1000, max=1000)
        return kl.mean()


class scDHMapModel(BaseModel):
    """
    scDHMap: Hyperbolic VAE for single-cell data
    
    Features:
    - Latent space on Lorentz hyperboloid (encode/decode via Poincaré ball)
    - ZINB reconstruction for count data
    - Optional t-SNE repulsion in hyperbolic space
    - Two-phase training: pretrain (recon+KL) → main (+ t-SNE)
    
    Batch format: (x, x_raw, size_factors, [idx])
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        encoder_layers: Optional[list] = None,
        decoder_layers: Optional[list] = None,
        dropout: float = 0.0,
        likelihood: str = "zinb",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        perplexity: float = 30.0,
        model_name: str = "scDHMap",
    ):
        """
        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension (Poincaré ball)
            encoder_layers: Encoder hidden dimensions
            decoder_layers: Decoder hidden dimensions
            dropout: Dropout rate
            alpha: t-SNE loss weight
            beta: KL loss weight
            gamma: t-SNE repulsion parameter
            perplexity: t-SNE perplexity
        """
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=encoder_layers or [], model_name=model_name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.perplexity = perplexity
        self.likelihood = likelihood

        self.core = scDHMapCore(
            input_dim=input_dim,
            encode_layers=encoder_layers or [128, 64, 32, 16],
            decode_layers=decoder_layers or [16, 32, 64, 128],
            z_dim=latent_dim,
            dropout=dropout,
            device=torch.device("cpu"),
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        dev = next(self.parameters()).device
        self.core.device = dev
        return self

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to Poincaré ball [B, latent_dim]"""
        _, _, z_mu, _, _, _ = self.core.ae_forward(x)
        return lorentz2poincare(z_mu)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode from Poincaré [B, latent_dim] to mean expression [B, input_dim]"""
        z_lorentz = poincare2lorentz(z)
        dec_h = self.core.decoder(z_lorentz)
        mean = self.core.dec_mean(dec_h)
        return mean

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        q_z, z, z_mu, mean, disp, pi = self.core.ae_forward(x)
        return {
            "q_z": q_z,
            "z": z,
            "z_mu": z_mu,
            "mean": mean,
            "disp": disp,
            "pi": pi,
            "reconstruction": mean,
        }
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], x_raw=None, sf=None, p_tensor=None, **kwargs):
        """Compute loss: ZINB + KL + optional t-SNE"""
        if x_raw is None:
            x_raw = x

        if self.likelihood != "zinb":
            raise ValueError("Only ZINB is supported")

        recon = self.core.zinb(x=x_raw, mean=outputs["mean"], disp=outputs["disp"], pi=outputs["pi"], scale_factor=sf)
        kld = self.core.kld_loss(outputs["q_z"], outputs["z"])

        tsne = torch.tensor(0.0, device=x.device)
        if p_tensor is not None:
            tsne = self.core.tsne_repel(outputs["z_mu"], p_tensor, gamma=self.gamma)

        total = recon + self.alpha * tsne + self.beta * kld
        return {"total_loss": total, "recon_loss": recon, "kld_loss": kld, "tsne_loss": tsne}

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 10,
        verbose: int = 1,
        pretrain_epochs: int = 400,
        x_pca: Optional[np.ndarray] = None,
        enable_batch_tsne: bool = False,
        **kwargs,
    ):
        """
        Two-phase training:
        1. Pretrain: ZINB + KL only
        2. Main: ZINB + KL + t-SNE (if enable_batch_tsne=True)
        """
        self.to(device)
        weight_decay = 1e-3
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        def _unpack_batch(batch):
            if not isinstance(batch, (list, tuple)):
                raise ValueError("scDHMapModel.fit expects batch as tuple/list")
            if len(batch) == 2:
                x, x_raw = batch
                sf = None
                idx = None
            elif len(batch) == 3:
                x, x_raw, sf = batch
                idx = None
            else:
                x, x_raw, sf, idx = batch[0], batch[1], batch[2], batch[3]
            return x, x_raw, sf, idx

        def _ensure_sf(x_raw_tensor: torch.Tensor, sf_tensor: Optional[torch.Tensor]):
            if sf_tensor is not None:
                return sf_tensor
            lib = x_raw_tensor.sum(dim=1).clamp_min(1.0)
            lib = lib / lib.mean().clamp_min(1e-6)
            return lib

        # Phase 1: Pretrain
        for ep in range(pretrain_epochs):
            self.train()
            tot = 0.0
            n = 0
            for batch in train_loader:
                x, x_raw, sf, idx = _unpack_batch(batch)
                x = x.to(device).float()
                x_raw = x_raw.to(device).float()
                sf = _ensure_sf(x_raw, sf.to(device).float() if sf is not None else None)

                out = self.forward(x)
                loss_dict = self.compute_loss(x, out, x_raw=x_raw, sf=sf, p_tensor=None)
                loss = loss_dict["recon_loss"] + loss_dict["kld_loss"]

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                opt.step()

                tot += float(loss.item())
                n += 1

            if verbose >= 1 and (ep + 1) % 50 == 0:
                print(f"Pretrain {ep+1:3d}/{pretrain_epochs} | Loss: {tot/max(n,1):.6f}")

        # Phase 2: Main training
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr / 2,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        for ep in range(epochs):
            self.train()
            tot = 0.0
            n = 0

            for batch in train_loader:
                x, x_raw, sf, idx = _unpack_batch(batch)
                x = x.to(device).float()
                x_raw = x_raw.to(device).float()
                sf = _ensure_sf(x_raw, sf.to(device).float() if sf is not None else None)

                p_tensor = None
                if enable_batch_tsne:
                    x_np = x.detach().cpu().numpy()
                    dist = pairwise_distances(x_np, metric="euclidean").astype(np.double)
                    p = compute_gaussian_perplexity(dist, float(self.perplexity))
                    p_tensor = torch.tensor(p, dtype=torch.float32, device=device)

                out = self.forward(x)
                loss_dict = self.compute_loss(x, out, x_raw=x_raw, sf=sf, p_tensor=p_tensor)
                loss = loss_dict["total_loss"]

                if torch.isnan(loss):
                    continue

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                opt.step()

                tot += float(loss.item())
                n += 1

            if verbose >= 1 and (ep + 1) % 50 == 0:
                print(f"Train {ep+1:3d}/{epochs} | Loss: {tot/max(n,1):.6f}")

        return {"train_loss": []}


def create_scdhmap_model(input_dim: int, latent_dim: int = 10, **kwargs) -> scDHMapModel:
    """
    Create scDHMap model
    
    Example:
        >>> model = create_scdhmap_model(2000, latent_dim=10, alpha=1.0, beta=1.0)
    """
    return scDHMapModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)