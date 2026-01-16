"""
scDiffusion: Denoising diffusion probabilistic model for single-cell data

Note: Diffusion models operate in data space, not explicit latent space like VAE.
- encode(): UNet bottleneck features projected to [B, embedding_dim]
- decode(): Maps embedding to initial noise → reverse diffusion → [B, input_dim]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional
from .base_model import BaseModel


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def get_num_groups(channels):
    for g in [32, 16, 8, 4, 2]:
        if channels % g == 0:
            return g
    return 1


class ClassEmbedder(nn.Module):
    """Conditional class embedding with dropout for classifier-free guidance"""
    def __init__(self, embed_dim, n_classes, cond_drop_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_classes + 1, embed_dim)
        self.n_classes = n_classes
        self.cond_drop_prob = cond_drop_prob

    def forward(self, y):
        if self.training and self.cond_drop_prob > 0:
            drop = torch.rand(y.size(0), device=y.device) < self.cond_drop_prob
            y = torch.where(drop, torch.full_like(y, self.n_classes), y)
        return self.embedding(y)


class ResBlock(nn.Module):
    """Residual block with time/condition embedding"""
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()

        self.norm1 = nn.GroupNorm(get_num_groups(in_channels), in_channels)
        self.conv1 = nn.Linear(in_channels, out_channels)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels),
        )

        self.norm2 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Linear(out_channels, out_channels)

        self.skip = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.norm1(x.unsqueeze(-1)).squeeze(-1)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.time_proj(emb)

        h = self.norm2(h.unsqueeze(-1)).squeeze(-1)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class DenoisingUNet(nn.Module):
    """1D UNet for noise prediction"""
    def __init__(
        self,
        input_dim,
        model_channels,
        num_res_blocks,
        dropout,
        channel_mult,
        n_classes=0,
        cond_drop_prob=0.1,
    ):
        super().__init__()

        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.model_channels = model_channels

        time_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.class_embedder = (
            ClassEmbedder(time_dim, n_classes, cond_drop_prob)
            if n_classes > 0
            else None
        )

        self.input_proj = nn.Linear(input_dim, model_channels)

        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.skip_channels = []

        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                self.skip_channels.append(ch)
            self.downsample.append(nn.Linear(ch, ch) if level < len(channel_mult) - 1 else None)

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        self.bottleneck_dim = ch

        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                ch = out_ch
            self.upsample.append(nn.Linear(ch, ch) if level > 0 else None)

        self.out_norm = nn.GroupNorm(get_num_groups(ch), ch)
        self.out_proj = nn.Linear(ch, input_dim)

    def _build_cond_embedding(self, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))
        if self.class_embedder is not None and y is not None:
            emb = emb + self.class_embedder(y)
        return emb

    def extract_bottleneck(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract bottleneck features for encode()"""
        emb = self._build_cond_embedding(t, y)
        h = self.input_proj(x)

        down_idx = 0
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[down_idx](h,emb)
                down_idx += 1
            if self.downsample[level] is not None:
                h = self.downsample[level](h)

        h = self.mid1(h, emb)
        h = self.mid2(h, emb)
        return h

    def forward(self, x, t, y=None):
        """Predict noise"""
        emb = self._build_cond_embedding(t, y)

        h = self.input_proj(x)
        hs = []

        down_idx = 0
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[down_idx](h,emb)
                hs.append(h)
                down_idx += 1
            if self.downsample[level] is not None:
                h = self.downsample[level](h)

        h = self.mid1(h, emb)
        h = self.mid2(h, emb)

        up_idx = 0
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = torch.cat([h, hs.pop()], dim=-1)
                h = self.up_blocks[up_idx](h,emb)
                up_idx += 1
            if self.upsample[level] is not None:
                h = self.upsample[level](h)

        h = self.out_norm(h.unsqueeze(-1)).squeeze(-1)
        h = F.silu(h)
        return self.out_proj(h)


class scDiffusionModel(BaseModel):
    """
    scDiffusion: DDPM for single-cell data generation
    
    Features:
    - Forward diffusion: q(x_t | x_0) adds noise progressively
    - Reverse diffusion: p(x_{t-1} | x_t) denoises via UNet
    - Conditional generation with class labels
    - Encode: UNet bottleneck → embedding_dim
    - Decode: embedding → initial noise → reverse diffusion
    """

    def __init__(
        self,
        input_dim,
        latent_dim=128,          # UNet capacity
        embedding_dim: int = 10, # Returned by encode()
        hidden_dims=None,
        n_timesteps=1000,
        beta_schedule="linear",
        n_classes=0,
        cond_drop_prob=0.1,
        loss_type="mse",
        model_name="scDiffusion",
    ):
        """
        Args:
            input_dim: Gene dimension
            latent_dim: UNet base channels
            embedding_dim: Dimension of extracted latent (BaseModel.latent_dim)
            hidden_dims: UNet channel multipliers
            n_timesteps: Number of diffusion steps
            beta_schedule: 'linear' or 'cosine'
            n_classes: Number of cell types (0 = unconditional)
            cond_drop_prob: Classifier-free guidance dropout
            loss_type: 'mse', 'l1', or 'hybrid'
        """
        hidden_dims = hidden_dims or [1, 2, 3, 4]

        super().__init__(input_dim=input_dim, latent_dim=embedding_dim, hidden_dims=hidden_dims, model_name=model_name)

        self.n_timesteps = n_timesteps
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.model_channels = latent_dim

        self.denoising_net = DenoisingUNet(
            input_dim=input_dim,
            model_channels=self.model_channels,
            num_res_blocks=2,
            dropout=0.1,
            channel_mult=tuple(hidden_dims),
            n_classes=n_classes,
            cond_drop_prob=cond_drop_prob,
        )

        self.latent_head = nn.Sequential(
            nn.Linear(self.denoising_net.bottleneck_dim, embedding_dim),
            nn.SiLU(),
        )

        self.noise_head = nn.Linear(embedding_dim, input_dim)

        self._setup_diffusion_schedule(beta_schedule)

    def _prepare_batch(self, batch_data, device):
        """Extract x and optional integer labels"""
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
            x = batch_data[0].to(device).float()

            if len(batch_data) >= 2 and torch.is_tensor(batch_data[1]):
                b1 = batch_data[1]
                if b1.dtype in (torch.int32, torch.int64) and b1.ndim == 1:
                    return x, {"y": b1.to(device).long()}

            return x, {}

        x = batch_data.to(device).float()
        return x, {}

    def _setup_diffusion_schedule(self, schedule_type: str = "linear"):
        """Setup forward diffusion variances"""
        if schedule_type == "linear":
            betas = torch.linspace(1e-4, 0.02, self.n_timesteps)
        elif schedule_type == "cosine":
            steps = self.n_timesteps + 1
            x = torch.linspace(0, self.n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.n_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Forward diffusion: add noise to x_0"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def encode(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract low-dimensional embedding [B, embedding_dim] from UNet bottleneck"""
        if timestep is None:
            timestep = self.n_timesteps // 2
        t = torch.full((x.size(0),), int(timestep), device=x.device, dtype=torch.long)

        x_t = self.q_sample(x, t)
        h = self.denoising_net.extract_bottleneck(x_t, t, y=y)
        z = self.latent_head(h)
        return z

    def decode(self, z: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Map embedding → initial noise → reverse diffusion → generated sample"""
        init_x = self.noise_head(z)
        return self.p_sample_loop(batch_size=z.size(0), device=init_x.device, init_x=init_x, y=y)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Training forward pass: predict noise"""
        batch_size = x.size(0)
        device = x.device

        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        predicted_noise = self.denoising_net(x_noisy, t, y)

        return {"predicted_noise": predicted_noise, "true_noise": noise, "x_noisy": x_noisy, "t": t, "y": y}

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Compute noise prediction loss"""
        predicted_noise = outputs["predicted_noise"]
        true_noise = outputs["true_noise"]

        if self.loss_type == "mse":
            loss = F.mse_loss(predicted_noise, true_noise, reduction="mean")
        elif self.loss_type == "l1":
            loss = F.l1_loss(predicted_noise, true_noise, reduction="mean")
        elif self.loss_type == "hybrid":
            loss = F.mse_loss(predicted_noise, true_noise, reduction="mean") + 0.1 * F.l1_loss(predicted_noise, true_noise, reduction="mean")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return {"total_loss": loss, "recon_loss": loss, "diffusion_loss": loss}

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None, clip_denoised: bool = True):
        """Single reverse diffusion step"""
        batch_size = x.size(0)
        t_tensor = torch.full((batch_size,), t, device=x.device, dtype=torch.long)

        predicted_noise = self.denoising_net(x, t_tensor, y)

        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]

        mean = (x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
        if clip_denoised:
            mean = torch.clamp(mean, -1, 1)

        if t > 0:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(beta) * noise
        return mean

    @torch.no_grad()
    def p_sample_loop(
        self,
        batch_size: int,
        device: str = "cuda",
        init_x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ):
        """Full reverse diffusion process"""
        if init_x is None:
            x = torch.randn(batch_size, self.input_dim, device=device)
        else:
            x = init_x.to(device)

        for t in reversed(range(self.n_timesteps)):
            x = self.p_sample(x, t, y, clip_denoised)

        return x

    def extract_latent(self, data_loader, device="cuda", timestep: Optional[int] = None, return_reconstructions: bool = False):
        """
        Extract latent representations and optionally reconstructions
        
        Returns:
            dict with 'latent' [N, embedding_dim], optional 'reconstruction' and 'labels'
        """
        self.eval()
        self.to(device)

        if timestep is None:
            timestep = self.n_timesteps // 2

        latents = []
        reconstructions = [] if return_reconstructions else None
        labels = []

        with torch.no_grad():
            for x, batch_kwargs in self._iter_loader(data_loader, device):
                y = batch_kwargs.get("y", None)

                z = self.encode(x, y=y, timestep=timestep)
                latents.append(z.detach().cpu().numpy())

                if y is not None:
                    labels.append(y.detach().cpu().numpy())

                if return_reconstructions:
                    t = torch.full((x.size(0),), int(timestep), device=device, dtype=torch.long)
                    x_t = self.q_sample(x, t)
                    recon = self.p_sample_loop(x.size(0), device, init_x=x_t, y=y)
                    reconstructions.append(recon.detach().cpu().numpy())

        result = {"latent": np.concatenate(latents, axis=0)}
        if len(labels) > 0:
            result["labels"] = np.concatenate(labels, axis=0)
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(reconstructions, axis=0)
        return result


def create_scdiffusion_model(
    input_dim: int,
    latent_dim: int = 128,
    n_classes: int = 0,
    embedding_dim: int = 10,
    **kwargs,
):
    """
    Create scDiffusion model
    
    Example:
        >>> model = create_scdiffusion_model(2000, latent_dim=128, embedding_dim=10, n_classes=5)
    """
    return scDiffusionModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_classes=n_classes,
        embedding_dim=embedding_dim,
        **kwargs,
    )