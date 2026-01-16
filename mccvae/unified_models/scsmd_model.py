"""
scSMD: ResNet-based autoencoder with clustering via mutual information
Reshapes 1D gene data → 2D images for CNN processing
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Optional, Any
from sklearn.cluster import KMeans
from scipy.sparse import issparse
from .base_model import BaseModel

class myBottleneck(nn.Module):
    """ResNet bottleneck block"""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class NBLoss(nn.Module):
    """Negative Binomial loss"""
    def forward(self, x, mean, disp):
        eps = 1e-8

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        mean = torch.nan_to_num(mean, nan=eps, posinf=1e6, neginf=eps).clamp(min=eps, max=1e6)
        disp = torch.nan_to_num(disp, nan=eps, posinf=1e4, neginf=eps).clamp(min=eps, max=1e4)

        # compute in float64 for stability
        x64, mean64, disp64 = x.double(), mean.double(), disp.double()
        t1 = torch.lgamma(disp64 + eps) + torch.lgamma(x64 + 1.0) - torch.lgamma(x64 + disp64 + eps)
        t2 = (disp64 + x64) * torch.log1p(mean64 / (disp64 + eps)) + x64 * (torch.log(disp64 + eps) - torch.log(mean64 + eps))
        out = t1 + t2
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out.mean().to(x.dtype)


class AutoEncoder(nn.Module):
    """ResNet-based convolutional autoencoder"""
    def __init__(self, block, layers, input_dim, latent_dim, img_size: int):
        super().__init__()
        self.img_size = img_size
        self.adjusted_input_dim = img_size * img_size
        self.latent_dim = latent_dim

        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.flatten_size = self._get_flatten_size()
        self.conv_feat_shape = self._get_conv_feat_shape()
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)

        self.fc_mean = nn.Linear(latent_dim, self.adjusted_input_dim)
        self.fc_disp = nn.Linear(latent_dim, self.adjusted_input_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)

        self.decoder_channels = 256 * 4
        self.upsample1 = nn.ConvTranspose2d(self.decoder_channels, 128, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=7, padding=3)

    def _get_conv_feat_shape(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.img_size, self.img_size)
            x = self.relu(self.bn1(self.conv1(dummy)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return (x.size(1), x.size(2), x.size(3))
        
    def decode_latent(self, u: torch.Tensor) -> torch.Tensor:
        """Decode latent to image [B, 1, img_size, img_size]"""
        b = u.size(0)
        c, h, w = self.conv_feat_shape
        dec = self.fc_decode(u).view(b, c, h, w)

        dec = torch.relu(self.upsample1(dec))
        dec = torch.relu(self.upsample2(dec))
        dec = torch.relu(self.upsample3(dec))
        dec = torch.relu(self.upsample4(dec))
        y = torch.sigmoid(self.final_conv(dec))

        if y.size(2) != self.img_size or y.size(3) != self.img_size:
            y = nn.functional.interpolate(
                y, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
        return y       
     
    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, st))
            self.in_planes = planes * 4
        return nn.Sequential(*layers)

    def _get_flatten_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.img_size, self.img_size)
            x = self.relu(self.bn1(self.conv1(dummy)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x.numel()

    def forward(self, x):
        b = x.size(0)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(b, -1)
        u = self.fc_encode(out)
        
        eps = 1e-6
        mean = (F.softplus(self.fc_mean(u)) + eps).clamp(max=1e6)
        disp = (F.softplus(self.fc_disp(u)) + eps).clamp(max=1e4)

        y = self.decode_latent(u)

        return mean, disp, u, y


class MutualNet(nn.Module):
    """MLP for soft clustering via mutual information"""
    def __init__(self, num_cluster: int, latent_dim: int, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers = []
        d = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, num_cluster), nn.Softmax(dim=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class scSMDModel(BaseModel):
    """
    scSMD: ResNet autoencoder with mutual information clustering
    
    Features:
    - Reshapes 1D gene data [B, input_dim] → 2D images [B, 1, H, W]
    - ResNet encoder/decoder with bottleneck blocks
    - NB loss for count data reconstruction
    - Three-phase training:
      1. Pretrain: AE with reconstruction + NB
      2. Phase1: AE + cluster center pulling
      3. Phase2: Mutual information maximization
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        n_clusters: int = 10,
        img_size: Optional[int] = None,
        model_name: str = "scSMD",
    ):
        """
        Args:
            input_dim: Gene dimension
            latent_dim: Latent dimension
            n_clusters: Number of clusters
            img_size: Image size for reshaping (default: sqrt(input_dim))
        """
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[], model_name=model_name)
        self.n_clusters = n_clusters
        self.img_size = img_size or max(64, int(np.sqrt(input_dim)))
        self.adjusted_input_dim = self.img_size * self.img_size

        self.autoencoder = AutoEncoder(myBottleneck, layers=[1, 1, 1], input_dim=self.adjusted_input_dim, latent_dim=latent_dim, img_size=self.img_size)
        self.mutual_net = MutualNet(num_cluster=n_clusters, latent_dim=latent_dim)
        self.nb_loss = NBLoss()

        self.is_trained = False

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape and normalize [B, input_dim] → [B, 1, img_size, img_size]"""
        x = x - x.min(dim=1, keepdim=True).values
        denom = x.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        x = x / denom

        if x.size(1) < self.adjusted_input_dim:
            pad = torch.zeros(x.size(0), self.adjusted_input_dim - x.size(1), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif x.size(1) > self.adjusted_input_dim:
            x = x[:, :self.adjusted_input_dim]

        return x.view(-1, 1, self.img_size, self.img_size)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to latent space [B, latent_dim]"""
        x_img = self._preprocess(x)
        _, _, u, _ = self.autoencoder(x_img)
        return u

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latent → gene-space reconstruction [B, input_dim]"""
        y = self.autoencoder.decode_latent(z)
        flat = y.view(y.size(0), -1)
        return flat[:, : self.input_dim]
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        x_img = self._preprocess(x)
        mean, disp, u, y = self.autoencoder(x_img)
        return {"mean": mean, "disp": disp, "latent": u, "reconstruction_img": y, "input_img": x_img}

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], cluster_centers: Optional[torch.Tensor] = None, **kwargs):
        """Compute loss: MSE + NB (+ optional cluster pulling)"""
        x_img = outputs["input_img"]
        y = outputs["reconstruction_img"]
        u = outputs["latent"]

        x_flat = x_img.view(x_img.size(0), -1)
        y_flat = y.view(y.size(0), -1)

        loss_recon = nn.MSELoss()(y_flat, x_flat)
        loss_nb = self.nb_loss(x_flat, outputs["mean"], outputs["disp"])
        loss = loss_recon + 0.1 * loss_nb

        loss_cluster = torch.tensor(0.0, device=x.device)
        if cluster_centers is not None:
            dist = torch.cdist(u, cluster_centers)
            closest = torch.argmin(dist, dim=1)
            loss_cluster = nn.MSELoss()(u, cluster_centers[closest])
            loss = loss_recon + 0.1 * loss_cluster

        return {"total_loss": loss, "recon_loss": loss_recon, "nb_loss": loss_nb, "cluster_loss": loss_cluster}

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
        pretrain_epochs: int = 200,
        **kwargs,
    ):
        """
        Three-phase training:
        1. Pretrain: AE with reconstruction + NB loss
        2. Phase1: Add cluster center pulling
        3. Phase2: Mutual information optimization
        """
        self.to(device)
        opt_ae = optim.Adam(self.autoencoder.parameters(), lr=lr)

        # Phase 0: Pretrain
        for ep in range(pretrain_epochs):
            self.train()
            tot = 0.0
            n = 0
            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                out = self.forward(x, **batch_kwargs)
                loss = self.compute_loss(x, out, **batch_kwargs)["total_loss"]
                if not torch.isfinite(loss):
                    if verbose >= 1:
                        print("Non-finite loss encountered; skipping batch.")
                    opt_ae.zero_grad(set_to_none=True)
                    continue
                opt_ae.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                opt_ae.step()

                tot += float(loss.item())
                n += 1

            if verbose >= 1 and (ep + 1) % 50 == 0:
                print(f"Pretrain {ep+1:3d}/{pretrain_epochs} | Loss: {tot/max(n,1):.4f}")

        # Initialize clusters with KMeans
        self.eval()
        z_all = []
        with torch.no_grad():
            for batch in train_loader:
                x, _ = self._prepare_batch(batch, device)
                z_all.append(self.encode(x).cpu().numpy())
        z_all = np.concatenate(z_all, axis=0)
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42).fit(z_all)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)

        # Phase 1: AE + cluster center pulling
        phase1_epochs = max(50, (epochs - pretrain_epochs) // 2)
        for ep in range(phase1_epochs):
            self.train()
            tot = 0.0
            n = 0
            for batch in train_loader:
                x, _ = self._prepare_batch(batch, device)
                out = self.forward(x)
                loss = self.compute_loss(x, out, cluster_centers=centers)["total_loss"]
                opt_ae.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                opt_ae.step()
                tot += float(loss.item())
                n += 1
            if verbose >= 1 and (ep + 1) % 25 == 0:
                print(f"Phase1 {ep+1:3d}/{phase1_epochs} | Loss: {tot/max(n,1):.4f}")

        # Phase 2: Mutual information optimization
        phase2_epochs = max(0, epochs - pretrain_epochs - phase1_epochs)
        opt_mn = optim.Adam(self.mutual_net.parameters(), lr=lr)
        opt_ae2 = optim.Adam(self.autoencoder.parameters(), lr=lr)

        for ep in range(phase2_epochs):
            self.train()
            tot = 0.0
            n = 0
            for batch in train_loader:
                x, _ = self._prepare_batch(batch, device)
                u = self.encode(x)
                q = self.mutual_net(u)
                loss = -torch.mean(torch.sum(q * torch.log(q + 1e-8), dim=1))

                opt_ae2.zero_grad()
                opt_mn.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.mutual_net.parameters(), max_norm=1.0)
                opt_ae2.step()
                opt_mn.step()

                tot += float(loss.item())
                n += 1

            if verbose >= 1 and (ep + 1) % 10 == 0:
                print(f"Phase2 {ep+1:3d}/{phase2_epochs} | Loss: {tot/max(n,1):.4f}")

        self.is_trained = True
        return {"train_loss": []}


def create_scsmd_model(input_dim: int, latent_dim: int = 10, n_clusters: int = 10, **kwargs) -> scSMDModel:
    """
    Create scSMD model
    
    Example:
        >>> model = create_scsmd_model(2000, latent_dim=10, n_clusters=10, img_size=64)
    """
    return scSMDModel(input_dim=input_dim, latent_dim=latent_dim, n_clusters=n_clusters, **kwargs)