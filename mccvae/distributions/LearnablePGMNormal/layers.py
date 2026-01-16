import torch
import geoopt
from torch import nn
from torch.nn import functional as F


# ============================================================================
# Hyperbolic Geometry Coordinate Transformations
# Replacing dgnn.utils functions with standalone implementations
# ============================================================================

def halfplane2disk(x, c):
    """
    Convert from Poincaré half-plane to Poincaré disk model
    
    Half-plane: {(x1, x2) | x2 > 0}
    Disk: {x | ||x||^2 < 1/c}
    
    Args:
        x: [..., 2] tensor, half-plane coordinates (x1, x2 with x2 > 0)
        c: curvature parameter (positive scalar)
    
    Returns:
        [..., 2] tensor, disk coordinates
    """
    x1, x2 = x[..., 0:1], x[..., 1:2]
    
    # Transformation formula
    denominator = x1.pow(2) + (x2 + 1).pow(2)
    u1 = 2 * x1 / denominator
    u2 = (x1.pow(2) + x2.pow(2) - 1) / denominator
    
    disk_coords = torch.cat([u1, u2], dim=-1)
    
    # Scale by curvature
    disk_coords = disk_coords * torch.sqrt(c)
    
    return disk_coords


def disk2halfplane(x, c):
    """
    Convert from Poincaré disk to Poincaré half-plane model
    
    Inverse of halfplane2disk
    
    Args:
        x: [..., 2] tensor, disk coordinates
        c: curvature parameter (positive scalar)
    
    Returns:
        [..., 2] tensor, half-plane coordinates
    """
    # Unscale by curvature
    x = x / torch.sqrt(c)
    
    u1, u2 = x[..., 0:1], x[..., 1:2]
    
    # Transformation formula
    denominator = u1.pow(2) + (1 - u2).pow(2)
    x1 = 2 * u1 / denominator
    x2 = (1 - u1.pow(2) - u2.pow(2)) / denominator
    
    halfplane_coords = torch.cat([x1, x2], dim=-1)
    
    # Ensure x2 > 0 (numerical stability)
    halfplane_coords[..., 1:2] = torch.clamp(halfplane_coords[..., 1:2], min=1e-7)
    
    return halfplane_coords


def disk2lorentz(x, c):
    """
    Convert from Poincaré disk to Lorentz (hyperboloid) model
    
    Disk: {x | ||x||^2 < 1/c}
    Lorentz: {x | <x, x>_L = -1/c, x0 > 0}
    
    Args:
        x: [..., n] tensor, disk coordinates
        c: curvature parameter (positive scalar)
    
    Returns:
        [..., n+1] tensor, Lorentz coordinates
    """
    # Unscale by curvature
    x = x / torch.sqrt(c)
    
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    
    # Formula: (1 + ||x||^2) / (1 - ||x||^2), 2x / (1 - ||x||^2)
    x0 = (1 + x_norm_sq) / (1 - x_norm_sq + 1e-7)
    x_rest = 2 * x / (1 - x_norm_sq + 1e-7)
    
    lorentz_coords = torch.cat([x0, x_rest], dim=-1)
    
    # Scale by curvature
    lorentz_coords = lorentz_coords / torch.sqrt(c)
    
    return lorentz_coords


def lorentz2disk(x, c):
    """
    Convert from Lorentz (hyperboloid) to Poincaré disk model
    
    Inverse of disk2lorentz
    
    Args:
        x: [..., n+1] tensor, Lorentz coordinates
        c: curvature parameter (positive scalar)
    
    Returns:
        [..., n] tensor, disk coordinates
    """
    # Unscale by curvature
    x = x * torch.sqrt(c)
    
    x0 = x[..., 0:1]
    x_rest = x[..., 1:]
    
    # Formula: x_rest / (1 + x0)
    disk_coords = x_rest / (1 + x0 + 1e-7)
    
    # Scale by curvature
    disk_coords = disk_coords * torch.sqrt(c)
    
    return disk_coords


# ============================================================================
# Original Encoder/Decoder Layers
# ============================================================================

class EncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.feature_dim = feature_dim

        self.variational = nn.Linear(
            self.feature_dim,
            4 * self.latent_dim
        )

    def forward(self, feature):
        feature = self.variational(feature)
        alpha, beta, logc, gamma = torch.split(
            feature,
            [
                self.latent_dim, 
                self.latent_dim, 
                self.latent_dim,
                self.latent_dim
            ],
            dim=-1
        )

        return torch.stack([alpha, beta, logc], dim=-1), gamma


class VanillaEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()
        
        self.encoder = EncoderLayer(args, feature_dim)

    def forward(self, feature):
        return self.encoder(feature)


class ExpEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()
        
        self.c = torch.tensor([args.c], device=args.device)

        self.manifold = geoopt.manifolds.PoincareBall(-args.c)
        self.encoder = EncoderLayer(args, feature_dim)

    def forward(self, feature):
        mean, gamma = self.encoder(feature)
        mean = disk2halfplane(
            self.manifold.expmap0(mean),
            self.c
        )
        mean = torch.stack([mean[..., 0], mean[..., 1].log() * 2], dim=-1)

        return mean, gamma


class VanillaDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

    def forward(self, z):
        a, b = z[..., 0], (z[..., 1] * 0.5).exp()
        z = torch.concat([a, b], dim=-1)
        return z


class LogDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.c = torch.tensor([args.c], device=args.device)

        self.manifold = geoopt.manifolds.PoincareBall(-args.c)

    def forward(self, z):
        a, b = z[..., 0], (z[..., 1] * 0.5).exp()
        z = torch.stack([a, b], dim=-1)
        z = self.manifold.logmap0(
            halfplane2disk(z, self.c)
        )

        return z.reshape(*z.shape[:-2], -1)