"""
GM-VAE: Geometric Manifold Variational Autoencoder
Supports 5 geometric distributions: Euclidean, Poincaré, PGM, LearnablePGM, HW
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple
import numpy as np
from .base_model import BaseModel

from ..distributions.EuclideanNormal import (
    Distribution as EuclideanDistribution,
    VanillaEncoderLayer as EuclideanEncoderLayer,
    VanillaDecoderLayer as EuclideanDecoderLayer,
    get_prior as get_euclidean_prior
)

from ..distributions.PoincareNormal import (
    Distribution as PoincareDistribution,
    VanillaEncoderLayer as PoincareEncoderLayer,
    VanillaDecoderLayer as PoincareDecoderLayer,
    get_prior as get_poincare_prior
)

from ..distributions.PGMNormal import (
    Distribution as PGMDistribution,
    VanillaEncoderLayer as PGMVanillaEncoderLayer,
    GeoEncoderLayer as PGMGeoEncoderLayer,
    VanillaDecoderLayer as PGMVanillaDecoderLayer,
    GeoDecoderLayer as PGMGeoDecoderLayer,
    get_prior as get_pgm_prior
)

from ..distributions.LearnablePGMNormal import (
    Distribution as LearnablePGMDistribution,
    VanillaEncoderLayer as LearnablePGMVanillaEncoderLayer,
    ExpEncoderLayer as LearnablePGMExpEncoderLayer,
    VanillaDecoderLayer as LearnablePGMVanillaDecoderLayer,
    LogDecoderLayer as LearnablePGMLogDecoderLayer,
)

from ..distributions.HWNormal import (
    Distribution as HWDistribution,
    VanillaEncoderLayer as HWEncoderLayer,
    VanillaDecoderLayer as HWDecoderLayer,
    get_prior as get_hw_prior
)


class SimpleArgs:
    """Argument container for distribution modules"""
    def __init__(self, latent_dim, device, c=-1.0):
        self.latent_dim = latent_dim
        self.device = device
        self.c = c


def get_learnable_pgm_prior_fixed(args):
    """Fixed prior for LearnablePGM: [1, D, 3] for (alpha, log_beta_square, log_c)"""
    mean = torch.zeros([1, args.latent_dim, 3], device=args.device)
    mean[..., 0] = 0.0  # alpha
    mean[..., 1] = 0.0  # log_beta_square
    mean[..., 2] = torch.log(torch.tensor(abs(args.c), device=args.device))  # log_c
    covar = torch.full([1, args.latent_dim], 0.0, device=args.device)
    return LearnablePGMDistribution(mean, covar)


DISTRIBUTION_CONFIG = {
    'euclidean': {
        'distribution_class': EuclideanDistribution,
        'encoder_layers': {'Vanilla': EuclideanEncoderLayer},
        'decoder_layers': {'Vanilla': EuclideanDecoderLayer},
        'get_prior': get_euclidean_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,
        'loc_shape': 'flat',
        'scale_shape': 'flat',
        'sample_shape': 'flat',
        'requires_even': False,
        'internal_dim_factor': 1.0,
        'decoder_output_shape': 'doubled_flat',
    },
    'poincare': {
        'distribution_class': PoincareDistribution,
        'encoder_layers': {'Vanilla': PoincareEncoderLayer},
        'decoder_layers': {'Vanilla': PoincareDecoderLayer},
        'get_prior': get_poincare_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,
        'loc_shape': '2d',
        'scale_shape': 'vector',
        'sample_shape': '2d',
        'requires_even': True,
        'internal_dim_factor': 0.5,
        'decoder_output_shape': 'geometry_2d',
    },
    'pgm': {
        'distribution_class': PGMDistribution,
        'encoder_layers': {
            'Vanilla': PGMVanillaEncoderLayer,
            'Geo': PGMGeoEncoderLayer
        },
        'decoder_layers': {
            'Vanilla': PGMVanillaDecoderLayer,
            'Geo': PGMGeoDecoderLayer
        },
        'get_prior': get_pgm_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,
        'loc_shape': '2d',
        'scale_shape': 'vector',
        'sample_shape': '2d',
        'requires_even': True,
        'internal_dim_factor': 0.5,
        'decoder_output_shape': 'geometry_2d',
    },
    'learnable_pgm': {
        'distribution_class': LearnablePGMDistribution,
        'encoder_layers': {
            'Vanilla': LearnablePGMVanillaEncoderLayer,
            'Exp': LearnablePGMExpEncoderLayer
        },
        'decoder_layers': {
            'Vanilla': LearnablePGMVanillaDecoderLayer,
            'Log': LearnablePGMLogDecoderLayer
        },
        'get_prior': get_learnable_pgm_prior_fixed,
        'default_layer': 'Vanilla',
        'param_count': 2,
        'loc_shape': '3d',
        'scale_shape': 'vector',
        'sample_shape': '2d',
        'requires_even': True,
        'internal_dim_factor': 0.5,
        'decoder_output_shape': 'flat',
    },
    'hw': {
        'distribution_class': HWDistribution,
        'encoder_layers': {'Vanilla': HWEncoderLayer},
        'decoder_layers': {'Vanilla': HWDecoderLayer},
        'get_prior': get_hw_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,
        'loc_shape': '3d',
        'scale_shape': '2d',
        'sample_shape': '3d',
        'requires_even': True,
        'internal_dim_factor': 0.5,
        'decoder_output_shape': 'geometry_3d',
    },
}


class GMVAEEncoder(nn.Module):
    """Encoder supporting 5 geometric distributions"""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 distribution: str = 'euclidean', layer_type: str = None,
                 device='cuda', c=-1.0):
        super().__init__()
        
        if distribution not in DISTRIBUTION_CONFIG:
            raise ValueError(f"Unknown distribution: {distribution}. "
                           f"Choose from {list(DISTRIBUTION_CONFIG.keys())}")
        
        self.distribution = distribution
        self.config = DISTRIBUTION_CONFIG[distribution]
        
        if self.config['requires_even'] and latent_dim % 2 != 0:
            raise ValueError(f"{distribution} requires even latent_dim (got {latent_dim})")
        
        if layer_type is None:
            layer_type = self.config['default_layer']
        
        if layer_type not in self.config['encoder_layers']:
            raise ValueError(f"Invalid layer_type '{layer_type}' for {distribution}. "
                           f"Choose from {list(self.config['encoder_layers'].keys())}")
        
        self.layer_type = layer_type
        self.latent_dim = latent_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = prev_dim
        
        internal_latent_dim = int(latent_dim * self.config['internal_dim_factor'])
        args = SimpleArgs(internal_latent_dim, device, c)
        
        encoder_class = self.config['encoder_layers'][layer_type]
        self.variational_layer = encoder_class(args, self.feature_dim)
    
    def forward(self, x):
        """Returns (loc, scale) distribution parameters"""
        features = self.feature_extractor(x)
        return self.variational_layer(features)


class GMVAEDecoder(nn.Module):
    """Decoder supporting 5 geometric distributions"""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 distribution: str = 'euclidean', layer_type: str = None,
                 device='cuda', c=-1.0, loss_type: str = 'MSE'):
        super().__init__()
        
        if distribution not in DISTRIBUTION_CONFIG:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        self.distribution = distribution
        self.config = DISTRIBUTION_CONFIG[distribution]
        
        if self.config['requires_even'] and latent_dim % 2 != 0:
            raise ValueError(f"{distribution} requires even latent_dim (got {latent_dim})")
        
        if layer_type is None:
            layer_type = self.config['default_layer']
        
        if layer_type not in self.config['decoder_layers']:
            raise ValueError(f"Invalid layer_type '{layer_type}' for {distribution}. "
                           f"Choose from {list(self.config['decoder_layers'].keys())}")
        
        self.layer_type = layer_type
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        
        internal_latent_dim = int(latent_dim * self.config['internal_dim_factor'])
        args = SimpleArgs(internal_latent_dim, device, c)
        
        decoder_class = self.config['decoder_layers'][layer_type]
        self.decode_layer = decoder_class(args)
        
        decoder_output_shape = self.config.get('decoder_output_shape', 'flat')
        
        if decoder_output_shape == 'doubled_flat':
            decoder_input_dim = latent_dim
        elif decoder_output_shape == 'geometry_2d':
            decoder_input_dim = latent_dim
        elif decoder_output_shape == 'geometry_3d':
            decoder_input_dim = latent_dim
        elif decoder_output_shape == 'flat':
            decoder_input_dim = latent_dim
        else:
            decoder_input_dim = latent_dim
        
        layers = []
        prev_dim = decoder_input_dim
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        output_features = output_dim * 2 if loss_type == 'NLL' else output_dim
        layers.append(nn.Linear(prev_dim, output_features))
        
        self.decoder_net = nn.Sequential(*layers)
    
    def forward(self, z):
        """Decode from geometry-specific latent representation"""
        z_decoded = self.decode_layer(z)
        
        decoder_output_shape = self.config.get('decoder_output_shape', 'flat')
        batch_size = z_decoded.size(0) if z_decoded.dim() > 0 else 1
        
        if decoder_output_shape == 'doubled_flat':
            if z_decoded.dim() == 2:
                z_decoded = z_decoded[..., :z_decoded.size(-1)//2]
        
        elif decoder_output_shape == 'geometry_2d':
            if z_decoded.dim() == 3:
                z_decoded = z_decoded.reshape(batch_size, -1)
        
        elif decoder_output_shape == 'geometry_3d':
            if z_decoded.dim() == 3:
                z_decoded = z_decoded[..., :2]
                z_decoded = z_decoded.reshape(batch_size, -1)
        
        elif decoder_output_shape == 'flat':
            if z_decoded.dim() > 2:
                z_decoded = z_decoded.reshape(batch_size, -1)
        
        if z_decoded.dim() > 2:
            z_decoded = z_decoded.reshape(batch_size, -1)
        
        return self.decoder_net(z_decoded)


class GMVAEModel(BaseModel):
    """
    Geometric Manifold VAE supporting 5 geometric spaces:
    - Euclidean: Standard Euclidean space
    - Poincaré: Poincaré ball (hyperbolic)
    - PGM: Pseudo-Gaussian Manifold
    - LearnablePGM: PGM with learnable curvature
    - HW: Hyperboloid Wrapped Normal (Lorentz model)
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 distribution: str = 'euclidean',
                 encoder_layer: str = None,
                 decoder_layer: str = None,
                 curvature: float = -1.0,
                 loss_type: str = 'MSE',
                 model_name: str = "GMVAE"):
        """
        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension (must be even for non-Euclidean)
                - Euclidean: directly uses D dimensions
                - Poincaré/PGM/LearnablePGM: D dimensions = D//2 points × 2 coords
                - HW: D dimensions = D//2 points × 3 Lorentz coords
            hidden_dims: Hidden layer dimensions
            distribution: 'euclidean', 'poincare', 'pgm', 'learnable_pgm', 'hw'
            encoder_layer: Encoder layer type (None = default)
            decoder_layer: Decoder layer type (None = default)
            curvature: Curvature parameter (for PGM/LearnablePGM/HW)
            loss_type: 'BCE', 'MSE', or 'NLL'
        """
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        if distribution not in DISTRIBUTION_CONFIG:
            raise ValueError(f"Unknown distribution: {distribution}. "
                           f"Choose from {list(DISTRIBUTION_CONFIG.keys())}")
        
        config = DISTRIBUTION_CONFIG[distribution]
        
        if config['requires_even'] and latent_dim % 2 != 0:
            raise ValueError(f"latent_dim must be even for {distribution} (got {latent_dim})")
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.distribution = distribution
        self.config = config
        self.encoder_layer = encoder_layer or config['default_layer']
        self.decoder_layer = decoder_layer or config['default_layer']
        self.curvature = curvature
        self.loss_type = loss_type
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder_net = GMVAEEncoder(
            input_dim, hidden_dims, latent_dim,
            distribution, self.encoder_layer, self.device, curvature
        )
        
        self.decoder_net = GMVAEDecoder(
            latent_dim, hidden_dims, input_dim,
            distribution, self.decoder_layer, self.device, curvature, loss_type
        )
        
        internal_latent_dim = int(latent_dim * config['internal_dim_factor'])
        args = SimpleArgs(internal_latent_dim, self.device, curvature)
        
        self.prior = config['get_prior'](args)
    
    def _create_distribution(self, *params):
        """Create distribution from parameters"""
        return self.config['distribution_class'](*params)
    
    def _reshape_for_output(self, z):
        """Reshape geometry-specific samples to [B, latent_dim]"""
        if z.dim() == 2:
            return z
        
        batch_size = z.size(0)
        
        if self.config['sample_shape'] == '2d':
            return z.reshape(batch_size, -1)
        elif self.config['sample_shape'] == '3d':
            if self.distribution == 'hw':
                return z[..., :2].reshape(batch_size, -1)
            return z.reshape(batch_size, -1)
        
        return z.reshape(batch_size, -1)
    
    def _reshape_for_decoder(self, z):
        """Reshape [B, latent_dim] to geometry-specific format"""
        batch_size = z.size(0)
        
        if self.config['sample_shape'] == 'flat':
            return z
        elif self.config['sample_shape'] == '2d':
            return z.reshape(batch_size, -1, 2)
        elif self.config['sample_shape'] == '3d':
            if self.distribution == 'hw':
                z_2d = z.reshape(batch_size, -1, 2)
                z_3d = torch.cat([
                    z_2d,
                    torch.zeros(batch_size, z_2d.size(1), 1, device=z.device)
                ], dim=-1)
                return z_3d
            return z.reshape(batch_size, -1, 3)
        
        return z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to [B, latent_dim]"""
        params = self.encoder_net(x)
        variational = self._create_distribution(*params)
        z = variational.rsample(1).squeeze(0)
        return self._reshape_for_output(z)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from [B, latent_dim]"""
        z = self._reshape_for_decoder(z)
        output = self.decoder_net(z)
        
        if self.loss_type == 'NLL':
            return output[..., :output.size(-1)//2]
        return output
    
    def forward(self, x: torch.Tensor, n_samples: int = 1, beta: float = 1.0, 
                iwae: int = 0, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with optional IWAE sampling"""
        params = self.encoder_net(x)
        variational = self._create_distribution(*params)
        
        z = variational.rsample(n_samples)
        
        if n_samples > 1:
            original_shape = z.shape
            
            if z.dim() == 3:
                z_flat = z.reshape(-1, z.size(-1))
                z_reshaped = z_flat
            elif z.dim() == 4:
                z_flat = z.reshape(-1, original_shape[-2], original_shape[-1])
                z_reshaped = z_flat
            else:
                z_flat = z.reshape(n_samples * x.size(0), -1)
                z_reshaped = self._reshape_for_decoder(z_flat)
            
            x_generated = self.decoder_net(z_reshaped)
            x_generated = x_generated.view(n_samples, x.size(0), -1)
        else:
            z = z.squeeze(0)
            x_generated = self.decoder_net(z)
        
        return {
            'reconstruction': x_generated,
            'latent': z,
            'variational': variational,
            'params': params
        }
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     beta: float = 1.0, n_samples: int = 1, iwae: int = 0,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """Compute VAE loss (reconstruction + KL) with optional IWAE"""
        x_generated = outputs['reconstruction']
        z = outputs['latent']
        variational = outputs['variational']
        
        if self.loss_type == 'BCE':
            if n_samples > 1:
                recon_loss = F.binary_cross_entropy_with_logits(
                    x_generated, x.unsqueeze(0).expand(x_generated.size()), reduction='mean'
                )
            else:
                recon_loss = F.binary_cross_entropy_with_logits(x_generated, x, reduction='mean')
        elif self.loss_type == 'MSE':
            if n_samples > 1:
                recon_loss = F.mse_loss(
                    x_generated, x.unsqueeze(0).expand(x_generated.size()), reduction='mean'
                )
            else:
                recon_loss = F.mse_loss(x_generated, x, reduction='mean')
        elif self.loss_type == 'NLL':
            if n_samples > 1:
                mu = x_generated[..., :x.size(-1)]
                logvar = x_generated[..., x.size(-1):]
                recon_loss = 0.5 * ((x.unsqueeze(0) - mu).pow(2) / logvar.exp() + logvar).mean()
            else:
                mu = x_generated[..., :x.size(-1)]
                logvar = x_generated[..., x.size(-1):]
                recon_loss = 0.5 * ((x - mu).pow(2) / logvar.exp() + logvar).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Prepare z for KL computation (manifold distributions need [K, B, ...] shape)
        z_kl = z
        if (self.distribution in ('poincare', 'pgm', 'learnable_pgm', 'hw')
                and n_samples == 1
                and z.dim() >= 2
                and z.shape[0] == x.shape[0]):
            z_kl = z.unsqueeze(0)
        
        if iwae == 0 or n_samples == 1:
            if hasattr(variational, 'kl_div') and variational.kl_div is not None:
                kl_div = variational.kl_div
                kl_loss = kl_div(self.prior).mean() if callable(kl_div) else kl_div.mean()
            else:
                log_q = variational.log_prob(z_kl)
                log_p = self.prior.log_prob(z_kl)
                
                if log_q.dim() > 1:
                    kl_loss = (log_q - log_p).sum(dim=-1).mean()
                else:
                    kl_loss = (log_q - log_p).mean()
            
            total_loss = recon_loss + beta * kl_loss
            recon_loss_sum = recon_loss
            kl_loss_sum = kl_loss
        else:
            log_q = variational.log_prob(z)
            log_p = self.prior.log_prob(z)
            
            if log_q.dim() > 2:
                log_q = log_q.sum(dim=-1)
                log_p = log_p.sum(dim=-1)
            
            kl_loss = log_q - log_p
            total_loss_sum = -recon_loss - beta * kl_loss
            
            loss = total_loss_sum.logsumexp(dim=0)
            loss = loss - np.log(n_samples)
            total_loss = -loss.mean()
            
            recon_loss_sum = recon_loss.mean(dim=0).sum()
            kl_loss_sum = kl_loss.mean(dim=0).sum()
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss_sum,
            'kl_loss': kl_loss_sum
        }


def create_gmvae_model(input_dim: int, latent_dim: int = 10, 
                       distribution: str = 'euclidean', **kwargs):
    """
    Create GM-VAE model
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension (must be even for non-Euclidean)
        distribution: 'euclidean', 'poincare', 'pgm', 'learnable_pgm', 'hw'
    
    Examples:
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='euclidean')
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='poincare')
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='learnable_pgm',
        ...                           encoder_layer='Exp', decoder_layer='Log')
    """
    return GMVAEModel(input_dim=input_dim, latent_dim=latent_dim, 
                     distribution=distribution, **kwargs)