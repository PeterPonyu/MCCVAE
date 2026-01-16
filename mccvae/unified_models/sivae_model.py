"""
siVAE: Supervised interpretable VAE for gene regulatory network inference
Key: Interpretable linear encoder + supervised classification

Reference: Kopf et al. (2021) Mixture-of-Experts VAE for clustering and 
generating from similarity-based representations on single cell data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np
from .base_model import BaseModel


class InterpretableLinearEncoder(nn.Module):
    """Linear encoder with sparsity constraints for gene-factor interpretability"""
    def __init__(self, input_dim: int, latent_dim: int, 
                 constraint: str = 'l1', constraint_weight: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.constraint = constraint
        self.constraint_weight = constraint_weight
        
        self.fc = nn.Linear(input_dim, latent_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight, gain=0.01)
    
    def forward(self, x):
        return self.fc(x)
    
    def get_constraint_loss(self):
        """Sparsity/smoothness constraint for interpretability"""
        if self.constraint == 'l1':
            return self.constraint_weight * torch.abs(self.fc.weight).sum()
        elif self.constraint == 'l2':
            return self.constraint_weight * (self.fc.weight ** 2).sum()
        else:
            return 0.0


class siVAEEncoder(nn.Module):
    """Encoder with optional interpretable linear layer"""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 use_interpretable: bool = True,
                 constraint: str = 'l1',
                 constraint_weight: float = 0.01,
                 batch_norm: bool = True, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.use_interpretable = use_interpretable
        self.batch_norm = batch_norm
        
        if use_interpretable:
            self.ile = InterpretableLinearEncoder(
                input_dim, latent_dim, constraint, constraint_weight
            )
            encoder_input_dim = latent_dim
        else:
            self.ile = None
            encoder_input_dim = input_dim
        
        if len(hidden_dims) > 0:
            layers = []
            prev_dim = encoder_input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            self.encoder = nn.Sequential(*layers)
            self.fc_mu = nn.Linear(prev_dim, latent_dim)
            self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        else:
            self.encoder = nn.Identity()
            self.fc_mu = nn.Identity()
            self.fc_logvar = nn.Linear(encoder_input_dim, latent_dim)
        
        if isinstance(self.fc_logvar, nn.Linear):
            with torch.no_grad():
                self.fc_logvar.weight.fill_(0.0)
                self.fc_logvar.bias.fill_(-5.0)
    
    def forward(self, x):
        if self.use_interpretable:
            ile_output = self.ile(x)
            h = self.encoder(ile_output)
            
            if isinstance(self.fc_mu, nn.Identity):
                mu = ile_output
            else:
                mu = self.fc_mu(h)
            
            logvar = self.fc_logvar(h)
            return mu, logvar, ile_output
        else:
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar, None


class siVAEDecoder(nn.Module):
    """Decoder with ZINB/NB/Gaussian output"""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 batch_norm: bool = True, dropout: float = 0.1,
                 output_distribution: str = 'nb'):
        super().__init__()
        
        self.batch_norm = batch_norm
        self.output_distribution = output_distribution
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        if output_distribution == 'zinb':
            self.fc_mean = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softmax(dim=-1)
            )
            self.fc_disp = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softplus()
            )
            self.fc_pi = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Sigmoid()
            )
        elif output_distribution == 'nb':
            self.fc_mean = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softmax(dim=-1)
            )
            self.fc_disp = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softplus()
            )
        elif output_distribution == 'gaussian':
            self.fc_mu = nn.Linear(prev_dim, output_dim)
            self.fc_logvar = nn.Linear(prev_dim, output_dim)
        else:
            raise ValueError(f"Unknown distribution: {output_distribution}")
    
    def forward(self, z):
        h = self.decoder(z)
        if self.output_distribution == 'zinb':
            mean = self.fc_mean(h)
            disp = self.fc_disp(h)
            pi = self.fc_pi(h)
            return {'mean': mean, 'disp': disp, 'pi': pi}
        elif self.output_distribution == 'nb':
            mean = self.fc_mean(h)
            disp = self.fc_disp(h)
            return {'mean': mean, 'disp': disp}
        elif self.output_distribution == 'gaussian':
            x_mu = self.fc_mu(h)
            x_logvar = self.fc_logvar(h)
            return {'x_mu': x_mu, 'x_logvar': x_logvar}


class SupervisedClassifier(nn.Module):
    """Predicts cell type labels from latent (supervised component)"""
    def __init__(self, latent_dim: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, z):
        return self.classifier(z)


class siVAEModel(BaseModel):
    """
    siVAE: Supervised interpretable VAE
    
    Features:
    - Interpretable Linear Encoder: sparse gene-factor mapping
    - Supervised classification: cell type prediction from latent
    - Library size normalization for count data
    - ZINB/NB/Gaussian reconstruction
    - Gene relevance scoring for GRN inference
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 50,
                 hidden_dims: list = None,
                 n_classes: int = 0,
                 use_interpretable: bool = True,
                 constraint: str = 'l1',
                 constraint_weight: float = 0.01,
                 batch_norm: bool = True,
                 dropout: float = 0.1,
                 output_distribution: str = 'nb',
                 use_batch: bool = False,
                 n_batches: int = 1,
                 supervised_weight: float = 1.0,
                 model_name: str = "siVAE"):
        """
        Args:
            input_dim: Number of genes
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions
            n_classes: Number of cell types (0 = unsupervised)
            use_interpretable: Use interpretable linear encoder
            constraint: 'l1' (sparsity) or 'l2' (smoothness)
            constraint_weight: Constraint loss weight
            output_distribution: 'zinb', 'nb', or 'gaussian'
            use_batch: Batch correction
            n_batches: Number of batches
            supervised_weight: Supervised loss weight
        """
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.n_classes = n_classes
        self.use_interpretable = use_interpretable
        self.constraint = constraint
        self.constraint_weight = constraint_weight
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_distribution = output_distribution
        self.use_batch = use_batch
        self.n_batches = n_batches
        self.supervised_weight = supervised_weight
        
        self.encoder_net = siVAEEncoder(
            input_dim, hidden_dims, latent_dim,
            use_interpretable, constraint, constraint_weight,
            batch_norm, dropout
        )
        
        decoder_input_dim = latent_dim
        if use_batch:
            decoder_input_dim += n_batches
        
        self.decoder_net = siVAEDecoder(
            decoder_input_dim, hidden_dims, input_dim,
            batch_norm, dropout, output_distribution
        )
        
        if n_classes > 0:
            self.classifier = SupervisedClassifier(latent_dim, n_classes)
        else:
            self.classifier = None
        
        if use_batch:
            self.batch_embedding = nn.Embedding(n_batches, n_batches)
            self.batch_embedding.weight.data = torch.eye(n_batches)
            self.batch_embedding.weight.requires_grad = False
    
    def _prepare_batch(self, batch_data, device):
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            metadata = {}

            if len(batch_data) >= 2 and torch.is_tensor(batch_data[1]):
                second_item = batch_data[1]

                if torch.is_floating_point(second_item) and second_item.shape == x.shape:
                    metadata["x_raw"] = second_item.to(device).float()

                elif second_item.dtype in (torch.int32, torch.int64) and second_item.ndim == 1:
                    if second_item.max() < self.n_batches and self.use_batch:
                        metadata["batch_id"] = second_item.to(device).long()
                    elif self.n_classes > 0 and second_item.max() < self.n_classes:
                        metadata["labels"] = second_item.to(device).long()

            if len(batch_data) >= 3 and torch.is_tensor(batch_data[2]):
                third_item = batch_data[2]
                if third_item.dtype in (torch.int32, torch.int64) and third_item.ndim == 1:
                    if self.n_classes > 0 and third_item.max() < self.n_classes:
                        metadata["labels"] = third_item.to(device).long()

            if self.use_batch and "batch_id" not in metadata:
                metadata["batch_id"] = torch.zeros(x.size(0), dtype=torch.long, device=device)

            return x, metadata

        x = batch_data.to(device).float()
        metadata = {}
        if self.use_batch:
            metadata["batch_id"] = torch.zeros(x.size(0), dtype=torch.long, device=device)
        return x, metadata
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space [B, latent_dim]"""
        mu, logvar, _ = self.encoder_net(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z: torch.Tensor, batch_id: Optional[torch.Tensor] = None):
        """Decode latent to gene expression"""
        if self.use_batch and batch_id is not None:
            batch_emb = self.batch_embedding(batch_id)
            z = torch.cat([z, batch_emb], dim=-1)

        decoder_output = self.decoder_net(z)

        if self.output_distribution in ['zinb', 'nb']:
            return decoder_output['mean']
        else:
            return decoder_output['x_mu']
    
    def forward(self, x: torch.Tensor, batch_id: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, x_raw: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass with library size normalization"""
        x_counts = x_raw if x_raw is not None else x

        library_size = x_counts.sum(dim=1, keepdim=True)
        x_norm = x_counts / (library_size + 1e-6)

        mu_z, logvar_z, ile_output = self.encoder_net(x_norm)
        z = self.reparameterize(mu_z, logvar_z)

        decoder_input = z
        if self.use_batch and batch_id is not None:
            batch_emb = self.batch_embedding(batch_id)
            decoder_input = torch.cat([z, batch_emb], dim=-1)

        decoder_output = self.decoder_net(decoder_input)

        logits = self.classifier(z) if (self.classifier is not None and labels is not None) else None

        output = {
            'latent': z,
            'z_mu': mu_z,
            'z_logvar': logvar_z,
            'library_size': library_size,
            'ile_output': ile_output,
            'logits': logits
        }
        output.update(decoder_output)
        return output
    
    def _zinb_loss(self, x: torch.Tensor, mean: torch.Tensor,
                   disp: torch.Tensor, pi: torch.Tensor,
                   library_size: torch.Tensor) -> torch.Tensor:
        """Zero-Inflated Negative Binomial loss"""
        eps = 1e-10
        mean_scaled = mean * library_size
        
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean_scaled / (disp + eps))) + \
             (x * (torch.log(disp + eps) - torch.log(mean_scaled + eps)))
        nb_case = t1 + t2 - torch.log(1.0 - pi + eps)
        
        zero_nb = torch.pow(disp / (disp + mean_scaled + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        
        result = torch.where(x < 1e-8, zero_case, nb_case)
        return result.mean()
    
    def _nb_loss(self, x: torch.Tensor, mean: torch.Tensor,
                 disp: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """Negative Binomial loss"""
        eps = 1e-10
        mean_scaled = mean * library_size
        
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean_scaled / (disp + eps))) + \
             (x * (torch.log(disp + eps) - torch.log(mean_scaled + eps)))
        
        return (t1 + t2).mean()
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], beta: float = 1, labels: Optional[torch.Tensor] = None, x_raw: Optional[torch.Tensor] = None, **kwargs):
        """Compute loss: reconstruction + KL + supervised + constraint"""
        x_counts = x_raw if x_raw is not None else x
        z_mu = outputs['z_mu']
        z_logvar = outputs['z_logvar']
        library_size = outputs['library_size']

        z_logvar_clamped = torch.clamp(z_logvar, min=-5, max=5)
        kl_loss = -0.5 * torch.sum(1 + z_logvar_clamped - z_mu.pow(2) - z_logvar_clamped.exp()) / x.size(0)
        kl_loss = torch.clamp(kl_loss, min=0.0)

        if self.output_distribution == 'zinb':
            recon_loss = self._zinb_loss(x_counts, outputs['mean'], outputs['disp'], outputs['pi'], library_size)
        elif self.output_distribution == 'nb':
            recon_loss = self._nb_loss(x_counts, outputs['mean'], outputs['disp'], library_size)
        else:
            recon_loss = F.mse_loss(outputs['x_mu'], x_counts, reduction='mean')

        supervised_loss = F.cross_entropy(outputs['logits'], labels) if (self.classifier is not None and labels is not None and outputs['logits'] is not None) else torch.tensor(0.0, device=x.device)
        constraint_loss = self.encoder_net.ile.get_constraint_loss() if (self.use_interpretable and self.encoder_net.ile is not None) else torch.tensor(0.0, device=x.device)

        total_loss = recon_loss + beta * kl_loss + self.supervised_weight * supervised_loss + constraint_loss
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'supervised_loss': supervised_loss,
            'constraint_loss': constraint_loss
        }
    
    def compute_gene_relevance(self, x: torch.Tensor, 
                              latent_dim_idx: int = 0) -> torch.Tensor:
        """
        Compute gene relevance scores for GRN inference
        
        Returns gradient-based importance of each gene for a latent factor
        """
        x.requires_grad = True
        
        mu_z, logvar_z, ile_output = self.encoder_net(x)
        target_latent = mu_z[:, latent_dim_idx].sum()
        target_latent.backward()
        
        relevance = torch.abs(x.grad).mean(dim=0)
        
        return relevance
    
    def extract_latent(self, data_loader, device: str = "cuda", return_reconstructions: bool = False):
        return super().extract_latent(
            data_loader=data_loader,
            device=device,
            return_reconstructions=return_reconstructions,
        )


def create_sivae_model(input_dim: int, latent_dim: int = 50,
                      n_classes: int = 0, **kwargs):
    """
    Create siVAE model
    
    Example:
        >>> model = create_sivae_model(2000, latent_dim=50, n_classes=10, 
        ...                            use_interpretable=True, output_distribution='zinb')
    """
    return siVAEModel(input_dim=input_dim, latent_dim=latent_dim,
                     n_classes=n_classes, **kwargs)