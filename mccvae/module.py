
"""
MoCo Coupling Variational Autoencoder Implementation

This module provides a comprehensive MCCVAE implementation supporting:
- Multiple loss functions (MSE, Negative Binomial, Zero-Inflated Negative Binomial)
- Momentum Contrast (MoCo) for contrastive representation learning
- Information bottleneck for learning coupled latent representations that capture
  biological correlations intrinsic to single-cell data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union, Literal, Optional


class Encoder(nn.Module):
    """
    Variational encoder network that maps input states to latent distributions.

    This encoder learns to map single-cell expression profiles to a latent space
    that captures the underlying biological structure and relationships.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state space (number of genes/features)
    hidden_dim : int
        Dimension of hidden layers in the network
    action_dim : int
        Dimension of the latent space
    """

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int
    ):
        super().__init__()

        # Shared feature extraction network
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent distribution parameters (mean and log variance)
        self.latent_params = nn.Linear(hidden_dim, action_dim * 2)

        # Initialize network weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize network weights using Xavier normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.01)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, state_dim)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - sampled_latent: Sampled latent vectors
            - latent_mean: Mean of the latent distribution
            - latent_log_var: Log variance of the latent distribution
        """
        # Apply log1p transformation for numerical stability with count data
        x_transformed = torch.log1p(x)

        # Extract features through the base network
        hidden_features = self.feature_network(x_transformed)

        # Compute latent distribution parameters
        latent_output = self.latent_params(hidden_features)
        latent_mean, latent_log_var = torch.split(
            latent_output, latent_output.size(-1) // 2, dim=-1
        )

        # Ensure positive variance using softplus
        latent_std = F.softplus(latent_log_var)

        # Sample from the latent distribution
        latent_distribution = Normal(latent_mean, latent_std)
        sampled_latent = latent_distribution.rsample()

        return sampled_latent, latent_mean, latent_log_var


class Decoder(nn.Module):
    """
    Decoder network that maps latent vectors back to the original expression space.

    Supports three loss modes for different single-cell data characteristics:
    - 'mse': Mean Squared Error for normalized continuous data
    - 'nb': Negative Binomial for UMI count data
    - 'zinb': Zero-Inflated Negative Binomial for sparse count data

    Parameters
    ----------
    state_dim : int
        Dimension of the original expression space (number of genes)
    hidden_dim : int
        Dimension of hidden layers in the network
    action_dim : int
        Dimension of the latent space
    loss_mode : Literal['mse', 'nb', 'zinb'], default='nb'
        Loss function mode determining output structure
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
    ):
        super().__init__()
        self.loss_mode = loss_mode

        # Shared feature extraction network
        self.feature_network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Configure output layers based on loss mode
        if loss_mode in ["nb", "zinb"]:
            # Negative binomial parameters
            self.dispersion_param = nn.Parameter(torch.randn(state_dim))
            # Mean parameters with softmax normalization for count data
            self.mean_decoder = nn.Sequential(
                nn.Linear(hidden_dim, state_dim), 
                nn.Softmax(dim=-1)
            )
        else:  # MSE mode
            # Direct linear output for continuous data
            self.mean_decoder = nn.Linear(hidden_dim, state_dim)

        # Zero-inflation parameters (only for ZINB mode)
        if loss_mode == "zinb":
            self.dropout_decoder = nn.Linear(hidden_dim, state_dim)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize network weights using Xavier normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.01)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        x : torch.Tensor
            Latent vectors of shape (batch_size, action_dim)

        Returns
        -------
        For 'mse' and 'nb' modes:
            torch.Tensor: Reconstructed gene expression

        For 'zinb' mode:
            Tuple[torch.Tensor, torch.Tensor]: (reconstructed_mean, dropout_logits)
        """
        # Extract features
        hidden_features = self.feature_network(x)

        # Compute mean output
        reconstructed_mean = self.mean_decoder(hidden_features)

        # Return additional zero-inflation parameters for ZINB mode
        if self.loss_mode == "zinb":
            dropout_logits = self.dropout_decoder(hidden_features)
            return reconstructed_mean, dropout_logits

        return reconstructed_mean


class MomentumContrast(nn.Module):
    """
    Momentum Contrast (MoCo) for unsupervised representation learning in single-cell data.

    Implements the MoCo framework for learning representations through
    contrastive learning with a momentum-updated key encoder. This approach
    helps learn robust representations that capture biological similarities
    and differences between cells.

    Parameters
    ----------
    encoder_query : nn.Module
        Query encoder network
    encoder_key : nn.Module
        Key encoder network (momentum-updated)
    state_dim : int
        Dimension of input state (number of genes)
    embedding_dim : int, default=128
        Dimension of output embeddings
    queue_size : int, default=65536
        Size of the negative sample queue
    momentum : float, default=0.999
        Momentum coefficient for key encoder updates
    temperature : float, default=0.2
        Temperature parameter for contrastive loss
    device : torch.device
        Device for computation
    """

    def __init__(
        self,
        encoder_query: nn.Module,
        encoder_key: nn.Module,
        state_dim: int,
        embedding_dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.2,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.encoder_query = encoder_query
        self.encoder_key = encoder_key
        self.n_genes = state_dim

        # Initialize key encoder parameters with query encoder values
        for param_query, param_key in zip(
            self.encoder_query.parameters(), self.encoder_key.parameters()
        ):
            param_key.data.copy_(param_query.data)
            param_key.requires_grad = False  # Key encoder is not trained directly

        # Initialize queue for negative samples
        self.register_buffer(
            "queue", torch.randn(embedding_dim, queue_size, device=device)
        )
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer(
            "queue_ptr", torch.zeros(1, dtype=torch.long, device=device)
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Update key encoder parameters using momentum."""
        for param_query, param_key in zip(
            self.encoder_query.parameters(), self.encoder_key.parameters()
        ):
            param_key.data = (
                param_key.data * self.momentum + 
                param_query.data * (1.0 - self.momentum)
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """
        Update the queue with new key vectors.

        Parameters
        ----------
        keys : torch.Tensor
            New key vectors to add to the queue
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Handle queue updates with potential wrap-around
        if ptr + batch_size <= self.queue_size:
            # Standard case: batch fits without wrapping
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            # Wrap-around case: split batch into two parts
            part1_size = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:part1_size].T

            part2_size = batch_size - part1_size
            self.queue[:, :part2_size] = keys[part1_size:].T

        # Update pointer for next batch
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(
        self, query_data: torch.Tensor, key_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning.

        Parameters
        ----------
        query_data : torch.Tensor
            Query samples (augmented cell expressions)
        key_data : torch.Tensor
            Key samples (differently augmented cell expressions)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - logits: Contrastive logits
            - labels: Target labels (zeros for positive pairs)
        """
        # Encode queries
        query_features = self.encoder_query(query_data)[1]  # Use mean features
        query_features = F.normalize(query_features, dim=1)

        # Encode keys with momentum update
        with torch.no_grad():
            self._momentum_update_key_encoder()
            key_features = self.encoder_key(key_data)[1]
            key_features = F.normalize(key_features, dim=1)

        # Compute positive logits
        positive_logits = torch.einsum("nc,nc->n", [query_features, key_features]).unsqueeze(-1)

        # Compute negative logits
        negative_logits = torch.einsum("nc,ck->nk", [query_features, self.queue.clone().detach()])

        # Combine logits and apply temperature scaling
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        logits /= self.temperature

        # Create labels (positive pairs have label 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # Update queue
        self._dequeue_and_enqueue(key_features)

        return logits, labels


class VAE(nn.Module):
    """
    MoCo Coupling Variational Autoencoder for single-cell representation learning.

    This implementation combines several techniques for learning meaningful
    representations of single-cell data:
    - Standard VAE with multiple loss functions tailored for single-cell data
    - Momentum Contrast (MoCo) for contrastive representation learning
    - Information bottleneck regularization that learns coupled latent 
      representations capturing biological correlations intrinsic to single-cell data
    - Support for various single-cell specific loss functions

    The information bottleneck component is specifically designed to learn
    coupled representations that capture biological correlations and 
    relationships between genes and cell types, enabling better understanding
    of cellular states and transitions.

    Parameters
    ----------
    state_dim : int
        Dimension of input state space (number of genes/features)
    hidden_dim : int
        Dimension of hidden layers
    action_dim : int
        Dimension of latent/action space
    i_dim : int
        Dimension of information bottleneck for coupled biological representations
    use_moco : bool
        Whether to enable Momentum Contrast
    loss_mode : Literal["mse", "nb", "zinb"], default="nb"
        Loss function mode for reconstruction
    moco_temperature : float, default=0.2
        Temperature parameter for MoCo contrastive loss
    device : torch.device
        Computation device
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        i_dim: int,
        use_moco: bool,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        moco_temperature: float = 0.2,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        super().__init__()
        self.use_moco = use_moco
        self.device = device

        # Initialize core VAE components
        self.encoder = Encoder(state_dim, hidden_dim, action_dim).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_mode).to(device)

        # Initialize Momentum Contrast if enabled
        if self.use_moco:
            self.encoder_key = Encoder(state_dim, hidden_dim, action_dim).to(device)
            self.momentum_contrast = MomentumContrast(
                self.encoder,
                self.encoder_key,
                state_dim,
                embedding_dim=action_dim,
                temperature=moco_temperature,
                device=device,
            )

        # Information bottleneck components for learning coupled biological representations
        # These components learn to compress the latent space while preserving
        # essential biological correlations and cellular relationships
        self.bottleneck_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.bottleneck_decoder = nn.Linear(i_dim, action_dim).to(device)

    def forward(
        self,
        x: torch.Tensor,
        x_query: Optional[torch.Tensor] = None,
        x_key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Primary input tensor (cell expression profiles)
        x_query : torch.Tensor, optional
            Query samples for contrastive learning (required if use_moco=True)
        x_key : torch.Tensor, optional
            Key samples for contrastive learning (required if use_moco=True)

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Variable-length tuple containing:
            - Latent samples, means, log variances
            - Reconstructions (direct and bottleneck paths)
            - Contrastive learning outputs (if use_moco=True)
            - Bottleneck encoded representations (coupled biological features)
        """
        # Standard encoding
        latent_sample, latent_mean, latent_log_var = self.encoder(x)

        # Information bottleneck processing for coupled biological representations
        # This learns compressed representations that maintain biological correlations
        bottleneck_encoded = self.bottleneck_encoder(latent_sample)
        bottleneck_decoded = self.bottleneck_decoder(bottleneck_encoded)

        # Decode through both paths
        if self.decoder.loss_mode == "zinb":
            # Zero-inflated negative binomial outputs for sparse count data
            reconstruction_direct, dropout_logits_direct = self.decoder(latent_sample)
            reconstruction_bottleneck, dropout_logits_bottleneck = self.decoder(bottleneck_decoded)

            outputs = (
                latent_sample,
                latent_mean,
                latent_log_var,
                reconstruction_direct,
                dropout_logits_direct,
                bottleneck_encoded,
                reconstruction_bottleneck,
                dropout_logits_bottleneck,
            )
        else:
            # Standard outputs for MSE or NB modes
            reconstruction_direct = self.decoder(latent_sample)
            reconstruction_bottleneck = self.decoder(bottleneck_decoded)

            outputs = (
                latent_sample,
                latent_mean,
                latent_log_var,
                reconstruction_direct,
                bottleneck_encoded,
                reconstruction_bottleneck,
            )

        # Add contrastive learning outputs if enabled
        if self.use_moco and x_query is not None and x_key is not None:
            contrastive_logits, contrastive_labels = self.momentum_contrast(x_query, x_key)
            outputs = outputs + (contrastive_logits, contrastive_labels)

        return outputs
