
"""
Variational Autoencoder Implementation with ODE Integration and Momentum Contrast

This module provides a comprehensive VAE implementation supporting:
- Multiple loss functions (MSE, Negative Binomial, Zero-Inflated Negative Binomial)
- Neural ODE integration for latent dynamics
- Momentum Contrast (MoCo) for contrastive learning
- Information bottleneck regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union, Literal, Optional
import numpy as np

from .mixin import NODEMixin


class Encoder(nn.Module):
    """
    Variational encoder network that maps input states to latent distributions.

    This encoder supports both standard VAE encoding and ODE-enhanced encoding
    with time prediction for temporal dynamics modeling.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state space
    hidden_dim : int
        Dimension of hidden layers in the network
    action_dim : int
        Dimension of the latent space (action space)
    use_ode : bool, default=False
        Whether to use ODE mode with time parameter prediction
    """

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        use_ode: bool = False
    ):
        super().__init__()
        self.use_ode = use_ode

        # Shared feature extraction network
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent distribution parameters (mean and log variance)
        self.latent_params = nn.Linear(hidden_dim, action_dim * 2)

        # Time encoder for ODE dynamics (only used in ODE mode)
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Ensures time values are in [0, 1] range
            )

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
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, state_dim)

        Returns
        -------
        If use_ode=False:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                - sampled_latent: Sampled latent vectors
                - latent_mean: Mean of the latent distribution
                - latent_log_var: Log variance of the latent distribution

        If use_ode=True:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                - sampled_latent: Sampled latent vectors
                - latent_mean: Mean of the latent distribution
                - latent_log_var: Log variance of the latent distribution
                - time_param: Predicted time parameter in [0, 1] range
        """
        # Apply log1p transformation for numerical stability
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

        # Return time parameter if in ODE mode
        if self.use_ode:
            time_param = self.time_encoder(hidden_features).squeeze(-1)
            return sampled_latent, latent_mean, latent_log_var, time_param

        return sampled_latent, latent_mean, latent_log_var


class Decoder(nn.Module):
    """
    Decoder network that maps latent vectors back to the original space.

    Supports three loss modes for different data types:
    - 'mse': Mean Squared Error for continuous data
    - 'nb': Negative Binomial for discrete count data
    - 'zinb': Zero-Inflated Negative Binomial for sparse count data

    Parameters
    ----------
    state_dim : int
        Dimension of the original data space
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
            # Mean parameters with softmax normalization
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
            torch.Tensor: Reconstructed output

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


class LatentODEFunction(nn.Module):
    """
    Neural ODE function for modeling latent space dynamics.

    This module defines the continuous-time dynamics in the latent space,
    allowing the model to learn temporal evolution patterns.

    Parameters
    ----------
    n_latent : int, default=10
        Dimension of the latent space
    n_hidden : int, default=25
        Dimension of the hidden layer
    """

    def __init__(self, n_latent: int = 10, n_hidden: int = 25):
        super().__init__()
        self.activation = nn.ELU()
        self.input_layer = nn.Linear(n_latent, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_latent)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative at time t and state x.

        Parameters
        ----------
        t : torch.Tensor
            Time point(s)
        x : torch.Tensor
            Latent state(s)

        Returns
        -------
        torch.Tensor
            Computed gradients/derivatives
        """
        hidden = self.activation(self.input_layer(x))
        gradient = self.output_layer(hidden)
        return gradient


class MomentumContrast(nn.Module):
    """
    Momentum Contrast (MoCo) for unsupervised representation learning.

    Implements the MoCo framework for learning representations through
    contrastive learning with a momentum-updated key encoder.

    Parameters
    ----------
    encoder_query : nn.Module
        Query encoder network
    encoder_key : nn.Module
        Key encoder network (momentum-updated)
    state_dim : int
        Dimension of input state
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
        self.augmentation_prob = 0.5
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
            Query samples
        key_data : torch.Tensor
            Key samples

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


class VAE(nn.Module, NODEMixin):
    """
    Variational Autoencoder with Neural ODE and Momentum Contrast support.

    This implementation combines several advanced techniques:
    - Standard VAE with multiple loss functions
    - Neural ODE integration for temporal dynamics
    - Momentum Contrast for representation learning
    - Information bottleneck regularization

    Parameters
    ----------
    state_dim : int
        Dimension of input state space
    hidden_dim : int
        Dimension of hidden layers
    action_dim : int
        Dimension of latent/action space
    i_dim : int
        Dimension of information bottleneck
    use_ode : bool
        Whether to enable Neural ODE integration
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
        use_ode: bool,
        use_moco: bool,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        moco_temperature: float = 0.2,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        super().__init__()
        self.use_moco = use_moco
        self.device = device

        # Initialize core VAE components
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_ode).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_mode).to(device)

        # Initialize Neural ODE solver if enabled
        if use_ode:
            self.ode_function = LatentODEFunction(action_dim)

        # Initialize Momentum Contrast if enabled
        if self.use_moco:
            self.encoder_key = Encoder(state_dim, hidden_dim, action_dim, use_ode).to(device)
            self.momentum_contrast = MomentumContrast(
                self.encoder,
                self.encoder_key,
                state_dim,
                embedding_dim=action_dim,
                temperature=moco_temperature,
                device=device,
            )

        # Information bottleneck components
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
            Primary input tensor
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
            - ODE-based results (if use_ode=True)
            - Contrastive learning outputs (if use_moco=True)
        """
        # Encode input
        if self.encoder.use_ode:
            return self._forward_with_ode(x, x_query, x_key)
        else:
            return self._forward_standard(x, x_query, x_key)

    def _forward_standard(
        self,
        x: torch.Tensor,
        x_query: Optional[torch.Tensor] = None,
        x_key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass without ODE integration."""
        # Standard encoding
        latent_sample, latent_mean, latent_log_var = self.encoder(x)

        # Information bottleneck processing
        bottleneck_encoded = self.bottleneck_encoder(latent_sample)
        bottleneck_decoded = self.bottleneck_decoder(bottleneck_encoded)

        # Decode through both paths
        if self.decoder.loss_mode == "zinb":
            # Zero-inflated negative binomial outputs
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
            # Standard outputs
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

    def _forward_with_ode(
        self,
        x: torch.Tensor,
        x_query: Optional[torch.Tensor] = None,
        x_key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with ODE integration."""
        # Encoding with time prediction
        latent_sample, latent_mean, latent_log_var, time_param = self.encoder(x)

        # Sort by time for ODE integration
        time_sorted_indices = torch.argsort(time_param)
        time_sorted = time_param[time_sorted_indices]
        latent_sorted = latent_sample[time_sorted_indices]
        latent_mean_sorted = latent_mean[time_sorted_indices]
        latent_log_var_sorted = latent_log_var[time_sorted_indices]
        x_sorted = x[time_sorted_indices]

        # Remove duplicate time points
        unique_mask = torch.ones_like(time_sorted, dtype=torch.bool)
        unique_mask[1:] = time_sorted[1:] != time_sorted[:-1]

        time_unique = time_sorted[unique_mask]
        latent_unique = latent_sorted[unique_mask]
        latent_mean_unique = latent_mean_sorted[unique_mask]
        latent_log_var_unique = latent_log_var_sorted[unique_mask]
        x_unique = x_sorted[unique_mask]

        # Solve ODE from initial condition
        initial_state = latent_unique[0]
        ode_solution = self.solve_ode(self.ode_function, initial_state, time_unique)

        # Information bottleneck processing
        bottleneck_encoded = self.bottleneck_encoder(latent_unique)
        bottleneck_decoded = self.bottleneck_decoder(bottleneck_encoded)

        bottleneck_encoded_ode = self.bottleneck_encoder(ode_solution)
        bottleneck_decoded_ode = self.bottleneck_decoder(bottleneck_encoded_ode)

        # Generate reconstructions
        if self.decoder.loss_mode == "zinb":
            # ZINB outputs
            reconstruction_direct, dropout_direct = self.decoder(latent_unique)
            reconstruction_bottleneck, dropout_bottleneck = self.decoder(bottleneck_decoded)
            reconstruction_ode, dropout_ode = self.decoder(ode_solution)
            reconstruction_bottleneck_ode, dropout_bottleneck_ode = self.decoder(bottleneck_decoded_ode)

            outputs = (
                latent_unique,
                latent_mean_unique,
                latent_log_var_unique,
                x_unique,
                reconstruction_direct,
                dropout_direct,
                bottleneck_encoded,
                bottleneck_encoded_ode,
                reconstruction_bottleneck,
                dropout_bottleneck,
                ode_solution,
                reconstruction_ode,
                dropout_ode,
                reconstruction_bottleneck_ode,
                dropout_bottleneck_ode,
            )
        else:
            # Standard outputs
            reconstruction_direct = self.decoder(latent_unique)
            reconstruction_bottleneck = self.decoder(bottleneck_decoded)
            reconstruction_ode = self.decoder(ode_solution)
            reconstruction_bottleneck_ode = self.decoder(bottleneck_decoded_ode)

            outputs = (
                latent_unique,
                latent_mean_unique,
                latent_log_var_unique,
                x_unique,
                reconstruction_direct,
                bottleneck_encoded,
                bottleneck_encoded_ode,
                reconstruction_bottleneck,
                ode_solution,
                reconstruction_ode,
                reconstruction_bottleneck_ode,
            )

        # Add contrastive learning outputs if enabled
        if self.use_moco and x_query is not None and x_key is not None:
            contrastive_logits, contrastive_labels = self.momentum_contrast(x_query, x_key)
            outputs = outputs + (contrastive_logits, contrastive_labels)

        return outputs
