
"""
MCCVAE: MoCo Coupling Variational Autoencoder with Multiple Regularization Techniques

This module implements a sophisticated VAE that combines multiple advanced techniques:
- Momentum Contrast (MoCo) for representation learning
- Multiple regularization methods (DIP, β-TC-VAE, InfoVAE)
- Support for various loss functions (MSE, NB, ZINB)
- Information bottleneck for entangled representations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE


class MCCVAE(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    MoCo Coupling Variational Autoencoder with Multiple Regularization Techniques.
    
    This implementation combines several state-of-the-art techniques:
    - Standard VAE with β-VAE regularization
    - Momentum Contrast (MoCo) for contrastive representation learning
    - DIP (Disentangled Inferred Prior) regularization
    - β-TC-VAE (Total Correlation) regularization
    - InfoVAE with MMD regularization
    - Information bottleneck for disentangled representations
    - Multiple reconstruction loss modes (MSE, NB, ZINB)
    
    Parameters
    ----------
    recon : float
        Weight for reconstruction loss
    irecon : float
        Weight for information bottleneck reconstruction loss
    beta : float
        Weight for KL divergence regularization (β-VAE)
    dip : float
        Weight for DIP regularization
    tc : float
        Weight for total correlation regularization
    info : float
        Weight for InfoVAE MMD regularization
    state_dim : int
        Dimension of input state space
    hidden_dim : int
        Dimension of hidden layers
    latent_dim : int
        Dimension of latent space
    i_dim : int
        Dimension of information bottleneck
    use_moco : bool
        Whether to enable Momentum Contrast learning
    loss_mode : str
        Loss function mode ('mse', 'nb', 'zinb')
    lr : float
        Learning rate for optimization
    vae_reg : float
        Regularization weight for VAE latent representations
    moco_weight : float
        Weight for contrastive learning loss
    use_qm : bool
        Whether to use mean (True) or samples (False) for latent representations
    moco_T : float
        Temperature parameter for MoCo contrastive loss
    device : torch.device
        Computation device
    """
    
    # Constants for numerical stability
    NUMERICAL_STABILITY_EPS = 1e-8
    GRADIENT_CLIP_VALUE = 1.0
    
    def __init__(
        self,
        recon: float,
        irecon: float,
        beta: float,
        dip: float,
        tc: float,
        info: float,
        state_dim: int,
        hidden_dim: int,
        latent_dim: int,
        i_dim: int,
        use_moco: bool,
        loss_mode: str,
        lr: float,
        vae_reg: float,
        moco_weight: float,
        use_qm: bool,
        moco_T: float,
        device: torch.device,
        *args,
        **kwargs,
    ):
        # Store configuration parameters
        self.use_moco = use_moco
        self.use_qm = use_qm
        self.loss_mode = loss_mode
        
        # Loss weights
        self.recon_weight = recon
        self.irecon_weight = irecon
        self.beta_weight = beta
        self.dip_weight = dip
        self.tc_weight = tc
        self.info_weight = info
        self.moco_weight = moco_weight
        
        # Regularization parameters
        self.vae_regularization = vae_reg
        self.device = device
        
        # Initialize the VAE model
        self.vae_model = VAE(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=latent_dim,
            i_dim=i_dim,
            use_moco=use_moco,
            loss_mode=loss_mode,
            moco_temperature=moco_T,
            device=device,
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.vae_model.parameters(), lr=lr)
        
        # Loss tracking
        self.training_losses = []
        
        # Initialize contrastive learning loss criterion
        self.contrastive_criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def take_latent(self, state: np.ndarray) -> np.ndarray:
        """
        Extract latent representations from input states.
        
        Returns either samples or means based on use_qm flag for standard VAE mode.
        
        Parameters
        ----------
        state : np.ndarray
            Input states to encode
            
        Returns
        -------
        np.ndarray
            Latent representations
        """
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        encoder_outputs = self.vae_model.encoder(state_tensor)
        latent_samples, latent_means, latent_log_vars = encoder_outputs
        
        if self.use_qm:
            return latent_means.cpu().numpy()
        else:
            return latent_samples.cpu().numpy()

    @torch.no_grad()
    def take_iembed(self, state: np.ndarray) -> np.ndarray:
        """
        Extract information bottleneck embeddings from input states.
        
        Parameters
        ----------
        state : np.ndarray
            Input states to encode
            
        Returns
        -------
        np.ndarray
            Information bottleneck embeddings
        """
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        model_outputs = self.vae_model(state_tensor)
        
        # Parse outputs based on configuration
        if self.use_moco:
            if self.loss_mode == "zinb":
                bottleneck_encoded = model_outputs[5]
            else:
                bottleneck_encoded = model_outputs[4]
        else:
            if self.loss_mode == "zinb":
                bottleneck_encoded = model_outputs[5]
            else:
                bottleneck_encoded = model_outputs[4]
                
        return bottleneck_encoded.cpu().numpy()

    @torch.no_grad()
    def take_refined(self, state: np.ndarray) -> np.ndarray:
        """
        Extract refined latent representations (l_d) from input states.
        
        The refined representations are obtained by:
        1. Encoding input to latent space (z)
        2. Compressing through bottleneck encoder to get l_e
        3. Expanding through bottleneck decoder to get l_d
        
        These l_d representations have the same dimensionality as the original
        latent space (latent_dim) but have been refined through the information
        bottleneck, capturing essential biological correlations while filtering
        out noise.
        
        Note: This method requires the VAE model to have bottleneck_encoder and
        bottleneck_decoder components, which are always present in the MCCVAE
        architecture as defined in module.py.
        
        Parameters
        ----------
        state : np.ndarray
            Input states to encode
            
        Returns
        -------
        np.ndarray
            Refined latent representations (l_d) with shape (n_cells, latent_dim)
        """
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        
        # Get latent representation
        encoder_outputs = self.vae_model.encoder(state_tensor)
        latent_samples, latent_means, latent_log_vars = encoder_outputs
        
        # Use mean or sample based on use_qm flag
        latent_repr = latent_means if self.use_qm else latent_samples
        
        # Apply bottleneck encoding then decoding to get refined representation
        # These components are guaranteed to exist in the MCCVAE architecture
        bottleneck_encoded = self.vae_model.bottleneck_encoder(latent_repr)
        bottleneck_decoded = self.vae_model.bottleneck_decoder(bottleneck_encoded)
        
        return bottleneck_decoded.cpu().numpy()

    def update(self, *states: tuple) -> None:
        """
        Perform one training step with the given states.
        
        This method handles the complete training pipeline:
        1. Forward pass through the model
        2. Compute all loss components
        3. Backward pass and optimization
        4. Loss tracking
        
        Parameters
        ----------
        *states : tuple
            Variable number of state tensors depending on configuration
            - Standard mode: (states,)
            - MoCo mode: (states_x, states_q, states_k)
        """
        # Initialize contrastive loss
        contrastive_loss = torch.zeros(1).to(self.device)
        
        # Prepare inputs and forward pass
        if self.use_moco:
            states_x = torch.tensor(states[0], dtype=torch.float).to(self.device)
            states_query = torch.tensor(states[1], dtype=torch.float).to(self.device)
            states_key = torch.tensor(states[2], dtype=torch.float).to(self.device)
            model_outputs = self.vae_model(states_x, states_query, states_key)
        else:
            states_query = torch.tensor(states[0], dtype=torch.float).to(self.device)
            model_outputs = self.vae_model(states_query)
        
        # Compute losses
        loss_components = self._compute_losses(model_outputs, states_query)
        
        # Extract loss components
        (reconstruction_loss, info_reconstruction_loss, kl_divergence, 
         dip_loss, tc_loss, mmd_loss, contrastive_loss) = loss_components
        
        # Compute total loss
        total_loss = (
            self.recon_weight * reconstruction_loss +
            info_reconstruction_loss +
            kl_divergence +
            dip_loss +
            tc_loss +
            mmd_loss +
            self.moco_weight * contrastive_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae_model.parameters(), self.GRADIENT_CLIP_VALUE)
        self.optimizer.step()
        
        # Track losses
        self.training_losses.append((
            total_loss.item(),
            reconstruction_loss.item(),
            info_reconstruction_loss.item(),
            kl_divergence.item(),
            dip_loss.item(),
            tc_loss.item(),
            mmd_loss.item(),
            contrastive_loss.item(),
        ))

    def _compute_losses(self, outputs: tuple, states_query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute all loss components."""
        if self.loss_mode == "zinb":
            return self._compute_zinb_losses(outputs, states_query)
        else:
            return self._compute_standard_losses(outputs, states_query)

    def _compute_zinb_losses(self, outputs: tuple, states_query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute loss components for ZINB loss mode."""
        # Parse outputs
        if self.use_moco:
            (latent_samples, latent_means, latent_log_vars, reconstruction_direct,
             dropout_logits_direct, bottleneck_encoded, reconstruction_bottleneck,
             dropout_logits_bottleneck, contrastive_logits, contrastive_labels) = outputs
            contrastive_loss = self.contrastive_criterion(contrastive_logits, contrastive_labels)
        else:
            (latent_samples, latent_means, latent_log_vars, reconstruction_direct,
             dropout_logits_direct, bottleneck_encoded, reconstruction_bottleneck,
             dropout_logits_bottleneck) = outputs
            contrastive_loss = torch.zeros(1).to(self.device)
        
        # Prepare normalization and dispersion
        normalization_factor = states_query.sum(-1).view(-1, 1)
        reconstruction_normalized = reconstruction_direct * normalization_factor + self.NUMERICAL_STABILITY_EPS
        dispersion = torch.exp(self.vae_model.decoder.dispersion_param)
        
        # Reconstruction loss
        reconstruction_loss = -self._log_zinb(
            states_query, reconstruction_normalized, dispersion, dropout_logits_direct
        ).sum(-1).mean()
        
        # Information reconstruction loss
        if self.irecon_weight:
            reconstruction_bottleneck_normalized = reconstruction_bottleneck * normalization_factor + self.NUMERICAL_STABILITY_EPS
            info_reconstruction_loss = -self.irecon_weight * self._log_zinb(
                states_query, reconstruction_bottleneck_normalized, dispersion, dropout_logits_bottleneck
            ).sum(-1).mean()
        else:
            info_reconstruction_loss = torch.zeros(1).to(self.device)
        
        # Compute regularization losses
        kl_divergence, dip_loss, tc_loss, mmd_loss = self._compute_regularization_losses(
            latent_samples, latent_means, latent_log_vars
        )
        
        return (reconstruction_loss, info_reconstruction_loss, kl_divergence,
                dip_loss, tc_loss, mmd_loss, contrastive_loss)

    def _compute_standard_losses(self, outputs: tuple, states_query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute loss components for NB or MSE loss modes."""
        # Parse outputs
        if self.use_moco:
            (latent_samples, latent_means, latent_log_vars, reconstruction_direct,
             bottleneck_encoded, reconstruction_bottleneck,
             contrastive_logits, contrastive_labels) = outputs
            contrastive_loss = self.contrastive_criterion(contrastive_logits, contrastive_labels)
        else:
            (latent_samples, latent_means, latent_log_vars, reconstruction_direct,
             bottleneck_encoded, reconstruction_bottleneck) = outputs
            contrastive_loss = torch.zeros(1).to(self.device)
        
        # Compute reconstruction losses based on mode
        if self.loss_mode == "nb":
            normalization_factor = states_query.sum(-1).view(-1, 1)
            reconstruction_normalized = reconstruction_direct * normalization_factor + self.NUMERICAL_STABILITY_EPS
            dispersion = torch.exp(self.vae_model.decoder.dispersion_param)
            
            reconstruction_loss = -self._log_nb(
                states_query, reconstruction_normalized, dispersion
            ).sum(-1).mean()
            
            if self.irecon_weight:
                # BUG FIX: Add numerical stability epsilon to prevent log(0) errors
                reconstruction_bottleneck_normalized = reconstruction_bottleneck * normalization_factor + self.NUMERICAL_STABILITY_EPS
                info_reconstruction_loss = -self.irecon_weight * self._log_nb(
                    states_query, reconstruction_bottleneck_normalized, dispersion
                ).sum(-1).mean()
            else:
                info_reconstruction_loss = torch.zeros(1).to(self.device)
        else:  # MSE mode
            reconstruction_loss = F.mse_loss(
                states_query, reconstruction_direct, reduction="none"
            ).sum(-1).mean()
            
            if self.irecon_weight:
                info_reconstruction_loss = self.irecon_weight * F.mse_loss(
                    states_query, reconstruction_bottleneck, reduction="none"
                ).sum(-1).mean()
            else:
                info_reconstruction_loss = torch.zeros(1).to(self.device)
        
        # Compute regularization losses
        kl_divergence, dip_loss, tc_loss, mmd_loss = self._compute_regularization_losses(
            latent_samples, latent_means, latent_log_vars
        )
        
        return (reconstruction_loss, info_reconstruction_loss, kl_divergence,
                dip_loss, tc_loss, mmd_loss, contrastive_loss)

    def _compute_regularization_losses(
        self,
        latent_samples: torch.Tensor,
        latent_means: torch.Tensor,
        latent_log_vars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all regularization loss components."""
        # Prior parameters (standard normal)
        prior_means = torch.zeros_like(latent_means)
        prior_log_vars = torch.zeros_like(latent_log_vars)
        
        # KL divergence loss (β-VAE)
        kl_divergence = self.beta_weight * self._normal_kl(
            latent_means, latent_log_vars, prior_means, prior_log_vars
        ).sum(-1).mean()
        
        # DIP loss
        if self.dip_weight:
            dip_loss = self.dip_weight * self._dip_loss(latent_means, latent_log_vars)
        else:
            dip_loss = torch.zeros(1).to(self.device)
        
        # Total correlation loss (β-TC-VAE)
        if self.tc_weight:
            tc_loss = self.tc_weight * self._betatc_compute_total_correlation(
                latent_samples, latent_means, latent_log_vars
            )
        else:
            tc_loss = torch.zeros(1).to(self.device)
        
        # MMD loss (InfoVAE)
        if self.info_weight:
            mmd_loss = self.info_weight * self._compute_mmd(
                latent_samples, torch.randn_like(latent_samples)
            )
        else:
            mmd_loss = torch.zeros(1).to(self.device)
        
        return kl_divergence, dip_loss, tc_loss, mmd_loss
