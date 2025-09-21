
"""
CODEVAE: Comprehensive Variational Autoencoder with Multiple Regularization Techniques

This module implements a sophisticated VAE that combines multiple advanced techniques:
- Neural Ordinary Differential Equations (Neural ODE)
- Momentum Contrast (MoCo) for representation learning
- Multiple regularization methods (DIP, β-TC-VAE, InfoVAE)
- Support for various loss functions (MSE, NB, ZINB)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from typing import Tuple, Optional, Union, List
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE


class CODEVAE(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    Comprehensive Variational Autoencoder with ODE and Contrastive Learning.
    
    This implementation combines several state-of-the-art techniques:
    - Standard VAE with β-VAE regularization
    - Neural ODE for modeling temporal dynamics
    - Momentum Contrast (MoCo) for representation learning
    - DIP (Disentangled Inferred Prior) regularization
    - β-TC-VAE (Total Correlation) regularization
    - InfoVAE with MMD regularization
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
    use_ode : bool
        Whether to enable Neural ODE integration
    use_moco : bool
        Whether to enable Momentum Contrast learning
    loss_mode : str
        Loss function mode ('mse', 'nb', 'zinb')
    lr : float
        Learning rate for optimization
    vae_reg : float
        Regularization weight for VAE latent representations
    ode_reg : float
        Regularization weight for ODE latent representations
    moco_weight : float
        Weight for contrastive learning loss
    use_qm : bool
        Whether to use mean (True) or samples (False) for latent representations
    moco_T : float
        Temperature parameter for MoCo contrastive loss
    device : torch.device
        Computation device
    """
    
    # Constants for numerical stability and ODE integration
    NUMERICAL_STABILITY_EPS = 1e-8
    ODE_INTEGRATION_STEP = 1e-2
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
        use_ode: bool,
        use_moco: bool,
        loss_mode: str,
        lr: float,
        vae_reg: float,
        ode_reg: float,
        moco_weight: float,
        use_qm: bool,
        moco_T: float,
        device: torch.device,
        *args,
        **kwargs,
    ):
        # Store configuration parameters
        self.use_ode = use_ode
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
        self.ode_regularization = ode_reg
        self.device = device
        
        # Initialize the VAE model
        self.vae_model = VAE(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
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
        
        For ODE mode, combines VAE and ODE latent representations.
        For standard mode, returns either samples or means based on use_qm flag.
        
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
        
        if self.use_ode:
            return self._extract_ode_latents(state_tensor)
        else:
            return self._extract_standard_latents(state_tensor)

    def _extract_ode_latents(self, state_tensor: torch.Tensor) -> np.ndarray:
        """Extract latent representations with ODE integration."""
        encoder_outputs = self.vae_model.encoder(state_tensor)
        latent_samples, latent_means, latent_log_vars, time_params = encoder_outputs
        
        # Sort by time and remove duplicates for ODE integration
        time_cpu = time_params.cpu()
        unique_times, sort_indices, inverse_indices = np.unique(
            time_cpu, return_index=True, return_inverse=True
        )
        unique_times_tensor = torch.tensor(unique_times)
        
        # Choose between means and samples for ODE integration
        if self.use_qm:
            sorted_representations = latent_means[sort_indices]
        else:
            sorted_representations = latent_samples[sort_indices]
            
        # Solve ODE from initial condition
        initial_condition = sorted_representations[0]
        ode_solution = self.vae_model.solve_ode(
            self.vae_model.ode_function, initial_condition, unique_times_tensor
        )
        
        # Map back to original ordering
        ode_representations = ode_solution[inverse_indices]
        
        # Combine VAE and ODE representations
        if self.use_qm:
            vae_representations = latent_means
        else:
            vae_representations = latent_samples
            
        combined_representations = (
            self.vae_regularization * vae_representations + 
            self.ode_regularization * ode_representations
        )
        
        return combined_representations.cpu().numpy()

    def _extract_standard_latents(self, state_tensor: torch.Tensor) -> np.ndarray:
        """Extract standard latent representations without ODE."""
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
        
        if self.use_ode:
            return self._extract_ode_embeddings(state_tensor)
        else:
            return self._extract_standard_embeddings(state_tensor)

    def _extract_ode_embeddings(self, state_tensor: torch.Tensor) -> np.ndarray:
        """Extract information embeddings with ODE integration."""
        encoder_outputs = self.vae_model.encoder(state_tensor)
        latent_samples, _, _, time_params = encoder_outputs
        
        # Process time parameters for ODE integration
        time_cpu = time_params.cpu()
        unique_times, sort_indices, inverse_indices = np.unique(
            time_cpu, return_index=True, return_inverse=True
        )
        unique_times_tensor = torch.tensor(unique_times)
        
        sorted_latents = latent_samples[sort_indices]
        initial_condition = sorted_latents[0]
        ode_solution = self.vae_model.solve_ode(
            self.vae_model.ode_function, initial_condition, unique_times_tensor
        )
        ode_latents = ode_solution[inverse_indices]
        
        # Apply information bottleneck to both representations
        bottleneck_vae = self.vae_model.bottleneck_encoder(latent_samples)
        bottleneck_ode = self.vae_model.bottleneck_encoder(ode_latents)
        
        # Combine representations
        combined_embeddings = (
            self.vae_regularization * bottleneck_vae + 
            self.ode_regularization * bottleneck_ode
        )
        
        return combined_embeddings.cpu().numpy()

    def _extract_standard_embeddings(self, state_tensor: torch.Tensor) -> np.ndarray:
        """Extract standard information embeddings without ODE."""
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
    def take_time(self, state: np.ndarray) -> np.ndarray:
        """
        Extract predicted time parameters from input states.
        
        Parameters
        ----------
        state : np.ndarray
            Input states to encode
            
        Returns
        -------
        np.ndarray
            Predicted time parameters
        """
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        encoder_outputs = self.vae_model.encoder(state_tensor)
        _, _, _, time_params = encoder_outputs
        return time_params.detach().cpu().numpy()

    @torch.no_grad()
    def take_grad(self, state: np.ndarray) -> np.ndarray:
        """
        Compute gradients in latent space using ODE function.
        
        Parameters
        ----------
        state : np.ndarray
            Input states to encode
            
        Returns
        -------
        np.ndarray
            Computed gradients in latent space
        """
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        encoder_outputs = self.vae_model.encoder(state_tensor)
        latent_samples, _, _, time_params = encoder_outputs
        
        gradients = self.vae_model.ode_function(time_params, latent_samples.cpu()).numpy()
        return gradients

    @torch.no_grad()
    def take_transition(self, state: np.ndarray, top_k: int = 30) -> np.ndarray:
        """
        Compute transition probability matrix based on latent dynamics.
        
        This method computes transitions by:
        1. Encoding states to latent space
        2. Computing future states using ODE gradients
        3. Building similarity matrix based on distances
        4. Sparsifying to keep only top-k transitions
        
        Parameters
        ----------
        state : np.ndarray
            Input states to encode
        top_k : int, default=30
            Number of top transitions to keep for each state
            
        Returns
        -------
        np.ndarray
            Sparse transition probability matrix
        """
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        encoder_outputs = self.vae_model.encoder(state_tensor)
        latent_samples, _, _, time_params = encoder_outputs
        
        # Compute gradients and future states
        gradients = self.vae_model.ode_function(time_params, latent_samples.cpu()).numpy()
        current_latents = latent_samples.cpu().numpy()
        future_latents = current_latents + self.ODE_INTEGRATION_STEP * gradients
        
        # Compute pairwise distances and similarities
        distances = pairwise_distances(current_latents, future_latents)
        median_distance = np.median(distances)
        similarities = np.exp(-(distances**2) / (2 * median_distance**2))
        
        # Normalize to get transition probabilities
        transition_matrix = similarities / similarities.sum(axis=1, keepdims=True)
        
        # Sparsify transition matrix
        sparse_transition_matrix = self._sparsify_transitions(transition_matrix, top_k)
        
        return sparse_transition_matrix

    def _sparsify_transitions(self, transition_matrix: np.ndarray, top_k: int) -> np.ndarray:
        """
        Sparsify transition matrix to keep only top-k transitions.
        
        Parameters
        ----------
        transition_matrix : np.ndarray
            Full transition probability matrix
        top_k : int
            Number of top transitions to keep
            
        Returns
        -------
        np.ndarray
            Sparsified transition matrix
        """
        n_cells = transition_matrix.shape[0]
        sparse_matrix = np.zeros_like(transition_matrix)
        
        for cell_idx in range(n_cells):
            # Find top-k transitions for this cell
            top_indices = np.argsort(transition_matrix[cell_idx])[::-1][:top_k]
            sparse_matrix[cell_idx, top_indices] = transition_matrix[cell_idx, top_indices]
            
            # Renormalize to ensure probabilities sum to 1
            total_prob = sparse_matrix[cell_idx].sum()
            if total_prob > 0:
                sparse_matrix[cell_idx] /= total_prob
                
        return sparse_matrix

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
            states_x = torch.tensor(states[1], dtype=torch.float).to(self.device)
            states_query = torch.tensor(states[1], dtype=torch.float).to(self.device)
            states_key = torch.tensor(states[2], dtype=torch.float).to(self.device)
            model_outputs = self.vae_model(states_x, states_query, states_key)
        else:
            states_query = torch.tensor(states[0], dtype=torch.float).to(self.device)
            model_outputs = self.vae_model(states_query)
        
        # Compute losses based on configuration
        if self.use_ode:
            loss_components = self._compute_ode_losses(model_outputs, states_query)
        else:
            loss_components = self._compute_standard_losses(model_outputs, states_query)
        
        # Extract loss components
        (reconstruction_loss, info_reconstruction_loss, kl_divergence, 
         dip_loss, tc_loss, mmd_loss, ode_divergence, contrastive_loss) = loss_components
        
        # Compute total loss
        total_loss = (
            self.recon_weight * reconstruction_loss +
            info_reconstruction_loss +
            kl_divergence +
            dip_loss +
            tc_loss +
            mmd_loss +
            ode_divergence +
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

    def _compute_ode_losses(self, outputs: tuple, states_query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute all loss components for ODE mode."""
        if self.loss_mode == "zinb":
            return self._compute_ode_zinb_losses(outputs, states_query)
        else:
            return self._compute_ode_standard_losses(outputs, states_query)

    def _compute_ode_zinb_losses(self, outputs: tuple, states_query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute loss components for ODE mode with ZINB loss."""
        # Parse outputs based on MoCo configuration
        if self.use_moco:
            (latent_samples, latent_means, latent_log_vars, input_states,
             reconstruction_direct, dropout_logits_direct,
             bottleneck_encoded, bottleneck_encoded_ode,
             reconstruction_bottleneck, dropout_logits_bottleneck,
             ode_samples, reconstruction_ode, dropout_logits_ode,
             reconstruction_bottleneck_ode, dropout_logits_bottleneck_ode,
             contrastive_logits, contrastive_labels) = outputs
            contrastive_loss = self.contrastive_criterion(contrastive_logits, contrastive_labels)
        else:
            (latent_samples, latent_means, latent_log_vars, input_states,
             reconstruction_direct, dropout_logits_direct,
             bottleneck_encoded, bottleneck_encoded_ode,
             reconstruction_bottleneck, dropout_logits_bottleneck,
             ode_samples, reconstruction_ode, dropout_logits_ode,
             reconstruction_bottleneck_ode, dropout_logits_bottleneck_ode) = outputs
            contrastive_loss = torch.zeros(1).to(self.device)
        
        # Compute ODE divergence
        ode_divergence = F.mse_loss(latent_samples, ode_samples, reduction="none").sum(-1).mean()
        
        # Prepare normalization factor
        normalization_factor = input_states.sum(-1).view(-1, 1)
        reconstruction_direct_normalized = reconstruction_direct * normalization_factor
        reconstruction_ode_normalized = reconstruction_ode * normalization_factor
        
        # Compute dispersion parameter
        dispersion = torch.exp(self.vae_model.decoder.dispersion_param)
        
        # Reconstruction losses
        reconstruction_loss = -self._log_zinb(
            input_states, reconstruction_direct_normalized, dispersion, dropout_logits_direct
        ).sum(-1).mean()
        reconstruction_loss += -self._log_zinb(
            input_states, reconstruction_ode_normalized, dispersion, dropout_logits_ode
        ).sum(-1).mean()
        
        # Information reconstruction loss
        if self.irecon_weight:
            reconstruction_bottleneck_normalized = reconstruction_bottleneck * normalization_factor + self.NUMERICAL_STABILITY_EPS
            reconstruction_bottleneck_ode_normalized = reconstruction_bottleneck_ode * normalization_factor
            
            info_reconstruction_loss = -self.irecon_weight * self._log_zinb(
                input_states, reconstruction_bottleneck_normalized, dispersion, dropout_logits_bottleneck
            ).sum(-1).mean()
            info_reconstruction_loss += -self.irecon_weight * self._log_zinb(
                input_states, reconstruction_bottleneck_ode_normalized, dispersion, dropout_logits_bottleneck_ode
            ).sum(-1).mean()
        else:
            info_reconstruction_loss = torch.zeros(1).to(self.device)
        
        # Compute regularization losses
        kl_divergence, dip_loss, tc_loss, mmd_loss = self._compute_regularization_losses(
            latent_samples, latent_means, latent_log_vars
        )
        
        return (reconstruction_loss, info_reconstruction_loss, kl_divergence,
                dip_loss, tc_loss, mmd_loss, ode_divergence, contrastive_loss)

    def _compute_ode_standard_losses(self, outputs: tuple, states_query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute loss components for ODE mode with standard losses."""
        # Parse outputs based on MoCo configuration
        if self.use_moco:
            (latent_samples, latent_means, latent_log_vars, input_states,
             reconstruction_direct, bottleneck_encoded, bottleneck_encoded_ode,
             reconstruction_bottleneck, ode_samples, reconstruction_ode,
             reconstruction_bottleneck_ode, contrastive_logits, contrastive_labels) = outputs
            contrastive_loss = self.contrastive_criterion(contrastive_logits, contrastive_labels)
        else:
            (latent_samples, latent_means, latent_log_vars, input_states,
             reconstruction_direct, bottleneck_encoded, bottleneck_encoded_ode,
             reconstruction_bottleneck, ode_samples, reconstruction_ode,
             reconstruction_bottleneck_ode) = outputs
            contrastive_loss = torch.zeros(1).to(self.device)
        
        # Compute ODE divergence
        ode_divergence = F.mse_loss(latent_samples, ode_samples, reduction="none").sum(-1).mean()
        
        # Compute reconstruction losses based on loss mode
        if self.loss_mode == "nb":
            reconstruction_loss, info_reconstruction_loss = self._compute_nb_reconstruction_losses(
                input_states, reconstruction_direct, reconstruction_ode,
                reconstruction_bottleneck, reconstruction_bottleneck_ode
            )
        else:  # MSE mode
            reconstruction_loss, info_reconstruction_loss = self._compute_mse_reconstruction_losses(
                input_states, reconstruction_direct, reconstruction_ode,
                reconstruction_bottleneck, reconstruction_bottleneck_ode
            )
        
        # Compute regularization losses
        kl_divergence, dip_loss, tc_loss, mmd_loss = self._compute_regularization_losses(
            latent_samples, latent_means, latent_log_vars
        )
        
        return (reconstruction_loss, info_reconstruction_loss, kl_divergence,
                dip_loss, tc_loss, mmd_loss, ode_divergence, contrastive_loss)

    def _compute_standard_losses(self, outputs: tuple, states_query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute all loss components for standard (non-ODE) mode."""
        if self.loss_mode == "zinb":
            return self._compute_standard_zinb_losses(outputs, states_query)
        else:
            return self._compute_standard_other_losses(outputs, states_query)

    def _compute_standard_zinb_losses(self, outputs: tuple, states_query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute loss components for standard mode with ZINB loss."""
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
        
        # No ODE divergence in standard mode
        ode_divergence = torch.zeros(1).to(self.device)
        
        return (reconstruction_loss, info_reconstruction_loss, kl_divergence,
                dip_loss, tc_loss, mmd_loss, ode_divergence, contrastive_loss)

    def _compute_standard_other_losses(self, outputs: tuple, states_query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute loss components for standard mode with NB or MSE loss."""
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
                reconstruction_bottleneck_normalized = reconstruction_bottleneck * normalization_factor
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
        
        # No ODE divergence in standard mode
        ode_divergence = torch.zeros(1).to(self.device)
        
        return (reconstruction_loss, info_reconstruction_loss, kl_divergence,
                dip_loss, tc_loss, mmd_loss, ode_divergence, contrastive_loss)

    def _compute_nb_reconstruction_losses(
        self, 
        input_states: torch.Tensor,
        reconstruction_direct: torch.Tensor,
        reconstruction_ode: torch.Tensor,
        reconstruction_bottleneck: torch.Tensor,
        reconstruction_bottleneck_ode: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute NB reconstruction losses for ODE mode."""
        normalization_factor = input_states.sum(-1).view(-1, 1)
        reconstruction_direct_normalized = reconstruction_direct * normalization_factor
        reconstruction_ode_normalized = reconstruction_ode * normalization_factor
        dispersion = torch.exp(self.vae_model.decoder.dispersion_param)
        
        # Main reconstruction loss
        reconstruction_loss = -self._log_nb(
            input_states, reconstruction_direct_normalized, dispersion
        ).sum(-1).mean()
        reconstruction_loss += -self._log_nb(
            input_states, reconstruction_ode_normalized, dispersion
        ).sum(-1).mean()
        
        # Information reconstruction loss
        if self.irecon_weight:
            reconstruction_bottleneck_normalized = reconstruction_bottleneck * normalization_factor
            reconstruction_bottleneck_ode_normalized = reconstruction_bottleneck_ode * normalization_factor
            
            info_reconstruction_loss = -self.irecon_weight * self._log_nb(
                input_states, reconstruction_bottleneck_normalized, dispersion
            ).sum(-1).mean()
            info_reconstruction_loss += -self.irecon_weight * self._log_nb(
                input_states, reconstruction_bottleneck_ode_normalized, dispersion
            ).sum(-1).mean()
        else:
            info_reconstruction_loss = torch.zeros(1).to(self.device)
            
        return reconstruction_loss, info_reconstruction_loss

    def _compute_mse_reconstruction_losses(
        self,
        input_states: torch.Tensor,
        reconstruction_direct: torch.Tensor,
        reconstruction_ode: torch.Tensor,
        reconstruction_bottleneck: torch.Tensor,
        reconstruction_bottleneck_ode: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MSE reconstruction losses for ODE mode."""
        # Main reconstruction loss
        reconstruction_loss = F.mse_loss(
            input_states, reconstruction_direct, reduction="none"
        ).sum(-1).mean()
        reconstruction_loss += F.mse_loss(
            input_states, reconstruction_ode, reduction="none"
        ).sum(-1).mean()
        
        # Information reconstruction loss
        if self.irecon_weight:
            info_reconstruction_loss = self.irecon_weight * F.mse_loss(
                input_states, reconstruction_bottleneck, reduction="none"
            ).sum(-1).mean()
            info_reconstruction_loss += self.irecon_weight * F.mse_loss(
                input_states, reconstruction_bottleneck_ode, reduction="none"
            ).sum(-1).mean()
        else:
            info_reconstruction_loss = torch.zeros(1).to(self.device)
            
        return reconstruction_loss, info_reconstruction_loss

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
