
"""
Training Environment for MCCVAE Model

This module provides a comprehensive training environment that combines the MCCVAE model
with evaluation capabilities. It handles data loading, augmentation, batch sampling,
and provides a unified interface for training and evaluation.

The environment supports:
- Batch sampling with configurable batch sizes
- Data augmentation for contrastive learning (MoCo)
- Automatic evaluation metric computation
- Integration with AnnData objects for single-cell analysis
- Information bottleneck learning for coupled latent representations that capture
  biological correlations intrinsic to single-cell data
"""

from .model import MCCVAE
from .mixin import envMixin
import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import Tuple, Optional, Union
import anndata


class Env(MCCVAE, envMixin):
    """
    Training environment for MCCVAE with integrated evaluation capabilities.
    
    This class combines the MCCVAE model with data management and evaluation
    functionality. It provides a streamlined interface for training MoCo Coupling
    Variational Autoencoders with various regularization techniques while automatically
    tracking performance metrics.
    
    The environment handles:
    - Data sampling and batch creation
    - Data augmentation for contrastive learning
    - Training step execution
    - Performance metric computation and tracking
    - Information bottleneck learning for coupled biological representations
    
    The information bottleneck component learns coupled latent representations
    that capture biological correlations intrinsic to single-cell data, enabling
    better understanding of cellular relationships, gene expression patterns,
    and biological processes.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing single-cell dataset
    layer : str
        Layer name in adata to use for training data
    percent : float
        Fraction of dataset to use per batch (0.0 to 1.0)
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
    hidden_dim : int
        Dimension of hidden layers in the network
    latent_dim : int
        Dimension of the latent space
    i_dim : int
        Dimension of information bottleneck for coupled biological representations
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
    moco_T : float
        Temperature parameter for MoCo contrastive loss
    aug_prob : float
        Probability of applying data augmentation
    mask_prob : float
        Probability of masking each feature during augmentation
    noise_prob : float
        Probability of adding noise to each feature during augmentation
    use_qm : bool
        Whether to use mean (True) or samples (False) for latent representations
    device : torch.device
        Computation device
    """
    
    def __init__(
        self,
        adata: anndata.AnnData,
        layer: str,
        percent: float,
        recon: float,
        irecon: float,
        beta: float,
        dip: float,
        tc: float,
        info: float,
        hidden_dim: int,
        latent_dim: int,
        i_dim: int,
        use_moco: bool,
        loss_mode: str,
        lr: float,
        vae_reg: float,
        moco_weight: float,
        moco_T: float,
        aug_prob: float,
        mask_prob: float,
        noise_prob: float,
        use_qm: bool,
        device: torch.device,
        *args,
        **kwargs,
    ):
        # Register and process the input single-cell data
        self._register_anndata(adata, layer, latent_dim)
        
        # Configure batch sampling
        self.batch_size = int(percent * self.n_observations)
        
        # Store augmentation parameters for single-cell data
        self.augmentation_probability = aug_prob
        self.masking_probability = mask_prob
        self.noise_probability = noise_prob
        
        # Initialize the parent MCCVAE model
        super().__init__(
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            state_dim=self.n_variables,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_moco=use_moco,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            moco_weight=moco_weight,
            moco_T=moco_T,
            use_qm=use_qm,
            device=device,
        )
        
        # Initialize performance tracking
        self.evaluation_scores = []

    def load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load and sample training data for one batch.
        
        This method samples a batch of single-cell data from the dataset and applies
        augmentation if contrastive learning (MoCo) is enabled. For MoCo,
        it returns three versions: original data and two augmented versions
        for query and key encoders.
        
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
            - data: Original or primary training data (cell expression profiles)
            - data_query: Query data for MoCo (None if not using MoCo)
            - data_key: Key data for MoCo (None if not using MoCo)
        """
        if self.use_moco:
            primary_data, query_data, key_data, sample_indices = self._sample_data()
            self.current_batch_indices = sample_indices
            return primary_data, query_data, key_data
        else:
            primary_data, sample_indices = self._sample_data()
            self.current_batch_indices = sample_indices
            return primary_data, None, None

    def step(self, *data_batch: np.ndarray) -> None:
        """
        Execute one training step with evaluation.
        
        This method performs a complete training iteration:
        1. Updates model parameters using the provided single-cell data
        2. Extracts latent representations including coupled biological features
        3. Computes evaluation metrics
        4. Stores the results for tracking
        
        Parameters
        ----------
        *data_batch : np.ndarray
            Variable number of data arrays:
            - Standard mode: (primary_data,)
            - MoCo mode: (primary_data, query_data, key_data)
        """
        if self.use_moco:
            # Update model with contrastive learning
            self.update(data_batch[0], data_batch[1], data_batch[2])
            # Extract latent representations from primary data
            latent_representations = self.take_latent(data_batch[0])
        else:
            # Standard VAE training
            self.update(data_batch[0])
            latent_representations = self.take_latent(data_batch[0])
        
        # Compute evaluation metrics
        evaluation_metrics = self._calc_score(latent_representations)
        self.evaluation_scores.append(evaluation_metrics)

    def _sample_data(self) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Sample a batch of single-cell data from the dataset.
        
        This method randomly samples cell indices and extracts the corresponding
        gene expression profiles. For MoCo training, it also creates augmented versions.
        
        Returns
        -------
        Union[Tuple, Tuple]
            For standard mode: (data_batch, sample_indices)
            For MoCo mode: (data_batch, query_batch, key_batch, sample_indices)
        """
        # Generate random permutation of all cell indices
        shuffled_indices = np.random.permutation(self.n_observations)
        
        # Select batch_size indices randomly
        # NOTE: This is redundant since we're already shuffling above
        # Could be simplified to: selected_indices = shuffled_indices[:self.batch_size]
        # However, keeping for backward compatibility and explicit randomness
        selected_indices = np.random.choice(shuffled_indices, self.batch_size, replace=False)
        
        # Extract the corresponding cell expression data
        data_batch = self.dataset[selected_indices, :]
        
        if self.use_moco:
            # Create augmented versions for contrastive learning
            query_batch = self._augment_data(data_batch)
            key_batch = self._augment_data(data_batch)
            return data_batch, query_batch, key_batch, selected_indices
        else:
            return data_batch, selected_indices

    def _augment_data(self, data_profile: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply data augmentation for contrastive learning in single-cell data.
        
        This method applies stochastic augmentations tailored for single-cell
        gene expression data, including:
        - Random gene masking (setting gene expressions to zero)
        - Gaussian noise injection to simulate technical noise
        
        These augmentations help the model learn robust representations
        by forcing it to be invariant to technical variations while preserving
        biological signal for coupled representation learning.
        
        Parameters
        ----------
        data_profile : Union[np.ndarray, torch.Tensor]
            Input single-cell expression data to augment, shape (batch_size, n_genes)
            
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Augmented single-cell data with the same type and shape as input
        """
        # Handle tensor inputs by converting to numpy
        is_tensor_input = isinstance(data_profile, torch.Tensor)
        if is_tensor_input:
            profile_array = data_profile.cpu().numpy()
            original_device = data_profile.device
        else:
            profile_array = data_profile.copy()
        
        # Ensure float32 precision for numerical stability
        profile_array = profile_array.astype(np.float32)
        
        # Apply augmentation with specified probability
        if np.random.rand() < self.augmentation_probability:
            
            # Gene masking: randomly set some gene expressions to zero
            # This simulates dropout events common in single-cell data
            masking_pattern = np.random.choice(
                [True, False], 
                size=self.n_variables, 
                p=[self.masking_probability, 1 - self.masking_probability]
            )
            profile_array[:, masking_pattern] = 0
            
            # Gaussian noise injection: add noise to randomly selected genes
            # This simulates technical noise in single-cell measurements
            # NOTE: Noise is applied AFTER masking, so masked genes remain zero
            # and only non-masked genes receive noise
            noise_pattern = np.random.choice(
                [True, False], 
                size=self.n_variables, 
                p=[self.noise_probability, 1 - self.noise_probability]
            )
            
            if np.any(noise_pattern):
                noise_values = np.random.normal(
                    loc=0.0, 
                    scale=0.2, 
                    size=(profile_array.shape[0], np.sum(noise_pattern))
                )
                profile_array[:, noise_pattern] += noise_values
        
        # Ensure non-negative values for count data compatibility
        # NOTE: This clipping is important because Gaussian noise can produce negative values
        # which would be invalid for count-based loss functions (NB, ZINB)
        profile_array = np.clip(profile_array, 0, None)
        
        # Return in the same format as input
        if is_tensor_input:
            return torch.from_numpy(profile_array).to(original_device)
        else:
            return profile_array

    def _register_anndata(self, adata: anndata.AnnData, layer: str, latent_dim: int) -> None:
        """
        Register and preprocess AnnData object for single-cell training.
        
        This method extracts the single-cell expression data from the specified layer,
        stores dataset dimensions, and creates initial cluster labels for evaluation
        purposes. The data is prepared for learning coupled biological representations.
        
        Parameters
        ----------
        adata : anndata.AnnData
            Annotated single-cell data object containing the dataset
        layer : str
            Name of the layer to extract (e.g., 'X', 'raw', 'counts')
        latent_dim : int
            Dimension of latent space (used for initial clustering)
        """
        # Extract single-cell expression data from specified layer
        self.dataset = adata.layers[layer].toarray()
        
        # Store dataset dimensions
        self.n_observations = adata.shape[0]  # Number of cells
        self.n_variables = adata.shape[1]     # Number of genes
        
        # Create initial cluster labels for evaluation
        # This provides reference clusters for computing evaluation metrics
        # that assess how well the coupled biological representations
        # capture cellular heterogeneity and biological structure
        clustering_model = KMeans(n_clusters=latent_dim, random_state=42)
        self.labels = clustering_model.fit_predict(self.dataset)
        
        return
