
"""
Agent Class for MCCVAE Training and Analysis

This module provides a high-level interface for training and analyzing MCCVAE models.
It includes comprehensive functionality for:
- Model training with progress tracking
- Latent representation extraction for biological correlation learning
- Information bottleneck analysis for coupled representations
- Visualization and analysis utilities for single-cell genomics
"""

from .environment import Env
from anndata import AnnData
import torch
import tqdm
from typing import Literal
import numpy as np
import warnings



class Agent(Env):
    """
    High-level agent for MCCVAE model training and analysis.
    
    This class provides a comprehensive interface for training MoCo Coupling Variational 
    Autoencoders with contrastive learning capabilities designed for single-cell genomics. 
    It offers methods for:
    
    - Model training with automatic progress tracking
    - Latent representation and embedding extraction
    - Information bottleneck analysis for coupled biological representations
    - Visualization-ready outputs for single-cell analysis
    
    The agent integrates seamlessly with AnnData objects commonly used in
    single-cell analysis and focuses on learning coupled latent representations
    that capture biological correlations intrinsic to single-cell data.
    
    The information bottleneck component specifically learns compressed 
    representations that preserve essential biological correlations while 
    filtering out technical noise, enabling better understanding of cellular 
    states and biological processes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated single-cell data object containing the dataset
    layer : str, default="counts"
        Layer name in adata to use for training data
    percent : float, default=0.01
        Fraction of dataset to use per batch (0.0 to 1.0)
    recon : float, default=1.0
        Weight for reconstruction loss
    irecon : float, default=0.0
        Weight for information bottleneck reconstruction loss
    beta : float, default=1.0
        Weight for KL divergence regularization (β-VAE)
    dip : float, default=0.0
        Weight for DIP regularization
    tc : float, default=0.0
        Weight for total correlation regularization
    info : float, default=0.0
        Weight for InfoVAE MMD regularization
    hidden_dim : int, default=128
        Dimension of hidden layers in the network
    latent_dim : int, default=10
        Dimension of the latent space
    i_dim : int, default=2
        Dimension of information bottleneck for coupled biological representations
    use_moco : bool, default=False
        Whether to enable Momentum Contrast learning
    loss_mode : Literal["mse", "nb", "zinb"], default="nb"
        Loss function mode for reconstruction
    lr : float, default=1e-4
        Learning rate for optimization
    vae_reg : float, default=0.5
        Regularization weight for VAE latent representations
    moco_weight : float, default=1.0
        Weight for contrastive learning loss
    moco_T : float, default=0.2
        Temperature parameter for MoCo contrastive loss
    aug_prob : float, default=0.5
        Probability of applying data augmentation
    mask_prob : float, default=0.2
        Probability of masking each gene during augmentation
    noise_prob : float, default=0.7
        Probability of adding noise to each gene during augmentation
    use_qm : bool, default=True
        Whether to use mean (True) or samples (False) for latent representations
    device : torch.device
        Computation device (automatically selects CUDA if available)
    """
    
    def __init__(
        self,
        adata: AnnData,
        layer: str = "counts",
        percent: float = 0.01,
        recon: float = 1.0,
        irecon: float = 0.0,
        beta: float = 1.0,
        dip: float = 0.0,
        tc: float = 0.0,
        info: float = 0.0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        use_moco: bool = False,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        lr: float = 1e-4,
        vae_reg: float = 0.5,
        moco_weight: float = 1.0,
        moco_T: float = 0.2,
        aug_prob: float = 0.5,
        mask_prob: float = 0.2,
        noise_prob: float = 0.7,
        use_qm: bool = True,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        # Initialize the parent environment with all parameters
        super().__init__(
            adata=adata,
            layer=layer,
            percent=percent,
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_moco=use_moco,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            moco_weight=moco_weight,
            moco_T=moco_T,
            aug_prob=aug_prob,
            mask_prob=mask_prob,
            noise_prob=noise_prob,
            use_qm=use_qm,
            device=device,
        )

    def fit(self, epochs: int = 1000) -> 'Agent':
        """
        Train the MCCVAE model with progress tracking.
        
        This method performs the complete training loop with:
        - Automatic batch sampling and data loading
        - Progress bar with real-time metrics display
        - Periodic evaluation metric computation
        - Loss and performance tracking
        - Learning of coupled biological representations
        
        The training progress shows key metrics including:
        - Total loss value
        - ARI (Adjusted Rand Index) for clustering accuracy
        - NMI (Normalized Mutual Information) for clustering quality
        - ASW (Average Silhouette Width) for cluster separation
        - C_H (Calinski-Harabasz Index) for cluster validity
        - D_B (Davies-Bouldin Index) for cluster compactness
        - P_C (Pearson Correlation) for disentanglement measure
        
        Parameters
        ----------
        epochs : int, default=1000
            Number of training epochs to perform
            
        Returns
        -------
        Agent
            Self (for method chaining)
        """
        with tqdm.tqdm(total=int(epochs), desc="Training MCCVAE", ncols=150) as progress_bar:
            for epoch in range(int(epochs)):
                # Load batch data (with augmentation if using MoCo)
                primary_data, query_data, key_data = self.load_data()
                
                # Perform training step
                if self.use_moco:
                    self.step(primary_data, query_data, key_data)
                else:
                    self.step(primary_data)
                
                # Update progress display every 10 epochs
                if (epoch + 1) % 10 == 0:
                    latest_loss = self.training_losses[-1]
                    latest_scores = self.evaluation_scores[-1]
                    
                    progress_bar.set_postfix({
                        "Loss": f"{latest_loss[0]:.2f}",
                        "ARI": f"{latest_scores[0]:.2f}",
                        "NMI": f"{latest_scores[1]:.2f}",
                        "ASW": f"{latest_scores[2]:.2f}",
                        "C_H": f"{latest_scores[3]:.2f}",
                        "D_B": f"{latest_scores[4]:.2f}",
                        "P_C": f"{latest_scores[5]:.2f}",
                    })
                
                progress_bar.update(1)
        
        return self

    def get_iembed(self) -> np.ndarray:
        """
        Extract information bottleneck embeddings from the full dataset.
        
        These embeddings represent the coupled latent representations learned
        through the information bottleneck layer, specifically designed to capture
        biological correlations intrinsic to single-cell data. They provide a 
        compressed view of the latent space that preserves essential biological
        relationships while filtering out technical noise.
        
        Returns
        -------
        np.ndarray
            Information bottleneck embeddings with shape (n_cells, i_dim)
            representing coupled biological correlations
        """
        information_embeddings = self.take_iembed(self.dataset)
        return information_embeddings

    def get_latent(self) -> np.ndarray:
        """
        Extract latent representations from the full dataset.
        
        These are the primary latent space representations learned by the MCCVAE,
        capturing the underlying biological structure and cellular relationships
        in single-cell data. They provide a comprehensive view of cellular states
        encoded in the gene expression patterns.
        
        Returns
        -------
        np.ndarray
            Latent representations with shape (n_cells, latent_dim)
        """
        latent_representations = self.take_latent(self.dataset)
        return latent_representations

    def get_bottleneck(self) -> np.ndarray:
        """
        Extract the compressed bottleneck embeddings (l_e) from the full dataset.
        
        This is an alias for get_iembed() provided for consistency with the README
        documentation. The bottleneck embeddings represent the compressed information
        bottleneck layer that captures biological correlations.
        
        Returns
        -------
        np.ndarray
            Bottleneck embeddings with shape (n_cells, i_dim)
        """
        return self.get_iembed()

    def get_refined(self) -> np.ndarray:
        """
        Extract the refined latent representations (l_d) from the full dataset.
        
        The refined representations are obtained by passing the primary latent
        representations through the information bottleneck:
        1. Encoding data to latent space (z)
        2. Compressing through bottleneck encoder to get l_e
        3. Expanding through bottleneck decoder to get l_d
        4. l_d is used for secondary reconstruction in the coupling component
        
        These refined representations (l_d) have the same dimensionality as the
        original latent space (latent_dim) but have been refined through the
        information bottleneck, capturing essential biological correlations
        while filtering out technical noise.
        
        Returns
        -------
        np.ndarray
            Refined latent representations (l_d) with shape (n_cells, latent_dim)
        
        See Also
        --------
        get_iembed : Get information bottleneck embeddings (l_e) - the compressed representation
        get_bottleneck : Alias for get_iembed()
        get_latent : Get primary latent representations (z)
        """
        refined_representations = self.take_refined(self.dataset)
        return refined_representations
