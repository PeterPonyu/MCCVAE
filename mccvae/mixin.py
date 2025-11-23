
"""
Mixin Classes for Variational Autoencoder Components

This module provides a collection of mixin classes that implement various advanced
techniques for variational autoencoders, including:
- Statistical distributions and loss functions (scVI-style)
- Neural Ordinary Differential Equations (Neural ODE)
- β-TC-VAE total correlation computation
- InfoVAE with Maximum Mean Discrepancy (MMD)
- Disentangled Inferred Prior (DIP) regularization
- Comprehensive evaluation metrics
"""

import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from typing import Optional, Tuple, Dict, Any
import math


class scviMixin:
    """
    Mixin class providing statistical distribution functions for single-cell variational inference.
    
    This mixin implements key probability distributions commonly used in single-cell analysis:
    - Normal distribution KL divergence
    - Negative Binomial (NB) log-likelihood
    - Zero-Inflated Negative Binomial (ZINB) log-likelihood
    
    These functions are essential for modeling count data in single-cell genomics applications.
    """

    def _normal_kl(
        self, 
        mean_1: torch.Tensor, 
        log_var_1: torch.Tensor, 
        mean_2: torch.Tensor, 
        log_var_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between two multivariate normal distributions.
        
        This function calculates KL(N(μ₁, σ₁²) || N(μ₂, σ₂²)) for each dimension
        independently, which is commonly used in VAE training for regularization.
        
        Parameters
        ----------
        mean_1 : torch.Tensor
            Mean of the first (posterior) distribution
        log_var_1 : torch.Tensor
            Log variance of the first (posterior) distribution
        mean_2 : torch.Tensor
            Mean of the second (prior) distribution
        log_var_2 : torch.Tensor
            Log variance of the second (prior) distribution
            
        Returns
        -------
        torch.Tensor
            KL divergence values for each dimension
        """
        variance_1 = torch.exp(log_var_1)
        variance_2 = torch.exp(log_var_2)
        log_std_1 = log_var_1 / 2.0
        log_std_2 = log_var_2 / 2.0
        
        kl_divergence = (
            log_std_2 - log_std_1 + 
            (variance_1 + (mean_1 - mean_2) ** 2.0) / (2.0 * variance_2) - 
            0.5
        )
        
        return kl_divergence

    def _log_nb(
        self, 
        x: torch.Tensor, 
        mean_param: torch.Tensor, 
        dispersion_param: torch.Tensor, 
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute log-likelihood under Negative Binomial distribution.
        
        The Negative Binomial distribution is parameterized by mean μ and dispersion θ,
        which is particularly suitable for modeling overdispersed count data in genomics.
        
        Parameters
        ----------
        x : torch.Tensor
            Observed count data
        mean_param : torch.Tensor
            Mean parameter (μ) of the NB distribution
        dispersion_param : torch.Tensor
            Dispersion parameter (θ) of the NB distribution
        eps : float, default=1e-8
            Small constant for numerical stability
            
        Returns
        -------
        torch.Tensor
            Log-likelihood values under the NB distribution
        """
        log_theta_mu_eps = torch.log(dispersion_param + mean_param + eps)
        
        log_likelihood = (
            dispersion_param * (torch.log(dispersion_param + eps) - log_theta_mu_eps) +
            x * (torch.log(mean_param + eps) - log_theta_mu_eps) +
            torch.lgamma(x + dispersion_param) -
            torch.lgamma(dispersion_param) -
            torch.lgamma(x + 1)
        )
        
        return log_likelihood

    def _log_zinb(
        self, 
        x: torch.Tensor, 
        mean_param: torch.Tensor, 
        dispersion_param: torch.Tensor, 
        dropout_logits: torch.Tensor, 
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute log-likelihood under Zero-Inflated Negative Binomial distribution.
        
        ZINB is a mixture model combining a point mass at zero (dropout component)
        and a Negative Binomial distribution. It's ideal for sparse count data
        where excess zeros occur due to technical or biological dropout.
        
        Parameters
        ----------
        x : torch.Tensor
            Observed count data
        mean_param : torch.Tensor
            Mean parameter (μ) of the NB component
        dispersion_param : torch.Tensor
            Dispersion parameter (θ) of the NB component
        dropout_logits : torch.Tensor
            Logits for the dropout probability (higher = more dropout)
        eps : float, default=1e-8
            Small constant for numerical stability
            
        Returns
        -------
        torch.Tensor
            Log-likelihood values under the ZINB distribution
        """
        softplus_dropout = F.softplus(-dropout_logits)
        log_theta_eps = torch.log(dispersion_param + eps)
        log_theta_mu_eps = torch.log(dispersion_param + mean_param + eps)
        dropout_theta_log = -dropout_logits + dispersion_param * (log_theta_eps - log_theta_mu_eps)

        # Case when x = 0 (zero inflation + NB zeros)
        zero_case = F.softplus(dropout_theta_log) - softplus_dropout
        zero_mask = (x < eps).type(torch.float32)
        masked_zero_case = torch.mul(zero_mask, zero_case)

        # Case when x > 0 (only NB component)
        non_zero_case = (
            -softplus_dropout +
            dropout_theta_log +
            x * (torch.log(mean_param + eps) - log_theta_mu_eps) +
            torch.lgamma(x + dispersion_param) -
            torch.lgamma(dispersion_param) -
            torch.lgamma(x + 1)
        )
        non_zero_mask = (x > eps).type(torch.float32)
        masked_non_zero_case = torch.mul(non_zero_mask, non_zero_case)

        log_likelihood = masked_zero_case + masked_non_zero_case
        return log_likelihood


class betatcMixin:
    """
    Mixin class implementing β-TC-VAE total correlation computation.
    
    β-TC-VAE (β-Total Correlation VAE) provides a principled approach to
    disentangled representation learning by decomposing the KL term into
    three components: index-code mutual information, total correlation,
    and dimension-wise KL divergence.
    
    This implementation focuses on the total correlation component, which
    measures statistical dependence between latent dimensions.
    """

    def _betatc_compute_gaussian_log_density(
        self, 
        samples: torch.Tensor, 
        mean: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log density of samples under multivariate Gaussian distributions.
        
        This function efficiently computes log p(z|μ, σ²) for multiple
        Gaussian distributions simultaneously, which is needed for the
        total correlation computation.
        
        Parameters
        ----------
        samples : torch.Tensor
            Sample points to evaluate, shape (n_samples, n_dims)
        mean : torch.Tensor
            Mean parameters, shape (n_distributions, n_dims)
        log_var : torch.Tensor
            Log variance parameters, shape (n_distributions, n_dims)
            
        Returns
        -------
        torch.Tensor
            Log density values for each sample under each distribution
        """
        pi_constant = torch.tensor(math.pi, requires_grad=False)
        normalization_term = torch.log(2 * pi_constant)
        inverse_variance = torch.exp(-log_var)
        deviation = samples - mean
        
        log_density = -0.5 * (
            deviation * deviation * inverse_variance + 
            log_var + 
            normalization_term
        )
        
        return log_density

    def _betatc_compute_total_correlation(
        self, 
        latent_samples: torch.Tensor, 
        latent_means: torch.Tensor, 
        latent_log_vars: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total correlation component of the β-TC-VAE objective.
        
        Total correlation TC(z) measures how much the joint distribution
        q(z) deviates from the product of marginals ∏ᵢq(zᵢ). Lower TC
        indicates more disentangled representations.
        
        TC(z) = 𝔼[log q(z) - ∑ᵢ log qᵢ(zᵢ)]
        
        Parameters
        ----------
        latent_samples : torch.Tensor
            Sampled latent codes, shape (batch_size, latent_dim)
        latent_means : torch.Tensor
            Mean parameters of posterior, shape (batch_size, latent_dim)
        latent_log_vars : torch.Tensor
            Log variance parameters of posterior, shape (batch_size, latent_dim)
            
        Returns
        -------
        torch.Tensor
            Total correlation estimate
        """
        # Compute log q(z_j|x_i) for all pairs of samples and data points
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            latent_samples.unsqueeze(dim=1),    # (batch_size, 1, latent_dim)
            latent_means.unsqueeze(dim=0),      # (1, batch_size, latent_dim)
            latent_log_vars.unsqueeze(dim=0),   # (1, batch_size, latent_dim)
        )
        
        # Compute log ∏ᵢ qᵢ(zᵢ) = ∑ᵢ log qᵢ(zᵢ)
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        
        # Compute log q(z) = log ∑ⱼ q(z|xⱼ)
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        
        # Total correlation: 𝔼[log q(z) - log ∏ᵢ qᵢ(zᵢ)]
        total_correlation = (log_qz - log_qz_product).mean()
        
        return total_correlation


class infoMixin:
    """
    Mixin class implementing InfoVAE with Maximum Mean Discrepancy (MMD).
    
    InfoVAE addresses posterior collapse in VAEs by replacing the KL divergence
    with Maximum Mean Discrepancy (MMD), which measures the distance between
    distributions in a reproducing kernel Hilbert space (RKHS).
    
    MMD provides a more flexible way to match the aggregated posterior
    to the prior distribution while preserving mutual information.
    """

    def _compute_mmd(
        self, 
        posterior_samples: torch.Tensor, 
        prior_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy between posterior and prior samples.
        
        MMD²(P, Q) = 𝔼[k(x, x')] - 2𝔼[k(x, y)] + 𝔼[k(y, y')]
        where x ~ P, y ~ Q, and k is the kernel function.
        
        Parameters
        ----------
        posterior_samples : torch.Tensor
            Samples from the posterior distribution q(z|x)
        prior_samples : torch.Tensor
            Samples from the prior distribution p(z)
            
        Returns
        -------
        torch.Tensor
            MMD² estimate between the two distributions
        """
        # Compute kernel matrices
        prior_prior_kernel = self._compute_kernel(prior_samples, prior_samples)
        prior_posterior_kernel = self._compute_kernel(prior_samples, posterior_samples)
        posterior_posterior_kernel = self._compute_kernel(posterior_samples, posterior_samples)
        
        # Compute unbiased estimates of expectations
        mean_prior_prior = self._compute_unbiased_mean(prior_prior_kernel, unbiased=True)
        mean_prior_posterior = self._compute_unbiased_mean(prior_posterior_kernel, unbiased=False)
        mean_posterior_posterior = self._compute_unbiased_mean(posterior_posterior_kernel, unbiased=True)
        
        # MMD² = E[k(p,p)] - 2E[k(p,q)] + E[k(q,q)]
        mmd_squared = mean_prior_prior - 2 * mean_prior_posterior + mean_posterior_posterior
        
        return mmd_squared

    def _compute_unbiased_mean(
        self, 
        kernel_matrix: torch.Tensor, 
        unbiased: bool
    ) -> torch.Tensor:
        """
        Compute (un)biased estimate of kernel expectation.
        
        For unbiased estimation, we exclude diagonal terms to avoid
        the bias introduced by k(x, x) terms in finite samples.
        
        Parameters
        ----------
        kernel_matrix : torch.Tensor
            Kernel matrix K with K[i,j] = k(x_i, x_j)
        unbiased : bool
            Whether to compute unbiased estimate (exclude diagonal)
            
        Returns
        -------
        torch.Tensor
            Mean kernel value
        """
        n_samples, m_samples = kernel_matrix.shape
        
        if unbiased:
            # Exclude diagonal terms: (∑K - ∑diag(K)) / (N(N-1))
            total_sum = kernel_matrix.sum(dim=(0, 1))
            diagonal_sum = torch.diagonal(kernel_matrix, dim1=0, dim2=1).sum(dim=-1)
            mean_kernel = (total_sum - diagonal_sum) / (n_samples * (n_samples - 1))
        else:
            # Include all terms: ∑K / (N*M)
            mean_kernel = kernel_matrix.mean(dim=(0, 1))
            
        return mean_kernel

    def _compute_kernel(
        self, 
        samples_1: torch.Tensor, 
        samples_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise kernel matrix between two sets of samples.
        
        Parameters
        ----------
        samples_1 : torch.Tensor
            First set of samples, shape (n_samples, feature_dim)
        samples_2 : torch.Tensor
            Second set of samples, shape (m_samples, feature_dim)
            
        Returns
        -------
        torch.Tensor
            Kernel matrix with shape (n_samples, m_samples)
        """
        batch_size, feature_dim = samples_1.shape
        
        # Expand for pairwise computation
        expanded_samples_1 = samples_1.unsqueeze(-2)  # (batch_size, 1, feature_dim)
        expanded_samples_2 = samples_2.unsqueeze(-3)  # (1, batch_size, feature_dim)
        
        expanded_samples_1 = expanded_samples_1.expand(batch_size, batch_size, feature_dim)
        expanded_samples_2 = expanded_samples_2.expand(batch_size, batch_size, feature_dim)
        
        kernel_matrix = self._kernel_rbf(expanded_samples_1, expanded_samples_2)
        
        return kernel_matrix

    def _kernel_rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF (Gaussian) kernel between input tensors.
        
        k(x, y) = exp(-||x - y||² / σ²)
        The bandwidth σ² is set to 2 * 2 * feature_dim following common practice.
        
        Parameters
        ----------
        x : torch.Tensor
            First input tensor
        y : torch.Tensor
            Second input tensor
            
        Returns
        -------
        torch.Tensor
            RBF kernel values
        """
        feature_dim = x.shape[-1]
        bandwidth = 2 * 2 * feature_dim  # Heuristic bandwidth selection
        
        squared_distance = (x - y).pow(2).sum(dim=-1)
        kernel_values = torch.exp(-squared_distance / bandwidth)
        
        return kernel_values


class dipMixin:
    """
    Mixin class implementing Disentangled Inferred Prior (DIP) regularization.
    
    DIP-VAE encourages disentangled representations by matching the covariance
    matrix of the aggregated posterior to an identity matrix. This promotes
    both unit variance (diagonal) and independence (off-diagonal) of latent factors.
    
    The DIP loss has two components:
    - Diagonal regularization: encourages unit variance for each factor
    - Off-diagonal regularization: encourages independence between factors
    """

    def _dip_loss(
        self, 
        latent_means: torch.Tensor, 
        latent_log_vars: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Disentangled Inferred Prior (DIP) loss.
        
        DIP loss = λ_d * ||diag(Cov) - I||² + λ_od * ||off_diag(Cov)||²
        where Cov is the covariance matrix of the aggregated posterior.
        
        Parameters
        ----------
        latent_means : torch.Tensor
            Mean parameters of the posterior, shape (batch_size, latent_dim)
        latent_log_vars : torch.Tensor
            Log variance parameters of the posterior, shape (batch_size, latent_dim)
            
        Returns
        -------
        torch.Tensor
            DIP loss value
        """
        covariance_matrix = self._dip_cov_matrix(latent_means, latent_log_vars)
        
        # Extract diagonal and off-diagonal elements
        diagonal_elements = torch.diagonal(covariance_matrix)
        off_diagonal_matrix = covariance_matrix - torch.diag(diagonal_elements)
        
        # Diagonal regularization: ||diag(Cov) - 1||²
        diagonal_loss = torch.sum((diagonal_elements - 1) ** 2)
        
        # Off-diagonal regularization: ||off_diag(Cov)||²
        off_diagonal_loss = torch.sum(off_diagonal_matrix ** 2)
        
        # Weighted combination (λ_d = 10, λ_od = 5 as per original paper)
        total_dip_loss = 10 * diagonal_loss + 5 * off_diagonal_loss
        
        return total_dip_loss

    def _dip_cov_matrix(
        self, 
        latent_means: torch.Tensor, 
        latent_log_vars: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute covariance matrix of the aggregated posterior.
        
        The covariance matrix combines:
        1. Sample covariance of the means: Cov[𝔼[z|x]]
        2. Expected variance: 𝔼[Var[z|x]]
        
        This gives the total covariance: Cov[z] = Cov[𝔼[z|x]] + 𝔼[Var[z|x]]
        
        Parameters
        ----------
        latent_means : torch.Tensor
            Mean parameters, shape (batch_size, latent_dim)
        latent_log_vars : torch.Tensor
            Log variance parameters, shape (batch_size, latent_dim)
            
        Returns
        -------
        torch.Tensor
            Covariance matrix of shape (latent_dim, latent_dim)
        """
        # Sample covariance of means
        mean_covariance = torch.cov(latent_means.T)
        
        # Expected diagonal variance
        expected_variances = torch.mean(torch.exp(latent_log_vars), dim=0)
        
        # Total covariance matrix
        total_covariance = mean_covariance + torch.diag(expected_variances)
        
        return total_covariance


class envMixin:
    """
    Mixin class providing comprehensive evaluation metrics for representation learning.
    
    This mixin implements various metrics commonly used to assess the quality
    of learned representations, including:
    - Clustering-based metrics (ARI, NMI, Silhouette)
    - Cluster validity indices (Calinski-Harabasz, Davies-Bouldin)
    - Correlation-based disentanglement measure
    
    These metrics are essential for evaluating both the quality and
    disentanglement properties of learned latent representations.
    """

    def _calc_score(self, latent_representation: np.ndarray, ifall: bool = False) -> Tuple:
        """
        Calculate comprehensive evaluation scores for latent representations.
        
        Parameters
        ----------
        latent_representation : np.ndarray
            Learned latent representations, shape (n_samples, latent_dim)
        ifall : bool, default=False
            Whether to use all samples (True) or only indexed samples (False)
            
        Returns
        -------
        Tuple
            Evaluation metrics: (ARI, NMI, ASW, C_H, D_B, P_C)
        """
        n_clusters = latent_representation.shape[1]  # Use latent dimension as cluster count
        cluster_labels = self._calc_label(latent_representation)
        evaluation_scores = self._compute_metrics(latent_representation, cluster_labels, ifall)
        
        return evaluation_scores

    def _calc_label(self, latent_representation: np.ndarray) -> np.ndarray:
        """
        Generate cluster labels using K-means clustering.
        
        Parameters
        ----------
        latent_representation : np.ndarray
            Latent representations to cluster
            
        Returns
        -------
        np.ndarray
            Cluster labels for each sample
        """
        n_clusters = latent_representation.shape[1]
        cluster_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(latent_representation)
        
        return cluster_labels

    def _calc_corr(self, latent_representation: np.ndarray) -> float:
        """
        Calculate correlation-based disentanglement measure.
        
        This metric measures the average absolute correlation between
        latent dimensions. Lower values indicate better disentanglement
        (more independent factors).
        
        Parameters
        ----------
        latent_representation : np.ndarray
            Latent representations, shape (n_samples, latent_dim)
            
        Returns
        -------
        float
            Average absolute off-diagonal correlation
        """
        correlation_matrix = abs(np.corrcoef(latent_representation.T))
        
        # Average correlation excluding diagonal (subtract 1 for self-correlation)
        # BUG FIX: The original implementation incorrectly computed the mean
        # It should properly exclude the diagonal elements (which are always 1)
        # Original: correlation_matrix.sum(axis=1).mean().item() - 1
        # Corrected: Explicitly exclude diagonal
        n_dims = correlation_matrix.shape[0]
        if n_dims <= 1:
            # Edge case: only one dimension, no off-diagonal correlations
            return 0.0
        
        # Sum all correlations and subtract diagonal, then divide by number of off-diagonal elements
        total_correlation = correlation_matrix.sum()
        diagonal_sum = np.trace(correlation_matrix)
        off_diagonal_sum = total_correlation - diagonal_sum
        n_off_diagonal = n_dims * n_dims - n_dims
        mean_correlation = off_diagonal_sum / n_off_diagonal if n_off_diagonal > 0 else 0.0
        
        return mean_correlation

    def _compute_metrics(
        self, 
        latent_representation: np.ndarray, 
        cluster_labels: np.ndarray, 
        ifall: bool
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Compute comprehensive evaluation metrics for clustering quality.
        
        Parameters
        ----------
        latent_representation : np.ndarray
            Latent representations
        cluster_labels : np.ndarray
            Predicted cluster labels
        ifall : bool
            Whether to use all labels or indexed subset
            
        Returns
        -------
        Tuple[float, float, float, float, float, float]
            Evaluation metrics: (ARI, NMI, ASW, C_H, D_B, P_C)
            - ARI: Adjusted Rand Index (clustering accuracy)
            - NMI: Normalized Mutual Information (clustering accuracy)  
            - ASW: Average Silhouette Width (cluster separation)
            - C_H: Calinski-Harabasz Index (cluster validity)
            - D_B: Davies-Bouldin Index (cluster compactness, lower is better)
            - P_C: Pearson Correlation (disentanglement measure)
        """
        # Select appropriate ground truth labels
        true_labels = self.labels if ifall else self.labels[self.current_batch_indices]
        
        # Clustering accuracy metrics
        ari_score = adjusted_mutual_info_score(true_labels, cluster_labels)
        nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
        
        # Cluster quality metrics
        silhouette_avg = silhouette_score(latent_representation, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(latent_representation, cluster_labels)
        davies_bouldin = davies_bouldin_score(latent_representation, cluster_labels)
        
        # Disentanglement metric
        correlation_measure = self._calc_corr(latent_representation)
        
        return ari_score, nmi_score, silhouette_avg, calinski_harabasz, davies_bouldin, correlation_measure
