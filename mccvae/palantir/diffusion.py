"""
Core diffusion map computation and eigenvalue extraction for Palantir
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import warnings


def compute_adaptive_kernel(data, knn=30, alpha=0.0, use_adjacency=False, distances=None):
    """
    Compute adaptive anisotropic kernel matrix.
    
    Parameters
    ----------
    data : ndarray or sparse matrix
        (n_cells, n_features) array or PCA projections
    knn : int, default=30
        Number of nearest neighbors for k-NN graph
    alpha : float, default=0.0
        Anisotropy degree (0 = no anisotropy, 1 = full anisotropy)
    use_adjacency : bool, default=False
        If True, use precomputed distances
    distances : csr_matrix, optional
        Precomputed distance matrix
        
    Returns
    -------
    K : csr_matrix
        (n_cells, n_cells) adaptive kernel matrix
    """
    n_cells = data.shape[0]
    
    # Compute pairwise distances
    if use_adjacency and distances is not None:
        # Extract distances from adjacency matrix
        D = distances.copy()
    else:
        # Convert sparse data to dense for distance computation
        if issparse(data):
            data_dense = data.toarray()
        else:
            data_dense = data
        
        # Compute Euclidean distances
        D = cdist(data_dense, data_dense, metric='euclidean')
    
    # Find k-nearest neighbors for each cell
    if not use_adjacency:
        # Use actual distances
        knn_indices = np.argsort(D, axis=1)[:, 1:knn+1]  # exclude self
        knn_distances = np.sort(D, axis=1)[:, 1:knn+1]
    else:
        # From sparse distance matrix
        knn_indices = np.zeros((n_cells, knn), dtype=int)
        knn_distances = np.zeros((n_cells, knn))
        
        if issparse(D):
            for i in range(n_cells):
                row = D.getrow(i).toarray().flatten()
                sorted_indices = np.argsort(row)
                knn_indices[i] = sorted_indices[1:knn+1]
                knn_distances[i] = row[sorted_indices[1:knn+1]]
        else:
            for i in range(n_cells):
                sorted_indices = np.argsort(D[i])
                knn_indices[i] = sorted_indices[1:knn+1]
                knn_distances[i] = D[i, sorted_indices[1:knn+1]]
    
    # Compute adaptive bandwidth (sigma) for each cell
    # Use k/3-th nearest neighbor distance
    sigma_idx = max(0, knn // 3 - 1)
    sigma = knn_distances[:, sigma_idx].copy()
    sigma[sigma == 0] = 1.0  # Avoid division by zero
    
    # Build kernel matrix using adaptive anisotropic kernel
    # K_ij = exp(-(d_ij^2) / (sigma_i * sigma_j))
    K_data = []
    K_row = []
    K_col = []
    
    for i in range(n_cells):
        for k, j in enumerate(knn_indices[i]):
            d_ij = knn_distances[i, k]
            denom = np.sqrt(sigma[i] * sigma[j])
            if denom > 0:
                k_val = np.exp(-(d_ij ** 2) / (denom ** 2))
            else:
                k_val = 0.0
            
            if k_val > 1e-10:  # Only store non-negligible values
                K_data.append(k_val)
                K_row.append(i)
                K_col.append(j)
                
                # Add symmetric entry
                K_data.append(k_val)
                K_row.append(j)
                K_col.append(i)
    
    K = csr_matrix((K_data, (K_row, K_col)), shape=(n_cells, n_cells))
    K.eliminate_zeros()
    
    # Apply alpha-normalization if specified
    if alpha > 0:
        # D_alpha normalization: K' = D^(-alpha) K D^(-alpha)
        degrees = np.asarray(K.sum(axis=1)).flatten()
        d_alpha = np.power(degrees + 1e-10, -alpha)
        D_alpha = diags(d_alpha)
        K = D_alpha @ K @ D_alpha
        K = csr_matrix(K)
    
    return K


def compute_transition_matrix(K):
    """
    Compute transition matrix from kernel matrix.
    
    Parameters
    ----------
    K : csr_matrix
        (n_cells, n_cells) kernel matrix
        
    Returns
    -------
    T : csr_matrix
        (n_cells, n_cells) row-stochastic transition matrix
    """
    # Compute degree (row sums)
    degrees = np.asarray(K.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0  # Avoid division by zero
    
    # T = D^(-1) K
    inv_degrees = 1.0 / degrees
    D_inv = diags(inv_degrees)
    T = D_inv @ K
    
    return csr_matrix(T)


def run_diffusion_maps(data, n_components=10, knn=30, alpha=0.0, 
                       use_adjacency=False, distances=None):
    """
    Run diffusion maps on the data.
    
    Parameters
    ----------
    data : ndarray or sparse matrix
        (n_cells, n_features) array
    n_components : int, default=10
        Number of eigenvectors to compute
    knn : int, default=30
        Number of nearest neighbors
    alpha : float, default=0.0
        Anisotropy normalization parameter
    use_adjacency : bool, default=False
        If True, use precomputed adjacency
    distances : csr_matrix, optional
        Precomputed distances
        
    Returns
    -------
    eigenvectors : ndarray
        (n_cells, n_components) diffusion components
    eigenvalues : ndarray
        (n_components,) sorted eigenvalues
    transition_matrix : csr_matrix
        (n_cells, n_cells) transition matrix
    """
    # Compute kernel
    K = compute_adaptive_kernel(data, knn=knn, alpha=alpha, 
                                use_adjacency=use_adjacency, distances=distances)
    
    # Compute transition matrix
    T = compute_transition_matrix(K)
    
    # Compute eigendecomposition of T
    # We want the top n_components+1 eigenvectors (skip first trivial eigenvalue=1)
    n_eigs = min(n_components + 1, T.shape[0] - 2)
    
    try:
        eigenvalues, eigenvectors = eigsh(T, k=n_eigs, which='LM', v0=np.ones(T.shape[0]))
    except Exception as e:
        warnings.warn(f"Eigendecomposition failed: {e}. Using dense eigendecomposition.")
        # Fallback to dense computation
        T_dense = T.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(T_dense)
        # Sort by descending eigenvalues
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx][:n_eigs]
        eigenvectors = eigenvectors[:, idx][:, :n_eigs]
    
    # Sort by descending eigenvalues
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Remove the first trivial eigenvector (eigenvalue ≈ 1)
    # and keep only n_components
    eigenvectors = eigenvectors[:, 1:n_components+1]
    eigenvalues = eigenvalues[1:n_components+1]
    
    return eigenvectors, eigenvalues, T


def determine_multiscale_space(eigenvectors, eigenvalues, n_eigs=None):
    """
    Determine the multiscale diffusion space.
    
    This creates a weighted combination of diffusion components where weights
    are determined by eigenvalues: weight_k = lambda_k / (1 - lambda_k)
    
    Parameters
    ----------
    eigenvectors : ndarray
        (n_cells, n_components) diffusion components
    eigenvalues : ndarray
        (n_components,) eigenvalues
    n_eigs : int, optional
        Number of eigenvectors to use. If None, determined by eigen gap.
        
    Returns
    -------
    multiscale_data : ndarray
        (n_cells, n_eigs) multiscale diffusion space
    n_eigs_used : int
        Number of eigenvectors used
    """
    # Determine number of eigenvectors using eigen gap if not specified
    if n_eigs is None:
        # Find gap in eigenvalues
        gaps = np.diff(eigenvalues)
        n_eigs = np.argmax(gaps) + 1
    else:
        n_eigs = min(n_eigs, len(eigenvalues))
    
    # Compute multiscale weights
    # weight_k = lambda_k / (1 - lambda_k)
    weights = np.zeros(n_eigs)
    for i in range(n_eigs):
        lam = eigenvalues[i]
        if lam < 1:
            weights[i] = lam / (1 - lam)
        else:
            weights[i] = 1.0
    
    # Create multiscale data
    multiscale_data = eigenvectors[:, :n_eigs] * weights[np.newaxis, :]
    
    return multiscale_data, n_eigs
