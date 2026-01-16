"""
Main Palantir API - equivalent to scanpy.external.tl.palantir
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from scipy.spatial.distance import cdist
import warnings

from .diffusion import (
    run_diffusion_maps,
    determine_multiscale_space,
)
from .pseudotime import (
    compute_pseudotime,
    compute_branch_probabilities,
    compute_entropy,
    find_terminal_states_automatic,
)
from .magic import run_magic_imputation
from .utils import early_cell, find_terminal_states, read_pca, scale_pca


class PResults:
    """Container for Palantir results."""
    
    def __init__(self, pseudotime, entropy, branch_probs, waypoints=None, 
                 terminal_states=None):
        """
        Initialize PResults object.
        
        Parameters
        ----------
        pseudotime : pd.Series
            Pseudotime values
        entropy : pd.Series
            Differentiation entropy
        branch_probs : pd.DataFrame
            Branch probabilities
        waypoints : pd.Index, optional
            Waypoint indices
        terminal_states : list, optional
            Terminal state indices
        """
        self.pseudotime = pseudotime
        self.entropy = entropy
        self.branch_probs = branch_probs
        self.waypoints = waypoints
        self.terminal_states = terminal_states
    
    def __repr__(self):
        return (f"PResults(n_cells={len(self.pseudotime)}, "
                f"n_states={self.branch_probs.shape[1]})")


def palantir(adata, *, n_components=10, knn=30, alpha=0.0,
             use_adjacency_matrix=False, distances_key=None, n_eigs=None,
             impute_data=True, n_steps=3, copy=False):
    """
    Run Palantir algorithm for pseudotime and lineage inference.
    
    This is equivalent to scanpy.external.tl.palantir.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with .X containing expression data
    n_components : int, default=10
        Number of diffusion components to compute
    knn : int, default=30
        Number of nearest neighbors for graph construction
    alpha : float, default=0.0
        Normalization parameter for diffusion operator (0-1)
    use_adjacency_matrix : bool, default=False
        If True, use precomputed adjacency matrix
    distances_key : str, optional
        Key in .obsp containing distance matrix if using adjacency
    n_eigs : int, optional
        Number of eigenvectors to use. If None, determined by eigen gap.
    impute_data : bool, default=True
        Whether to perform MAGIC imputation
    n_steps : int, default=3
        Number of diffusion steps for imputation
    copy : bool, default=False
        Return copy of adata instead of modifying in place
        
    Returns
    -------
    adata : AnnData
        Modified or copied adata with palantir results stored in:
        - .obsm['X_palantir_diff_comp']: Diffusion components
        - .obsm['X_palantir_multiscale']: Multiscale diffusion space
        - .obsp['palantir_diff_op']: Transition matrix
        - .layers['palantir_imp']: Imputed expression (if impute_data=True)
        - .uns['palantir_EigenValues']: Eigenvalues
    """
    if copy:
        adata = adata.copy()
    
    # Validate input
    if adata.n_obs == 0:
        raise ValueError("adata has no cells")
    if adata.n_vars == 0:
        raise ValueError("adata has no genes")
    
    # Get PCA representation (use existing or compute)
    if 'X_pca' in adata.obsm:
        pca_data = adata.obsm['X_pca'][:, :n_components]
    else:
        # Compute PCA
        from sklearn.decomposition import PCA
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        # Handle very small datasets
        n_comp_pca = min(n_components, min(X.shape) - 1)
        pca = PCA(n_components=n_comp_pca)
        pca_data = pca.fit_transform(X)
        adata.obsm['X_pca'] = pca_data
    
    # Ensure we have enough components
    if pca_data.shape[1] < n_components:
        n_components = pca_data.shape[1]
        warnings.warn(f"Reducing n_components to {n_components} based on data dimensionality")
    
    # Scale PCA data
    pca_scaled, pca_mean, pca_std = scale_pca(pca_data)
    
    # Get distance matrix if using adjacency
    distances = None
    if use_adjacency_matrix and distances_key is not None:
        if distances_key in adata.obsp:
            distances = adata.obsp[distances_key]
    
    # Run diffusion maps
    eigenvectors, eigenvalues, transition_matrix = run_diffusion_maps(
        pca_scaled,
        n_components=n_components,
        knn=knn,
        alpha=alpha,
        use_adjacency=use_adjacency_matrix,
        distances=distances
    )
    
    # Determine multiscale space
    multiscale_data, n_eigs_used = determine_multiscale_space(
        eigenvectors, eigenvalues, n_eigs=n_eigs
    )
    
    # Store results in adata
    adata.obsm['X_palantir_diff_comp'] = eigenvectors
    adata.obsm['X_palantir_multiscale'] = multiscale_data
    adata.obsp['palantir_diff_op'] = transition_matrix
    adata.uns['palantir_EigenValues'] = eigenvalues
    
    # Perform MAGIC imputation if requested
    if impute_data:
        if issparse(adata.X):
            X = adata.X.copy()
        else:
            X = adata.X.copy()
        
        imputed_data = run_magic_imputation(X, transition_matrix, n_steps=n_steps)
        adata.layers['palantir_imp'] = imputed_data
    
    return adata


def palantir_results(adata, early_cell, *, ms_data='X_palantir_multiscale',
                    terminal_states=None, knn=30, num_waypoints=1200,
                    n_jobs=-1, scale_components=True, 
                    use_early_cell_as_start=False, max_iterations=25,
                    seed=None):
    """
    Compute pseudotime and branch probabilities from Palantir diffusion space.
    
    This is equivalent to scanpy.external.tl.palantir_results.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with palantir results
    early_cell : str or int
        Cell barcode/name or index of early cell. If str, looks in adata.obs_names.
    ms_data : str, default='X_palantir_multiscale'
        Key in .obsm containing multiscale diffusion space
    terminal_states : list, optional
        List of terminal state indices. If None, attempts to identify automatically.
    knn : int, default=30
        Number of nearest neighbors for trajectory graph
    num_waypoints : int, default=1200
        Number of waypoints for trajectory refinement
    n_jobs : int, default=-1
        Number of parallel jobs (not used in this implementation)
    scale_components : bool, default=True
        Whether to scale diffusion components
    use_early_cell_as_start : bool, default=False
        Whether to use early cell as start or compute from all cells
    max_iterations : int, default=25
        Maximum iterations for pseudotime refinement
    seed : int, optional
        Random seed
        
    Returns
    -------
    pr : PResults
        Object with attributes: pseudotime, entropy, branch_probs
    """
    # Validate that palantir has been run
    if ms_data not in adata.obsm:
        raise ValueError(f"Palantir results not found. Key '{ms_data}' not in .obsm. "
                        "Run palantir() first.")
    
    # Get multiscale data
    multiscale_data = adata.obsm[ms_data]
    
    # Parse early cell
    if isinstance(early_cell, str):
        if early_cell in adata.obs_names:
            early_cell_idx = adata.obs_names.get_loc(early_cell)
        else:
            raise ValueError(f"Cell '{early_cell}' not found in adata.obs_names")
    else:
        early_cell_idx = int(early_cell)
    
    # Find terminal states if not provided
    if terminal_states is None:
        terminal_states = find_terminal_states_automatic(multiscale_data, 
                                                         np.zeros(len(multiscale_data)))
        if len(terminal_states) == 0:
            # Fallback: use cells at 95th percentile of first diffusion component
            dc = adata.obsm.get('X_palantir_diff_comp', multiscale_data)
            if dc.shape[1] > 0:
                first_comp = dc[:, 0]
                threshold = np.percentile(first_comp, 95)
                terminal_states = np.where(first_comp >= threshold)[0]
            else:
                # Last resort: use farthest cells from early cell
                from scipy.spatial.distance import cdist
                distances = cdist([multiscale_data[early_cell_idx]], multiscale_data)[0]
                terminal_states = np.argsort(distances)[-5:]
    
    # Compute pseudotime
    pseudotime = compute_pseudotime(
        multiscale_data,
        early_cell_idx,
        terminal_states=terminal_states,
        knn=knn,
        n_waypoints=num_waypoints,
        n_iterations=max_iterations,
        use_early_cell_as_start=use_early_cell_as_start,
        scale_components=scale_components,
        seed=seed
    )
    
    # Compute branch probabilities
    branch_probs, terminal_set = compute_branch_probabilities(
        multiscale_data,
        pseudotime,
        early_cell_idx,
        terminal_states,
        knn=knn
    )
    
    # Compute entropy
    entropy = compute_entropy(branch_probs)
    
    # Store results in adata
    adata.obs['palantir_pseudotime'] = pseudotime
    adata.obs['palantir_entropy'] = entropy
    
    # Store branch probabilities
    terminal_names = [f"Terminal_{i}" for i in range(len(terminal_states))]
    adata.obsm['palantir_branch_probs'] = branch_probs
    
    # Create PResults object
    pr = PResults(
        pseudotime=pd.Series(pseudotime, index=adata.obs_names),
        entropy=pd.Series(entropy, index=adata.obs_names),
        branch_probs=pd.DataFrame(branch_probs, index=adata.obs_names, 
                                  columns=terminal_names),
        waypoints=pd.Index(range(len(terminal_states))),
        terminal_states=terminal_states
    )
    
    return pr
