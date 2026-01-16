"""
Utility functions for Palantir analysis
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def early_cell(adata, cell_type_key='cell_type', cell_type_label=None, 
               diffusion_components=None, n_components=10):
    """
    Find an early cell based on cell type and diffusion components.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    cell_type_key : str, default='cell_type'
        Key in .obs containing cell type labels
    cell_type_label : str, optional
        Specific cell type to use as early cells. If None, uses first type.
    diffusion_components : ndarray, optional
        Diffusion components from palantir run. If None, loads from adata.obsm
    n_components : int, default=10
        Number of components (if not found in adata)
        
    Returns
    -------
    early_cell_index : int
        Index of selected early cell
    """
    if diffusion_components is None:
        if 'X_palantir_diff_comp' in adata.obsm:
            diffusion_components = adata.obsm['X_palantir_diff_comp']
        else:
            raise ValueError("Diffusion components not found. Run palantir first.")
    
    # Get cell type labels
    if cell_type_key not in adata.obs:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in adata.obs")
    
    cell_types = adata.obs[cell_type_key]
    
    # Select cell type
    if cell_type_label is None:
        cell_type_label = cell_types.unique()[0]
    
    # Find cells of this type
    type_mask = cell_types == cell_type_label
    type_indices = np.where(type_mask)[0]
    
    if len(type_indices) == 0:
        raise ValueError(f"No cells found for cell type '{cell_type_label}'")
    
    # Select cell with most negative first diffusion component
    type_dcs = diffusion_components[type_indices, 0]
    best_idx = type_indices[np.argmin(type_dcs)]
    
    return best_idx


def find_terminal_states(adata, cell_type_key='cell_type', 
                        diffusion_components=None, pseudotime=None,
                        exclude_cell_types=None):
    """
    Find terminal states based on cell type and diffusion components.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    cell_type_key : str, default='cell_type'
        Key in .obs containing cell type labels
    diffusion_components : ndarray, optional
        Diffusion components from palantir run
    pseudotime : ndarray, optional
        Pseudotime values
    exclude_cell_types : list, optional
        Cell types to exclude from terminal state candidates
        
    Returns
    -------
    terminal_state_indices : list
        Indices of selected terminal states (one per cell type)
    """
    if diffusion_components is None:
        if 'X_palantir_diff_comp' in adata.obsm:
            diffusion_components = adata.obsm['X_palantir_diff_comp']
        else:
            raise ValueError("Diffusion components not found. Run palantir first.")
    
    if pseudotime is None:
        if 'palantir_pseudotime' in adata.obs:
            pseudotime = adata.obs['palantir_pseudotime'].values
        else:
            raise ValueError("Pseudotime not found. Run palantir first.")
    
    cell_types = adata.obs[cell_type_key]
    unique_types = cell_types.unique()
    
    if exclude_cell_types is not None:
        unique_types = [ct for ct in unique_types if ct not in exclude_cell_types]
    
    terminal_indices = []
    
    for cell_type in unique_types:
        type_mask = cell_types == cell_type
        type_indices = np.where(type_mask)[0]
        
        # Select cell with highest pseudotime in this cell type
        type_pseudotime = pseudotime[type_indices]
        best_idx = type_indices[np.argmax(type_pseudotime)]
        terminal_indices.append(best_idx)
    
    return terminal_indices


def read_pca(adata, n_components=10, use_rep='X_pca', recalculate=False):
    """
    Read or compute PCA representation for use in palantir.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    n_components : int, default=10
        Number of PCA components
    use_rep : str, default='X_pca'
        Key in .obsm to use or store PCA
    recalculate : bool, default=False
        If True, recalculate PCA
        
    Returns
    -------
    pca_data : ndarray
        (n_cells, n_components) PCA representation
    """
    if use_rep in adata.obsm and not recalculate:
        pca_data = adata.obsm[use_rep]
        if pca_data.shape[1] >= n_components:
            return pca_data[:, :n_components]
    
    # Compute PCA
    from sklearn.decomposition import PCA
    
    if issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(X)
    
    adata.obsm[use_rep] = pca_data
    
    return pca_data


def scale_pca(data, mean=None, std=None):
    """
    Standardize data (0 mean, unit variance).
    
    Parameters
    ----------
    data : ndarray
        (n_cells, n_features) data matrix
    mean : ndarray, optional
        Pre-computed mean
    std : ndarray, optional
        Pre-computed standard deviation
        
    Returns
    -------
    scaled_data : ndarray
        Scaled data
    mean : ndarray
        Mean values
    std : ndarray
        Standard deviation values
    """
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
        std[std == 0] = 1.0
    
    scaled_data = (data - mean) / std
    return scaled_data, mean, std


# Helper import
from scipy.sparse import issparse
