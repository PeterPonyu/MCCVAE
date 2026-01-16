"""
Magic imputation for Palantir
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix


def run_magic_imputation(data, transition_matrix, n_steps=3):
    """
    Perform MAGIC (Markov Affinity-based Graph Imputation of Cells) imputation.
    
    This uses the diffusion operator to smooth gene expression values through
    the cell graph. Imputed values are computed as:
    X_imputed = T^n_steps * X
    
    Parameters
    ----------
    data : ndarray or sparse matrix
        (n_cells, n_genes) expression matrix
    transition_matrix : csr_matrix
        (n_cells, n_cells) transition matrix from diffusion maps
    n_steps : int, default=3
        Number of diffusion steps
        
    Returns
    -------
    imputed_data : ndarray or sparse matrix
        (n_cells, n_genes) imputed expression matrix
    """
    is_sparse = issparse(data)
    
    # Compute T^n_steps
    T_power = transition_matrix.copy().astype(float)
    
    for _ in range(n_steps - 1):
        T_power = T_power @ transition_matrix
        # Keep as sparse
        T_power = csr_matrix(T_power)
    
    # Apply to data
    if is_sparse:
        imputed_data = T_power @ data
        imputed_data = csr_matrix(imputed_data)
    else:
        imputed_data = T_power @ data
        if issparse(imputed_data):
            imputed_data = imputed_data.toarray()
    
    return imputed_data
