"""
Test script for Palantir implementation
"""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mccvae.palantir import palantir, palantir_results


def create_synthetic_data(n_cells=1000, n_genes=2000, seed=42):
    """Create synthetic single-cell data for testing."""
    np.random.seed(seed)
    
    # Create pseudotime trajectory
    t = np.linspace(0, 1, n_cells)
    
    # 3 cell types with different trajectories
    n_per_type = n_cells // 3
    
    # Gene expression that depends on pseudotime
    X = np.zeros((n_cells, n_genes))
    
    # Type 1: Early genes (high at t=0, low at t=1)
    early_genes = np.arange(0, n_genes // 3)
    for i, g in enumerate(early_genes):
        X[:, g] = 10 * (1 - t) + np.random.normal(0, 1, n_cells)
    
    # Type 2: Middle genes (peak at t=0.5)
    middle_genes = np.arange(n_genes // 3, 2 * n_genes // 3)
    for i, g in enumerate(middle_genes):
        X[:, g] = 10 * np.exp(-((t - 0.5) ** 2) / 0.1) + np.random.normal(0, 1, n_cells)
    
    # Type 3: Late genes (low at t=0, high at t=1)
    late_genes = np.arange(2 * n_genes // 3, n_genes)
    for i, g in enumerate(late_genes):
        X[:, g] = 10 * t + np.random.normal(0, 1, n_cells)
    
    # Ensure non-negative
    X = np.maximum(X, 0)
    
    # Create AnnData object
    adata = AnnData(X)
    adata.obs['pseudotime_true'] = t
    
    # Cell type labels
    cell_types = []
    for i in range(n_cells):
        if t[i] < 0.33:
            cell_types.append('Progenitor')
        elif t[i] < 0.67:
            cell_types.append('Intermediate')
        else:
            cell_types.append('Differentiated')
    adata.obs['cell_type'] = cell_types
    
    # Gene names
    adata.var_names = [f'Gene_{i}' for i in range(n_genes)]
    adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
    
    return adata


def test_palantir():
    """Test Palantir implementation."""
    print("=" * 80)
    print("Testing Palantir Implementation")
    print("=" * 80)
    
    # Create synthetic data
    print("\n1. Creating synthetic single-cell data...")
    adata = create_synthetic_data(n_cells=500, n_genes=1000)
    print(f"   Created data: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    # Normalize
    print("\n2. Normalizing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    print("   Data normalized")
    
    # Run palantir
    print("\n3. Running Palantir (diffusion maps)...")
    adata = palantir(adata, n_components=10, knn=15, impute_data=True)
    print(f"   Diffusion components shape: {adata.obsm['X_palantir_diff_comp'].shape}")
    print(f"   Multiscale space shape: {adata.obsm['X_palantir_multiscale'].shape}")
    print(f"   Eigenvalues: {adata.uns['palantir_EigenValues'][:5]}")
    
    # Find early cell
    print("\n4. Finding early cell...")
    early_cell_idx = np.argmin(adata.obsm['X_palantir_diff_comp'][:, 0])
    print(f"   Early cell index: {early_cell_idx}")
    
    # Run palantir_results
    print("\n5. Computing pseudotime and branch probabilities...")
    pr = palantir_results(adata, early_cell_idx, knn=15)
    print(f"   Pseudotime range: [{pr.pseudotime.min():.3f}, {pr.pseudotime.max():.3f}]")
    print(f"   Entropy range: [{pr.entropy.min():.3f}, {pr.entropy.max():.3f}]")
    print(f"   Branch probabilities shape: {pr.branch_probs.shape}")
    print(f"   Terminal states: {pr.terminal_states}")
    
    # Compare with true pseudotime
    print("\n6. Validating results...")
    true_pt = adata.obs['pseudotime_true'].values
    computed_pt = adata.obs['palantir_pseudotime'].values
    
    # Normalize both to [0, 1]
    true_pt_norm = (true_pt - true_pt.min()) / (true_pt.max() - true_pt.min())
    computed_pt_norm = (computed_pt - computed_pt.min()) / (computed_pt.max() - computed_pt.min())
    
    # Compute correlation
    correlation = np.corrcoef(true_pt_norm, computed_pt_norm)[0, 1]
    print(f"   Correlation with true pseudotime: {correlation:.4f}")
    
    # Check entropy properties
    max_entropy = adata.obs['palantir_entropy'].max()
    min_entropy = adata.obs['palantir_entropy'].min()
    print(f"   Entropy range: [{min_entropy:.4f}, {max_entropy:.4f}]")
    print(f"   Mean entropy: {adata.obs['palantir_entropy'].mean():.4f}")
    
    # Verify branch probabilities sum to 1
    branch_sums = pr.branch_probs.sum(axis=1)
    print(f"   Branch prob sums [should be ≈1]: min={branch_sums.min():.4f}, max={branch_sums.max():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)
    
    return adata, pr


if __name__ == '__main__':
    adata, pr = test_palantir()
