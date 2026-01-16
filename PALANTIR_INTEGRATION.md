# Palantir Integration Guide

## Quick Start

**1. Import and Use**

```python
import scanpy as sc
from mccvae.palantir import palantir, palantir_results

# Load your data
adata = sc.read_h5ad("data.h5ad")

# Preprocess
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata)

# Run Palantir (diffusion maps)
adata = palantir(adata, n_components=10, knn=30)

# Specify early cell index
early_cell_idx = 0  # Update with appropriate cell index

# Compute pseudotime and fate
pr = palantir_results(adata, early_cell_idx)
```

**2. Access Results**

```python
# From adata.obs
pseudotime = adata.obs['palantir_pseudotime']
entropy = adata.obs['palantir_entropy']

# From PResults object
branch_probs = pr.branch_probs
terminal_states = pr.terminal_states

# From adata.obsm
diffusion_components = adata.obsm['X_palantir_diff_comp']
multiscale_space = adata.obsm['X_palantir_multiscale']
branch_probs_matrix = adata.obsm['palantir_branch_probs']
```

## Parameters

**`palantir()` Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 10 | Number of diffusion components to compute |
| `knn` | 30 | Number of nearest neighbors for graph |
| `alpha` | 0.0 | Kernel normalization (0=no anisotropy, 1=full) |
| `n_eigs` | None | Number of eigenvectors (computed automatically if None) |
| `impute_data` | True | Perform MAGIC expression imputation |
| `n_steps` | 3 | MAGIC diffusion steps |

**`palantir_results()` Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `early_cell` | - | Start cell (barcode string or index) |
| `knn` | 30 | Neighbors for trajectory graph |
| `terminal_states` | None | Terminal state indices (auto-detect if None) |
| `num_waypoints` | 1200 | Waypoint sampling for refinement |
| `max_iterations` | 25 | Pseudotime refinement iterations |

## Common Workflows

**1. Cell Type-Based Early Cell Selection**

```python
from mccvae.palantir import early_cell

# Find early cell from stem cell population
early_idx = early_cell(adata, cell_type_key='cell_type', 
                       cell_type_label='Progenitor')
pr = palantir_results(adata, early_idx)
```

**2. Finding Terminal States**

```python
from mccvae.palantir import find_terminal_states

# Identify terminal differentiated states
terminal_states = find_terminal_states(adata, cell_type_key='cell_type')
pr = palantir_results(adata, early_idx, terminal_states=terminal_states)
```

**3. Downstream Analysis: Gene Trends**

```python
# Identify genes that vary along pseudotime
pseudotime = adata.obs['palantir_pseudotime']

# Calculate correlation with pseudotime
correlations = []
for gene in adata.var_names:
    corr = np.corrcoef(pseudotime, adata[:, gene].X.toarray().flatten())[0, 1]
    correlations.append(corr)

# Find top pseudotime-varying genes
top_genes_idx = np.argsort(np.abs(correlations))[-20:]
top_genes = adata.var_names[top_genes_idx]
```

**4. Visualization on Embeddings**

```python
# Plot on UMAP
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Color by pseudotime
sc.pl.umap(adata, color='palantir_pseudotime', cmap='viridis')

# Color by entropy (fate commitment)
sc.pl.umap(adata, color='palantir_entropy', cmap='plasma')
```

## Troubleshooting

**Issue: "Diffusion components not found. Run palantir first."**

Solution: Call `palantir()` before `palantir_results()`:
```python
adata = palantir(adata)  # Required first step
pr = palantir_results(adata, early_cell_idx)
```

**Issue: Poor pseudotime results**

Possible causes and solutions:
- Too few neighbors: Increase `knn` to 30-50
- Wrong early cell: Select a progenitor/stem cell, not differentiated
- Poor preprocessing: Ensure proper normalization and scaling
- Too few components: Increase `n_components` to 15-20

Example:
```python
adata = palantir(adata, n_components=15, knn=50)
pr = palantir_results(adata, early_cell_idx, knn=50)
```

**Issue: Memory error on large datasets**

Solutions:
- Reduce `n_components` (minimum recommended: 10)
- Reduce `num_waypoints` (e.g., 500 for 100k cells)
- Use only highly variable genes before running

```python
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]
```
## Performance

Typical performance on laptop with 8GB RAM:
- 5,000 cells × 2,000 genes: ~30 seconds
- 20,000 cells × 5,000 genes: ~2 minutes
- 50,000 cells × 10,000 genes: ~10 minutes

For larger datasets, use highly variable gene selection or sparse representations.

## Citation

If you use this implementation, cite the original Palantir paper:

Setty et al. (2019) "Characterization of cell fate probabilities in single-cell data with Palantir" *Nature Biotechnology* https://doi.org/10.1038/s41587-019-0068-4

## References

- Setty et al. (2019) Nature Biotechnology: https://doi.org/10.1038/s41587-019-0068-4
- Original Palantir: https://github.com/dpeerlab/Palantir
- Scanpy documentation: https://scanpy.readthedocs.io

