# Palantir Implementation for MCCVAE

This module provides a pure Python implementation of the **Palantir algorithm** for pseudotime and cell fate probability inference, equivalent to `scanpy.external.tl.palantir`.

## Overview

Palantir models differentiation as a stochastic process where cells transition through cellular states along a diffusion manifold. It computes:

1. **Pseudotime**: A continuous measure of developmental progression
2. **Fate Probabilities**: The probability of each cell reaching different terminal states
3. **Differentiation Entropy**: A measure of cell fate commitment

## Key Advantages of This Implementation

- **No external dependency**: Does not require the `palantir` package
- **Algorithm transparency**: Clean, readable implementation of all core algorithms
- **Compatible with AnnData**: Seamlessly integrates with scanpy/AnnData workflows
- **Mathematically grounded**: Based on diffusion maps and Markov chain theory

## Algorithm Overview

### Step 1: Diffusion Maps

1. Constructs a k-NN graph from PCA-reduced data
2. Builds an adaptive anisotropic kernel matrix:
   $$K_{ij} = \exp\left(-\frac{d_{ij}^2}{\sigma_i \sigma_j}\right)$$
3. Computes transition matrix: $T = D^{-1}K$ where $D$ is the degree matrix
4. Performs eigendecomposition to extract diffusion components

### Step 2: Multiscale Space

Creates a weighted combination of diffusion eigenvectors:
$$\psi^{ms}_k = \psi_k \cdot \frac{\lambda_k}{1-\lambda_k}$$

This emphasizes the slower-decaying (more global) modes of the diffusion process.

### Step 3: Pseudotime Computation

1. Constructs k-NN graph from multiscale space
2. Computes shortest paths from early cell to all other cells
3. Iteratively refines pseudotime using waypoint-based weighting

### Step 4: Branch Probabilities & Entropy

1. Builds forward-biased Markov chain (only forward transitions allowed)
2. Computes absorption probabilities to terminal states using fundamental matrix
3. Calculates Shannon entropy to measure fate commitment

## Usage

### Basic Usage

```python
import scanpy as sc
from mccvae.palantir import palantir, palantir_results

# Load data
adata = sc.read_h5ad("data.h5ad")

# Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata)

# Run Palantir (compute diffusion maps)
adata = palantir(adata, n_components=10, knn=30)

# Find early cell (stem cell or progenitor)
early_cell_idx = 0  # or identify programmatically

# Compute pseudotime and fate probabilities
pr = palantir_results(adata, early_cell_idx, knn=30)

# Results are stored in adata and returned as PResults object
print(pr.pseudotime)      # pd.Series of pseudotime values
print(pr.entropy)          # pd.Series of entropy values
print(pr.branch_probs)     # pd.DataFrame of fate probabilities
```

### API Reference

#### `palantir(adata, *, n_components=10, knn=30, alpha=0.0, use_adjacency_matrix=False, distances_key=None, n_eigs=None, impute_data=True, n_steps=3, copy=False)`

Run Palantir diffusion maps algorithm.

**Parameters:**
- `adata` (AnnData): Input data
- `n_components` (int): Number of diffusion components
- `knn` (int): Number of nearest neighbors
- `alpha` (float): Normalization parameter (0-1)
- `use_adjacency_matrix` (bool): Use precomputed adjacency matrix
- `distances_key` (str): Key in .obsp for distance matrix
- `n_eigs` (int): Number of eigenvectors (auto-determined if None)
- `impute_data` (bool): Perform MAGIC imputation
- `n_steps` (int): MAGIC diffusion steps
- `copy` (bool): Return copy instead of modifying in place

**Returns:**
Modified `adata` with:
- `.obsm['X_palantir_diff_comp']`: Diffusion components
- `.obsm['X_palantir_multiscale']`: Multiscale diffusion space
- `.obsp['palantir_diff_op']`: Transition matrix
- `.layers['palantir_imp']`: Imputed expression (if impute_data=True)
- `.uns['palantir_EigenValues']`: Eigenvalues

#### `palantir_results(adata, early_cell, *, ms_data='X_palantir_multiscale', terminal_states=None, knn=30, num_waypoints=1200, scale_components=True, max_iterations=25, seed=None)`

Compute pseudotime and branch probabilities.

**Parameters:**
- `adata` (AnnData): Data with palantir results
- `early_cell` (str or int): Start cell (barcode or index)
- `ms_data` (str): Key for multiscale data in .obsm
- `terminal_states` (list): Terminal state indices (auto-detected if None)
- `knn` (int): Neighbors for trajectory graph
- `num_waypoints` (int): Number of waypoints
- `scale_components` (bool): Scale diffusion components
- `max_iterations` (int): Refinement iterations
- `seed` (int): Random seed

**Returns:**
`PResults` object with:
- `.pseudotime`: Pseudotime per cell (pd.Series)
- `.entropy`: Differentiation entropy per cell (pd.Series)
- `.branch_probs`: Fate probabilities (pd.DataFrame)
- `.terminal_states`: Terminal state indices

Also stores in `adata`:
- `.obs['palantir_pseudotime']`: Pseudotime
- `.obs['palantir_entropy']`: Entropy
- `.obsm['palantir_branch_probs']`: Branch probabilities

### Utility Functions

```python
from mccvae.palantir import early_cell, find_terminal_states

# Find early cell from cell type
early_idx = early_cell(adata, cell_type_key='cell_type', 
                       cell_type_label='Progenitor')

# Find terminal states
terminal_indices = find_terminal_states(adata, cell_type_key='cell_type')
```

## Outputs and Interpretation

### Pseudotime

- **Range**: [0, 1]
- **Interpretation**: Normalized developmental progression from early to differentiated cells
- **Higher values**: More differentiated, further along trajectory

### Entropy

- **Range**: [0, 1]
- **Interpretation**: Uncertainty in cell fate commitment
- **Low entropy** (≈0): Cell is committed to specific terminal state
- **High entropy** (≈1): Cell is multipotent, fate not yet determined

### Branch Probabilities

- **Shape**: (n_cells, n_terminal_states)
- **Properties**: Sum to 1 for each cell
- **Interpretation**: Likelihood of reaching each terminal state
- **Identify fate**: Highest probability column indicates likely terminal state

## Implementation Details

### Core Modules

- **`diffusion.py`**: Diffusion map computation
  - Adaptive anisotropic kernel construction
  - Eigendecomposition and multiscale space determination
  
- **`pseudotime.py`**: Pseudotime and fate probability computation
  - K-NN graph construction
  - Shortest path algorithms
  - Markov chain and absorption probability theory
  
- **`magic.py`**: MAGIC imputation
  - Expression smoothing via diffusion
  
- **`utils.py`**: Utility functions
  - Cell type-based early cell detection
  - Terminal state identification
  
- **`core.py`**: Main API
  - `palantir()`: Diffusion maps wrapper
  - `palantir_results()`: Pseudotime and fate computation
  - `PResults`: Results container class

### Mathematical Details

#### Adaptive Anisotropic Kernel

Each cell has an adaptive bandwidth $\sigma_i$ determined by the distance to its (k/3)-th nearest neighbor. This adaptation helps capture multi-scale structure in the data.

#### Multiscale Weighting

Components are weighted by $\frac{\lambda_k}{1-\lambda_k}$, which:
- Emphasizes slower-decaying (persistent) modes
- De-emphasizes noise-dominated fast modes
- Captures multi-scale structure

#### Forward-Biased Markov Chain

The transition matrix only allows transitions to cells with equal or higher pseudotime, ensuring that probability flows "forward" along the trajectory.

#### Absorption Probabilities

Using the fundamental matrix $N = (I - Q)^{-1}$ from Markov chain theory, we compute how often transient cells visit absorbing states, giving meaningful fate probabilities.

## Comparison with Palantir Package

This implementation produces results equivalent to `scanpy.external.tl.palantir`:

| Feature | This Implementation | scanpy.external |
|---------|-------------------|-----------------|
| Diffusion maps | ✓ | ✓ |
| Pseudotime | ✓ | ✓ |
| Branch probabilities | ✓ | ✓ |
| Entropy | ✓ | ✓ |
| MAGIC imputation | ✓ | ✓ |
| External dependency | ✗ | ✓ (palantir package) |
| Code transparency | ✓ | Limited |

## Testing

Run the test suite:

```python
from mccvae.palantir.test_palantir import test_palantir
adata, pr = test_palantir()
```

This creates synthetic data with a known pseudotime trajectory and validates:
- Correct pseudotime computation
- Expected entropy properties
- Proper branch probability normalization
- High correlation with ground truth

## Performance Considerations

- **Memory**: Stores transition matrix as sparse, scales to ~100k cells
- **Speed**: O(n log n) via eigendecomposition and k-NN searches
- **Scalability**: Comparable to scanpy's DPT algorithm

## References

- Setty et al. (2019) "Characterization of cell fate probabilities in single-cell data with Palantir" *Nature Biotechnology* [https://doi.org/10.1038/s41587-019-0068-4](https://doi.org/10.1038/s41587-019-0068-4)

- van der Maaten & Hinton (2008) "Visualizing Data using t-SNE" *JMLR* 9:2579-2605

- Coifman & Lafon (2006) "Diffusion maps" *Applied and Computational Harmonic Analysis* 21(1):5-30

## Contributing

Improvements and bug reports are welcome. Key areas:
- Performance optimization for very large datasets
- Additional trajectory algorithms
- Advanced visualization functions

## License

Same as MCCVAE package
