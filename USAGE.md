# MCCVAE Usage Guide

This guide provides comprehensive instructions for using the Momentum Contrastive Coupling Variational Autoencoder (MCCVAE) framework for single-cell genomics analysis.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [API Server Usage](#api-server-usage)
6. [Parameter Reference](#parameter-reference)
7. [Understanding Outputs](#understanding-outputs)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for large datasets)

### Step 1: Clone the Repository

```bash
git clone https://github.com/PeterPonyu/MCCVAE.git
cd MCCVAE
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `scanpy` - Single-cell analysis toolkit
- `torch` - PyTorch deep learning framework
- `tqdm` - Progress bar library
- `uvicorn` - ASGI server (for API)
- `fastapi` - Web framework (for API)

### Step 3: Verify Installation

```python
import mccvae
print(mccvae.__version__)  # Should print 0.0.1
```

---

## Quick Start

Here's the simplest way to get started with MCCVAE:

```python
import scanpy as sc
from mccvae import Agent

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Create an agent with default parameters
agent = Agent(adata=adata, layer="counts", latent_dim=10)

# Train the model
agent.fit(epochs=1000)

# Extract latent representations
latent = agent.get_latent()
print(f"Latent shape: {latent.shape}")
```

---

## Basic Usage

### Loading Data

MCCVAE works with AnnData objects from the Scanpy ecosystem:

```python
import scanpy as sc

# Read from H5AD file
adata = sc.read_h5ad("pbmc3k.h5ad")

# Or load from other formats
adata = sc.read_10x_mtx("path/to/10x/data/")
adata = sc.read_csv("expression_matrix.csv")
```

### Creating an Agent

The `Agent` class is the main interface for MCCVAE:

```python
from mccvae import Agent

agent = Agent(
    adata=adata,
    layer="counts",           # Which layer to use from adata
    latent_dim=10,           # Dimension of latent space
    use_moco=True,           # Enable Momentum Contrast
    irecon=1.0               # Enable information bottleneck
)
```

### Training the Model

```python
# Train for 1000 epochs with progress tracking
agent.fit(epochs=1000)

# The training progress will show:
# - Total loss value
# - ARI (Adjusted Rand Index) - clustering accuracy
# - NMI (Normalized Mutual Information) - clustering quality
# - ASW (Average Silhouette Width) - cluster separation
# - C_H (Calinski-Harabasz Index) - cluster validity
# - D_B (Davies-Bouldin Index) - cluster compactness
# - P_C (Pearson Correlation) - disentanglement measure
```

### Extracting Results

MCCVAE provides three types of embeddings:

```python
# 1. Primary latent representations (z)
latent = agent.get_latent()
print(f"Latent representations: {latent.shape}")

# 2. Information bottleneck embeddings (l_e)
# Two equivalent methods are available:
bottleneck = agent.get_iembed()      # Original method name
bottleneck = agent.get_bottleneck()  # Alias for consistency with README
print(f"Bottleneck embeddings: {bottleneck.shape}")

# 3. Refined latent representations (l_d)
# These are obtained by decoding the bottleneck embeddings back to latent dimension
refined = agent.get_refined()  # Returns l_d with shape (n_cells, latent_dim)
print(f"Refined representations: {refined.shape}")

# All can be used for downstream analysis
# Add to AnnData object for visualization
adata.obsm['X_mccvae'] = latent
adata.obsm['X_mccvae_bottleneck'] = bottleneck
adata.obsm['X_mccvae_refined'] = refined
```

### Visualization with Scanpy

```python
import scanpy as sc

# Use latent representations
adata.obsm['X_mccvae'] = agent.get_latent()

# Compute UMAP on latent space
sc.pp.neighbors(adata, use_rep='X_mccvae')
sc.tl.umap(adata)

# Visualize
sc.pl.umap(adata, color='cell_type')

# Use bottleneck embeddings for trajectory analysis
adata.obsm['X_bottleneck'] = agent.get_iembed()
sc.pp.neighbors(adata, use_rep='X_bottleneck')
sc.tl.umap(adata)
sc.pl.umap(adata, color='pseudotime')
```

---

## Advanced Usage

### Customizing Training Parameters

MCCVAE provides extensive control over the training process:

```python
agent = Agent(
    adata=adata,
    layer="counts",
    
    # Architecture parameters
    hidden_dim=128,          # Hidden layer dimension
    latent_dim=10,           # Latent space dimension
    i_dim=2,                 # Information bottleneck dimension
    
    # Loss weights
    recon=1.0,              # Reconstruction loss weight
    irecon=1.0,             # Information bottleneck reconstruction weight
    beta=1.0,               # KL divergence weight (β-VAE)
    dip=0.0,                # DIP regularization weight
    tc=0.0,                 # Total correlation weight
    info=0.0,               # InfoVAE MMD weight
    
    # Momentum Contrast (MoCo) parameters
    use_moco=True,          # Enable contrastive learning
    moco_weight=1.0,        # Contrastive loss weight
    moco_T=0.2,             # Temperature for contrastive loss
    
    # Data augmentation (for MoCo)
    aug_prob=0.5,           # Probability of augmentation
    mask_prob=0.2,          # Probability of masking each gene
    noise_prob=0.7,         # Probability of adding noise
    
    # Training parameters
    percent=0.01,           # Fraction of data per batch
    lr=1e-4,                # Learning rate
    vae_reg=0.5,            # VAE regularization
    loss_mode="nb",         # Loss function: "mse", "nb", or "zinb"
    use_qm=True             # Use mean (True) or samples (False)
)
```

### Choosing Loss Functions

MCCVAE supports three loss functions for different data types:

```python
# 1. Mean Squared Error (MSE) - for normalized continuous data
agent = Agent(adata=adata, layer="normalized", loss_mode="mse")

# 2. Negative Binomial (NB) - for UMI count data (default)
agent = Agent(adata=adata, layer="counts", loss_mode="nb")

# 3. Zero-Inflated Negative Binomial (ZINB) - for sparse count data
agent = Agent(adata=adata, layer="counts", loss_mode="zinb")
```

### Enabling Disentanglement Techniques

```python
# β-VAE: Increase beta for more disentangled representations
agent = Agent(adata=adata, beta=4.0)

# β-TC-VAE: Focus on total correlation
agent = Agent(adata=adata, tc=6.0)

# DIP-VAE: Disentangled Inferred Prior
agent = Agent(adata=adata, dip=1.0)

# InfoVAE: Maximum Mean Discrepancy
agent = Agent(adata=adata, info=1.0)

# Combine multiple techniques
agent = Agent(adata=adata, beta=2.0, dip=0.5, tc=1.0)
```

### Working with Different Data Modalities

MCCVAE is modality-agnostic and works with various single-cell data types:

```python
# scRNA-seq data
adata_rna = sc.read_h5ad("scrna_data.h5ad")
agent_rna = Agent(adata=adata_rna, layer="counts", loss_mode="nb")

# scATAC-seq data
adata_atac = sc.read_h5ad("scatac_data.h5ad")
agent_atac = Agent(adata=adata_atac, layer="counts", loss_mode="nb")

# Normalized data
adata_norm = sc.read_h5ad("normalized_data.h5ad")
agent_norm = Agent(adata=adata_norm, layer="X", loss_mode="mse")
```

### Batch Training with Custom Evaluation

```python
# Train with intermediate checkpoints
for epoch_batch in range(10):
    agent.fit(epochs=100)  # Train for 100 epochs
    
    # Save intermediate results
    latent = agent.get_latent()
    np.save(f'latent_epoch_{(epoch_batch+1)*100}.npy', latent)
    
    # Custom evaluation
    print(f"Epoch {(epoch_batch+1)*100}: Completed")
```

---

## API Server Usage

MCCVAE includes a web-based interface for training and visualization.

### Starting the Servers

```bash
python start_servers.py
```

This will start:
- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs

The browser will automatically open to the frontend interface.

### API Endpoints

The FastAPI backend provides several REST endpoints:

#### 1. Upload Data
```bash
POST /upload-data
Content-Type: multipart/form-data

# Upload an H5AD file
curl -X POST "http://localhost:8000/upload-data" \
  -F "file=@your_data.h5ad"
```

#### 2. Get Data Summary
```bash
GET /data-summary

# Returns information about the loaded dataset
curl "http://localhost:8000/data-summary"
```

#### 3. Start Training
```bash
POST /start-training
Content-Type: application/json

{
  "parameters": {
    "layer": "counts",
    "latent_dim": 10,
    "use_moco": true,
    "irecon": 1.0
  },
  "config": {
    "epochs": 1000
  }
}
```

#### 4. Get Training Progress
```bash
GET /training-progress

# Monitor real-time training metrics
curl "http://localhost:8000/training-progress"
```

#### 5. Download Embeddings
```bash
GET /download/embeddings/{embedding_type}

# Download latent embeddings
curl "http://localhost:8000/download/embeddings/latent" -o latent.csv

# Download bottleneck embeddings (also called 'interpretable' in the API)
curl "http://localhost:8000/download/embeddings/interpretable" -o bottleneck.csv
```

**Note**: The API endpoint uses the term "interpretable" for the bottleneck embeddings,
while the Python code uses `get_bottleneck()` or `get_iembed()`. These refer to the same embeddings.

#### 6. Application Status
```bash
GET /status

# Check if data is loaded, model is trained, etc.
curl "http://localhost:8000/status"
```

### Using the Frontend

The web interface provides an intuitive way to:

1. **Upload Data**: Drag and drop H5AD files
2. **Configure Parameters**: Interactive forms for all model parameters
3. **Monitor Training**: Real-time progress bars and metric plots
4. **Download Results**: Export embeddings as CSV files

---

## Parameter Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | AnnData | Required | Annotated data object containing single-cell dataset |
| `layer` | str | "counts" | Layer name in adata to use for training |
| `latent_dim` | int | 10 | Dimension of the primary latent space |
| `i_dim` | int | 2 | Dimension of information bottleneck (for coupled representations) |
| `hidden_dim` | int | 128 | Dimension of hidden layers in encoder/decoder networks |

### Loss Weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recon` | float | 1.0 | Weight for primary reconstruction loss |
| `irecon` | float | 0.0 | Weight for information bottleneck reconstruction loss |
| `beta` | float | 1.0 | Weight for KL divergence (β-VAE regularization) |
| `dip` | float | 0.0 | Weight for Disentangled Inferred Prior (DIP) regularization |
| `tc` | float | 0.0 | Weight for total correlation (β-TC-VAE) |
| `info` | float | 0.0 | Weight for InfoVAE MMD regularization |
| `moco_weight` | float | 1.0 | Weight for contrastive learning loss |

### Momentum Contrast (MoCo) Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_moco` | bool | False | Enable Momentum Contrast learning |
| `moco_T` | float | 0.2 | Temperature parameter for contrastive loss (lower = harder negatives) |
| `aug_prob` | float | 0.5 | Probability of applying augmentation to data |
| `mask_prob` | float | 0.2 | Probability of masking each gene during augmentation |
| `noise_prob` | float | 0.7 | Probability of adding noise to each gene |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `percent` | float | 0.01 | Fraction of dataset to use per batch (0.0-1.0) |
| `lr` | float | 1e-4 | Learning rate for Adam optimizer |
| `vae_reg` | float | 0.5 | Regularization weight for VAE latent representations |
| `loss_mode` | str | "nb" | Loss function: "mse", "nb", or "zinb" |
| `use_qm` | bool | True | Use mean (True) or samples (False) from latent distribution |
| `device` | torch.device | Auto | Computation device (auto-detects CUDA) |

---

## Understanding Outputs

### Latent Representations

The primary latent representations (`get_latent()`) capture the overall biological structure:

- **Shape**: `(n_cells, latent_dim)`
- **Use cases**:
  - Cell clustering and identification
  - UMAP/t-SNE visualization
  - Batch correction assessment
  - Cell type classification

```python
latent = agent.get_latent()
# Shape: (10000, 10) for 10,000 cells with latent_dim=10
```

### Information Bottleneck Embeddings

The bottleneck embeddings (`get_iembed()`) provide compressed, coupled representations:

- **Shape**: `(n_cells, i_dim)`
- **Use cases**:
  - Trajectory inference
  - Pseudotime analysis
  - Identifying continuous transitions
  - Capturing biological correlations

```python
bottleneck = agent.get_iembed()
# Shape: (10000, 2) for 10,000 cells with i_dim=2
```

### Training Metrics

During training, MCCVAE reports several evaluation metrics:

| Metric | Full Name | Range | Better When | Interpretation |
|--------|-----------|-------|-------------|----------------|
| ARI | Adjusted Rand Index | [-1, 1] | Higher | Clustering agreement with ground truth |
| NMI | Normalized Mutual Information | [0, 1] | Higher | Information shared between clusterings |
| ASW | Average Silhouette Width | [-1, 1] | Higher | How well-separated clusters are |
| C_H | Calinski-Harabasz Index | [0, ∞) | Higher | Ratio of between-cluster to within-cluster variance |
| D_B | Davies-Bouldin Index | [0, ∞) | Lower | Average similarity between clusters |
| P_C | Pearson Correlation | [0, ∞) | Lower | Average correlation between latent dimensions (disentanglement) |

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Problem**: GPU runs out of memory during training.

**Solution**:
```python
# Reduce batch size
agent = Agent(adata=adata, percent=0.005)  # Smaller batches

# Or use CPU
import torch
agent = Agent(adata=adata, device=torch.device('cpu'))
```

#### 2. Layer Not Found

**Problem**: `KeyError: 'counts'`

**Solution**:
```python
# Check available layers
print(adata.layers.keys())

# Use the correct layer name
agent = Agent(adata=adata, layer="raw_counts")

# Or use .X directly (not recommended for raw counts)
adata.layers['X'] = adata.X
agent = Agent(adata=adata, layer="X")
```

#### 3. Slow Training

**Problem**: Training is very slow.

**Solutions**:
- Enable GPU acceleration (CUDA)
- Increase batch size: `percent=0.02`
- Reduce hidden dimension: `hidden_dim=64`
- Reduce dataset size before training
- Disable unnecessary regularization terms

#### 4. Poor Clustering Results

**Problem**: Low ARI/NMI scores during training.

**Solutions**:
```python
# Enable MoCo for better representations
agent = Agent(adata=adata, use_moco=True)

# Increase latent dimension
agent = Agent(adata=adata, latent_dim=20)

# Try different loss functions
agent = Agent(adata=adata, loss_mode="zinb")  # For sparse data

# Adjust beta for better separation
agent = Agent(adata=adata, beta=2.0)
```

#### 5. NaN/Inf Losses

**Problem**: Loss becomes NaN or Inf during training.

**Solutions**:
```python
# Reduce learning rate
agent = Agent(adata=adata, lr=1e-5)

# Use more stable loss function
agent = Agent(adata=adata, loss_mode="mse")

# Check data quality (no zeros only, no extreme values)
import numpy as np
print(f"Data range: {adata.X.min()} to {adata.X.max()}")
```

### Best Practices

1. **Data Preprocessing**:
   - Always check data quality before training
   - Remove low-quality cells and genes
   - Consider log-normalization for RNA-seq data

2. **Parameter Selection**:
   - Start with default parameters
   - Enable `use_moco=True` for better representations
   - Set `irecon=1.0` for trajectory analysis
   - Use `loss_mode="nb"` for raw count data

3. **Training Strategy**:
   - Start with fewer epochs (100-500) for testing
   - Monitor metrics every 10 epochs
   - Use GPU for large datasets (>10,000 cells)
   - Save intermediate results for long training runs

4. **Validation**:
   - Visualize embeddings with UMAP/t-SNE
   - Compare results with known cell types
   - Check biological interpretation of latent dimensions
   - Validate on held-out test data

---

## Example Workflows

### Workflow 1: Cell Type Identification

```python
import scanpy as sc
from mccvae import Agent

# Load and preprocess data
adata = sc.read_h5ad("pbmc_data.h5ad")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Train MCCVAE with MoCo
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=10,
    use_moco=True,
    loss_mode="nb"
)
agent.fit(epochs=1000)

# Extract and visualize
adata.obsm['X_mccvae'] = agent.get_latent()
sc.pp.neighbors(adata, use_rep='X_mccvae')
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color='leiden')
```

### Workflow 2: Trajectory Analysis

```python
import scanpy as sc
from mccvae import Agent

# Load developmental data
adata = sc.read_h5ad("differentiation_data.h5ad")

# Train with information bottleneck
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=10,
    i_dim=2,            # 2D bottleneck for trajectories
    irecon=1.0,         # Enable bottleneck reconstruction
    use_moco=True
)
agent.fit(epochs=1000)

# Use bottleneck for trajectory
adata.obsm['X_trajectory'] = agent.get_iembed()
sc.pp.neighbors(adata, use_rep='X_trajectory')
sc.tl.umap(adata)
sc.tl.diffmap(adata)
sc.pl.umap(adata, color='pseudotime')
```

### Workflow 3: Batch Integration

```python
import scanpy as sc
from mccvae import Agent

# Load multi-batch data
adata = sc.read_h5ad("multi_batch_data.h5ad")

# Train with strong contrastive learning
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=15,
    use_moco=True,
    moco_weight=2.0,    # Stronger contrastive learning
    aug_prob=0.7,       # More augmentation
    loss_mode="nb"
)
agent.fit(epochs=1500)

# Evaluate batch mixing
adata.obsm['X_mccvae'] = agent.get_latent()
sc.pp.neighbors(adata, use_rep='X_mccvae')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['batch', 'cell_type'])
```

---

## Additional Resources

- **GitHub Repository**: https://github.com/PeterPonyu/MCCVAE
- **Paper**: [Link to publication when available]
- **Issues**: Report bugs at https://github.com/PeterPonyu/MCCVAE/issues

## Citation

If you use MCCVAE in your research, please cite:

```
[Citation information to be added]
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
