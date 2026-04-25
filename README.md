
<div align="center">
  <a href="https://peterponyu.github.io/">
    <img src="https://peterponyu.github.io/assets/badges/MCCVAE.svg" width="64" alt="ZF Lab · MCCVAE">
  </a>
</div>

# MCC: Momentum Contrastive Coupling

[![GitHub Pages](https://img.shields.io/badge/Web%20Portal-Live-brightgreen)](https://peterponyu.github.io/MCCVAE/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Interactive Benchmarking Portal:** [https://peterponyu.github.io/MCCVAE/](https://peterponyu.github.io/MCCVAE/)

**Publication status:** Published in *Biomedical Signal Processing and Control*.

**Fu, Z.**, Chen, C., Zhang, K. (2026). Islands and bridges: Momentum contrastive coupling unifies discrete and continuous structure in single-cell omics. *Biomedical Signal Processing and Control*, 122, 110376. [DOI](https://doi.org/10.1016/j.bspc.2026.110376) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1746809426009304)

MCC is a deep learning framework for single-cell genomics analysis. It combines a Momentum Contrastive (MoCo) module with an information-bottleneck coupling within a Variational Autoencoder (VAE) framework.

## Features

- **Momentum Contrastive Learning:** Uses MoCo with InfoNCE loss, momentum-updated encoders, and memory queues for contrastive representation learning.
- **Information Bottleneck Coupling:** Compresses and refines latent codes through a low-dimensional bottleneck layer.
- **Dual Reconstruction Pathways:** Two reconstruction paths - direct from primary latent representation and through the information bottleneck.
- **Multiple Modalities:** Supports scRNA-seq and scATAC-seq data.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PeterPonyu/MCCVAE.git
    cd MCCVAE
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## UI

Visualization training

```bash
python start_servers.py
```

## Usage

Use the `Agent` class to train models and extract latent representations.

```python
import scanpy as sc
from mccvae import Agent

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize agent
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=10,
    use_moco=True,
    irecon=1.
)

# Train
agent.fit(epochs=1000)

# Extract representations
latent_representations = agent.get_latent()       # Shape: (n_cells, latent_dim)
bottleneck_embeddings = agent.get_bottleneck()    # Shape: (n_cells, i_dim)
refined_embeddings = agent.get_refined()          # Shape: (n_cells, latent_dim)
```

**Embeddings:**
- `get_latent()`: Primary latent representations (z) with dimension `latent_dim`
- `get_bottleneck()` or `get_iembed()`: Compressed bottleneck (l_e) with dimension `i_dim`
- `get_refined()`: Refined representations (l_d) with dimension `latent_dim`

## Architecture

The framework consists of three components:

### VAE Backbone
- **Query Encoder**: Maps input to latent representation z
- **Decoder**: Reconstructs data from latent representations
- **Loss**: Negative binomial reconstruction + KL divergence

### MoCo Component
- **Momentum Encoder**: Parameters updated via exponential moving average
- **Memory Queue**: FIFO mechanism for negative samples
- **InfoNCE Loss**: Contrastive loss with L2-normalized embeddings

### Coupling Component
- **Information Bottleneck**: z → l_e → l_d pathway
  - l_e: Compressed representation via `bottleneck_encoder` (dimension: `i_dim`)
  - l_d: Refined representation via `bottleneck_decoder` (dimension: `latent_dim`)
- **Secondary Reconstruction**: Reconstruction through l_d

## Loss Function

The MCC objective combines four loss components:

```
L_MCC = L_NB + β·L_KLD + α·L_MoCo + γ·L_Cou
```

- **L_NB**: Negative binomial reconstruction loss (z → reconstruction)
- **L_KLD**: KL divergence regularization
- **L_MoCo**: InfoNCE contrastive loss
- **L_Cou**: Coupling reconstruction loss (z → l_e → l_d → reconstruction)


## Citation

If you use MCCVAE or the MCC benchmark portal, please cite:

```bibtex
@article{fu2026islands,
  title = {Islands and bridges: Momentum contrastive coupling unifies discrete and continuous structure in single-cell omics},
  author = {Fu, Zeyu and Chen, Chunlin and Zhang, Keyang},
  journal = {Biomedical Signal Processing and Control},
  volume = {122},
  pages = {110376},
  year = {2026},
  doi = {10.1016/j.bspc.2026.110376}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
