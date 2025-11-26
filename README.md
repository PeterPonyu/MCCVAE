
# MCC: Momentum Contrastive Coupling

MCC is a sophisticated deep learning framework for single-cell genomics analysis that unifies discrete cell identities and continuous trajectories. It combines a Momentum Contrastive (MoCo) module with an information-bottleneck coupling within a Variational Autoencoder (VAE) framework to learn rich, disentangled representations of single-cell data.

## Features

- **Momentum Contrastive Learning:** Utilizes MoCo to impose geometric discipline in latent space, cleanly separating cell populations through InfoNCE loss with momentum-updated encoders and memory queues.
- **Information Bottleneck Coupling:** Implements a coupling component that compresses and refines latent codes to form smooth, low-dimensional manifolds tracing gradual cellular transitions.
- **Dual Reconstruction Pathways:** Features two complementary reconstruction paths - direct from primary latent representation and through the information bottleneck.
- **Modality-Agnostic Architecture:** Seamlessly transfers across different single-cell modalities (scRNA-seq, scATAC-seq) without modification.
- **Unified Framework:** Harmonizes generative fidelity with contrastive discriminability to respect both population boundaries and the continua that bridge them.

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

The primary interface for using MCC is the `Agent` class. Here's a basic example of how to train a model and extract latent representations:

```python
import scanpy as sc
from mccvae import Agent

# Load your single-cell data into an AnnData object
adata = sc.read_h5ad("your_data.h5ad")

# Initialize the agent with your data and desired parameters
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=10,
    use_moco=True,
    irecon=1.
)

# Train the model
agent.fit(epochs=1000)

# Extract the primary latent representations (z)
# Shape: (n_cells, latent_dim) - captures overall biological structure
latent_representations = agent.get_latent()

# Extract the compressed bottleneck embeddings (l_e)
# Shape: (n_cells, i_dim) - low-dimensional coupled representations
bottleneck_embeddings = agent.get_bottleneck()

# Extract the refined latent representations (l_d)
# Shape: (n_cells, latent_dim) - bottleneck-refined representations
refined_embeddings = agent.get_refined()
```

**Note on Embeddings:**
- **z (via `get_latent()`)**: Primary latent space representations with dimension `latent_dim`
- **l_e (via `get_bottleneck()` or `get_iembed()`)**: Compressed bottleneck with dimension `i_dim` (typically 2)
- **l_d (via `get_refined()`)**: Refined representations obtained by decoding l_e back to `latent_dim`

## Architecture Overview

The MCC framework integrates three key components:

### 1. VAE Backbone
- **Query Encoder**: Maps input data to primary latent representation z
- **Decoder**: Reconstructs data from latent representations
- **Standard VAE Loss**: Negative binomial reconstruction loss + KL divergence

### 2. MoCo Component
- **Momentum Encoder**: Updates parameters via exponential moving average
- **Memory Queue**: FIFO mechanism storing negative samples for contrastive learning
- **InfoNCE Loss**: Contrasts query with positive key and negative samples
- **L2 Normalization**: Applied to embeddings before similarity computation

### 3. Coupling Component
- **Information Bottleneck**: Compresses z → l_e → l_d pathway
  - **l_e (bottleneck_encoded)**: Compressed representation via `bottleneck_encoder`, dimension `i_dim`
  - **l_d (bottleneck_decoded)**: Refined representation via `bottleneck_decoder`, dimension `latent_dim`
- **Latent Encoder/Decoder**: Linear transformations `f_enc: latent_dim → i_dim` and `f_dec: i_dim → latent_dim`
- **Coupling Reconstruction**: Secondary reconstruction path through refined latent l_d
- **Accessible via**: `get_bottleneck()` for l_e, `get_refined()` for l_d

## Loss Function

The complete MCC objective combines four loss components:

```
L_MCC = L_NB + β·L_KLD + α·L_MoCo + γ·L_Cou
```

Where:
- **L_NB**: Negative binomial reconstruction loss (primary path: z → reconstruction)
- **L_KLD**: KL divergence regularization
- **L_MoCo**: InfoNCE contrastive loss with momentum updates
- **L_Cou**: Coupling reconstruction loss (bottleneck path: z → l_e → l_d → reconstruction)


## License

This project is licensed under the MIT License - see the LICENSE file for details.
