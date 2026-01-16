"""
scGCC: Graph contrastive learning with MoCo for single-cell data
Dual mode: PyG Data loader or tensor loader with automatic kNN graph construction
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Dict, Optional, Tuple, Callable
from .base_model import BaseModel

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Batch, Data
except Exception:
    GATConv = None
    Batch = None
    Data = None

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


class TwoViewAugmenter(nn.Module):
    """Stochastic augmentation: feature masking + Gaussian noise + scale jitter"""
    def __init__(
        self,
        feature_mask_prob: float = 0.2,
        gaussian_noise_std: float = 0.1,
        scale_jitter: float = 0.0,
    ):
        super().__init__()
        self.feature_mask_prob = float(feature_mask_prob)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.scale_jitter = float(scale_jitter)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        v = x
        if self.feature_mask_prob > 0:
            mask = (torch.rand_like(v) < self.feature_mask_prob)
            v = v.masked_fill(mask, 0.0)
        if self.gaussian_noise_std > 0:
            v = v + torch.randn_like(v) * self.gaussian_noise_std
        if self.scale_jitter > 0:
            s = (1.0 + (torch.rand(v.size(0), 1, device=v.device) * 2 - 1.0) * self.scale_jitter)
            v = v * s
        return v

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._augment(x), self._augment(x)


def dropout_edges(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    """Edge dropout for graph augmentation"""
    if p <= 0:
        return edge_index
    e = edge_index.size(1)
    keep = torch.rand(e, device=edge_index.device) >= p
    out = edge_index[:, keep]
    return out if out.size(1) > 0 else edge_index


class GATEncoder(nn.Module):
    """GAT encoder with optional MLP projection"""
    def __init__(self, in_channels, latent_dim, num_heads=4, dropout=0.4, use_mlp=False):
        super().__init__()
        if GATConv is None:
            raise ImportError("torch_geometric required for scGCCModel (pip install torch-geometric).")

        self.gat1 = GATConv(in_channels, 128, heads=num_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(128 * num_heads, latent_dim, heads=num_heads, dropout=dropout, concat=False)
        self.fc = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, latent_dim)) if use_mlp else None
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return self.fc(x) if self.fc is not None else x


class MoCoGraph(nn.Module):
    """MoCo with momentum encoder and queue for graph contrastive learning"""
    def __init__(self, num_genes, latent_dim, r=1024, m=0.99, T=0.2, heads=4, mlp=False):
        super().__init__()
        self.r = r
        self.m = m
        self.T = T

        self.encoder_q = GATEncoder(num_genes, latent_dim, num_heads=heads, use_mlp=mlp)
        self.encoder_k = GATEncoder(num_genes, latent_dim, num_heads=heads, use_mlp=mlp)

        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(latent_dim, r), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data = pk.data * self.m + pq.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        if self.r % batch_size != 0:
            return
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.r

    def forward(self, im_q, im_k, edge_index, num_seed_nodes: int):
        q_all = self.encoder_q(im_q, edge_index)
        q_seed = F.normalize(q_all[:num_seed_nodes], dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_all = self.encoder_k(im_k, edge_index)
            k_seed = F.normalize(k_all[:num_seed_nodes], dim=1)

        l_pos = torch.einsum("nc,nc->n", q_seed, k_seed).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", q_seed, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k_seed)
        return logits, labels

    @torch.no_grad()
    def infer(self, x, edge_index):
        self.encoder_k.eval()
        return self.encoder_k(x, edge_index)


def build_knn_edge_index(x: torch.Tensor, k: int = 15, metric: str = "cosine") -> torch.Tensor:
    """Build undirected kNN edge_index [2, E] from features [N, F]"""
    if NearestNeighbors is None:
        raise ImportError("scGCCModel requires scikit-learn (pip install scikit-learn).")

    x_np = x.detach().cpu().numpy()
    nnm = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nnm.fit(x_np)
    neigh = nnm.kneighbors(return_distance=False)

    src = np.repeat(np.arange(neigh.shape[0]), neigh.shape[1] - 1)
    dst = neigh[:, 1:].reshape(-1)

    edges = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(edges, dtype=torch.long)


class scGCCModel(BaseModel):
    """
    scGCC: Graph contrastive learning with MoCo
    
    Features:
    - MoCo: momentum encoder + queue-based negative sampling
    - GAT encoder for graph representation
    - Two-view augmentation (masking, noise, edge dropout)
    - Dual training mode:
      A. PyG NeighborLoader with Data objects
      B. Standard tensor loader with automatic kNN graph construction
    - Contrastive-only (no decoder)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        queue_size: int = 512,
        heads: int = 4,
        mlp: bool = False,
        model_name: str = "scGCC",
        view_augmenter: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        feature_mask_prob: float = 0.2,
        gaussian_noise_std: float = 0.1,
        scale_jitter: float = 0.0,
        edge_drop_prob: float = 0.0,
    ):
        """
        Args:
            input_dim: Gene dimension
            latent_dim: Latent dimension
            queue_size: MoCo queue size for negatives
            heads: GAT attention heads
            mlp: Use MLP projection head
            view_augmenter: Custom augmentation function
            feature_mask_prob: Feature masking probability
            gaussian_noise_std: Gaussian noise std
            scale_jitter: Scale jittering magnitude
            edge_drop_prob: Edge dropout probability
        """
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[], model_name=model_name)
        self.moco = MoCoGraph(num_genes=input_dim, latent_dim=latent_dim, r=queue_size, heads=heads, mlp=mlp)
        self.criterion = nn.CrossEntropyLoss()

        self.edge_drop_prob = float(edge_drop_prob)
        if view_augmenter is None:
            self.view_augmenter = TwoViewAugmenter(
                feature_mask_prob=feature_mask_prob,
                gaussian_noise_std=gaussian_noise_std,
                scale_jitter=scale_jitter,
            )
        else:
            self.view_augmenter = view_augmenter

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode nodes to latent space [N, latent_dim]"""
        return self.moco.infer(x, edge_index)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("scGCC is contrastive-only; no decoder.")

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Use fit() with Data batches for scGCC.")

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Use fit() for scGCC.")

    def _full_x_from_loader(self, loader) -> torch.Tensor:
        """Extract full feature matrix from loader"""
        ds = getattr(loader, "dataset", None)
        tensors = getattr(ds, "tensors", None)
        if isinstance(tensors, (tuple, list)) and len(tensors) >= 1 and torch.is_tensor(tensors[0]):
            return tensors[0]
        
        xs = []
        for b in loader:
            if isinstance(b, (list, tuple)):
                xs.append(b[0])
            else:
                xs.append(b)
        return torch.cat(xs, dim=0)

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 20,
        lr: float = 0.1,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 10,
        verbose: int = 1,
        edge_index: Optional[torch.Tensor] = None,
        full_x: Optional[torch.Tensor] = None,
        knn_k: int = 15,
        knn_metric: str = "cosine",
        **kwargs,
    ):
        """
        Dual-mode training:
        - Mode A: PyG loader yields Data objects
        - Mode B: Tensor loader → build kNN graph automatically
        """
        self.to(device)
        opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        def _is_pyg_data(obj) -> bool:
            return hasattr(obj, "x") and hasattr(obj, "edge_index") and hasattr(obj, "to")

        def _to_pyg_batch(obj):
            if _is_pyg_data(obj):
                return obj
            if isinstance(obj, (list, tuple)) and len(obj) > 0 and all(_is_pyg_data(d) for d in obj):
                if Batch is None:
                    raise ImportError("torch_geometric required for list[Data].")
                return Batch.from_data_list(list(obj))
            return None

        # Mode A: PyG Data loader
        first = next(iter(train_loader))
        pyg_batch = _to_pyg_batch(first)
        if pyg_batch is not None:
            for ep in range(epochs):
                self.train()
                tot = 0.0
                n = 0
                for batch in train_loader:
                    data = _to_pyg_batch(batch).to(device)

                    im_q, im_k = self.view_augmenter(data.x)
                    edge_k = dropout_edges(data.edge_index, self.edge_drop_prob) if self.edge_drop_prob > 0 else data.edge_index

                    num_seed_nodes = int(getattr(data, "batch_size", data.num_nodes))

                    logits, labels = self.moco(im_q, im_k, edge_k, num_seed_nodes)
                    loss = self.criterion(logits, labels)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    tot += float(loss.item())
                    n += 1

                if verbose >= 1:
                    print(f"Epoch {ep+1:3d}/{epochs} | Train Loss: {tot/max(n,1):.4f}")
            return {"train_loss": []}

        # Mode B: Tensor loader with kNN graph
        if full_x is None:
            full_x = self._full_x_from_loader(train_loader)

        if edge_index is None:
            edge_index = build_knn_edge_index(full_x, k=knn_k, metric=knn_metric)

        if Data is None:
            raise ImportError("torch_geometric required for scGCC.")

        graph = Data(x=full_x, edge_index=edge_index).to(device)

        if verbose >= 1:
            print(f"[scGCC] Tensor mode: nodes={graph.x.size(0)}, edges={graph.edge_index.size(1)}")

        for ep in range(epochs):
            self.train()
            im_q, im_k = self.view_augmenter(graph.x)
            edge_k = dropout_edges(graph.edge_index, self.edge_drop_prob) if self.edge_drop_prob > 0 else graph.edge_index

            logits, labels = self.moco(im_q, im_k, edge_k, int(graph.num_nodes))
            loss = self.criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if verbose >= 1:
                print(f"Epoch {ep+1:3d}/{epochs} | Train Loss: {float(loss.item()):.4f}")

        return {"train_loss": []}

    @torch.no_grad()
    def extract_latent(
        self,
        data_loader,
        device: str = "cuda",
        return_reconstructions: bool = False,
        edge_index: Optional[torch.Tensor] = None,
        full_x: Optional[torch.Tensor] = None,
        knn_k: int = 15,
        knn_metric: str = "cosine",
    ) -> Dict[str, np.ndarray]:
        """Extract latent with automatic kNN graph construction"""
        if return_reconstructions:
            warnings.warn("scGCC has no decoder; return_reconstructions ignored.")

        self.eval()
        self.to(device)

        if full_x is None:
            full_x = self._full_x_from_loader(data_loader)

        if edge_index is None:
            edge_index = build_knn_edge_index(full_x, k=knn_k, metric=knn_metric)
        
        z = self.encode(full_x.to(device).float(), edge_index.to(device))
        return {"latent": z.detach().cpu().numpy()}


def create_scgcc_model(input_dim: int, latent_dim: int = 256, **kwargs) -> scGCCModel:
    """
    Create scGCC model
    
    Example:
        >>> model = create_scgcc_model(2000, latent_dim=256, queue_size=512, 
        ...                            heads=4, knn_k=15)
    """
    return scGCCModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)