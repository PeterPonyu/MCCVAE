"""
scAGCL: Adversarial Graph Contrastive Learning for single-cell data
Uses adversarial perturbations on graph structure and features for contrastive learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Dict, Optional, Tuple, Any
from .base_model import BaseModel

try:
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import squareform, pdist
except ImportError:
    NearestNeighbors = None
    squareform = None
    pdist = None


class GCNConv(nn.Module):
    """Graph Convolution Layer"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class AGCLEncoder(nn.Module):
    """Multi-layer GCN encoder"""
    def __init__(self, dim_in: int, dim_out: int, num_layers: int = 2):
        super().__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.conv = nn.ModuleList()
        self.conv.append(GCNConv(dim_in, 2 * dim_out))
        for _ in range(1, num_layers - 1):
            self.conv.append(GCNConv(2 * dim_out, 2 * dim_out))
        self.conv.append(GCNConv(2 * dim_out, dim_out))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = F.relu(self.conv[i](x, adj))
        return x


class AGCLCore(nn.Module):
    """scAGCL core model with contrastive learning"""
    def __init__(self, encoder: AGCLEncoder, n_hidden: int, n_proj_hidden: int, tau: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc_layer1 = nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = nn.Linear(n_proj_hidden, n_hidden)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc_layer1(z))
        return self.fc_layer2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int = 0) -> torch.Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self._batched_semi_loss(h1, h2, batch_size)
            l2 = self._batched_semi_loss(h2, h1, batch_size)

        return ((l1 + l2) * 0.5).mean()

    def _batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int) -> torch.Tensor:
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))
            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)


def normalize_adj_tensor(adj: torch.Tensor) -> torch.Tensor:
    """Symmetrically normalize adjacency tensor: D^{-1/2} A D^{-1/2}"""
    rowsum = torch.sum(adj, 1)
    d_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


def adj_from_edge_index(num_nodes: int, edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert edge index to normalized dense adjacency matrix (no torch_geometric dependency)
    
    Args:
        num_nodes: Number of nodes
        edge_index: Edge index tensor [2, num_edges]
        device: Device for tensor
        
    Returns:
        Normalized adjacency matrix [num_nodes, num_nodes]
    """
    # Create adjacency matrix with self-loops
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    
    if edge_index.size(1) > 0:
        row, col = edge_index[0], edge_index[1]
        adj[row, col] = 1.0
    
    # Add self-loops
    adj = adj + torch.eye(num_nodes, device=device)
    
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = adj.sum(1)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    
    return D_inv_sqrt @ adj @ D_inv_sqrt


def edge_dropping(edge_index: torch.Tensor, p: float = 0.5, force_undirected: bool = False) -> torch.Tensor:
    """Drop edges with probability p"""
    if p <= 0. or not edge_index.size(1):
        return edge_index
    row, col = edge_index
    edge_mask = torch.bernoulli(torch.ones(row.size(0), device=edge_index.device) * (1 - p)).bool()
    if force_undirected:
        edge_mask[row > col] = False
    edge_index = edge_index[:, edge_mask]
    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def gene_dropping(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Drop genes (features) with probability"""
    drop_mask = torch.bernoulli(torch.ones(x.size(1), device=x.device) * (1 - drop_prob)).bool()
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def build_knn_graph(features: np.ndarray, num_clusters: int) -> Tuple[np.ndarray, torch.Tensor]:
    """Build kNN graph from features"""
    if NearestNeighbors is None or squareform is None:
        raise ImportError("scikit-learn and scipy required for kNN graph construction")

    cell_num = features.shape[0]
    average_num = cell_num // num_clusters
    neighbor_num = max(5, min(15, average_num // 10))

    dis_matrix = squareform(pdist(features, metric='correlation'))
    
    # Handle NaN values that can occur with correlation distance
    # Replace NaN with maximum finite distance or 2.0 (max correlation distance)
    if np.isnan(dis_matrix).any():
        max_dist = np.nanmax(dis_matrix) if not np.isnan(dis_matrix).all() else 2.0
        dis_matrix = np.nan_to_num(dis_matrix, nan=max_dist)
 
    nbrs = NearestNeighbors(n_neighbors=neighbor_num, metric='precomputed').fit(dis_matrix)
    _, indices = nbrs.kneighbors(dis_matrix)

    adj_matrix = np.zeros((cell_num, cell_num))
    for i in range(cell_num):
        for j in indices[i]:
            adj_matrix[i, j] = 1

    edge_list = []
    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1:
                edge_list.append([i, j])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.int64).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)

    return adj_matrix, edge_index


class scAGCLModel(BaseModel):
    """
    scAGCL: Adversarial Graph Contrastive Learning for single-cell data
    
    Features:
    - Adversarial graph perturbation for robust contrastive learning
    - Two-view augmentation (edge dropping, gene dropping)
    - Contrastive loss with projection head
    - Optional adversarial training for harder negative mining
    - No decoder (contrastive-only)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        proj_dim: int = 256,
        num_layers: int = 2,
        tau: float = 0.5,
        edge_drop_rate1: float = 0.4,
        edge_drop_rate2: float = 0.3,
        feature_drop_rate1: float = 0.3,
        feature_drop_rate2: float = 0.4,
        adversarial_weight: float = 1.0,
        num_clusters: int = 8,
        model_name: str = "scAGCL",
    ):
        """
        Args:
            input_dim: Gene dimension
            latent_dim: Hidden/latent dimension
            proj_dim: Projection head dimension
            num_layers: Number of GCN layers
            tau: Temperature for contrastive loss
            edge_drop_rate1: Edge drop rate for view 1
            edge_drop_rate2: Edge drop rate for view 2
            feature_drop_rate1: Feature drop rate for view 1
            feature_drop_rate2: Feature drop rate for view 2
            adversarial_weight: Weight for adversarial loss (0 to disable)
            num_clusters: Expected number of clusters (for graph construction)
        """
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[latent_dim * 2],
            model_name=model_name,
        )

        self.proj_dim = proj_dim
        self.num_layers = num_layers
        self.tau = tau
        self.edge_drop_rate1 = edge_drop_rate1
        self.edge_drop_rate2 = edge_drop_rate2
        self.feature_drop_rate1 = feature_drop_rate1
        self.feature_drop_rate2 = feature_drop_rate2
        self.adversarial_weight = adversarial_weight
        self.num_clusters = num_clusters

        encoder = AGCLEncoder(input_dim, latent_dim, num_layers)
        self.model = AGCLCore(encoder, latent_dim, proj_dim, tau)

    def _prepare_batch(self, batch_data: Any, device: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            return x, {}
        x = batch_data.to(device).float()
        return x, {}

    def encode(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Encode to latent space"""
        if adj is None:
            _, edge_index = build_knn_graph(x.detach().cpu().numpy(), self.num_clusters)
            adj = adj_from_edge_index(x.size(0), edge_index.to(x.device), x.device)
        return self.model(x, adj)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("scAGCL is contrastive-only; no decoder.")

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Use fit() for scAGCL training.")

    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Use fit() for scAGCL training.")

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 500,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 50,
        verbose: int = 1,
        subgraph_size: int = 400,
        num_iters_adv: int = 10,
        alpha: float = 100,
        beta: float = 0.01,
        **kwargs,
    ) -> Dict[str, list]:
        """
        Train scAGCL model
        
        Args:
            train_loader: DataLoader with full data
            epochs: Number of epochs
            lr: Learning rate
            device: Device
            subgraph_size: Size of subgraph for training
            num_iters_adv: Number of adversarial iterations
            alpha: Adversarial structure attack strength
            beta: Adversarial feature attack strength
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        history = {"train_loss": []}

        # Get full data
        full_x = self._full_x_from_loader(train_loader).to(device).float()
        adj_np, edge_index = build_knn_graph(full_x.detach().cpu().numpy(), self.num_clusters)
        edge_index = edge_index.to(device)
        adj = adj_from_edge_index(full_x.size(0), edge_index, device)

        best_loss = float("inf")
        patience_counter = 0

        # Prepare validation data if provided
        val_x = None
        val_adj = None
        if val_loader is not None:
            val_x = self._full_x_from_loader(val_loader).to(device).float()
            val_adj_np, val_edge_index = build_knn_graph(val_x.detach().cpu().numpy(), self.num_clusters)
            val_adj = adj_from_edge_index(val_x.size(0), val_edge_index.to(device), device)

        for epoch in range(epochs):
            self.train()

            # Sample subgraph
            n = full_x.size(0)
            subgraph_size_actual = min(subgraph_size, n)
            perm = torch.randperm(n)[:subgraph_size_actual]
            x_sub = full_x[perm]

            # Build subgraph adjacency
            adj_sub_np = adj_np[perm.cpu().numpy()][:, perm.cpu().numpy()]
            edge_list = []
            for i in range(adj_sub_np.shape[0]):
                for j in range(adj_sub_np.shape[1]):
                    if adj_sub_np[i, j] == 1:
                        edge_list.append([i, j])
            if edge_list:
                edge_sub = torch.tensor(edge_list, dtype=torch.int64, device=device).T
            else:
                edge_sub = torch.empty((2, 0), dtype=torch.int64, device=device)

            # Augmentations
            x_1 = gene_dropping(x_sub, self.feature_drop_rate1)
            x_2 = gene_dropping(x_sub, self.feature_drop_rate2)

            edge_1 = edge_dropping(edge_sub, self.edge_drop_rate1, force_undirected=True)
            edge_2 = edge_dropping(edge_sub, self.edge_drop_rate2, force_undirected=True)

            adj_1 = adj_from_edge_index(x_1.size(0), edge_1, device)
            adj_2 = adj_from_edge_index(x_2.size(0), edge_2, device)

            optimizer.zero_grad()
            z_1 = self.model(x_1, adj_1)
            z_2 = self.model(x_2, adj_2)
            loss = self.model.loss(z_1, z_2)

            loss.backward()
            optimizer.step()

            train_loss_val = float(loss.item())
            history["train_loss"].append(train_loss_val)

            # Compute validation loss if val_loader provided
            if val_loader is not None and val_x is not None:
                self.eval()
                with torch.no_grad():
                    # Use same augmentation for validation
                    v_x1 = gene_dropping(val_x, self.feature_drop_rate1)
                    v_x2 = gene_dropping(val_x, self.feature_drop_rate2)
                    v_z1 = self.model(v_x1, val_adj)
                    v_z2 = self.model(v_x2, val_adj)
                    val_loss = float(self.model.loss(v_z1, v_z2).item())
                current_loss = val_loss

                # Early stopping only when validation is available
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"\n✓ Early stopping at epoch {epoch+1}")
                    break
            else:
                current_loss = train_loss_val
                # No early stopping when val_loader is None, just save best train model
                if save_path and current_loss < best_loss:
                    best_loss = current_loss
                    self.save_model(save_path)

            if verbose >= 1 and (epoch + 1) % 20 == 0:
                msg = f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss_val:.4f}"
                if val_loader is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)

        if verbose >= 1:
            print("\n✓ Training finished!")
        return history

    def _full_x_from_loader(self, loader) -> torch.Tensor:
        """Extract full data tensor from loader"""
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

    @torch.no_grad()
    def extract_latent(
        self,
        data_loader,
        device: str = "cuda",
        return_reconstructions: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Extract latent representations"""
        if return_reconstructions:
            warnings.warn("scAGCL has no decoder; return_reconstructions ignored.")

        self.eval()
        self.to(device)

        full_x = self._full_x_from_loader(data_loader).to(device).float()
        _, edge_index = build_knn_graph(full_x.detach().cpu().numpy(), self.num_clusters)
        adj = adj_from_edge_index(full_x.size(0), edge_index.to(device), device)
        z = self.model(full_x, adj)

        return {"latent": z.detach().cpu().numpy()}


def create_scagcl_model(input_dim: int, latent_dim: int = 256, **kwargs) -> scAGCLModel:
    """
    Create scAGCL model
    
    Example:
        >>> model = create_scagcl_model(2000, latent_dim=256, num_clusters=8)
    """
    return scAGCLModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
