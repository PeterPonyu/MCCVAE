
"""
Evaluation Metrics and Utilities for Clustering and Dimensionality Reduction

This module provides comprehensive evaluation metrics for clustering algorithms
and dimensionality reduction techniques, with support for batch correction
assessment and graph-based connectivity analysis.
"""

import numpy as np
from numpy import ndarray
import pandas as pd
import scib
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.sparse import csr_matrix, issparse
from scipy.sparse import csgraph
from typing import List, Tuple, Union, Optional


def get_dfs(mode: str, agent_list: List) -> map:
    """
    Compute aggregated statistics from agent evaluation results.
    
    This function processes evaluation results from multiple agents and computes
    either mean or standard deviation statistics across different metrics.
    
    Args:
        mode: Statistical mode to compute ("mean" or "std")
        agent_list: List of agent objects containing evaluation scores
        
    Returns:
        Map object containing DataFrames with aggregated statistics
        
    Raises:
        ValueError: If mode is not "mean" or "std"
    """
    if mode not in ["mean", "std"]:
        raise ValueError("Mode must be either 'mean' or 'std'")
    
    # Compute statistics based on mode
    if mode == "mean":
        aggregated_stats = list(
            map(
                lambda agent_group: zip(
                    *(
                        np.array(metric_batch).mean(axis=0)
                        for metric_batch in zip(*((zip(*agent.score)) for agent in agent_group))
                    )
                ),
                list(zip(*agent_list)),
            )
        )
    elif mode == "std":
        aggregated_stats = list(
            map(
                lambda agent_group: zip(
                    *(
                        np.array(metric_batch).std(axis=0)
                        for metric_batch in zip(*((zip(*agent.score)) for agent in agent_group))
                    )
                ),
                list(zip(*agent_list)),
            )
        )
    
    # Convert to DataFrames with meaningful column names
    metric_columns = ["ARI", "NMI", "ASW", "C_H", "D_B", "P_C"]
    return map(
        lambda stats: pd.DataFrame(stats, columns=metric_columns),
        aggregated_stats,
    )


def moving_average(a: Union[List, ndarray], window_size: int) -> ndarray:
    """
    Compute moving average with centered window.
    
    Args:
        a: Input array or list of values
        window_size: Size of the moving window
        
    Returns:
        NumPy array containing the moving average values
    """
    series = pd.Series(a)
    return (
        series.rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def fetch_score(
    adata1, 
    q_z: ndarray, 
    label_true: ndarray, 
    label_mode: str = "KMeans", 
    batch: bool = False
) -> Tuple[float, ...]:
    """
    Compute comprehensive clustering evaluation metrics.
    
    This function evaluates clustering performance using various metrics including
    mutual information, silhouette score, and graph connectivity measures.
    
    Args:
        adata1: AnnData object containing the dataset
        q_z: Latent representation or probability matrix
        label_true: Ground truth labels
        label_mode: Method for obtaining cluster labels ("KMeans", "Max", or "Min")
        batch: Whether to compute batch correction metrics
        
    Returns:
        Tuple of evaluation metrics. If batch=False: (NMI, ARI, ASW, C_H, D_B)
        If batch=True: (NMI, ARI, ASW, C_H, D_B, G_C, clisi, ilisi, bASW)
        
    Raises:
        ValueError: If label_mode is not one of the supported options
    """
    # Subsample for computational efficiency if dataset is large
    if adata1.shape[0] > 3000:
        subsample_indices = np.random.choice(
            np.random.permutation(adata1.shape[0]), 3000, replace=False
        )
        adata1 = adata1[subsample_indices, :]
        q_z = q_z[subsample_indices, :]
        label_true = label_true[subsample_indices]
    
    # Generate cluster labels based on specified mode
    if label_mode == "KMeans":
        labels = KMeans(n_clusters=q_z.shape[1], random_state=42).fit_predict(q_z)
    elif label_mode == "Max":
        labels = np.argmax(q_z, axis=1)
    elif label_mode == "Min":
        labels = np.argmin(q_z, axis=1)
    else:
        raise ValueError("label_mode must be one of 'KMeans', 'Max', or 'Min'")

    # Store results in AnnData object
    adata1.obsm["X_qz"] = q_z
    adata1.obs["label"] = pd.Categorical(labels)

    # Compute core clustering metrics
    nmi_score = normalized_mutual_info_score(label_true, labels)
    ari_score = adjusted_mutual_info_score(label_true, labels)
    asw_score = silhouette_score(q_z, labels)
    
    # For non-KMeans methods, use absolute silhouette score
    if label_mode != "KMeans":
        asw_score = abs(asw_score)
    
    ch_score = calinski_harabasz_score(q_z, labels)
    db_score = davies_bouldin_score(q_z, labels)

    # Additional subsampling for graph-based metrics if needed
    if adata1.shape[0] > 5000:
        graph_indices = np.random.choice(
            np.random.permutation(adata1.shape[0]), 5000, replace=False
        )
        adata1 = adata1[graph_indices, :]
    
    # Compute graph connectivity metric
    knn_graph = kneighbors_graph(adata1.obsm["X_qz"], n_neighbors=15)
    graph_connectivity = graph_connection(knn_graph, adata1.obs["label"].values)
    
    # Compute cLISI (cluster LISI) score
    clisi_score = scib.metrics.clisi_graph(
        adata1, "label", "embed", "X_qz", n_cores=-2
    )
    
    if batch:
        # Compute batch correction metrics
        ilisi_score = scib.metrics.ilisi_graph(
            adata1, "batch", "embed", "X_qz", n_cores=-2
        )
        batch_asw = scib.metrics.silhouette_batch(
            adata1, "batch", "label", "X_qz"
        )
        return nmi_score, ari_score, asw_score, ch_score, db_score, graph_connectivity, clisi_score, ilisi_score, batch_asw
    
    return nmi_score, ari_score, asw_score, ch_score, db_score


def graph_connection(graph: csr_matrix, labels: ndarray) -> float:
    """
    Compute graph connectivity score for clustering evaluation.
    
    This function measures how well-connected clusters are in the neighborhood graph
    by analyzing the largest connected component within each cluster.
    
    Args:
        graph: Sparse adjacency matrix representing the neighborhood graph
        labels: Cluster labels for each node
        
    Returns:
        Average connectivity score across all clusters (higher is better)
    """
    connectivity_scores = []
    
    for cluster_label in np.unique(labels):
        # Extract nodes belonging to current cluster
        cluster_mask = np.where(labels == cluster_label)[0]
        cluster_subgraph = graph[cluster_mask, :][:, cluster_mask]
        
        # Find connected components within the cluster
        n_components, component_labels = csgraph.connected_components(
            cluster_subgraph, connection="strong"
        )
        
        # Compute size distribution of connected components
        _, component_sizes = np.unique(component_labels, return_counts=True)
        
        # Connectivity score: ratio of largest component to total cluster size
        connectivity_ratio = component_sizes.max() / component_sizes.sum()
        connectivity_scores.append(connectivity_ratio)
    
    return np.mean(connectivity_scores)

