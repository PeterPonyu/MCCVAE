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
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse import issparse


def get_dfs(mode, agent_list):
    if mode == "mean":
        ls = list(
            map(
                lambda x: zip(
                    *(
                        np.array(b).mean(axis=0)
                        for b in zip(*((zip(*a.score)) for a in x))
                    )
                ),
                list(zip(*agent_list)),
            )
        )
    elif mode == "std":
        ls = list(
            map(
                lambda x: zip(
                    *(
                        np.array(b).std(axis=0)
                        for b in zip(*((zip(*a.score)) for a in x))
                    )
                ),
                list(zip(*agent_list)),
            )
        )
    return map(
        lambda x: pd.DataFrame(x, columns=["ARI", "NMI", "ASW", "C_H", "D_B", "P_C"]),
        ls,
    )


def moving_average(a, window_size):
    series = pd.Series(a)
    return (
        series.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    )


def fetch_score(adata1, q_z, label_true, label_mode="KMeans", batch=False):
    if adata1.shape[0] > 3e3:
        idxs = np.random.choice(
            np.random.permutation(adata1.shape[0]), 3000, replace=False
        )
        adata1 = adata1[idxs, :]
        q_z = q_z[idxs, :]
        label_true = label_true[idxs]
    if label_mode == "KMeans":
        labels = KMeans(q_z.shape[1]).fit_predict(q_z)
    elif label_mode == "Max":
        labels = np.argmax(q_z, axis=1)
    elif label_mode == "Min":
        labels = np.argmin(q_z, axis=1)
    else:
        raise ValueError("Mode must be in one of KMeans, Max and Min")

    adata1.obsm["X_qz"] = q_z
    adata1.obs["label"] = pd.Categorical(labels)

    NMI = normalized_mutual_info_score(label_true, labels)
    ARI = adjusted_mutual_info_score(label_true, labels)
    ASW = silhouette_score(q_z, labels)
    if label_mode != "KMeans":
        ASW = abs(ASW)
    C_H = calinski_harabasz_score(q_z, labels)
    D_B = davies_bouldin_score(q_z, labels)

    if adata1.shape[0] > 5e3:
        idxs = np.random.choice(
            np.random.permutation(adata1.shape[0]), 5000, replace=False
        )
        adata1 = adata1[idxs, :]
    G_C = graph_connection(
        kneighbors_graph(adata1.obsm["X_qz"], 15), adata1.obs["label"].values
    )
    clisi = scib.metrics.clisi_graph(adata1, "label", "embed", "X_qz", n_cores=-2)
    if batch:
        ilisi = scib.metrics.ilisi_graph(adata1, "batch", "embed", "X_qz", n_cores=-2)
        bASW = scib.metrics.silhouette_batch(adata1, "batch", "label", "X_qz")
        return NMI, ARI, ASW, C_H, D_B, G_C, clisi, ilisi, bASW
    return NMI, ARI, ASW, C_H, D_B


def graph_connection(graph: csr_matrix, labels: ndarray):
    cg_res = []
    for l in np.unique(labels):
        mask = np.where(labels == l)[0]
        subgraph = graph[mask, :][:, mask]
        _, lab = csgraph.connected_components(subgraph, connection="strong")
        tab = np.unique(lab, return_counts=True)[1]
        cg_res.append(tab.max() / tab.sum())
    return np.mean(cg_res)


def quiver_autoscale(
    E: np.ndarray,
    V: np.ndarray,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()

    Q = ax.quiver(
        E[:, 0] / scale_factor,
        E[:, 1] / scale_factor,
        V[:, 0],
        V[:, 1],
        angles="xy",
        scale=None,
        scale_units="xy",
    )
    Q._init()
    fig.clf()
    plt.close(fig)
    return Q.scale / scale_factor


def l2_norm(x, axis=-1):
    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    else:
        return np.sqrt(np.sum(x * x, axis=axis))
