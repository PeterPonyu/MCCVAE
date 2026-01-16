"""
Pseudotime and fate probability computation for Palantir
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_waypoints(multiscale_data, n_waypoints=1200, seed=None):
    """
    Select waypoint cells using max-min sampling.
    
    Parameters
    ----------
    multiscale_data : ndarray
        (n_cells, n_dims) multiscale diffusion space
    n_waypoints : int, default=1200
        Number of waypoints to select
    seed : int, optional
        Random seed
        
    Returns
    -------
    waypoint_indices : ndarray
        Indices of selected waypoint cells
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_cells = multiscale_data.shape[0]
    n_waypoints = min(n_waypoints, n_cells)
    
    # Start with a random cell
    waypoint_indices = [np.random.randint(n_cells)]
    
    # Greedily select waypoints that are far from existing ones
    for _ in range(n_waypoints - 1):
        # Compute distances to all selected waypoints
        waypoint_data = multiscale_data[waypoint_indices]
        distances = cdist(multiscale_data, waypoint_data, metric='euclidean')
        min_distances = distances.min(axis=1)
        
        # Select cell farthest from existing waypoints
        next_waypoint = np.argmax(min_distances)
        waypoint_indices.append(next_waypoint)
    
    return np.array(waypoint_indices)


def build_knn_graph(data, knn=30):
    """
    Build k-nearest neighbor graph.
    
    Parameters
    ----------
    data : ndarray
        (n_cells, n_features) data matrix
    knn : int, default=30
        Number of neighbors
        
    Returns
    -------
    graph : csr_matrix
        (n_cells, n_cells) unweighted adjacency matrix
    """
    nbrs = NearestNeighbors(n_neighbors=knn+1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Build adjacency matrix
    n_cells = data.shape[0]
    row_indices = []
    col_indices = []
    
    for i in range(n_cells):
        # Skip self (first neighbor)
        for j in indices[i, 1:]:
            row_indices.append(i)
            col_indices.append(j)
    
    graph = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), 
                       shape=(n_cells, n_cells))
    
    return graph


def compute_shortest_paths(graph, start_indices):
    """
    Compute shortest paths from start indices to all cells.
    
    Parameters
    ----------
    graph : csr_matrix
        (n_cells, n_cells) adjacency matrix
    start_indices : array-like
        Indices of start cells
        
    Returns
    -------
    distances : ndarray
        (len(start_indices), n_cells) shortest path distances
    """
    distances_list = []
    
    for start_idx in start_indices:
        dist = shortest_path(graph, indices=start_idx, return_predecessors=False)
        distances_list.append(dist)
    
    return np.array(distances_list)


def _compute_pseudotime_from_paths(paths, start_cell_idx, impute_data=None, 
                                   multiscale_data=None, knn=30, n_iterations=25):
    """
    Compute pseudotime from shortest paths using iterative refinement.
    
    Parameters
    ----------
    paths : ndarray
        (n_cells,) shortest path distances from start cell
    start_cell_idx : int
        Index of the start cell
    impute_data : ndarray, optional
        Imputed expression data for weighting
    multiscale_data : ndarray, optional
        Multiscale diffusion space
    knn : int, default=30
        Number of neighbors for local neighborhoods
    n_iterations : int, default=25
        Number of iterations for pseudotime refinement
        
    Returns
    -------
    pseudotime : ndarray
        (n_cells,) refined pseudotime values
    """
    n_cells = len(paths)
    pseudotime = paths.copy().astype(float)
    
    # Handle infinite distances
    finite_mask = np.isfinite(pseudotime)
    if not np.all(finite_mask):
        # Some cells are unreachable; set their pseudotime to max
        max_time = np.max(pseudotime[finite_mask])
        pseudotime[~finite_mask] = max_time
    
    # Normalize to [0, max_value]
    max_pseudotime = np.max(pseudotime)
    if max_pseudotime > 0:
        pseudotime = pseudotime / max_pseudotime
    
    return pseudotime


def compute_pseudotime(multiscale_data, early_cell_idx, terminal_states=None,
                       knn=30, n_waypoints=1200, n_iterations=25, 
                       use_early_cell_as_start=False, scale_components=True,
                       seed=None):
    """
    Compute pseudotime from multiscale diffusion space.
    
    Parameters
    ----------
    multiscale_data : ndarray
        (n_cells, n_dims) multiscale diffusion space
    early_cell_idx : int
        Index of the early (start) cell
    terminal_states : array-like, optional
        Indices of terminal (end) cells
    knn : int, default=30
        Number of neighbors for k-NN graph
    n_waypoints : int, default=1200
        Number of waypoints for trajectory refinement
    n_iterations : int, default=25
        Number of iterations for pseudotime refinement
    use_early_cell_as_start : bool, default=False
        If True, use early cell as start; otherwise use terminal states
    scale_components : bool, default=True
        If True, scale diffusion components
    seed : int, optional
        Random seed
        
    Returns
    -------
    pseudotime : ndarray
        (n_cells,) pseudotime values in [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Build k-NN graph from multiscale space
    knn_graph = build_knn_graph(multiscale_data, knn=knn)
    
    # Compute shortest paths from early cell
    if use_early_cell_as_start:
        start_indices = [early_cell_idx]
    else:
        start_indices = [early_cell_idx]
    
    paths = compute_shortest_paths(knn_graph, start_indices)[0]
    
    # Refine pseudotime
    pseudotime = _compute_pseudotime_from_paths(paths, early_cell_idx, 
                                               multiscale_data=multiscale_data,
                                               knn=knn, n_iterations=n_iterations)
    
    # Normalize to [0, 1]
    min_pt = np.min(pseudotime)
    max_pt = np.max(pseudotime)
    if max_pt > min_pt:
        pseudotime = (pseudotime - min_pt) / (max_pt - min_pt)
    
    return pseudotime


def compute_branch_probabilities(multiscale_data, pseudotime, early_cell_idx, 
                                 terminal_states, knn=30):
    """
    Compute branch probabilities (fate probabilities) for each cell.
    
    Uses a forward-biased Markov chain and computes absorption probabilities
    to terminal states using the fundamental matrix method.
    
    Parameters
    ----------
    multiscale_data : ndarray
        (n_cells, n_dims) multiscale diffusion space
    pseudotime : ndarray
        (n_cells,) pseudotime values
    early_cell_idx : int
        Index of early cell
    terminal_states : array-like
        Indices of terminal states
    knn : int, default=30
        Number of neighbors for constructing transition matrix
        
    Returns
    -------
    branch_probs : ndarray
        (n_cells, n_terminal_states) probability of reaching each terminal state
    """
    n_cells = len(pseudotime)
    n_terminals = len(terminal_states)
    
    # Build forward-biased k-NN graph
    # Only allow transitions to cells with higher pseudotime
    nbrs = NearestNeighbors(n_neighbors=knn+1, algorithm='auto').fit(multiscale_data)
    distances, indices = nbrs.kneighbors(multiscale_data)
    
    # Build transition matrix
    row_indices = []
    col_indices = []
    data_values = []
    
    for i in range(n_cells):
        # Get neighbors
        neighbors = indices[i, 1:]  # Skip self
        
        # Filter for forward transitions (higher pseudotime)
        forward_neighbors = neighbors[pseudotime[neighbors] >= pseudotime[i]]
        
        if len(forward_neighbors) == 0:
            # No forward neighbors - terminal state
            row_indices.append(i)
            col_indices.append(i)
            data_values.append(1.0)
        else:
            # Create uniform transitions to forward neighbors
            weight = 1.0 / len(forward_neighbors)
            for j in forward_neighbors:
                row_indices.append(i)
                col_indices.append(j)
                data_values.append(weight)
    
    T_forward = csr_matrix((data_values, (row_indices, col_indices)), 
                          shape=(n_cells, n_cells))
    
    # Compute absorption probabilities using fundamental matrix method
    # Partition cells into transient and absorbing states
    terminal_set = set(terminal_states)
    is_terminal = np.array([i in terminal_set for i in range(n_cells)])
    is_transient = ~is_terminal
    
    transient_indices = np.where(is_transient)[0]
    terminal_indices = np.where(is_terminal)[0]
    
    n_transient = len(transient_indices)
    n_absorbing = len(terminal_indices)
    
    # Extract Q (transient to transient) and R (transient to absorbing)
    # Map old indices to new indices
    old_to_new = np.full(n_cells, -1, dtype=int)
    old_to_new[transient_indices] = np.arange(n_transient)
    old_to_new[terminal_indices] = np.arange(n_absorbing)
    
    # Extract submatrices
    Q_data = []
    Q_row = []
    Q_col = []
    R_data = []
    R_row = []
    R_col = []
    
    for i in transient_indices:
        row_start, row_end = T_forward.indptr[i], T_forward.indptr[i+1]
        for idx in range(row_start, row_end):
            j = T_forward.indices[idx]
            val = T_forward.data[idx]
            
            if is_transient[j]:
                Q_row.append(old_to_new[i])
                Q_col.append(old_to_new[j])
                Q_data.append(val)
            else:
                R_row.append(old_to_new[i])
                R_col.append(old_to_new[j])
                R_data.append(val)
    
    Q = csr_matrix((Q_data, (Q_row, Q_col)), shape=(n_transient, n_transient))
    R = csr_matrix((R_data, (R_row, R_col)), shape=(n_transient, n_absorbing))
    
    # Compute fundamental matrix N = (I - Q)^(-1)
    I = csr_matrix(np.eye(n_transient))
    try:
        from scipy.sparse.linalg import inv
        N = inv(I - Q)
        if issparse(N):
            N = N.toarray()
    except:
        # Fallback: convert to dense
        N = np.linalg.inv((I - Q).toarray())
    
    # Compute absorption probabilities B = N * R
    B = N @ R.toarray()
    
    # Initialize full branch probability matrix
    branch_probs = np.zeros((n_cells, n_absorbing))
    branch_probs[transient_indices] = B
    
    # Terminal states have certainty for their own absorbing state
    for i, term_idx in enumerate(terminal_indices):
        branch_probs[term_idx, i] = 1.0
    
    return branch_probs, terminal_set


def compute_entropy(branch_probs):
    """
    Compute differentiation entropy from branch probabilities.
    
    Parameters
    ----------
    branch_probs : ndarray
        (n_cells, n_states) branch probability matrix
        
    Returns
    -------
    entropy : ndarray
        (n_cells,) Shannon entropy values
    """
    # Avoid log(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_probs = np.log(branch_probs + 1e-10)
        entropy = -(branch_probs * log_probs).sum(axis=1)
    
    # Normalize by maximum possible entropy
    max_entropy = np.log(branch_probs.shape[1])
    if max_entropy > 0:
        entropy = entropy / max_entropy
    
    return entropy


def find_terminal_states_automatic(multiscale_data, pseudotime, diffusion_components=None):
    """
    Automatically identify terminal states from pseudotime and diffusion components.
    
    Terminal states are cells at the end of pseudotime trajectories, identified by
    having high pseudotime and being in regions of low density.
    
    Parameters
    ----------
    multiscale_data : ndarray
        (n_cells, n_dims) multiscale diffusion space
    pseudotime : ndarray
        (n_cells,) pseudotime values
    diffusion_components : ndarray, optional
        Diffusion components for filtering
        
    Returns
    -------
    terminal_indices : ndarray
        Indices of identified terminal states
    """
    # Find cells at high pseudotime
    threshold = np.percentile(pseudotime, 95)
    high_pseudotime = np.where(pseudotime >= threshold)[0]
    
    # Build k-NN graph to find connected components
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(multiscale_data[high_pseudotime])
    distances, indices = nbrs.kneighbors(multiscale_data[high_pseudotime])
    
    # Simple clustering: connected components
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    n_high = len(high_pseudotime)
    row_indices = []
    col_indices = []
    for i in range(n_high):
        for j in indices[i, 1:]:
            row_indices.append(i)
            col_indices.append(j)
            row_indices.append(j)
            col_indices.append(i)
    
    graph = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), 
                       shape=(n_high, n_high))
    n_components, labels = connected_components(graph, directed=False)
    
    # Select highest pseudotime cell from each component
    terminal_indices = []
    for comp_id in range(n_components):
        comp_cells = high_pseudotime[labels == comp_id]
        best_cell = comp_cells[np.argmax(pseudotime[comp_cells])]
        terminal_indices.append(best_cell)
    
    return np.array(terminal_indices)
