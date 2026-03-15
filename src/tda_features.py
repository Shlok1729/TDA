import numpy as np
import pandas as pd
from ripser import ripser

def get_time_delay_embedding(data, delay=2, dimension=3):
    """
    Constructs a time-delay embedding from 1D sequence or multivariate sequences.
    If data is 2D (n_steps, n_features), the embedding dimension increases by n_features.
    """
    # If data is 1D (n_steps,), convert to 2D (n_steps, 1)
    if data.ndim == 1:
        data = data[:, np.newaxis]
        
    n_steps, n_features = data.shape
    
    if n_steps - (dimension - 1) * delay <= 0:
        return data
    
    embedded = []
    for i in range(n_steps - (dimension - 1) * delay):
        # For each time step i, we concatenate the delayed samples for ALL features
        point = []
        for j in range(dimension):
            point.extend(data[i + j * delay, :])
        embedded.append(point)
    return np.array(embedded)

def _calculate_persistence_entropy(diagram):
    # Ripser outputs [birth, death] pairs. 
    # Death can be infinity for H0. Filter out infinity for entropy calculation.
    clean_diagram = diagram[diagram[:, 1] != np.inf]
    if len(clean_diagram) == 0:
        return 0.0
    
    lifespans = clean_diagram[:, 1] - clean_diagram[:, 0]
    total_life = np.sum(lifespans)
    if total_life == 0:
        return 0.0
        
    probabilities = lifespans / total_life
    # Persistence entropy formula: -sum(p * log(p))
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    return entropy

def _calculate_betti_curve_approx(diagram, n_bins=10, max_death=None):
    clean_diagram = diagram[diagram[:, 1] != np.inf]
    if len(clean_diagram) == 0:
        return np.zeros(n_bins)
        
    if max_death is None:
        max_death = np.max(clean_diagram[:, 1])
        if max_death == 0:
            max_death = 1.0
            
    bins = np.linspace(0, max_death, n_bins + 1)
    betti_curve = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_start = bins[i]
        bin_end = bins[i+1]
        # Count how many features are alive during this bin
        # A feature is alive if birth <= bin_start and death > bin_end (approximately)
        alive_count = np.sum((clean_diagram[:, 0] <= bin_end) & (clean_diagram[:, 1] >= bin_start))
        betti_curve[i] = alive_count
        
    return betti_curve

def extract_tda_features(X_windows, time_delay=2, dimension=3):
    # X_windows can be (n_samples, window_size) or (n_samples, window_size, n_features)
    n_samples = X_windows.shape[0]
    tda_features = []
    
    # Use a fixed max death for normalized point clouds
    fixed_max_death = 2.0 
    
    for i in range(n_samples):
        window_data = X_windows[i]
        
        # 1. Calculate Persistence Diagrams
        point_cloud = get_time_delay_embedding(window_data, delay=time_delay, dimension=dimension)
        
        # NORMALIZE point cloud to [0, 1] to ensure Betti curves are comparable across different scales
        pc_min = point_cloud.min(axis=0)
        pc_max = point_cloud.max(axis=0)
        # Avoid division by zero
        range_val = pc_max - pc_min
        range_val[range_val == 0] = 1.0
        point_cloud_norm = (point_cloud - pc_min) / range_val
        
        diagrams = ripser(point_cloud_norm, maxdim=1)['dgms']
        
        # H0 diagram
        h0_diagram = diagrams[0]
        h1_diagram = diagrams[1] if len(diagrams) > 1 else np.empty((0,2))
        
        # 2. Extract Scalar Features
        entropy_h0 = _calculate_persistence_entropy(h0_diagram)
        betti_h0 = _calculate_betti_curve_approx(h0_diagram, n_bins=5, max_death=fixed_max_death)
        
        entropy_h1 = _calculate_persistence_entropy(h1_diagram) if len(h1_diagram) > 0 else 0.0
        
        # 3. Basic Stats (on the first feature for consistency)
        stats_data = window_data[:, 0] if window_data.ndim > 1 else window_data
        mean_val = np.mean(stats_data)
        std_val = np.std(stats_data)
        min_val = np.min(stats_data)
        max_val = np.max(stats_data)
        
        features = np.hstack(([mean_val, std_val, min_val, max_val, entropy_h0, entropy_h1], betti_h0))
        tda_features.append(features)
        
    return np.array(tda_features)

def get_diagrams_for_visualization(X_window, time_delay=2, dimension=3):
    # A helper for the Streamlit app to plot a single window's persistence diagram
    point_cloud = get_time_delay_embedding(X_window, delay=time_delay, dimension=dimension)
    
    # Normalize for consistency with extract_tda_features
    pc_min = point_cloud.min(axis=0)
    pc_max = point_cloud.max(axis=0)
    range_val = pc_max - pc_min
    range_val[range_val == 0] = 1.0
    point_cloud_norm = (point_cloud - pc_min) / range_val
    
    diagrams = ripser(point_cloud_norm, maxdim=1)['dgms']
    return diagrams
