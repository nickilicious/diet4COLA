import numpy as np
import pandas as pd
from itertools import product
from openpiv.pyprocess import extended_search_area_piv, get_coordinates
from openpiv import validation, tools
from scipy.interpolate import RegularGridInterpolator
import czifile as czi
from points_df import Points

# Define parameter ranges to test
param_grid = {
    'window_size': [8, 16, 32, 64],
    'search_size': [16, 20, 32, 48],
    'overlap': [4, 8, 12, 16],
    'dt': [1.0],
    'sig2noise_threshold': [1.0, 1.5, 2.0]
}

def run_piv_with_params(points, cellids, window_size, search_size, overlap, dt, sig2noise_threshold):
    """Run PIV tracking with given parameters and return overall MSE"""
    overall_mse = []
    
    for cellid in cellids:
        cell_points = points.points_by_cell(cellid)
        if cell_points.empty:
            continue
            
        image_dir = f'../data/ablation-czi/{cellid}.czi'
        
        try:
            with czi.CziFile(image_dir) as image_czi:
                image_data = image_czi.asarray()
                channel_red = np.squeeze(image_data[:, :, 0, :, :, :, :, :])
                channel_red_norm = channel_red / channel_red.max()
                num_slices, height, width = channel_red_norm.shape
                
                rgb_stack = np.zeros((num_slices, height, width, 2), dtype=np.float32)
                rgb_stack[..., 0] = channel_red_norm
                
                tracked_frames = cell_points['frame'].unique()
                tracked_points = []
                active_points = {}
                
                for frame_idx in tracked_frames:
                    if tracked_frames[0] == frame_idx:
                        pts_frame = cell_points[cell_points['frame'] == frame_idx]
                        for _, row in pts_frame.iterrows():
                            active_points[row['point_id']] = (row['x'], row['y'])
                        continue
                    
                    frame_a = (rgb_stack[frame_idx][..., 0] * 255).astype(np.uint32)
                    frame_b = (rgb_stack[frame_idx+1][..., 0] * 255).astype(np.uint32)
                    
                    vel_x, vel_y, signal_to_noise = extended_search_area_piv(
                        frame_a, frame_b,
                        window_size=window_size,
                        overlap=overlap,
                        dt=dt,
                        search_area_size=search_size,
                        sig2noise_method='peak2peak'
                    )
                    
                    x_grid, y_grid = get_coordinates(
                        image_size=frame_a.shape,
                        search_area_size=search_size,
                        overlap=overlap
                    )
                    
                    vel_x, vel_y, invalid_mask = validation.sig2noise_val(
                        vel_x, vel_y, signal_to_noise, threshold=sig2noise_threshold
                    )
                    
                    x_grid, y_grid, vel_x, vel_y = tools.transform_coordinates(
                        x_grid, y_grid, vel_x, vel_y
                    )
                    
                    vel_x_interp = RegularGridInterpolator(
                        (y_grid[:,0], x_grid[0,:]), vel_x, 
                        bounds_error=False, fill_value=0
                    )
                    vel_y_interp = RegularGridInterpolator(
                        (y_grid[:,0], x_grid[0,:]), vel_y, 
                        bounds_error=False, fill_value=0
                    )
                    
                    new_active_points = {}
                    for pid, (x, y) in active_points.items():
                        vx = vel_x_interp((y, x))
                        vy = vel_y_interp((y, x))
                        vx = np.nan_to_num(vx, nan=0.0)
                        vy = np.nan_to_num(vy, nan=0.0)
                        new_x = x + vx * dt
                        new_y = y + vy * dt
                        tracked_points.append({
                            'frame': frame_idx+1, 
                            'point_id': pid, 
                            'x': new_x, 
                            'y': new_y
                        })
                        new_active_points[pid] = (new_x, new_y)
                    
                    active_points = new_active_points
                
                tracked_df = pd.DataFrame(tracked_points)
                annotated_df = cell_points[['frame', 'point_id', 'x', 'y']]
                merged = pd.merge(
                    tracked_df, annotated_df, 
                    on=['frame', 'point_id'], 
                    suffixes=('_tracked', '_annotated')
                )
                
                if len(merged) > 0:
                    mse = np.mean(
                        (merged['x_tracked'] - merged['x_annotated'])**2 + 
                        (merged['y_tracked'] - merged['y_annotated'])**2
                    )
                    if not np.isnan(mse):
                        overall_mse.append(mse)
        
        except Exception as e:
            print(f"Error processing {cellid}: {e}")
            continue
    
    return np.mean(overall_mse) if overall_mse else np.nan

# Load data
points = Points.from_csv('../out/points.csv')
cellids = points['cell_id'].unique()

# Option 1: Grid Search (comprehensive but slow)
def grid_search(sample_size=None):
    """Perform grid search over all parameter combinations"""
    results = []
    
    # Optionally use subset of cells for faster testing
    test_cellids = cellids[:sample_size] if sample_size else cellids
    
    param_combinations = list(product(
        param_grid['window_size'],
        param_grid['search_size'],
        param_grid['overlap'],
        param_grid['dt'],
        param_grid['sig2noise_threshold']
    ))
    
    # Filter valid combinations first
    valid_combinations = [
        (ws, ss, ov, dt, sn) 
        for ws, ss, ov, dt, sn in param_combinations
        if ov < ws and ss >= ws
    ]
    
    total = len(valid_combinations)
    print(f"Testing {total} valid parameter combinations...\n")
    
    for idx, (ws, ss, ov, dt, sn) in enumerate(valid_combinations, 1):
        print(f"[{idx}/{total}] Testing: ws={ws}, ss={ss}, ov={ov}, dt={dt}, sn={sn}")
        
        mse = run_piv_with_params(points, test_cellids, ws, ss, ov, dt, sn)
        
        results.append({
            'window_size': ws,
            'search_size': ss,
            'overlap': ov,
            'dt': dt,
            'sig2noise_threshold': sn,
            'mse': mse
        })
        
        print(f"    → MSE: {mse:.4f}\n")
    
    results_df = pd.DataFrame(results).sort_values('mse')
    return results_df

# Option 2: Random Search (faster, good coverage)
def random_search(n_iterations=50, sample_size=None):
    """Randomly sample parameter combinations"""
    results = []
    test_cellids = cellids[:sample_size] if sample_size else cellids
    
    print(f"Running random search with {n_iterations} iterations...\n")
    
    iteration = 0
    while iteration < n_iterations:
        ws = np.random.choice(param_grid['window_size'])
        ss = np.random.choice(param_grid['search_size'])
        ov = np.random.choice(param_grid['overlap'])
        dt = np.random.choice(param_grid['dt'])
        sn = np.random.choice(param_grid['sig2noise_threshold'])
        
        # Skip invalid combinations
        if ov >= ws or ss < ws:
            continue
        
        iteration += 1
        print(f"[{iteration}/{n_iterations}] Testing: ws={ws}, ss={ss}, ov={ov}, dt={dt}, sn={sn}")
        
        mse = run_piv_with_params(points, test_cellids, ws, ss, ov, dt, sn)
        
        results.append({
            'window_size': ws,
            'search_size': ss,
            'overlap': ov,
            'dt': dt,
            'sig2noise_threshold': sn,
            'mse': mse
        })
        
        print(f"    → MSE: {mse:.4f}\n")
    
    results_df = pd.DataFrame(results).sort_values('mse')
    return results_df

# Option 3: Test specific parameter sets
def test_specific_params(param_sets, sample_size=None):
    """Test a list of specific parameter combinations"""
    results = []
    test_cellids = cellids[:sample_size] if sample_size else cellids
    
    for idx, params in enumerate(param_sets):
        print(f"Testing {idx+1}/{len(param_sets)}: {params}")
        
        mse = run_piv_with_params(
            points, test_cellids,
            window_size=params['window_size'],
            search_size=params['search_size'],
            overlap=params['overlap'],
            dt=params['dt'],
            sig2noise_threshold=params['sig2noise_threshold']
        )
        
        results.append({**params, 'mse': mse})
        
        print(f"MSE: {mse:.4f}\n")
    
    results_df = pd.DataFrame(results).sort_values('mse')
    return results_df

# Example usage:

# 1. Quick test with random search on sample of cells
# results = random_search(n_iterations=20, sample_size=5)

# 2. Full grid search on sample
results = grid_search(sample_size=None)

# 3. Test specific configurations
specific_params = [
    {'window_size': 16, 'search_size': 20, 'overlap': 8, 'dt': 1.0, 'sig2noise_threshold': 1.0},
    {'window_size': 32, 'search_size': 48, 'overlap': 16, 'dt': 1.0, 'sig2noise_threshold': 1.5},
    {'window_size': 16, 'search_size': 32, 'overlap': 8, 'dt': 0.5, 'sig2noise_threshold': 1.5},
]
# results = test_specific_params(specific_params, sample_size=10)

# Display results
print("\nTop 10 parameter combinations:")
print(results.head(10))
results.to_csv('piv_parameter_optimization_results.csv', index=False)