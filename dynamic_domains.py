import ruptures as rpt
import numpy as np
from sklearn.decomposition import PCA
import random
def master_shift(config):
    """Main function to detect shifts in single or multiple features."""
    print("Running shift detection with configuration:")
    print({k: v for k, v in config.items() if k != 'data'})
    
    # Get the shift_col parameter
    shift_col = config['shift_col']
    
    # Case 1: Multifeature detection
    if str(shift_col) == 'multifeature':
        return _detect_shifts(
            data=config['data'],
            features=None,  # Will use all numeric columns
            penalty=config['penalty'],
            mode=config['mode'],
            model=config['model'],
            use_pca=config.get('use_pca', False),
            n_components=config.get('n_components', 1)
        )
    
    # Case 2: Any numeric column detection
    elif str(shift_col) == 'any_numeric':
        numeric_cols = config['data'].select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in the data.")
        # Select the first numeric column and update config
        selected_col = random.choice(numeric_cols)
        print(f"Selected numeric column: {selected_col}")
        return _detect_shifts(
            data=config['data'],
            features=[selected_col],
            penalty=config['penalty'],
            mode=config['mode'],
            model=config['model'],
            use_pca=False,
            n_components=None
        )
    
    # Case 3: Specific column detection
    else:
        # Validate the column exists and is numeric
        if shift_col not in config['data'].columns:
            raise ValueError(f"Column '{shift_col}' not found in data.")
        if not np.issubdtype(config['data'][shift_col].dtype, np.number):
            raise ValueError(f"Column '{shift_col}' must be numeric.")
            
        return _detect_shifts(
            data=config['data'],
            features=[shift_col],
            penalty=config['penalty'],
            mode=config['mode'],
            model=config['model'],
            use_pca=False,
            n_components=None
        )

def _detect_shifts(data, features, penalty, mode='pelt', model='rbf', use_pca=False, n_components=1):
    """
    Core function for shift detection that handles both single and multi-feature cases.
    
    Parameters:
    - data: pandas DataFrame
    - features: list of feature column names (None for all numeric columns)
    - penalty: penalty value for change point detection
    - mode: 'pelt' or 'binseg'
    - model: cost model for ruptures ('rbf', 'l2', etc.)
    - use_pca: if True, apply PCA before detection
    - n_components: number of PCA components to keep
    
    Returns:
    - DataFrame with 'domain' column indicating segments
    """
    # Prepare data
    if features is None:
        features = data.select_dtypes(include=[np.number]).columns.tolist()
        if not features:
            raise ValueError("No numeric columns found for shift detection.")
    
    data_sorted = data.sort_values(features[0] if len(features) == 1 else features[0])
    vals = data_sorted[features].values
    
    # Optional PCA
    if use_pca and len(features) > 1:
        print(f"Applying PCA with {n_components} components...")
        vals = PCA(n_components=n_components).fit_transform(vals)
    
    # Change point detection
    if mode == 'pelt':
        algo = rpt.Pelt(model=model).fit(vals)
        change_points = algo.predict(pen=penalty)
    elif mode == 'binseg':
        algo = rpt.Binseg(model=model).fit(vals)
        change_points = algo.predict(n_bkps=penalty)
    else:
        raise ValueError("Method must be either 'pelt' or 'binseg'.")
    
    print("Detected change points at indices:", change_points)
    
    # Assign domain labels
    labels = np.zeros(len(vals), dtype=int)
    prev_cp = 0
    for i, cp in enumerate(change_points):
        labels[prev_cp:cp] = i
        prev_cp = cp
    
    data_sorted['domain'] = labels
    return data_sorted