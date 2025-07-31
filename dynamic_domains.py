import ruptures as rpt
import numpy as np
from sklearn.decomposition import PCA


def master_shift(config):
    if str(config['shift_col']) == 'multifeature':
        # Handle multifeature shift detection (all numeric columns)
        return get_dynamic_shifts_multifeature(
            data = config['data'],
            penalty = config['penalty'],
            mode = config['mode'],
            model = config['model'],
            use_pca = config.get('use_pca', False),
            n_components = config.get('n_components', 1)
        )
    else:
        if config['mode'] == 'og':
            config['penalty'] = None
            config['model'] = None
            config['use_pca'] = False
            config['n_components'] = None
    
        return get_dynamic_shifts(
            data = config['data'],
            shift_col = config['shift_col'],
            penalty = config['penalty'],
            mode = config['mode'],
            model = config['model']
        )


def get_dynamic_shifts(data, shift_col, penalty, mode='pelt', model='rbf'):

    data.sort_values(shift_col, inplace=True)
    vals = data[shift_col].values
    if mode == 'pelt':
        algo = rpt.Pelt(model=model).fit(vals)
        change_points = algo.predict(pen=penalty)
    elif mode == 'binseg':
        algo = rpt.Binseg(model=model).fit(vals)
        change_points = model.predict(n_bkps=penalty)
    else:
        raise ValueError("Method must be either 'pelt' or 'binseg'.")

    print("Detected change points at indices:", change_points)
    labels = np.zeros_like(vals, dtype=int)
    shift_num = 0
    prev_cp = 0
    for cp in change_points:
        labels[prev_cp:cp] = shift_num
        shift_num += 1
        prev_cp = cp
    data['domain'] = labels
    return data

def get_dynamic_shifts_multifeature(data, penalty, mode='pelt', model='rbf', use_pca=False, n_components=1):
    """
    Detect dynamic shifts in multivariate data using change point detection.
    
    Parameters:
    - data: pandas DataFrame
    - features: list of numeric feature column names
    - penalty: penalty value for change point detection
    - method: 'pelt' or 'binseg'
    - model: cost model for ruptures ('rbf', 'l2', etc.)
    - use_pca: if True, apply PCA before detection
    - n_components: number of PCA components to keep (if use_pca=True)
    
    Returns:
    - data: DataFrame with an additional 'domain' column indicating segments
    """
    # get vals of all numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in the data for multifeature shift detection.")  
    vals = data[numeric_cols].values
    print("Dataset Shape:", vals.shape)
    print(data.head())
    # Optional PCA
    if use_pca:
        print(f"Applying PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
        vals = pca.fit_transform(vals)
    
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
    
    data['domain'] = labels
    return data