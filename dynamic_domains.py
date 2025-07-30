import ruptures as rpt
import numpy as np
from sklearn.decomposition import PCA

def get_dynamic_shifts(data, shift_col, penalty, method='pelt', model='rbf'):

    data.sort_values(shift_col, inplace=True)
    vals = data[shift_col].values
    if method == 'pelt':
        algo = rpt.Pelt(model=model).fit(vals)
        change_points = algo.predict(pen=penalty)
    elif method == 'binseg':
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

def get_dynamic_shifts_multifeature(data, features, penalty, method='pelt', model='rbf', use_pca=False, n_components=1):
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
    
    vals = data[features].values  # (n_samples, n_features)
    
    # Optional PCA
    if use_pca:
        print(f"Applying PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
        vals = pca.fit_transform(vals)
    
    # Change point detection
    if method == 'pelt':
        algo = rpt.Pelt(model=model).fit(vals)
        change_points = algo.predict(pen=penalty)
    elif method == 'binseg':
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