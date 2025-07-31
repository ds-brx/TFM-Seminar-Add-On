from new_datasets import (get_chess_data,
    get_parking_birmingham_data,
    get_housing_ames_data,
    get_folktables_data)

DATASET_DICT = {
    "ames" : {
        "data_func": get_housing_ames_data,
        "default_shift_col": "LastUpdated",
    },

    "chess" : {
        "data_func": get_chess_data,
        "default_shift_col": None,
    },

    "parking" : {
        "data_func": get_parking_birmingham_data,
        "default_shift_col": "YearBuilt",
    },

    "folktables" : {
        "data_func": get_folktables_data,
        "default_shift_col": "YearBuilt",   

    }
}

HP_GRID = {
    'mode': ['og', 'pelt', 'binseg'],
    'penalty': [3, 5, 7, None],  
    'model': ['rbf', 'linear', None],
    'shift_col': ['default', 'multifeature'],
    'use_pca': [True, False],
    'n_components': [0.3, 0.5, 0.7, 0.9, None],  
}