from new_datasets import (get_chess_data,
    get_parking_birmingham_data,
    get_housing_ames_data,
    get_folktables_data)

DATASET_DICT = {
    "ames" : {
        "data_func": get_housing_ames_data,
        "default_shift_col": "YearBuilt",
    },

    "chess" : {
        "data_func": get_chess_data,
        "default_shift_col": None,
    },

    "parking" : {
        "data_func": get_parking_birmingham_data,
        "default_shift_col": "LastUpdated",
    },

    "folktables" : {
        "data_func": get_folktables_data,
        "default_shift_col": "YearBuilt",   

    }
}

HP_GRID = {
    'mode': ['pelt', 'binseg'],
    'penalty': [3, 5, 7, 9, 11, 13],  
    'model': ['rbf'],
    'shift_col': ['multifeature', 'any_numeric'],
    'use_pca': [True, False],
    'n_components': [0.3, 0.5, 0.7, 0.9],  
}