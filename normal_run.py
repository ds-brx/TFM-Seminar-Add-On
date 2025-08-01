from tabpfn.datasets.dist_shift_datasets import get_housing_ames_data
from ablations import evaluate_model
from setup import models

old_data = get_housing_ames_data()
evaluate_model(models["tabpfn_dist_model_1"], old_data)
