from importlib import resources

import tabpfn
from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig

# Get the library path for the tabpfn package
libpath = resources.files(tabpfn)


# Helper function to load each pre-trained model with the corresponding configuration.
def get_model(task_type, model_path, model_type):
    model_path_config = TabPFNModelPathsConfig(
        paths=[f"{libpath}/model_cache/{model_path}.cpkt"], task_type=task_type
    )

    model = get_best_tabpfn(
        task_type=task_type,
        model_type=model_type,
        paths_config=model_path_config,
        debug=False,
        device="auto"
    )
    model.show_progress = False
    model.seed = 1

    return model


task_type = "dist_shift_multiclass"

models_to_load = [
    ("tabpfn_dist_model_1", "best_dist"),
    ("tabpfn_dist_model_2", "best_dist"),
    ("tabpfn_dist_model_3", "best_dist"),
    ("tabpfn_dist_ablation_no_t2v_model_1", "best_dist"),
    ("tabpfn_base_model_1", "best_base"),
    ("tabpfn_base_model_2", "best_base"),
    ("tabpfn_base_model_3", "best_base"),
]


models = {
    model_name: get_model(task_type, model_name, model_type)
    for model_name, model_type in models_to_load
}