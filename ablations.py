import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from config import DATASET_DICT, HP_GRID
from setup import models
import argparse
import json
import random
import os

def evaluate_model(model, dataset):
  clf = models["tabpfn_dist_model_1"]

  X_train, X_test, y_train, y_test, dist_shift_domain_train, dist_shift_domain_test = train_test_split(
      dataset.x, dataset.y, dataset.dist_shift_domain, test_size=0.50, shuffle=False, random_state=42)

  clf.fit(X_train, y_train, additional_x={"dist_shift_domain": dist_shift_domain_train})

  preds = clf.predict_proba(X_test, additional_x={"dist_shift_domain": dist_shift_domain_test})

  y_eval = np.argmax(preds, axis=1)
  return {
      "roc_auc": roc_auc_score(y_test, preds, multi_class='ovo'),
      "accuracy": accuracy_score(y_test, y_eval)
  }


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate Drift-Resilient TabPFN model on a dataset.')
    parser.add_argument('--automl', action='store_true', help='Run in AutoML mode with hyperparameter grid search.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--shift_col', type=str, default='default', choices=('default', 'multifeature'), help='Column name for shift detection in the dataset.')
    parser.add_argument('--penalty', type=int, default=None, help='Penalty for drift detection.')
    parser.add_argument('--mode', type=str, choices=('og', 'pelt', 'binseg') , default='og', help='Drift detection mode: og, pelt, or binseg.')
    parser.add_argument('--model', type=str, choices=('rbf', 'linear'), default=None, help='Model type for drift detection: rbf or linear.')
    parser.add_argument('--use_pca', type=bool, default=False, help='Whether to use PCA for dimensionality reduction before drift detection.')
    parser.add_argument('--n_components', type=float, default=None, help='Percent of PCA components to keep if use_pca is True.') 
    return parser.parse_args()

def main():
    args = get_args()
    if args.automl:
        # sample random hyperparameters from the grid for now
        # TODO: implement a proper AutoML strategy
        random.seed(42)  # For reproducibility
        config = {key: random.choice(values) for key, values in HP_GRID.items()}
    else:
        config = {
            "shift_col": args.shift_col,
            "penalty": args.penalty,
            "mode": args.mode, 
            "model": args.model,
            "use_pca": args.use_pca,
            "n_components": args.n_components,  
        }
    # Log the configuration
    print("Configuration:", config)

    if config['shift_col'] == 'default':
        config['shift_col'] = DATASET_DICT[args.dataset]["default_shift_col"]

    if args.dataset not in DATASET_DICT:
        raise ValueError(f"Dataset {args.dataset} is not supported. Choose from {list(DATASET_DICT.keys())}.")  
    
    dataset = DATASET_DICT[args.dataset]["data_func"](config = config)
    
    print(f"Evaluating model on {args.dataset} dataset...")
    results = evaluate_model(models["tabpfn_dist_model_1"], dataset)
    results.update(config)
    results.pop('data', None)  # Remove the data object from results to avoid serialization issues

    # Make Results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  
    # Save results to a JSON file   
    file_path = os.path.join(results_dir,f"results_{args.dataset}.json")

    # Load existing results if file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                all_results = json.load(f)
                if isinstance(all_results, dict):
                    # Convert old format (single dict) to list
                    all_results = [all_results]
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []

    # Append new result
    all_results.append(results)

    # Save back as a list
    with open(file_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Results appended to {file_path}")
if __name__ == "__main__":
    main()
    
    
