import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from config import DATASET_DICT, HP_GRID
from setup import models
import argparse
import json
import random
import os
import torch
from itertools import product

def evaluate_model(dataset, clf):
    X_train, X_test, y_train, y_test, dist_shift_domain_train, dist_shift_domain_test = train_test_split(
        dataset.x, dataset.y, dataset.dist_shift_domain, test_size=0.50, shuffle=False, random_state=42
    )

    clf.fit(X_train, y_train, additional_x={"dist_shift_domain": dist_shift_domain_train})
    preds = clf.predict_proba(X_test, additional_x={"dist_shift_domain": dist_shift_domain_test})

    y_eval = np.argmax(preds, axis=1)
    if torch.unique(y_test).numel() == 2:
        # Binary classification
        roc_auc = roc_auc_score(y_test, preds[:, 1])
    else:
        # Multiclass classification
        roc_auc = roc_auc_score(y_test, preds, multi_class='ovo')
    
    accuracy = accuracy_score(y_test, y_eval)
    
    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy
    }

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate Drift-Resilient TabPFN model on a dataset.')
    parser.add_argument('--automl', action='store_true', help='Run in AutoML mode with hyperparameter grid search.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--shift_col', type=str, default='default', 
                       choices=('any_numeric', 'multifeature'), 
                       help='Column name for shift detection in the dataset.')
    parser.add_argument('--penalty', type=int, default=None, help='Penalty for drift detection.')
    parser.add_argument('--mode', type=str, choices=('og', 'pelt', 'binseg'), default='og', 
                       help='Drift detection mode: og, pelt, or binseg.')
    parser.add_argument('--model', type=str, choices=('rbf', 'linear'), default=None, 
                       help='Model type for drift detection: rbf or linear.')
    parser.add_argument('--use_pca', type=bool, default=False, 
                       help='Whether to use PCA for dimensionality reduction before drift detection.')
    parser.add_argument('--n_components', type=float, default=None, 
                       help='Percent of PCA components to keep if use_pca is True.') 
    parser.add_argument('--max_configs', type=int, default=20,
                       help='Maximum number of configurations to try in AutoML mode.')
    return parser.parse_args()

def generate_configurations(HP_GRID, max_configs=None):
    """Generate all possible configurations from the HP_GRID or a random subset."""
    # Generate all possible combinations
    keys, values = zip(*HP_GRID.items())
    all_configs = [dict(zip(keys, v)) for v in product(*values)]
    
    # If max_configs is specified and smaller than total possible configs
    if max_configs is not None and len(all_configs) > max_configs:
        return random.sample(all_configs, max_configs)
    return all_configs

def automl_search(dataset, config_template, HP_GRID, max_configs=None):
    """Perform AutoML search for best hyperparameters."""
    best_score = -np.inf
    best_config = None
    best_results = None
    
    configs = generate_configurations(HP_GRID, max_configs)
    print(f"Evaluating {len(configs)} configurations...")
    
    for config in configs:
        # Update the base configuration with AutoML params
        current_config = config_template.copy()
        current_config.update(config)
        
        # Get the dataset with current configuration
        configured_dataset = dataset(current_config)
        
        # Evaluate model
        if isinstance(configured_dataset, list):
            # Handle list of datasets by averaging scores
            scores = []
            for ds in configured_dataset:
                res = evaluate_model(ds, models["tabpfn_dist_model_1"])
                scores.append(res["roc_auc"])  # Using ROC AUC as primary metric
            current_score = np.mean(scores)
        else:
            res = evaluate_model(configured_dataset, models["tabpfn_dist_model_1"])
            current_score = res["roc_auc"]
        
        # Update best configuration if current is better
        if current_score > best_score:
            best_score = current_score
            best_config = current_config
            best_results = res 
            print(f"New best score: {best_score:.4f}.")
    
    return best_config, best_results

def main():
    args = get_args()
    
    if args.dataset not in DATASET_DICT:
        raise ValueError(f"Dataset {args.dataset} is not supported. Choose from {list(DATASET_DICT.keys())}.")  
    
    # Base configuration
    base_config = {
        "shift_col": args.shift_col,
        "penalty": args.penalty,
        "mode": args.mode, 
        "model": args.model,
        "use_pca": args.use_pca,
        "n_components": args.n_components,  
    }
    
    if args.automl:
        print("Running in AutoML mode...")        
        # Get dataset function
        dataset_func = DATASET_DICT[args.dataset]["data_func"]
        
        # Perform AutoML search
        best_config, best_results = automl_search(
            dataset_func, 
            base_config, 
            HP_GRID, 
            max_configs=args.max_configs
        )
        
        results = best_results
        results.update(best_config)
    else:
        print("Running with manual configuration...")
        dataset = DATASET_DICT[args.dataset]["data_func"](config=base_config)
        
        if isinstance(dataset, list):
            results = []
            for ds in dataset:
                res = evaluate_model(ds, models["tabpfn_dist_model_1"])
                res.update(base_config)
                results.append(res)
        else:
            results = evaluate_model(dataset, models["tabpfn_dist_model_1"])
            results.update(base_config)
    
    # Clean results for saving
    if isinstance(results, list):
        clean_results = []
        for res in results:
            clean_res = res.copy()
            clean_res.pop('data', None)
            clean_results.append(clean_res)
    else:
        clean_results = results.copy()
        clean_results.pop('data', None)
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"results_{args.dataset}.json")
    
    # Load existing results if file exists
    all_results = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing = json.load(f)
                if isinstance(existing, dict):
                    all_results = [existing]
                else:
                    all_results = existing
            except json.JSONDecodeError:
                pass
    
    # Append new results
    if isinstance(clean_results, list):
        all_results.extend(clean_results)
    else:
        all_results.append(clean_results)
    
    # Save back to file
    with open(file_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Results saved to {file_path}")

if __name__ == "__main__":
    main()