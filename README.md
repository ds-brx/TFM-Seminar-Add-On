
# Add-On for Seminar: Tabular Foundation Models  
**Paper Reference**: [Drift-Resilient TabPFN](https://github.com/automl/Drift-Resilient_TabPFN)

## Motivation

The original **Drift-Resilient TabPFN** model requires manual selection of a domain indicator, typically based on a single, user-chosen shift column. This introduces subjectivity and reproducibility issues, as the model’s performance is highly sensitive to this choice.

Our add-on **automates domain indication** by detecting distribution shifts using classical change point detection algorithms. This not only improves the model’s reliability and performance but also eliminates the need for manual intervention.

---

## Setup Instructions

```bash
git clone https://github.com/ds-brx/TFM-Seminar-Add-On.git
cd TFM-Seminar-Add-On

# Create a new conda environment
conda create -n dft python=3.10
conda activate dft

# Remove conflicting packages
pip uninstall -y torch torchaudio torchvision torchtext numpy tsfresh transformers sentence-transformers peft

# Install required packages
pip install git+https://github.com/automl/Drift-Resilient_TabPFN.git
pip install ruptures
```

---

## Running the Add-On

We use the [`ruptures`](https://github.com/deepcharles/ruptures) library to detect change points in tabular data. You can run the change point detection in different modes:

- Run original settings from paper: `--mode og`
- Automl Mode: `--automl --a_configs 10`
- Shift detection on:
  - All numeric features: `all_numeric`
  - A random numeric feature: `any_numeric`
  -   A specific column name: <str>
- Optional PCA for dimensionality reduction: `use_pca=True`

Currently, we evaluate our method on real-world test datasets from the Drift-Resilient TabPFN paper. The dataset-specific domain indicator allocation functions, located in `new_datasets.py`, have been updated to use our dynamic domain allocation system in place of the original manual approach.

### Example Commands:
```bash
# Manual configuration with PCA and specific settings
python -m ablations --dataset chess --mode pelt --penalty 3 --model rbf --shift_col all_numeric --use_pca True --n_components 0.75

# Original settings
python -m ablations --dataset ames --mode og

# AutoML with 10 configurations
python -m ablations --dataset parking --automl --max_configs 10
```

---

## Arguments

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--dataset`      | Dataset name. Options: `chess`, `parking`, `ames` , `folktables`            |
| `--mode`         | Change point detection method. Options: `pelt`, `binseg`, `og`              |
| `--penalty`      | Penalty parameter (int) for controlling sensitivity of change detection     |
| `--model`        | Cost model for ruptures. Options: `rbf`, `linear`                           |
| `--shift_col`    | Feature(s) for shift detection: `all_numeric`, `any_numeric`,`default`, or column name |
| `--use_pca`      | Whether to apply PCA on selected features. Options: `True`, `False`         |
| `--n_components` | PCA: Fraction of components to keep (between 0.0 and 1.0)                   |

## Contribution

We welcome contributions! Please open an issue or submit a pull request if you have improvements, bug fixes, or new features to propose.
