# Project README

This repository implements a modular, configurable pipeline for protein sequence–based machine learning experiments. It uses Hydra/OmegaConf for configuration, separates concerns into distinct modules, and supports multiple ML models, hyperparameter optimization, and experiment tracking.

## Directory Structure

```text
project_root/
├── README.md                             # Project overview and instructions
├── pyproject.toml / setup.py / requirements.txt  # Project metadata and dependencies
│
├── configs/                              # Hydra/OmegaConf configuration files
│   ├── config.yaml                       # Main config, defines defaults
│   ├── data/                             # Data settings
│   │   ├── default.yaml                  # raw path, split strategy, seed
│   │   └── kfold.yaml                    # K‑Fold split override
│   ├── model/                            # Model and hyperparameter settings
│   │   ├── default.yaml                  # default model type, search strategy
│   │   ├── svm.yaml                      # SVM-specific overrides
│   │   ├── rf.yaml                       # Random Forest-specific overrides
│   │   └── nn.yaml                       # Neural Network-specific overrides
│   ├── experiment/                       # Experiment settings
│   │   ├── default.yaml                  # run count, run_id prefix, log backend
│   │   └── mlflow.yaml                   # MLflow-specific settings
│   └── logger/                           # Logger backends
│       ├── default.yaml                  # tensorboard/console defaults
│       └── wandb.yaml                    # Weights & Biases settings
│
├── data/                                 # Raw data directory
│   └── raw/                              # Single source: raw protein sequences and labels
│
├── src/                                  # Core pipeline code
│   ├── __main__.py                       # CLI entrypoint (e.g. `python -m src --config-name=svm`)
│   ├── settings.py                       # Hydra initialization and global constants
│   │
│   ├── data/                      # Data ingestion and splitting
│   │   ├── __init__.py
│   │   ├── base.py                       # DataModule interface
│   │   └── protein_data.py               # Implementation for protein data
│   │
│   ├── feature/                   # Descriptor extraction and preprocessing
│   │   ├── __init__.py
│   │   ├── base.py                       # FeatureModule interface
│   │   └── rdkit_descriptor.py           # RDKit descriptor implementation
│   │
│   ├── model/                     # Model, optimizer, and scheduler definitions
│   │   ├── __init__.py
│   │   ├── base.py                       # ModelModule interface
│   │   ├── svm.py                        # SVM module
│   │   ├── rf.py                         # Random Forest module
│   │   └── nn.py                         # Neural Network module
│   │
│   ├── runner/                           # Experiment orchestration
│   │   ├── __init__.py
│   │   └── experiment.py                 # ExperimentRunner that orchestrates modules
│   │
│   ├── logger/                    # Logging and visualization backends
│   │   ├── __init__.py
│   │   ├── base.py                       # Logger interface
│   │   ├── tensorboard.py                # TensorBoard implementation
│   │   └── wandb.py                      # Weights & Biases implementation
│   │
│   └── utils/                            # Utility functions
│       ├── __init__.py
│       ├── metrics.py                    # Metrics computation (accuracy, ROC AUC, MSE)
│       └── io.py     
│
├── scripts/                              # Utility scripts for batch experiments and reporting
│   ├── run_all.sh                        # Run multiple configs sequentially
│   └── report_generator.py               # Generate summary reports
│
├── experiments/                          # Local MLflow or W&B storage (if enabled locally)
│
├── tests/                                # Unit tests for each module
│   ├── test_data_module.py
│   ├── test_feature_module.py
│   └── test_model_module.py
│
├── outputs/                              # Artifacts for each run stored by run_id
│   ├── 20250522-123456/                  # Example run_id named by timestamp
│   └── 20250522-134501/
│
└── logs/                                 # Centralized pipeline logs
    └── project.log
````

## Installation

1. **Clone repository**

   ```bash
   git clone git@your-repo-url.git
   cd project_root
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   # Or using Poetry / pyproject.toml:
   pip install .
   ```

## Usage

### Running a Single Experiment

```bash
python -m src --config-name svm
```

### Overriding Configuration via CLI

```bash
python -m src --config-name rf experiment.run_id=myrun data.seed=42
```

### Batch Experiments

```bash
bash scripts/run_all.sh
```

## Configuration

All configurations live under `configs/`. Hydra will merge defaults and overrides:

* **data**: raw data path, split strategy, random seed
* **model**: model type and hyperparameter search space
* **experiment**: run counts, run\_id prefix, logging backend
* **logger**: TensorBoard or W\&B settings

## Outputs

Each experiment run creates `outputs/<run_id>/` containing:

* `config_used.yaml`: full merged configuration
* `model_checkpoint/`: saved model artifacts
* `metrics.json`: performance metrics on train/val/test
* `figures/`: ROC, PR curves, etc.

## Logging

Global logs written to `logs/project.log`. If MLflow or W\&B is enabled, experiment metadata and metrics are also pushed to the respective backend under `experiments/`.

## Testing

Run unit tests:

```bash
pytest tests/
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Implement features and write tests
4. Submit a pull request

