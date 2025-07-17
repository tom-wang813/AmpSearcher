# AmpSearcher

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![CI/CD Status](https://github.com/your-username/AmpSearcher/actions/workflows/ci.yaml/badge.svg)](https://github.com/your-username/AmpSearcher/actions/workflows/ci.yaml)

## Introduction

AmpSearcher is a modular, configuration-driven framework designed for the discovery and optimization of Antimicrobial Peptides (AMPs). It provides a flexible and extensible platform for researchers to build, train, and deploy machine learning models for AMP prediction and design.

### Key Features

*   **Modular Design:** Clear separation of concerns for data processing, feature extraction, model architectures, training loops, and optimization algorithms.
*   **Configuration-Driven:** Easily define and manage experiments using YAML configuration files, ensuring high reproducibility.
*   **Extensibility:** Simple mechanisms to integrate new featurizers, model architectures, optimization strategies, and data processing pipelines.
*   **PyTorch Lightning:** Leverages PyTorch Lightning for streamlined and efficient model training.
*   **MLflow Integration:** Supports experiment tracking, model versioning, and reproducibility of experimental results.
*   **DVC Integration:** Enables data version control, ensuring traceability and reproducibility of data pipelines.

## Quick Start

### Download and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/AmpSearcher.git
   cd AmpSearcher
   ```

2. **Set up the Environment:**
   ```bash
   conda create -n amp python=3.11 -y
   conda activate amp
   pip install -e .
   ```

### Running a Demo

To quickly see AmpSearcher in action, run the following demo:

```bash
python scripts/demo.py
```

This script will:
1. Load a pre-trained model
2. Process a sample sequence
3. Make a prediction
4. Display the results

## Detailed Usage Guide

AmpSearcher consists of several submodules, each handling a specific aspect of the AMP discovery and optimization process. Here's a brief overview of each submodule and how to use them:

### 1. Data Handling

The `data` submodule manages dataset loading, preprocessing, and validation.

```python
from amp_searcher.data import AMPDataset

dataset = AMPDataset("path/to/your/data.csv")
```

### 2. Featurizers

The `featurizers` submodule converts raw sequences into numerical features.

```python
from amp_searcher.featurizers import PhysicochemicalFeaturizer

featurizer = PhysicochemicalFeaturizer()
features = featurizer.featurize("ACDEFGHIKLMNPQRSTVWY")
```

### 3. Models

The `models` submodule contains various model architectures for AMP prediction.

```python
from amp_searcher.models import ScreeningFFNN

model = ScreeningFFNN(input_dim=128, output_dim=1)
```

### 4. Training

The `training` submodule handles the model training process.

```python
from amp_searcher.training import AMPTrainer

trainer = AMPTrainer(model, dataset, config)
trainer.train()
```

### 5. Optimizers

The `optimizers` submodule provides algorithms for sequence optimization.

```python
from amp_searcher.optimizers import GeneticAlgorithm

ga = GeneticAlgorithm(model, config)
optimized_sequences = ga.optimize()
```

### 6. Pipelines

The `pipelines` submodule combines multiple steps into cohesive workflows.

```python
from amp_searcher.pipelines import ScreeningPipeline

pipeline = ScreeningPipeline(config)
results = pipeline.run(input_sequences)
```

### Combined Usage Example

Here's an example that combines multiple submodules:

```python
from amp_searcher.data import AMPDataset
from amp_searcher.featurizers import PhysicochemicalFeaturizer
from amp_searcher.models import ScreeningFFNN
from amp_searcher.training import AMPTrainer
from amp_searcher.optimizers import GeneticAlgorithm

# Load and preprocess data
dataset = AMPDataset("data/amp_sequences.csv")

# Create featurizer and model
featurizer = PhysicochemicalFeaturizer()
model = ScreeningFFNN(input_dim=featurizer.output_dim, output_dim=1)

# Train the model
trainer = AMPTrainer(model, dataset, config)
trained_model = trainer.train()

# Optimize sequences
ga = GeneticAlgorithm(trained_model, config)
optimized_sequences = ga.optimize()

print(f"Optimized sequences: {optimized_sequences}")
```

For more detailed examples and tutorials, please refer to the Jupyter notebooks in the `examples/` directory.

[The rest of the README content remains the same...]
