# Getting Started with AmpSearcher

## Introduction

AmpSearcher is a powerful tool for searching and optimizing antimicrobial peptides (AMPs). This guide will help you set up and start using AmpSearcher for your research.

## Installation

### Using Conda

1. Create a new conda environment:
   ```bash
   conda create -n amp_searcher python=3.9
   ```

2. Activate the environment:
   ```bash
   conda activate amp_searcher
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Using Pip

1. It's recommended to use a virtual environment:
   ```bash
   python -m venv amp_searcher_env
   source amp_searcher_env/bin/activate  # On Windows, use `amp_searcher_env\Scripts\activate`
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t amp_searcher .
   ```

2. Run the Docker container:
   ```bash
   docker run -it amp_searcher
   ```

## Basic Usage

### Training a Model

Use a configuration file to train a model:

```bash
python src/amp_searcher/train.py --config_path configs/main/examples/screening_ffnn_physicochemical.yaml
```

### Making Predictions

Use a trained model to make predictions:

```bash
python src/amp_searcher/predict.py \
    --model_path models/your_model.ckpt \
    --model_config configs/main/examples/screening_ffnn_physicochemical.yaml \
    --featurizer_config configs/main/examples/screening_ffnn_physicochemical.yaml \
    --data_path data/your_sequences.csv \
    --sequence_col sequence
```

### Programmatic Interface

You can also use AmpSearcher in your Python scripts:

```python
from amp_searcher.pipelines.screening_pipeline import ScreeningPipeline
from amp_searcher.data.datasets import load_data_from_csv
import yaml

# Load configuration
with open("configs/main/examples/screening_ffnn_physicochemical.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize screening pipeline
pipeline = ScreeningPipeline(
    model_config=config["model"],
    model_checkpoint_path="models/your_model.ckpt",
    featurizer_config=config["featurizer"]
)

# Make predictions
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "FLPIIAKLLSGLL"]
predictions = pipeline.predict(sequences)

for seq, pred in zip(sequences, predictions):
    print(f"Sequence: {seq}")
    print(f"Predicted probability: {pred.item():.4f}")
```

## Project Structure

AmpSearcher is organized into several key modules:

- **Data**: Handles data loading, processing, and validation
- **Featurizers**: Convert peptide sequences into numerical features
- **Models**: Neural network architectures for different tasks
- **Optimizers**: Algorithms for peptide sequence optimization
- **Pipelines**: High-level interfaces combining multiple components
- **Training**: Training utilities and callbacks

## Configuration Files

AmpSearcher uses YAML configuration files to define experiments. Example configurations are available in `configs/main/examples/`:

- `screening_ffnn_physicochemical.yaml`: Feed-forward neural network for screening
- `screening_transformer_physicochemical.yaml`: Transformer model for screening
- `contrastive_simclr_physicochemical.yaml`: Contrastive learning setup
- `generative_vae_physicochemical.yaml`: Variational autoencoder for generation

## Next Steps

- Learn about [Data Handling](02_data_handling.md)
- Explore [Featurizers](03_featurizers.md)
- Understand [Models](04_models.md)
- Dive into [Optimizers](05_optimizers.md)
- Master [Pipelines](06_pipelines.md)
- Get started with [Training](07_training.md)

## Quick Example

Here's a complete example to get you started:

```python
import yaml
from amp_searcher.data.datasets import load_data_from_csv
from amp_searcher.training.trainer import AmpTrainer

# 1. Prepare your data (CSV with 'sequence' and 'label' columns)
# 2. Create a configuration file (see examples in configs/main/examples/)
# 3. Train a model

config_path = "configs/main/examples/screening_ffnn_physicochemical.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Add data configuration
config["data"] = {
    "path": "data/your_training_data.csv",
    "sequence_col": "sequence",
    "label_col": "label",
    "processor_config": {
        "name": "sequence_processor",
        "params": {
            "featurizer_config": config["featurizer"]
        }
    }
}

# Initialize trainer and start training
trainer = AmpTrainer(config)
trainer.train(config["data"])
```

This will train a model using the specified configuration and save checkpoints for later use in prediction or optimization tasks.
