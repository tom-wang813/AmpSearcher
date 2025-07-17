# Training in AmpSearcher

## Overview

AmpSearcher provides a comprehensive training system built on PyTorch Lightning for training various types of models for antimicrobial peptide research. The training module handles data loading, model training, validation, and checkpointing with support for different model types and training strategies.

## Training System Architecture

### AmpTrainer

The `AmpTrainer` class is the main interface for training models:

```python
from amp_searcher.training.trainer import AmpTrainer
import yaml

# Load configuration
with open("configs/main/examples/screening_ffnn_physicochemical.yaml", "r") as f:
    config = yaml.safe_load(f)

# Add data configuration
config["data"] = {
    "path": "data/training_data.csv",
    "sequence_col": "sequence",
    "label_col": "activity",
    "processor_config": {
        "name": "sequence_processor",
        "params": {
            "featurizer_config": config["featurizer"]
        }
    }
}

# Initialize trainer
trainer = AmpTrainer(config)

# Start training
trainer.train(config["data"])
```

## Training Configuration

### Complete Training Configuration

```yaml
# Data configuration
data:
  path: "data/training_data.csv"
  sequence_col: "sequence"
  label_col: "activity"
  validation_split: 0.2
  test_split: 0.1
  processor_config:
    name: "sequence_processor"
    params:
      featurizer_config:
        name: "PhysicochemicalFeaturizer"
        custom_features: ["length", "molecular_weight", "charge_at_ph_7"]

# Featurizer configuration
featurizer:
  name: PhysicochemicalFeaturizer
  custom_features: ["length", "molecular_weight", "charge_at_ph_7"]

# Model configuration
model:
  type: screening
  architecture:
    name: FFNN
    params:
      input_dim: 3
      output_dim: 1
      hidden_dims: [64, 32]
      dropout_rate: 0.2
      activation: relu
  lightning_module_params:
    task_type: classification
    optimizer_params:
      lr: 0.001
      weight_decay: 0.0001
    scheduler_params:
      name: StepLR
      step_size: 10
      gamma: 0.1

# Training configuration
trainer:
  max_epochs: 100
  batch_size: 32
  num_workers: 4
  accelerator: auto
  devices: auto
  logger_name: amp_screening_experiment
  log_dir: ./lightning_logs
  monitor_gradients: true
  early_stopping:
    monitor: val_loss
    patience: 10
    mode: min
  model_checkpoint:
    monitor: val_loss
    save_top_k: 3
    mode: min
```

## Training Different Model Types

### 1. Screening Model Training

```python
# Configuration for screening model
screening_config = {
    "data": {
        "path": "data/amp_dataset.csv",
        "sequence_col": "sequence",
        "label_col": "is_amp",
        "processor_config": {
            "name": "sequence_processor",
            "params": {
                "featurizer_config": {
                    "name": "PhysicochemicalFeaturizer"
                }
            }
        }
    },
    "featurizer": {
        "name": "PhysicochemicalFeaturizer"
    },
    "model": {
        "type": "screening",
        "architecture": {
            "name": "FFNN",
            "params": {
                "input_dim": 10,
                "output_dim": 1,
                "hidden_dims": [64, 32]
            }
        },
        "lightning_module_params": {
            "task_type": "classification",
            "optimizer_params": {"lr": 0.001}
        }
    },
    "trainer": {
        "max_epochs": 50,
        "batch_size": 32
    }
}

trainer = AmpTrainer(screening_config)
trainer.train(screening_config["data"])
```

### 2. Contrastive Learning Training

```python
# Configuration for contrastive learning
contrastive_config = {
    "data": {
        "path": "data/amp_sequences.csv",
        "sequence_col": "sequence",
        "processor_config": {
            "name": "sequence_processor",
            "params": {
                "featurizer_config": {
                    "name": "PhysicochemicalFeaturizer"
                }
            }
        }
    },
    "featurizer": {
        "name": "PhysicochemicalFeaturizer"
    },
    "model": {
        "type": "contrastive",
        "architecture": {
            "name": "SimCLRBackbone",
            "params": {
                "input_dim": 10,
                "hidden_dims": [128, 64],
                "projection_dim": 32
            }
        },
        "lightning_module_params": {
            "temperature": 0.1,
            "optimizer_params": {"lr": 0.001}
        }
    },
    "trainer": {
        "max_epochs": 100,
        "batch_size": 64
    }
}

trainer = AmpTrainer(contrastive_config)
trainer.train(contrastive_config["data"])
```

### 3. Generative Model Training (VAE)

```python
# Configuration for VAE training
vae_config = {
    "data": {
        "path": "data/amp_sequences.csv",
        "sequence_col": "sequence",
        "processor_config": {
            "name": "sequence_processor",
            "params": {
                "featurizer_config": {
                    "name": "PhysicochemicalFeaturizer"
                }
            }
        }
    },
    "featurizer": {
        "name": "PhysicochemicalFeaturizer"
    },
    "model": {
        "type": "generative",
        "architecture": {
            "name": "VAE",
            "params": {
                "input_dim": 10,
                "latent_dim": 16,
                "hidden_dims": [64, 32],
                "beta": 1.0
            }
        },
        "lightning_module_params": {
            "optimizer_params": {"lr": 0.001}
        }
    },
    "trainer": {
        "max_epochs": 200,
        "batch_size": 32
    }
}

trainer = AmpTrainer(vae_config)
trainer.train(vae_config["data"])
```

## Advanced Training Features

### Custom Training Callbacks

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class CustomMetricsCallback(Callback):
    """Custom callback for logging additional metrics."""
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Calculate custom metrics
        if hasattr(pl_module, 'validation_predictions'):
            predictions = pl_module.validation_predictions
            labels = pl_module.validation_labels
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(labels, predictions > 0.5)
            recall = recall_score(labels, predictions > 0.5)
            
            # Log metrics
            pl_module.log("val_precision", precision)
            pl_module.log("val_recall", recall)

class LearningRateMonitor(Callback):
    """Monitor and log learning rate changes."""
    
    def on_train_epoch_start(self, trainer, pl_module):
        # Get current learning rate
        optimizer = trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        pl_module.log("learning_rate", current_lr)

# Add callbacks to trainer configuration
config["trainer"]["callbacks"] = [
    CustomMetricsCallback(),
    LearningRateMonitor()
]
```

### Multi-GPU Training

```yaml
trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 2  # Use 2 GPUs
  strategy: ddp  # Distributed Data Parallel
  sync_batchnorm: true
  batch_size: 64  # Per GPU batch size
```

### Mixed Precision Training

```yaml
trainer:
  max_epochs: 100
  precision: 16  # Use 16-bit precision
  accelerator: gpu
  devices: 1
```

## Training Monitoring and Logging

### TensorBoard Integration

```python
from pytorch_lightning.loggers import TensorBoardLogger

# Configure TensorBoard logger
logger = TensorBoardLogger(
    save_dir="lightning_logs",
    name="amp_experiment",
    version="v1.0"
)

# Add to trainer configuration
config["trainer"]["logger"] = logger
```

### Weights & Biases Integration

```python
from pytorch_lightning.loggers import WandbLogger

# Configure Wandb logger
wandb_logger = WandbLogger(
    project="amp-searcher",
    name="screening-experiment",
    save_dir="wandb_logs"
)

config["trainer"]["logger"] = wandb_logger
```

### Custom Logging

```python
class DetailedLoggingCallback(Callback):
    """Detailed logging callback for training monitoring."""
    
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            # Log batch-level metrics
            loss = outputs['loss']
            pl_module.log("train_batch_loss", loss, on_step=True, on_epoch=False)
            
            # Log gradient norms
            total_norm = 0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            pl_module.log("gradient_norm", total_norm, on_step=True, on_epoch=False)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Log model parameters statistics
        for name, param in pl_module.named_parameters():
            pl_module.log(f"param_mean_{name}", param.mean())
            pl_module.log(f"param_std_{name}", param.std())
```

## Hyperparameter Optimization

### Grid Search

```python
from itertools import product

def grid_search_training():
    # Define hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    hidden_dims = [[32], [64, 32], [128, 64, 32]]
    
    best_score = 0
    best_config = None
    
    for lr, batch_size, hidden_dim in product(learning_rates, batch_sizes, hidden_dims):
        # Create configuration
        config = base_config.copy()
        config["model"]["lightning_module_params"]["optimizer_params"]["lr"] = lr
        config["trainer"]["batch_size"] = batch_size
        config["model"]["architecture"]["params"]["hidden_dims"] = hidden_dim
        
        # Train model
        trainer = AmpTrainer(config)
        results = trainer.train(config["data"])
        
        # Evaluate performance
        val_score = results["val_accuracy"]  # or whatever metric you're optimizing
        
        if val_score > best_score:
            best_score = val_score
            best_config = config.copy()
    
    return best_config, best_score

# Run grid search
best_config, best_score = grid_search_training()
print(f"Best configuration achieved score: {best_score}")
```

### Optuna Integration

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    
    # Create configuration
    config = base_config.copy()
    config["model"]["lightning_module_params"]["optimizer_params"]["lr"] = lr
    config["trainer"]["batch_size"] = batch_size
    config["model"]["architecture"]["params"]["hidden_dims"] = [hidden_size, hidden_size // 2]
    config["model"]["architecture"]["params"]["dropout_rate"] = dropout_rate
    
    # Train model
    trainer = AmpTrainer(config)
    results = trainer.train(config["data"])
    
    return results["val_accuracy"]

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value}")
```

## Cross-Validation Training

```python
from sklearn.model_selection import KFold
import numpy as np

def cross_validation_training(config, n_folds=5):
    """Perform k-fold cross-validation training."""
    
    # Load full dataset
    import pandas as pd
    df = pd.read_csv(config["data"]["path"])
    
    # Initialize k-fold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print(f"Training fold {fold + 1}/{n_folds}")
        
        # Split data
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Save temporary files
        train_path = f"temp_train_fold_{fold}.csv"
        val_path = f"temp_val_fold_{fold}.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        # Update config for this fold
        fold_config = config.copy()
        fold_config["data"]["path"] = train_path
        fold_config["data"]["validation_path"] = val_path
        fold_config["trainer"]["logger_name"] = f"fold_{fold}"
        
        # Train model
        trainer = AmpTrainer(fold_config)
        results = trainer.train(fold_config["data"])
        
        fold_results.append(results)
        
        # Cleanup temporary files
        import os
        os.remove(train_path)
        os.remove(val_path)
    
    # Calculate average performance
    avg_results = {}
    for key in fold_results[0].keys():
        if isinstance(fold_results[0][key], (int, float)):
            avg_results[key] = np.mean([result[key] for result in fold_results])
            avg_results[f"{key}_std"] = np.std([result[key] for result in fold_results])
    
    return avg_results, fold_results

# Run cross-validation
cv_results, fold_results = cross_validation_training(config, n_folds=5)
print(f"Cross-validation results: {cv_results}")
```

## Transfer Learning

### Fine-tuning Pre-trained Models

```python
def fine_tune_model(pretrained_checkpoint, new_data_config, fine_tune_config):
    """Fine-tune a pre-trained model on new data."""
    
    # Load pre-trained model
    from amp_searcher.models.lightning_module_factory import LightningModuleFactory
    
    pretrained_model = LightningModuleFactory.build_lightning_module(
        fine_tune_config["model"]["type"],
        **fine_tune_config["model"]["lightning_module_params"]
    )
    pretrained_model = pretrained_model.load_from_checkpoint(pretrained_checkpoint)
    
    # Freeze backbone layers (optional)
    if fine_tune_config.get("freeze_backbone", False):
        for param in pretrained_model.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last layer
        for param in pretrained_model.model.network[-1].parameters():
            param.requires_grad = True
    
    # Reduce learning rate for fine-tuning
    fine_tune_config["model"]["lightning_module_params"]["optimizer_params"]["lr"] *= 0.1
    
    # Create trainer with pre-trained model
    trainer = AmpTrainer(fine_tune_config)
    trainer.model = pretrained_model
    
    # Fine-tune on new data
    results = trainer.train(new_data_config)
    
    return results

# Fine-tuning configuration
fine_tune_config = config.copy()
fine_tune_config["freeze_backbone"] = True
fine_tune_config["trainer"]["max_epochs"] = 20

# Fine-tune model
results = fine_tune_model(
    pretrained_checkpoint="models/pretrained_model.ckpt",
    new_data_config=new_data_config,
    fine_tune_config=fine_tune_config
)
```

## Model Evaluation and Testing

### Comprehensive Model Evaluation

```python
def evaluate_trained_model(model_checkpoint, test_data_config):
    """Comprehensive evaluation of trained model."""
    
    from amp_searcher.pipelines.screening_pipeline import ScreeningPipeline
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    import pandas as pd
    
    # Load test data
    test_df = pd.read_csv(test_data_config["path"])
    sequences = test_df[test_data_config["sequence_col"]].tolist()
    true_labels = test_df[test_data_config["label_col"]].tolist()
    
    # Initialize pipeline
    pipeline = ScreeningPipeline(
        model_config=config["model"],
        model_checkpoint_path=model_checkpoint,
        featurizer_config=config["featurizer"]
    )
    
    # Make predictions
    predictions = pipeline.predict(sequences)
    pred_probs = predictions.numpy()
    pred_labels = (pred_probs > 0.5).astype(int)
    
    # Calculate metrics
    auc_score = roc_auc_score(true_labels, pred_probs)
    classification_rep = classification_report(true_labels, pred_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    # Create results dictionary
    results = {
        "auc_score": auc_score,
        "classification_report": classification_rep,
        "confusion_matrix": conf_matrix,
        "predictions": pred_probs,
        "true_labels": true_labels
    }
    
    return results

# Evaluate model
test_config = {
    "path": "data/test_data.csv",
    "sequence_col": "sequence",
    "label_col": "activity"
}

evaluation_results = evaluate_trained_model(
    model_checkpoint="lightning_logs/version_0/checkpoints/best.ckpt",
    test_data_config=test_config
)

print(f"AUC Score: {evaluation_results['auc_score']:.4f}")
print(f"Classification Report:\n{evaluation_results['classification_report']}")
```

## Training Best Practices

### 1. Data Preparation

```python
def prepare_training_data(raw_data_path, output_dir):
    """Prepare and validate training data."""
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load raw data
    df = pd.read_csv(raw_data_path)
    
    # Data validation and cleaning
    # Remove duplicates
    df = df.drop_duplicates(subset=['sequence'])
    
    # Filter by sequence length
    df = df[df['sequence'].str.len().between(5, 50)]
    
    # Validate amino acid sequences
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    df = df[df['sequence'].apply(lambda x: set(x.upper()).issubset(valid_aa))]
    
    # Balance dataset (if needed)
    if 'activity' in df.columns:
        # Balance positive and negative samples
        pos_samples = df[df['activity'] == 1]
        neg_samples = df[df['activity'] == 0]
        
        min_samples = min(len(pos_samples), len(neg_samples))
        balanced_df = pd.concat([
            pos_samples.sample(min_samples, random_state=42),
            neg_samples.sample(min_samples, random_state=42)
        ])
        df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['activity'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['activity'])
    
    # Save splits
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    return train_df, val_df, test_df

# Prepare data
train_df, val_df, test_df = prepare_training_data(
    raw_data_path="data/raw_amp_data.csv",
    output_dir="data/processed"
)
```

### 2. Training Configuration Best Practices

```yaml
# Recommended training configuration
trainer:
  max_epochs: 100
  batch_size: 32  # Adjust based on GPU memory
  num_workers: 4  # Adjust based on CPU cores
  
  # Early stopping to prevent overfitting
  early_stopping:
    monitor: val_loss
    patience: 15
    mode: min
    min_delta: 0.001
  
  # Model checkpointing
  model_checkpoint:
    monitor: val_loss
    save_top_k: 3
    mode: min
    save_last: true
  
  # Learning rate monitoring
  lr_monitor:
    logging_interval: epoch
  
  # Gradient clipping
  gradient_clip_val: 1.0
  
  # Deterministic training for reproducibility
  deterministic: true
  seed: 42
```

### 3. Monitoring Training Progress

```python
def monitor_training_progress(log_dir):
    """Monitor and analyze training progress."""
    
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    # Load TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Extract metrics
    train_loss = [(s.step, s.value) for s in event_acc.Scalars('train_loss')]
    val_loss = [(s.step, s.value) for s in event_acc.Scalars('val_loss')]
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    train_steps, train_losses = zip(*train_loss)
    val_steps, val_losses = zip(*val_loss)
    
    ax1.plot(train_steps, train_losses, label='Train Loss')
    ax1.plot(val_steps, val_losses, label='Validation Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Learning rate curve
    if 'learning_rate' in event_acc.Tags()['scalars']:
        lr_data = [(s.step, s.value) for s in event_acc.Scalars('learning_rate')]
        lr_steps, lr_values = zip(*lr_data)
        ax2.plot(lr_steps, lr_values)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Monitor training
fig = monitor_training_progress("lightning_logs/version_0")
```

## Troubleshooting Training Issues

### Common Issues and Solutions

1. **Overfitting**:
   - Increase dropout rate
   - Add weight decay
   - Use early stopping
   - Reduce model complexity

2. **Underfitting**:
   - Increase model capacity
   - Reduce regularization
   - Increase learning rate
   - Train for more epochs

3. **Slow convergence**:
   - Adjust learning rate
   - Use learning rate scheduling
   - Check data preprocessing
   - Verify loss function

4. **Memory issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

```python
# Example debugging configuration
debug_config = config.copy()
debug_config.update({
    "trainer": {
        "max_epochs": 5,  # Short training for debugging
        "batch_size": 8,  # Small batch size
        "limit_train_batches": 0.1,  # Use only 10% of training data
        "limit_val_batches": 0.1,
        "fast_dev_run": True,  # Quick sanity check
        "overfit_batches": 10  # Overfit on small subset
    }
})

# Run debug training
debug_trainer = AmpTrainer(debug_config)
debug_results = debug_trainer.train(debug_config["data"])
```

This comprehensive training system provides all the tools needed to effectively train, monitor, and evaluate models for antimicrobial peptide research with AmpSearcher.
