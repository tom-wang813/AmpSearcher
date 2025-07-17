# Models in AmpSearcher

## Overview

AmpSearcher provides a comprehensive model system built on PyTorch Lightning, supporting various neural network architectures for different tasks including screening, contrastive learning, and generative modeling of antimicrobial peptides.

## Model Architecture

### Base Lightning Module

All models inherit from `BaseLightningModule`, which provides common functionality:

```python
from amp_searcher.models.base import BaseLightningModule
import torch
import torch.nn as nn

class CustomModel(BaseLightningModule):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.model(x)
```

## Model Types

### 1. Screening Models

Screening models are designed for classification tasks (AMP vs non-AMP prediction).

#### Feed-Forward Neural Network (FFNN)

```python
from amp_searcher.models.screening.lightning_module import ScreeningLightningModule
from amp_searcher.models.architectures.feed_forward_nn import FeedForwardNN

# Create FFNN architecture
model_arch = FeedForwardNN(
    input_dim=10,
    output_dim=1,
    hidden_dims=[64, 32],
    dropout_rate=0.2,
    activation="relu"
)

# Create Lightning module
model = ScreeningLightningModule(
    model=model_arch,
    task_type="classification",
    optimizer_params={"lr": 0.001},
    scheduler_params={"name": "StepLR", "step_size": 10, "gamma": 0.1}
)
```

#### Transformer Encoder

```python
from amp_searcher.models.architectures.advanced.transformer_encoder import TransformerEncoder

# Create Transformer architecture
model_arch = TransformerEncoder(
    vocab_size=22,  # 20 amino acids + special tokens
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    max_seq_len=50,
    num_classes=1
)

# Create Lightning module
model = ScreeningLightningModule(
    model=model_arch,
    task_type="classification",
    optimizer_params={"lr": 0.0001},
    scheduler_params={"name": "CosineAnnealingLR", "T_max": 100}
)
```

### 2. Contrastive Learning Models

Contrastive models learn representations by comparing similar and dissimilar sequences.

```python
from amp_searcher.models.contrastive.lightning_module import ContrastiveLightningModule
from amp_searcher.models.contrastive.architectures.simclr_backbone import SimCLRBackbone

# Create SimCLR backbone
backbone = SimCLRBackbone(
    input_dim=10,
    hidden_dims=[128, 64],
    projection_dim=32
)

# Create contrastive model
model = ContrastiveLightningModule(
    backbone=backbone,
    temperature=0.1,
    optimizer_params={"lr": 0.001}
)
```

### 3. Generative Models

Generative models can create new peptide sequences with desired properties.

#### Variational Autoencoder (VAE)

```python
from amp_searcher.models.generative.lightning_module import GenerativeLightningModule
from amp_searcher.models.generative.architectures.vae import VAE

# Create VAE
vae = VAE(
    input_dim=10,
    latent_dim=16,
    hidden_dims=[64, 32],
    beta=1.0  # KL divergence weight
)

# Create generative model
model = GenerativeLightningModule(
    model=vae,
    optimizer_params={"lr": 0.001}
)
```

## Model Factory

Use the factory pattern to create models from configuration:

```python
from amp_searcher.models.model_factory import ModelFactory
from amp_searcher.models.lightning_module_factory import LightningModuleFactory

# Register custom architecture
@ModelFactory.register("custom_architecture")
def build_custom_model(input_dim, output_dim, **kwargs):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )

# Build model from configuration
model_config = {
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
}

# Create architecture
architecture = ModelFactory.build_model(
    model_config["architecture"]["name"],
    **model_config["architecture"]["params"]
)

# Create Lightning module
lightning_module = LightningModuleFactory.build_lightning_module(
    model_config["type"],
    model=architecture,
    **model_config["lightning_module_params"]
)
```

## Configuration Examples

### Screening FFNN Configuration

```yaml
model:
  type: screening
  architecture:
    name: FFNN
    params:
      input_dim: 10
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
```

### Transformer Configuration

```yaml
model:
  type: screening
  architecture:
    name: TransformerEncoder
    params:
      vocab_size: 22
      d_model: 128
      nhead: 8
      num_layers: 4
      dim_feedforward: 512
      max_seq_len: 50
      num_classes: 1
      dropout: 0.1
  lightning_module_params:
    task_type: classification
    optimizer_params:
      lr: 0.0001
    scheduler_params:
      name: CosineAnnealingLR
      T_max: 100
```

### Contrastive Learning Configuration

```yaml
model:
  type: contrastive
  architecture:
    name: SimCLRBackbone
    params:
      input_dim: 10
      hidden_dims: [128, 64]
      projection_dim: 32
      dropout_rate: 0.1
  lightning_module_params:
    temperature: 0.1
    optimizer_params:
      lr: 0.001
```

### VAE Configuration

```yaml
model:
  type: generative
  architecture:
    name: VAE
    params:
      input_dim: 10
      latent_dim: 16
      hidden_dims: [64, 32]
      beta: 1.0
  lightning_module_params:
    optimizer_params:
      lr: 0.001
```

## Training and Evaluation

### Basic Training Loop

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Prepare data
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices="auto",
    logger=pl.loggers.TensorBoardLogger("lightning_logs/"),
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1)
    ]
)

# Train model
trainer.fit(model, train_dataloader, val_dataloader)
```

### Custom Metrics

```python
from torchmetrics import Accuracy, F1Score, AUROC

class CustomScreeningModule(ScreeningLightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.accuracy = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")
        self.auroc = AUROC(task="binary")
    
    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        
        x, y = batch
        y_hat = self(x)
        y_pred = torch.sigmoid(y_hat)
        
        # Log metrics
        self.log("val_accuracy", self.accuracy(y_pred, y))
        self.log("val_f1", self.f1(y_pred, y))
        self.log("val_auroc", self.auroc(y_pred, y))
        
        return loss
```

## Model Architectures

### Feed-Forward Neural Network

```python
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.0, activation="relu"):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        return self.network(x)
```

### Transformer Encoder

```python
import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, 
                 max_seq_len, num_classes, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer expects (batch_size, seq_len, d_model)
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x.masked_fill(mask_expanded, 0)
            x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        return self.classifier(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)
```

## Model Optimization

### Learning Rate Scheduling

```python
# Step LR
scheduler_config = {
    "name": "StepLR",
    "step_size": 10,
    "gamma": 0.1
}

# Cosine Annealing
scheduler_config = {
    "name": "CosineAnnealingLR",
    "T_max": 100,
    "eta_min": 1e-6
}

# Reduce on Plateau
scheduler_config = {
    "name": "ReduceLROnPlateau",
    "mode": "min",
    "factor": 0.5,
    "patience": 5
}
```

### Optimizer Configuration

```python
# Adam optimizer
optimizer_config = {
    "name": "Adam",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "betas": [0.9, 0.999]
}

# AdamW optimizer
optimizer_config = {
    "name": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.01
}

# SGD optimizer
optimizer_config = {
    "name": "SGD",
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0001
}
```

## Model Evaluation

### Inference

```python
# Load trained model
model = ScreeningLightningModule.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(features)
    probabilities = torch.sigmoid(predictions)
```

### Model Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            features, labels = batch
            outputs = model(features)
            preds = torch.sigmoid(outputs)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to binary predictions
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc": auc
    }
```

## Advanced Features

### Model Ensembling

```python
class ModelEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Average predictions
        return torch.stack(outputs).mean(dim=0)

# Create ensemble
models = [model1, model2, model3]
ensemble = ModelEnsemble(models)
```

### Transfer Learning

```python
# Load pre-trained model
pretrained_model = ScreeningLightningModule.load_from_checkpoint("pretrained.ckpt")

# Freeze backbone layers
for param in pretrained_model.model.network[:-1].parameters():
    param.requires_grad = False

# Fine-tune only the last layer
pretrained_model.model.network[-1] = nn.Linear(64, new_num_classes)
```

## Best Practices

1. **Start with simple architectures** (FFNN) before moving to complex ones
2. **Use appropriate learning rates** (1e-3 for Adam, 1e-2 for SGD)
3. **Apply regularization** (dropout, weight decay) to prevent overfitting
4. **Monitor validation metrics** to detect overfitting early
5. **Use learning rate scheduling** for better convergence
6. **Save model checkpoints** regularly during training
7. **Validate on held-out test sets** for unbiased evaluation

## Troubleshooting

### Common Issues

1. **Gradient explosion**: Use gradient clipping
```python
trainer = pl.Trainer(gradient_clip_val=1.0)
```

2. **Overfitting**: Increase dropout, add weight decay, use early stopping
3. **Slow convergence**: Adjust learning rate, use learning rate scheduling
4. **Memory issues**: Reduce batch size, use gradient accumulation

### Debugging

```python
# Check model architecture
print(model)

# Monitor gradients
trainer = pl.Trainer(track_grad_norm=2)

# Log model parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad: {param.requires_grad}")
```

This comprehensive model system provides flexible and powerful tools for building, training, and evaluating neural networks for antimicrobial peptide research.
