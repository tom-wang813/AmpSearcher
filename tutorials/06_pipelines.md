# Pipelines in AmpSearcher

## Overview

Pipelines in AmpSearcher provide high-level interfaces that combine multiple components (featurizers, models, optimizers) into streamlined workflows. They simplify complex tasks like screening peptides for antimicrobial activity and searching for novel peptide sequences with desired properties.

## Available Pipelines

### 1. Screening Pipeline

The `ScreeningPipeline` is designed for predicting antimicrobial activity of peptide sequences using trained models.

```python
from amp_searcher.pipelines.screening_pipeline import ScreeningPipeline
import yaml

# Load configuration
with open("configs/main/examples/screening_ffnn_physicochemical.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize screening pipeline
pipeline = ScreeningPipeline(
    model_config=config["model"],
    model_checkpoint_path="models/trained_screening_model.ckpt",
    featurizer_config=config["featurizer"]
)

# Predict antimicrobial activity
sequences = [
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    "FLPIIAKLLSGLL",
    "GIGKFLHSAKKFGKAFVGEIMNS"
]

predictions = pipeline.predict(sequences)

for seq, pred in zip(sequences, predictions):
    probability = pred.item()
    print(f"Sequence: {seq}")
    print(f"AMP Probability: {probability:.4f}")
    print(f"Prediction: {'AMP' if probability > 0.5 else 'Non-AMP'}")
    print("-" * 50)
```

### 2. Search Pipeline

The `SearchPipeline` combines generative models, screening models, and optimizers to discover novel antimicrobial peptides.

```python
from amp_searcher.pipelines.search_pipeline import SearchPipeline

# Initialize search pipeline
search_pipeline = SearchPipeline(
    generative_model_config={
        "type": "generative",
        "architecture": {"name": "VAE", "params": {"input_dim": 10, "latent_dim": 16}}
    },
    generative_model_checkpoint="models/vae_model.ckpt",
    screening_model_config={
        "type": "screening", 
        "architecture": {"name": "FFNN", "params": {"input_dim": 10, "output_dim": 1}}
    },
    screening_model_checkpoint="models/screening_model.ckpt",
    featurizer_config={"name": "PhysicochemicalFeaturizer"},
    optimizer_config={
        "name": "GeneticAlgorithmOptimizer",
        "population_size": 100,
        "max_generations": 50
    },
    sequence_decoder_config={"name": "simple_decoder", "max_length": 30}
)

# Run search for novel AMPs
best_sequences = search_pipeline.search()

print("Top discovered sequences:")
for seq, score in best_sequences[:10]:
    print(f"Sequence: {seq}, Score: {score:.4f}")
```

## Pipeline Configuration

### Screening Pipeline Configuration

```yaml
# Complete configuration for screening pipeline
featurizer:
  name: PhysicochemicalFeaturizer
  custom_features:
    - length
    - molecular_weight
    - charge_at_ph_7
    - aromaticity
    - gravy

model:
  type: screening
  architecture:
    name: FFNN
    params:
      input_dim: 5  # Number of features
      output_dim: 1
      hidden_dims: [64, 32]
      dropout_rate: 0.2
  lightning_module_params:
    task_type: classification
    optimizer_params:
      lr: 0.001
    scheduler_params:
      name: StepLR
      step_size: 10
      gamma: 0.1

# Model checkpoint path
model_checkpoint: "models/screening_model.ckpt"
```

### Search Pipeline Configuration

```yaml
# Generative model configuration
generative_model:
  type: generative
  architecture:
    name: VAE
    params:
      input_dim: 10
      latent_dim: 16
      hidden_dims: [64, 32]
      beta: 1.0
  checkpoint_path: "models/vae_model.ckpt"

# Screening model configuration
screening_model:
  type: screening
  architecture:
    name: FFNN
    params:
      input_dim: 10
      output_dim: 1
      hidden_dims: [64, 32]
  checkpoint_path: "models/screening_model.ckpt"

# Featurizer configuration
featurizer:
  name: PhysicochemicalFeaturizer

# Optimizer configuration
optimizer:
  name: GeneticAlgorithmOptimizer
  population_size: 100
  mutation_rate: 0.1
  crossover_rate: 0.8
  max_generations: 50

# Sequence decoder configuration
sequence_decoder:
  name: simple_decoder
  max_length: 30
  amino_acids: "ACDEFGHIKLMNPQRSTVWY"
```

## Custom Pipeline Creation

You can create custom pipelines for specific workflows:

```python
from amp_searcher.pipelines.screening_pipeline import ScreeningPipeline
from amp_searcher.featurizers.featurizer_factory import FeaturizerFactory
from amp_searcher.models.lightning_module_factory import LightningModuleFactory
import torch

class CustomScreeningPipeline:
    """Custom pipeline with additional preprocessing and postprocessing."""
    
    def __init__(self, model_config, model_checkpoint_path, featurizer_config, 
                 min_length=5, max_length=50):
        self.min_length = min_length
        self.max_length = max_length
        
        # Initialize featurizer
        self.featurizer = FeaturizerFactory.build_featurizer(
            featurizer_config["name"],
            **{k: v for k, v in featurizer_config.items() if k != "name"}
        )
        
        # Load model
        self.model = LightningModuleFactory.build_lightning_module(
            model_config["type"],
            **model_config.get("lightning_module_params", {})
        )
        self.model = self.model.load_from_checkpoint(model_checkpoint_path)
        self.model.eval()
    
    def preprocess_sequences(self, sequences):
        """Filter sequences by length and validate amino acids."""
        valid_sequences = []
        valid_indices = []
        
        for i, seq in enumerate(sequences):
            # Length filter
            if not (self.min_length <= len(seq) <= self.max_length):
                continue
            
            # Amino acid validation
            if not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq.upper()):
                continue
            
            valid_sequences.append(seq.upper())
            valid_indices.append(i)
        
        return valid_sequences, valid_indices
    
    def predict(self, sequences):
        """Predict with preprocessing and confidence estimation."""
        # Preprocess sequences
        valid_sequences, valid_indices = self.preprocess_sequences(sequences)
        
        if not valid_sequences:
            return torch.zeros(len(sequences))
        
        # Featurize
        features = torch.stack([
            torch.tensor(self.featurizer.featurize(seq), dtype=torch.float32)
            for seq in valid_sequences
        ])
        
        # Predict
        with torch.no_grad():
            predictions = torch.sigmoid(self.model(features)).squeeze()
        
        # Map back to original indices
        full_predictions = torch.zeros(len(sequences))
        for i, idx in enumerate(valid_indices):
            full_predictions[idx] = predictions[i] if predictions.dim() > 0 else predictions
        
        return full_predictions
    
    def predict_with_confidence(self, sequences, num_samples=10):
        """Predict with uncertainty estimation using dropout."""
        self.model.train()  # Enable dropout
        
        valid_sequences, valid_indices = self.preprocess_sequences(sequences)
        
        if not valid_sequences:
            return torch.zeros(len(sequences)), torch.zeros(len(sequences))
        
        features = torch.stack([
            torch.tensor(self.featurizer.featurize(seq), dtype=torch.float32)
            for seq in valid_sequences
        ])
        
        # Multiple forward passes with dropout
        predictions_list = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = torch.sigmoid(self.model(features)).squeeze()
                predictions_list.append(pred)
        
        # Calculate mean and std
        predictions_tensor = torch.stack(predictions_list)
        mean_pred = predictions_tensor.mean(dim=0)
        std_pred = predictions_tensor.std(dim=0)
        
        # Map back to original indices
        full_predictions = torch.zeros(len(sequences))
        full_uncertainties = torch.zeros(len(sequences))
        
        for i, idx in enumerate(valid_indices):
            full_predictions[idx] = mean_pred[i] if mean_pred.dim() > 0 else mean_pred
            full_uncertainties[idx] = std_pred[i] if std_pred.dim() > 0 else std_pred
        
        self.model.eval()  # Disable dropout
        return full_predictions, full_uncertainties

# Usage
custom_pipeline = CustomScreeningPipeline(
    model_config=config["model"],
    model_checkpoint_path="models/trained_model.ckpt",
    featurizer_config=config["featurizer"],
    min_length=8,
    max_length=40
)

# Predict with confidence
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "INVALID_SEQ123"]
predictions, uncertainties = custom_pipeline.predict_with_confidence(sequences)

for seq, pred, unc in zip(sequences, predictions, uncertainties):
    print(f"Sequence: {seq}")
    print(f"Prediction: {pred:.4f} ± {unc:.4f}")
```

## Batch Processing Pipeline

For processing large datasets efficiently:

```python
class BatchScreeningPipeline:
    """Pipeline optimized for batch processing of large datasets."""
    
    def __init__(self, model_config, model_checkpoint_path, featurizer_config, 
                 batch_size=32, device="auto"):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        
        # Initialize components
        self.featurizer = FeaturizerFactory.build_featurizer(
            featurizer_config["name"],
            **{k: v for k, v in featurizer_config.items() if k != "name"}
        )
        
        self.model = LightningModuleFactory.build_lightning_module(
            model_config["type"],
            **model_config.get("lightning_module_params", {})
        )
        self.model = self.model.load_from_checkpoint(model_checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict_batch(self, sequences):
        """Predict in batches for memory efficiency."""
        all_predictions = []
        
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i + self.batch_size]
            
            # Featurize batch
            batch_features = torch.stack([
                torch.tensor(self.featurizer.featurize(seq), dtype=torch.float32)
                for seq in batch_sequences
            ]).to(self.device)
            
            # Predict batch
            with torch.no_grad():
                batch_predictions = torch.sigmoid(self.model(batch_features))
                all_predictions.append(batch_predictions.cpu())
        
        return torch.cat(all_predictions, dim=0)
    
    def predict_from_file(self, input_file, output_file, sequence_column="sequence"):
        """Process sequences from CSV file and save results."""
        import pandas as pd
        
        # Read data
        df = pd.read_csv(input_file)
        sequences = df[sequence_column].tolist()
        
        # Predict in batches
        predictions = self.predict_batch(sequences)
        
        # Add predictions to dataframe
        df["amp_probability"] = predictions.squeeze().numpy()
        df["amp_prediction"] = (df["amp_probability"] > 0.5).astype(int)
        
        # Save results
        df.to_csv(output_file, index=False)
        
        return df

# Usage
batch_pipeline = BatchScreeningPipeline(
    model_config=config["model"],
    model_checkpoint_path="models/trained_model.ckpt",
    featurizer_config=config["featurizer"],
    batch_size=64
)

# Process large dataset
results_df = batch_pipeline.predict_from_file(
    input_file="data/large_peptide_dataset.csv",
    output_file="results/predictions.csv",
    sequence_column="sequence"
)

print(f"Processed {len(results_df)} sequences")
print(f"Predicted AMPs: {results_df['amp_prediction'].sum()}")
```

## Multi-Model Ensemble Pipeline

Combine multiple models for more robust predictions:

```python
class EnsembleScreeningPipeline:
    """Pipeline that combines predictions from multiple models."""
    
    def __init__(self, model_configs, ensemble_method="average"):
        self.models = []
        self.featurizers = []
        self.ensemble_method = ensemble_method
        
        for config in model_configs:
            # Initialize featurizer
            featurizer = FeaturizerFactory.build_featurizer(
                config["featurizer"]["name"],
                **{k: v for k, v in config["featurizer"].items() if k != "name"}
            )
            
            # Load model
            model = LightningModuleFactory.build_lightning_module(
                config["model"]["type"],
                **config["model"].get("lightning_module_params", {})
            )
            model = model.load_from_checkpoint(config["checkpoint_path"])
            model.eval()
            
            self.featurizers.append(featurizer)
            self.models.append(model)
    
    def predict(self, sequences):
        """Predict using ensemble of models."""
        all_predictions = []
        
        for featurizer, model in zip(self.featurizers, self.models):
            # Featurize with model-specific featurizer
            features = torch.stack([
                torch.tensor(featurizer.featurize(seq), dtype=torch.float32)
                for seq in sequences
            ])
            
            # Predict
            with torch.no_grad():
                predictions = torch.sigmoid(model(features))
                all_predictions.append(predictions)
        
        # Combine predictions
        if self.ensemble_method == "average":
            ensemble_pred = torch.stack(all_predictions).mean(dim=0)
        elif self.ensemble_method == "max":
            ensemble_pred = torch.stack(all_predictions).max(dim=0)[0]
        elif self.ensemble_method == "weighted":
            # Simple equal weighting (can be made more sophisticated)
            weights = torch.ones(len(all_predictions)) / len(all_predictions)
            ensemble_pred = sum(w * pred for w, pred in zip(weights, all_predictions))
        
        return ensemble_pred
    
    def predict_with_individual_scores(self, sequences):
        """Return both ensemble and individual model predictions."""
        individual_predictions = []
        
        for featurizer, model in zip(self.featurizers, self.models):
            features = torch.stack([
                torch.tensor(featurizer.featurize(seq), dtype=torch.float32)
                for seq in sequences
            ])
            
            with torch.no_grad():
                predictions = torch.sigmoid(model(features))
                individual_predictions.append(predictions)
        
        ensemble_pred = torch.stack(individual_predictions).mean(dim=0)
        
        return ensemble_pred, individual_predictions

# Configuration for ensemble
ensemble_configs = [
    {
        "featurizer": {"name": "PhysicochemicalFeaturizer"},
        "model": {"type": "screening"},
        "checkpoint_path": "models/physicochemical_model.ckpt"
    },
    {
        "featurizer": {"name": "CompositionFeaturizer", "include_aac": True},
        "model": {"type": "screening"},
        "checkpoint_path": "models/composition_model.ckpt"
    },
    {
        "featurizer": {"name": "PseAACFeaturizer"},
        "model": {"type": "screening"},
        "checkpoint_path": "models/pseaac_model.ckpt"
    }
]

# Create ensemble pipeline
ensemble_pipeline = EnsembleScreeningPipeline(
    model_configs=ensemble_configs,
    ensemble_method="average"
)

# Predict with ensemble
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"]
ensemble_pred, individual_preds = ensemble_pipeline.predict_with_individual_scores(sequences)

print(f"Ensemble prediction: {ensemble_pred[0].item():.4f}")
for i, pred in enumerate(individual_preds):
    print(f"Model {i+1} prediction: {pred[0].item():.4f}")
```

## Pipeline Monitoring and Logging

Add monitoring capabilities to pipelines:

```python
import logging
import time
from datetime import datetime

class MonitoredScreeningPipeline(ScreeningPipeline):
    """Screening pipeline with monitoring and logging capabilities."""
    
    def __init__(self, *args, log_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Setup logging
        self.logger = logging.getLogger("ScreeningPipeline")
        self.logger.setLevel(logging.INFO)
        
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Monitoring metrics
        self.prediction_count = 0
        self.total_prediction_time = 0
        self.start_time = datetime.now()
    
    def predict(self, sequences):
        """Predict with monitoring."""
        start_time = time.time()
        
        self.logger.info(f"Starting prediction for {len(sequences)} sequences")
        
        try:
            predictions = super().predict(sequences)
            
            # Update metrics
            prediction_time = time.time() - start_time
            self.prediction_count += len(sequences)
            self.total_prediction_time += prediction_time
            
            # Log performance metrics
            avg_time_per_seq = prediction_time / len(sequences)
            self.logger.info(
                f"Prediction completed in {prediction_time:.2f}s "
                f"({avg_time_per_seq:.4f}s per sequence)"
            )
            
            # Log prediction statistics
            amp_count = (predictions > 0.5).sum().item()
            self.logger.info(
                f"Predicted {amp_count}/{len(sequences)} sequences as AMPs "
                f"({amp_count/len(sequences)*100:.1f}%)"
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_performance_stats(self):
        """Get performance statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_time_per_prediction = (
            self.total_prediction_time / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            "total_predictions": self.prediction_count,
            "total_prediction_time": self.total_prediction_time,
            "average_time_per_prediction": avg_time_per_prediction,
            "uptime_seconds": uptime,
            "predictions_per_second": self.prediction_count / uptime if uptime > 0 else 0
        }

# Usage with monitoring
monitored_pipeline = MonitoredScreeningPipeline(
    model_config=config["model"],
    model_checkpoint_path="models/trained_model.ckpt",
    featurizer_config=config["featurizer"],
    log_file="logs/screening_pipeline.log"
)

# Make predictions
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"] * 100
predictions = monitored_pipeline.predict(sequences)

# Check performance stats
stats = monitored_pipeline.get_performance_stats()
print(f"Performance Stats: {stats}")
```

## Best Practices

1. **Use appropriate batch sizes** for memory efficiency
2. **Implement proper error handling** for robust pipelines
3. **Add logging and monitoring** for production deployments
4. **Cache featurized data** when processing the same sequences multiple times
5. **Use ensemble methods** for critical applications
6. **Validate inputs** before processing
7. **Optimize for your specific hardware** (CPU vs GPU)

## Integration with External Systems

### REST API Integration

```python
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Initialize pipeline
pipeline = ScreeningPipeline(
    model_config=config["model"],
    model_checkpoint_path="models/trained_model.ckpt",
    featurizer_config=config["featurizer"]
)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        sequences = data.get('sequences', [])
        
        if not sequences:
            return jsonify({'error': 'No sequences provided'}), 400
        
        predictions = pipeline.predict(sequences)
        
        results = []
        for seq, pred in zip(sequences, predictions):
            results.append({
                'sequence': seq,
                'amp_probability': float(pred.item()),
                'prediction': 'AMP' if pred.item() > 0.5 else 'Non-AMP'
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Database Integration

```python
import sqlite3
import pandas as pd

class DatabaseScreeningPipeline(ScreeningPipeline):
    """Pipeline with database integration for storing results."""
    
    def __init__(self, db_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence TEXT NOT NULL,
                amp_probability REAL NOT NULL,
                prediction TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def predict_and_store(self, sequences):
        """Predict and store results in database."""
        predictions = self.predict(sequences)
        
        # Prepare data for database
        data = []
        for seq, pred in zip(sequences, predictions):
            prob = float(pred.item())
            prediction = 'AMP' if prob > 0.5 else 'Non-AMP'
            data.append((seq, prob, prediction))
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executemany(
            'INSERT INTO predictions (sequence, amp_probability, prediction) VALUES (?, ?, ?)',
            data
        )
        
        conn.commit()
        conn.close()
        
        return predictions
    
    def get_prediction_history(self, limit=100):
        """Retrieve prediction history from database."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f'SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}',
            conn
        )
        conn.close()
        return df

# Usage
db_pipeline = DatabaseScreeningPipeline(
    db_path="predictions.db",
    model_config=config["model"],
    model_checkpoint_path="models/trained_model.ckpt",
    featurizer_config=config["featurizer"]
)

# Predict and store
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"]
predictions = db_pipeline.predict_and_store(sequences)

# Retrieve history
history = db_pipeline.get_prediction_history()
print(history.head())
```

This comprehensive pipeline system provides flexible, scalable, and production-ready tools for antimicrobial peptide screening and discovery workflows.
