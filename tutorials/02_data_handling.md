# Data Handling in AmpSearcher

## Overview

AmpSearcher provides a comprehensive data handling system that includes data loading, processing, validation, and feature extraction. The data module is designed to work with peptide sequences and their associated labels for various machine learning tasks.

## Core Components

### AmpDataset

The `AmpDataset` class is the main data container that extends PyTorch's `Dataset` class:

```python
from amp_searcher.data.datasets import AmpDataset
from amp_searcher.data.processors.processor_factory import ProcessorFactory

# Create a processor
processor_config = {
    "name": "sequence_processor",
    "params": {
        "featurizer_config": {
            "name": "PhysicochemicalFeaturizer"
        }
    }
}
processor = ProcessorFactory.build_processor(
    processor_config["name"], 
    **processor_config["params"]
)

# Create dataset
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "FLPIIAKLLSGLL"]
labels = [1.0, 0.0]  # Optional, can be None for prediction

dataset = AmpDataset(sequences, labels, processor)
```

### Data Loading Functions

#### Loading from CSV

```python
from amp_searcher.data.datasets import load_data_from_csv

# Load data with labels (for training)
dataset = load_data_from_csv(
    filepath="data/training_data.csv",
    sequence_col="sequence",
    label_col="activity",
    processor_config={
        "name": "sequence_processor",
        "params": {
            "featurizer_config": {
                "name": "PhysicochemicalFeaturizer"
            }
        }
    },
    output_dir="data/features"  # Optional: save processed features
)

# Load data without labels (for prediction)
dataset = load_data_from_csv(
    filepath="data/sequences_to_predict.csv",
    sequence_col="sequence",
    processor_config={
        "name": "sequence_processor",
        "params": {
            "featurizer_config": {
                "name": "PhysicochemicalFeaturizer"
            }
        }
    }
)
```

#### Loading from Text File

```python
from amp_searcher.data.datasets import load_sequences_from_file

# Load sequences from text file (one sequence per line)
dataset = load_sequences_from_file(
    filepath="data/sequences.txt",
    processor_config={
        "name": "sequence_processor",
        "params": {
            "featurizer_config": {
                "name": "PhysicochemicalFeaturizer"
            }
        }
    }
)
```

## Data Processors

Data processors handle the conversion of raw sequences into numerical features suitable for machine learning models.

### Sequence Processor

The main processor that combines featurization with sequence processing:

```python
from amp_searcher.data.processors.sequence_processor import SequenceProcessor
from amp_searcher.featurizers.featurizer_factory import FeaturizerFactory

# Create featurizer
featurizer = FeaturizerFactory.build_featurizer(
    "PhysicochemicalFeaturizer",
    custom_features=["length", "molecular_weight", "charge_at_ph_7"]
)

# Create processor
processor = SequenceProcessor(featurizer=featurizer)

# Process sequences
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "FLPIIAKLLSGLL"]
features = processor.process(sequences)
```

### Processor Factory

Use the factory pattern to create processors from configuration:

```python
from amp_searcher.data.processors.processor_factory import ProcessorFactory

# Register custom processor (if needed)
@ProcessorFactory.register("custom_processor")
class CustomProcessor:
    def __init__(self, **kwargs):
        pass
    
    def process(self, sequences):
        # Custom processing logic
        pass

# Build processor from configuration
processor = ProcessorFactory.build_processor(
    "sequence_processor",
    featurizer_config={
        "name": "PhysicochemicalFeaturizer",
        "custom_features": ["length", "molecular_weight"]
    }
)
```

## Data Validation

AmpSearcher includes comprehensive data validation to ensure data quality and consistency.

### Available Validators

1. **Amino Acid Validator**: Checks for valid amino acid sequences
2. **Sequence Length Validator**: Validates sequence length constraints
3. **Missing Value Validator**: Handles missing or invalid data

```python
from amp_searcher.data.validators.validator_factory import ValidatorFactory

# Create validators
aa_validator = ValidatorFactory.build_validator(
    "amino_acid_validator",
    allowed_chars="ACDEFGHIKLMNPQRSTVWY"
)

length_validator = ValidatorFactory.build_validator(
    "sequence_length_validator",
    min_length=5,
    max_length=50
)

# Validate sequences
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "INVALID123"]
valid_sequences = []

for seq in sequences:
    if aa_validator.validate(seq) and length_validator.validate(seq):
        valid_sequences.append(seq)
```

### Schema Validation

Validate DataFrames against predefined schemas:

```python
import pandas as pd
from amp_searcher.data.validators.schemas import RawDataSchema, validate_dataframe

# Load data
df = pd.read_csv("data/training_data.csv")

# Validate against schema
try:
    validate_dataframe(df, RawDataSchema)
    print("Data validation passed!")
except Exception as e:
    print(f"Validation error: {e}")
```

## Feature Caching

Save processed features to avoid recomputation:

```python
# Save features during dataset creation
dataset = load_data_from_csv(
    filepath="data/training_data.csv",
    sequence_col="sequence",
    label_col="activity",
    processor_config=processor_config,
    output_dir="data/features"  # Features will be saved here
)

# Manually save features
dataset.save_features(
    output_dir="data/features",
    filename="custom_features.parquet"
)
```

## Working with PyTorch DataLoader

```python
from torch.utils.data import DataLoader

# Create DataLoader for training
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate through batches
for batch_idx, (features, labels) in enumerate(dataloader):
    # features: torch.Tensor of shape (batch_size, feature_dim)
    # labels: torch.Tensor of shape (batch_size,) or None
    print(f"Batch {batch_idx}: {features.shape}")
    if labels is not None:
        print(f"Labels: {labels.shape}")
```

## Data Configuration Examples

### Basic Configuration

```yaml
data:
  path: "data/training_data.csv"
  sequence_col: "sequence"
  label_col: "activity"
  processor_config:
    name: "sequence_processor"
    params:
      featurizer_config:
        name: "PhysicochemicalFeaturizer"
```

### Advanced Configuration with Validation

```yaml
data:
  path: "data/training_data.csv"
  sequence_col: "sequence"
  label_col: "activity"
  processor_config:
    name: "sequence_processor"
    params:
      featurizer_config:
        name: "PhysicochemicalFeaturizer"
        custom_features: ["length", "molecular_weight", "charge_at_ph_7"]
      validators:
        - name: "amino_acid_validator"
          params:
            allowed_chars: "ACDEFGHIKLMNPQRSTVWY"
        - name: "sequence_length_validator"
          params:
            min_length: 5
            max_length: 50
```

## Best Practices

1. **Always validate your data** before training or prediction
2. **Use feature caching** for large datasets to speed up repeated experiments
3. **Choose appropriate featurizers** based on your specific task
4. **Handle missing values** appropriately in your validation pipeline
5. **Use consistent column names** across your datasets

## Common Data Formats

### Training Data CSV Format

```csv
sequence,activity,source
KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK,1,database_A
FLPIIAKLLSGLL,0,database_B
GIGKFLHSAKKFGKAFVGEIMNS,1,database_A
```

### Prediction Data CSV Format

```csv
sequence,id
KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK,seq_001
FLPIIAKLLSGLL,seq_002
GIGKFLHSAKKFGKAFVGEIMNS,seq_003
```

## Troubleshooting

### Common Issues

1. **Invalid amino acid characters**: Use amino acid validator to filter sequences
2. **Inconsistent sequence lengths**: Apply length validation or padding
3. **Missing labels**: Ensure label column exists for training data
4. **Memory issues with large datasets**: Use feature caching and appropriate batch sizes

### Error Handling

```python
try:
    dataset = load_data_from_csv(
        filepath="data/training_data.csv",
        sequence_col="sequence",
        label_col="activity",
        processor_config=processor_config
    )
except FileNotFoundError:
    print("Data file not found. Please check the file path.")
except ValueError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

This comprehensive data handling system ensures that your peptide sequence data is properly processed, validated, and ready for machine learning tasks in AmpSearcher.
