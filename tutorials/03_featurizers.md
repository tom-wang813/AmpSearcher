# Featurizers in AmpSearcher

## Overview

Featurizers are essential components in AmpSearcher that convert peptide sequences into numerical representations suitable for machine learning models. AmpSearcher provides several built-in featurizers, each capturing different aspects of peptide properties.

## Base Featurizer

All featurizers inherit from the `BaseFeaturizer` abstract class:

```python
from amp_searcher.featurizers.base import BaseFeaturizer
import numpy as np

class CustomFeaturizer(BaseFeaturizer):
    def featurize(self, sequence: str) -> np.ndarray:
        # Implement your custom featurization logic
        pass
```

## Available Featurizers

### 1. PhysicochemicalFeaturizer

Calculates physicochemical properties of peptide sequences using BioPython.

```python
from amp_searcher.featurizers.physicochemical import PhysicochemicalFeaturizer

# Use all available features
featurizer = PhysicochemicalFeaturizer()

# Use specific features only
featurizer = PhysicochemicalFeaturizer(
    custom_features=["length", "molecular_weight", "charge_at_ph_7"]
)

# Featurize a sequence
sequence = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
features = featurizer.featurize(sequence)
print(f"Features shape: {features.shape}")
print(f"Feature names: {featurizer.feature_names}")
```

**Available Features:**
- `length`: Sequence length
- `molecular_weight`: Molecular weight in Daltons
- `charge_at_ph_7`: Net charge at pH 7.0
- `isoelectric_point`: Isoelectric point
- `aromaticity`: Fraction of aromatic amino acids
- `instability_index`: Protein instability index
- `gravy`: Grand average of hydropathy
- `helix_fraction`: Alpha helix fraction
- `turn_fraction`: Turn fraction
- `sheet_fraction`: Beta sheet fraction

### 2. CompositionFeaturizer

Calculates amino acid composition and dipeptide composition.

```python
from amp_searcher.featurizers.composition import CompositionFeaturizer

# Amino acid composition only
featurizer = CompositionFeaturizer(include_aac=True, include_dpc=False)

# Both amino acid and dipeptide composition
featurizer = CompositionFeaturizer(include_aac=True, include_dpc=True)

sequence = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
features = featurizer.featurize(sequence)
print(f"Features shape: {features.shape}")
```

**Features:**
- **AAC (Amino Acid Composition)**: 20 features representing the frequency of each amino acid
- **DPC (Dipeptide Composition)**: 400 features representing the frequency of each dipeptide

### 3. PseAACFeaturizer

Implements Pseudo Amino Acid Composition with physicochemical properties.

```python
from amp_searcher.featurizers.pse_aac import PseAACFeaturizer

# Default parameters
featurizer = PseAACFeaturizer()

# Custom parameters
featurizer = PseAACFeaturizer(
    lambda_val=10,  # Weight factor for pseudo components
    w=0.1          # Weight for pseudo amino acid composition
)

sequence = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
features = featurizer.featurize(sequence)
print(f"Features shape: {features.shape}")
```

### 4. SimpleSequenceFeaturizer

Converts sequences to integer tokens for neural network models.

```python
from amp_searcher.featurizers.simple_sequence import SimpleSequenceFeaturizer

# Define vocabulary (amino acids + special tokens)
vocab = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '<PAD>': 0, '<UNK>': 21
}

featurizer = SimpleSequenceFeaturizer(max_len=50, vocab=vocab)

sequence = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
features = featurizer.featurize(sequence)
print(f"Features shape: {features.shape}")
print(f"Features type: {type(features)}")

# Convert back to sequence
reconstructed = featurizer.defragment(features)
print(f"Reconstructed: {reconstructed}")
```

## Featurizer Factory

Use the factory pattern to create featurizers from configuration:

```python
from amp_searcher.featurizers.featurizer_factory import FeaturizerFactory

# Register a custom featurizer
@FeaturizerFactory.register("custom_featurizer")
class CustomFeaturizer(BaseFeaturizer):
    def __init__(self, param1=None, param2=None):
        self.param1 = param1
        self.param2 = param2
    
    def featurize(self, sequence: str) -> np.ndarray:
        # Custom implementation
        return np.array([len(sequence)])

# Build featurizer from name and parameters
featurizer = FeaturizerFactory.build_featurizer(
    "PhysicochemicalFeaturizer",
    custom_features=["length", "molecular_weight"]
)

# Build custom featurizer
custom_featurizer = FeaturizerFactory.build_featurizer(
    "custom_featurizer",
    param1="value1",
    param2="value2"
)
```

## Configuration Examples

### YAML Configuration

```yaml
featurizer:
  name: PhysicochemicalFeaturizer
  custom_features:
    - length
    - molecular_weight
    - charge_at_ph_7
    - aromaticity
```

```yaml
featurizer:
  name: CompositionFeaturizer
  include_aac: true
  include_dpc: false
```

```yaml
featurizer:
  name: PseAACFeaturizer
  lambda_val: 15
  w: 0.05
```

### Python Configuration

```python
# Physicochemical features
config = {
    "name": "PhysicochemicalFeaturizer",
    "custom_features": ["length", "molecular_weight", "charge_at_ph_7"]
}

# Composition features
config = {
    "name": "CompositionFeaturizer",
    "include_aac": True,
    "include_dpc": True
}

# Create featurizer from config
featurizer = FeaturizerFactory.build_featurizer(
    config["name"],
    **{k: v for k, v in config.items() if k != "name"}
)
```

## Combining Featurizers

You can combine multiple featurizers to create richer representations:

```python
import numpy as np

class CombinedFeaturizer(BaseFeaturizer):
    def __init__(self, featurizers):
        self.featurizers = featurizers
    
    def featurize(self, sequence: str) -> np.ndarray:
        features = []
        for featurizer in self.featurizers:
            feat = featurizer.featurize(sequence)
            features.append(feat)
        return np.concatenate(features)

# Combine physicochemical and composition features
phys_featurizer = PhysicochemicalFeaturizer()
comp_featurizer = CompositionFeaturizer(include_aac=True, include_dpc=False)

combined = CombinedFeaturizer([phys_featurizer, comp_featurizer])
features = combined.featurize("KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK")
print(f"Combined features shape: {features.shape}")
```

## Feature Scaling and Normalization

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Example with multiple sequences
sequences = [
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    "FLPIIAKLLSGLL",
    "GIGKFLHSAKKFGKAFVGEIMNS"
]

featurizer = PhysicochemicalFeaturizer()
features = np.array([featurizer.featurize(seq) for seq in sequences])

# Standardization (zero mean, unit variance)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Min-Max scaling (0-1 range)
minmax_scaler = MinMaxScaler()
minmax_features = minmax_scaler.fit_transform(features)

print(f"Original features shape: {features.shape}")
print(f"Scaled features shape: {scaled_features.shape}")
```

## Performance Considerations

### Batch Processing

```python
def batch_featurize(featurizer, sequences, batch_size=100):
    """Process sequences in batches for memory efficiency."""
    features = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_features = [featurizer.featurize(seq) for seq in batch]
        features.extend(batch_features)
    return np.array(features)

# Example usage
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"] * 1000
featurizer = PhysicochemicalFeaturizer()
features = batch_featurize(featurizer, sequences, batch_size=50)
```

### Caching Features

```python
import pickle
import os

def cache_features(sequences, featurizer, cache_file):
    """Cache computed features to disk."""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    features = [featurizer.featurize(seq) for seq in sequences]
    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)
    
    return features

# Usage
sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "FLPIIAKLLSGLL"]
featurizer = PhysicochemicalFeaturizer()
features = cache_features(sequences, featurizer, "features_cache.pkl")
```

## Choosing the Right Featurizer

### Task-Specific Recommendations

1. **Binary Classification (AMP vs Non-AMP)**:
   - `PhysicochemicalFeaturizer`: Good for interpretability
   - `CompositionFeaturizer`: Captures sequence composition patterns

2. **Multi-class Classification**:
   - `PseAACFeaturizer`: Balances composition and sequence order
   - Combined featurizers for richer representations

3. **Sequence Generation**:
   - `SimpleSequenceFeaturizer`: For neural language models
   - Token-based representations

4. **Similarity Search**:
   - `CompositionFeaturizer`: Good for composition-based similarity
   - `PhysicochemicalFeaturizer`: For property-based similarity

### Feature Dimensionality

| Featurizer | Dimensions | Use Case |
|------------|------------|----------|
| PhysicochemicalFeaturizer | 10 | Interpretable models |
| CompositionFeaturizer (AAC only) | 20 | Fast training |
| CompositionFeaturizer (AAC + DPC) | 420 | Rich composition info |
| PseAACFeaturizer | 20 + λ | Balanced approach |
| SimpleSequenceFeaturizer | max_len | Neural networks |

## Troubleshooting

### Common Issues

1. **Invalid amino acid characters**:
```python
# PhysicochemicalFeaturizer handles this automatically
sequence_with_invalid = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAKX"
features = featurizer.featurize(sequence_with_invalid)  # X is filtered out
```

2. **Empty sequences**:
```python
empty_sequence = ""
features = featurizer.featurize(empty_sequence)  # Returns zeros
```

3. **Memory issues with large datasets**:
```python
# Use batch processing or feature caching
features = batch_featurize(featurizer, large_sequence_list, batch_size=100)
```

## Custom Featurizer Example

```python
from amp_searcher.featurizers.base import BaseFeaturizer
from amp_searcher.featurizers.featurizer_factory import FeaturizerFactory
import numpy as np

@FeaturizerFactory.register("hydrophobicity_featurizer")
class HydrophobicityFeaturizer(BaseFeaturizer):
    """Custom featurizer based on hydrophobicity scales."""
    
    def __init__(self, scale="kyte_doolittle"):
        self.scale = scale
        self.hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
    
    def featurize(self, sequence: str) -> np.ndarray:
        if not sequence:
            return np.zeros(3)
        
        hydro_values = [self.hydrophobicity.get(aa, 0) for aa in sequence.upper()]
        
        return np.array([
            np.mean(hydro_values),      # Average hydrophobicity
            np.std(hydro_values),       # Hydrophobicity variance
            len(sequence)               # Sequence length
        ])

# Usage
featurizer = FeaturizerFactory.build_featurizer("hydrophobicity_featurizer")
features = featurizer.featurize("KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK")
print(f"Custom features: {features}")
```

This comprehensive featurizer system allows you to extract meaningful numerical representations from peptide sequences, enabling effective machine learning for antimicrobial peptide research.
