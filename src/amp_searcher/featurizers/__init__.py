"""
Featurizers module for AmpSearcher.

This module provides various featurization methods for converting peptide sequences
into numerical representations suitable for machine learning models.
"""

# Base featurizer
from .base import BaseFeaturizer

# Specific featurizers
from .physicochemical import PhysicochemicalFeaturizer
from .composition import CompositionFeaturizer
from .pse_aac import PseAACFeaturizer
from .simple_sequence import SimpleSequenceFeaturizer

# Factory for creating featurizers
from .featurizer_factory import FeaturizerFactory

__all__ = [
    # Base class
    "BaseFeaturizer",
    
    # Specific featurizers
    "PhysicochemicalFeaturizer",
    "CompositionFeaturizer",
    "PseAACFeaturizer",
    "SimpleSequenceFeaturizer",
    
    # Factory
    "FeaturizerFactory",
]

# Module-level convenience functions
def create_featurizer(name, **kwargs):
    """Create a featurizer by name with given parameters.
    
    Args:
        name: Name of the featurizer to create
        **kwargs: Parameters to pass to the featurizer constructor
        
    Returns:
        BaseFeaturizer: Configured featurizer instance
    """
    return FeaturizerFactory.build_featurizer(name, **kwargs)

def get_available_featurizers():
    """Get a list of available featurizer names.
    
    Returns:
        list: List of available featurizer names
    """
    return list(FeaturizerFactory._registry.keys())

def featurize_sequences(sequences, featurizer_config):
    """Featurize a list of sequences using specified configuration.
    
    Args:
        sequences: List of peptide sequences
        featurizer_config: Dictionary with featurizer configuration
                          Must contain 'name' key and optional parameters
        
    Returns:
        numpy.ndarray: Featurized sequences
    """
    import numpy as np
    
    # Create featurizer
    featurizer_name = featurizer_config.pop("name")
    featurizer = create_featurizer(featurizer_name, **featurizer_config)
    
    # Featurize sequences
    features = []
    for seq in sequences:
        feature_vector = featurizer.featurize(seq)
        features.append(feature_vector)
    
    return np.array(features)

def get_featurizer_info(name):
    """Get information about a specific featurizer.
    
    Args:
        name: Name of the featurizer
        
    Returns:
        dict: Information about the featurizer including description and parameters
    """
    featurizer_info = {
        "PhysicochemicalFeaturizer": {
            "description": "Calculates physicochemical properties using BioPython",
            "output_dim": "Variable (depends on selected features)",
            "default_features": [
                "length", "molecular_weight", "charge_at_ph_7", 
                "isoelectric_point", "aromaticity", "instability_index",
                "gravy", "helix_fraction", "turn_fraction", "sheet_fraction"
            ],
            "parameters": {
                "custom_features": "List of specific features to calculate"
            }
        },
        "CompositionFeaturizer": {
            "description": "Calculates amino acid and dipeptide composition",
            "output_dim": "20 (AAC only) or 420 (AAC + DPC)",
            "parameters": {
                "include_aac": "Include amino acid composition (default: True)",
                "include_dpc": "Include dipeptide composition (default: False)"
            }
        },
        "PseAACFeaturizer": {
            "description": "Pseudo amino acid composition with physicochemical properties",
            "output_dim": "20 + lambda_val",
            "parameters": {
                "lambda_val": "Number of pseudo components (default: 10)",
                "w": "Weight for pseudo components (default: 0.1)"
            }
        },
        "SimpleSequenceFeaturizer": {
            "description": "Converts sequences to integer tokens for neural networks",
            "output_dim": "max_len",
            "parameters": {
                "max_len": "Maximum sequence length (default: 50)",
                "vocab": "Vocabulary mapping amino acids to integers"
            }
        }
    }
    
    return featurizer_info.get(name, {"description": "Unknown featurizer"})

# Featurizer recommendations based on use case
FEATURIZER_RECOMMENDATIONS = {
    "binary_classification": [
        "PhysicochemicalFeaturizer",
        "CompositionFeaturizer"
    ],
    "multi_class_classification": [
        "PseAACFeaturizer",
        "CompositionFeaturizer"
    ],
    "sequence_generation": [
        "SimpleSequenceFeaturizer"
    ],
    "similarity_search": [
        "CompositionFeaturizer",
        "PhysicochemicalFeaturizer"
    ],
    "interpretable_models": [
        "PhysicochemicalFeaturizer"
    ],
    "deep_learning": [
        "SimpleSequenceFeaturizer",
        "CompositionFeaturizer"
    ]
}

def get_recommended_featurizers(use_case):
    """Get recommended featurizers for a specific use case.
    
    Args:
        use_case: The intended use case (e.g., 'binary_classification')
        
    Returns:
        list: List of recommended featurizer names
    """
    return FEATURIZER_RECOMMENDATIONS.get(use_case, [])
