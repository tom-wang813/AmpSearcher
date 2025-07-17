"""
AmpSearcher: A comprehensive toolkit for antimicrobial peptide discovery and analysis.

This package provides tools for:
- Data loading and preprocessing
- Feature extraction from peptide sequences
- Machine learning model training and inference
- Optimization algorithms for peptide design
- High-level pipelines for screening and discovery workflows
"""

__version__ = "0.1.0"
__author__ = "AmpSearcher Team"
__email__ = "contact@ampsearcher.org"
__license__ = "MIT"

# Core modules
from . import data
from . import featurizers
from . import models
from . import optimizers
from . import pipelines
from . import training
from . import utils

# Main classes and functions for easy access
from .data import AmpDataset, load_sequences_from_file, load_data_from_csv
from .featurizers import (
    BaseFeaturizer,
    PhysicochemicalFeaturizer,
    CompositionFeaturizer,
    PseAACFeaturizer,
    SimpleSequenceFeaturizer,
    FeaturizerFactory,
)
from .models import (
    BaseLightningModule,
    ScreeningLightningModule,
    GenerativeLightningModule,
    ContrastiveLightningModule,
    LightningModuleFactory,
    ModelFactory,
)
from .optimizers import BaseOptimizer, OptimizerFactory
from .pipelines import ScreeningPipeline, SearchPipeline
from .training import GradientMonitor, SchedulerFactory
from .utils import constants, SequenceDecoderFactory

# High-level API functions
def create_screening_pipeline(model_config, model_checkpoint_path, featurizer_config):
    """Create a screening pipeline for AMP prediction.
    
    Args:
        model_config: Model configuration dictionary
        model_checkpoint_path: Path to trained model checkpoint
        featurizer_config: Featurizer configuration dictionary
        
    Returns:
        ScreeningPipeline: Configured screening pipeline
    """
    return ScreeningPipeline(
        model_config=model_config,
        model_checkpoint_path=model_checkpoint_path,
        featurizer_config=featurizer_config
    )

def create_search_pipeline(generative_model_config, screening_model_config, 
                          optimizer_config, **kwargs):
    """Create a search pipeline for novel AMP discovery.
    
    Args:
        generative_model_config: Generative model configuration
        screening_model_config: Screening model configuration
        optimizer_config: Optimizer configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        SearchPipeline: Configured search pipeline
    """
    return SearchPipeline(
        generative_model_config=generative_model_config,
        screening_model_config=screening_model_config,
        optimizer_config=optimizer_config,
        **kwargs
    )

# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Submodules
    "data",
    "featurizers", 
    "models",
    "optimizers",
    "pipelines",
    "training",
    "utils",
    
    # Core classes - Data
    "AmpDataset",
    "load_sequences_from_file",
    "load_data_from_csv",
    
    # Core classes - Featurizers
    "BaseFeaturizer",
    "PhysicochemicalFeaturizer",
    "CompositionFeaturizer", 
    "PseAACFeaturizer",
    "SimpleSequenceFeaturizer",
    "FeaturizerFactory",
    
    # Core classes - Models
    "BaseLightningModule",
    "ScreeningLightningModule",
    "GenerativeLightningModule",
    "ContrastiveLightningModule",
    "LightningModuleFactory",
    "ModelFactory",
    
    # Core classes - Optimizers
    "BaseOptimizer",
    "OptimizerFactory",
    
    # Core classes - Pipelines
    "ScreeningPipeline",
    "SearchPipeline",
    
    # Core classes - Training
    "GradientMonitor",
    "SchedulerFactory",
    
    # Core classes - Utils
    "constants",
    "SequenceDecoderFactory",
    
    # High-level API
    "create_screening_pipeline",
    "create_search_pipeline",
]

# Package configuration
import logging

# Set up package-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Package-level constants
DEFAULT_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_MAX_SEQUENCE_LENGTH = 50
DEFAULT_MIN_SEQUENCE_LENGTH = 5

# Compatibility checks
def _check_dependencies():
    """Check if required dependencies are available."""
    try:
        import torch
        import pytorch_lightning
        import numpy
        import pandas
        import sklearn
        import Bio
    except ImportError as e:
        raise ImportError(
            f"Missing required dependency: {e}. "
            "Please install all required dependencies using: "
            "pip install -r requirements.txt"
        )

# Run dependency check on import
_check_dependencies()
