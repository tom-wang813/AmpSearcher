"""
Models module for AmpSearcher.

This module provides neural network architectures and PyTorch Lightning modules
for various tasks including screening, contrastive learning, and generative modeling.
"""

# Base classes
from .base import BaseLightningModule

# Lightning modules for different tasks
from .screening.lightning_module import ScreeningLightningModule
from .generative.lightning_module import GenerativeLightningModule
from .contrastive.lightning_module import ContrastiveLightningModule

# Factories
from .lightning_module_factory import LightningModuleFactory
from .model_factory import ModelFactory

# Core architectures
from .architectures.feed_forward_nn import FeedForwardNeuralNetwork
from .architectures.advanced.transformer_encoder import TransformerEncoder

# Screening architectures
from .screening.architectures.ffnn import FFNN

# Generative architectures
from .generative.architectures.vae import VAE
from .generative.architectures.vae_encoder import VAEEncoder
from .generative.architectures.vae_decoder import VAEDecoder

# Contrastive architectures
from .contrastive.architectures.simclr_backbone import SimCLRBackbone

__all__ = [
    # Base classes
    "BaseLightningModule",
    
    # Lightning modules
    "ScreeningLightningModule",
    "GenerativeLightningModule", 
    "ContrastiveLightningModule",
    
    # Factories
    "LightningModuleFactory",
    "ModelFactory",
    
    # Core architectures
    "FeedForwardNeuralNetwork",
    "TransformerEncoder",
    
    # Screening architectures
    "FFNN",
    
    # Generative architectures
    "VAE",
    "VAEEncoder",
    "VAEDecoder",
    
    # Contrastive architectures
    "SimCLRBackbone",
]

# Module-level convenience functions
def create_model(model_type, architecture_name, architecture_params, 
                lightning_module_params=None):
    """Create a complete model with Lightning module.
    
    Args:
        model_type: Type of model ('screening', 'generative', 'contrastive')
        architecture_name: Name of the architecture
        architecture_params: Parameters for the architecture
        lightning_module_params: Parameters for the Lightning module
        
    Returns:
        BaseLightningModule: Configured Lightning module with architecture
    """
    # Create architecture
    architecture = ModelFactory.build_model(architecture_name, **architecture_params)
    
    # Create Lightning module
    lightning_params = lightning_module_params or {}
    lightning_module = LightningModuleFactory.build_lightning_module(
        model_type, model=architecture, **lightning_params
    )
    
    return lightning_module

def get_available_architectures():
    """Get available architecture names.
    
    Returns:
        dict: Dictionary mapping model types to available architectures
    """
    return {
        "screening": ["FFNN", "TransformerEncoder"],
        "generative": ["VAE"],
        "contrastive": ["SimCLRBackbone"],
        "core": ["FeedForwardNN", "TransformerEncoder"]
    }

def get_model_info(architecture_name):
    """Get information about a specific model architecture.
    
    Args:
        architecture_name: Name of the architecture
        
    Returns:
        dict: Information about the architecture
    """
    model_info = {
        "FFNN": {
            "description": "Feed-forward neural network for screening tasks",
            "task_type": "screening",
            "parameters": {
                "input_dim": "Input feature dimension",
                "output_dim": "Output dimension (1 for binary classification)",
                "hidden_dims": "List of hidden layer dimensions",
                "dropout_rate": "Dropout rate (default: 0.0)",
                "activation": "Activation function (default: 'relu')"
            }
        },
        "TransformerEncoder": {
            "description": "Transformer encoder for sequence modeling",
            "task_type": "screening",
            "parameters": {
                "vocab_size": "Vocabulary size",
                "d_model": "Model dimension",
                "nhead": "Number of attention heads",
                "num_layers": "Number of transformer layers",
                "dim_feedforward": "Feedforward dimension",
                "max_seq_len": "Maximum sequence length",
                "num_classes": "Number of output classes"
            }
        },
        "VAE": {
            "description": "Variational autoencoder for generative modeling",
            "task_type": "generative",
            "parameters": {
                "input_dim": "Input feature dimension",
                "latent_dim": "Latent space dimension",
                "hidden_dims": "List of hidden layer dimensions",
                "beta": "KL divergence weight (default: 1.0)"
            }
        },
        "SimCLRBackbone": {
            "description": "SimCLR backbone for contrastive learning",
            "task_type": "contrastive",
            "parameters": {
                "input_dim": "Input feature dimension",
                "hidden_dims": "List of hidden layer dimensions",
                "projection_dim": "Projection head dimension",
                "dropout_rate": "Dropout rate (default: 0.0)"
            }
        }
    }
    
    return model_info.get(architecture_name, {"description": "Unknown architecture"})

# Model recommendations based on task and data
MODEL_RECOMMENDATIONS = {
    "binary_classification": {
        "small_dataset": ["FFNN"],
        "large_dataset": ["FFNN", "TransformerEncoder"],
        "sequence_data": ["TransformerEncoder"],
        "feature_data": ["FFNN"]
    },
    "multi_class_classification": {
        "small_dataset": ["FFNN"],
        "large_dataset": ["FFNN", "TransformerEncoder"],
        "sequence_data": ["TransformerEncoder"],
        "feature_data": ["FFNN"]
    },
    "sequence_generation": {
        "any": ["VAE"]
    },
    "representation_learning": {
        "any": ["SimCLRBackbone"]
    }
}

def get_recommended_models(task_type, data_characteristics=None):
    """Get recommended models for a specific task and data type.
    
    Args:
        task_type: The task type (e.g., 'binary_classification')
        data_characteristics: Characteristics of the data (e.g., 'small_dataset')
        
    Returns:
        list: List of recommended model names
    """
    if task_type not in MODEL_RECOMMENDATIONS:
        return []
    
    if data_characteristics and data_characteristics in MODEL_RECOMMENDATIONS[task_type]:
        return MODEL_RECOMMENDATIONS[task_type][data_characteristics]
    
    # Return all recommendations for the task
    all_recommendations = []
    for recommendations in MODEL_RECOMMENDATIONS[task_type].values():
        all_recommendations.extend(recommendations)
    
    return list(set(all_recommendations))

# Default configurations for common use cases
DEFAULT_CONFIGS = {
    "screening_ffnn_small": {
        "architecture": {
            "name": "FFNN",
            "params": {
                "input_dim": 10,
                "output_dim": 1,
                "hidden_dims": [64, 32],
                "dropout_rate": 0.2
            }
        },
        "lightning_module_params": {
            "task_type": "classification",
            "optimizer_params": {"lr": 0.001},
            "scheduler_params": {"name": "StepLR", "step_size": 10, "gamma": 0.1}
        }
    },
    "screening_transformer": {
        "architecture": {
            "name": "TransformerEncoder",
            "params": {
                "vocab_size": 22,
                "d_model": 128,
                "nhead": 8,
                "num_layers": 4,
                "dim_feedforward": 512,
                "max_seq_len": 50,
                "num_classes": 1
            }
        },
        "lightning_module_params": {
            "task_type": "classification",
            "optimizer_params": {"lr": 0.0001},
            "scheduler_params": {"name": "CosineAnnealingLR", "T_max": 100}
        }
    },
    "generative_vae": {
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
    }
}

def get_default_config(config_name):
    """Get a default configuration for common use cases.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_CONFIGS.get(config_name, {})
