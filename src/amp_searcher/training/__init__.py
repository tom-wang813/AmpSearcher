"""
Training module for AmpSearcher.

This module provides training utilities including callbacks, schedulers,
and training orchestration for PyTorch Lightning models.
"""

# Training callbacks
from .callbacks import GradientMonitor

# Scheduler factory
from .scheduler_factory import SchedulerFactory

# Main trainer
from .trainer import AmpTrainer as Trainer

__all__ = [
    # Callbacks
    "GradientMonitor",
    
    # Factories
    "SchedulerFactory",
    
    # Main trainer
    "Trainer",
]

# Module-level convenience functions
def create_scheduler(name, optimizer, **kwargs):
    """Create a learning rate scheduler by name.
    
    Args:
        name: Name of the scheduler to create
        optimizer: PyTorch optimizer instance
        **kwargs: Parameters to pass to the scheduler constructor
        
    Returns:
        torch.optim.lr_scheduler: Configured scheduler instance
    """
    return SchedulerFactory.build_scheduler(name, optimizer, **kwargs)

def get_available_schedulers():
    """Get a list of available scheduler names.
    
    Returns:
        list: List of available scheduler names
    """
    return list(SchedulerFactory._registry.keys())

def get_scheduler_info(name):
    """Get information about a specific scheduler.
    
    Args:
        name: Name of the scheduler
        
    Returns:
        dict: Information about the scheduler
    """
    scheduler_info = {
        "StepLR": {
            "description": "Decays learning rate by gamma every step_size epochs",
            "parameters": {
                "step_size": "Period of learning rate decay",
                "gamma": "Multiplicative factor of learning rate decay (default: 0.1)"
            },
            "use_cases": ["Simple step-wise decay", "Milestone-based training"]
        },
        "ExponentialLR": {
            "description": "Decays learning rate by gamma every epoch",
            "parameters": {
                "gamma": "Multiplicative factor of learning rate decay"
            },
            "use_cases": ["Smooth exponential decay", "Long training runs"]
        },
        "CosineAnnealingLR": {
            "description": "Cosine annealing learning rate schedule",
            "parameters": {
                "T_max": "Maximum number of iterations",
                "eta_min": "Minimum learning rate (default: 0)"
            },
            "use_cases": ["Smooth decay with restarts", "Fine-tuning"]
        },
        "ReduceLROnPlateau": {
            "description": "Reduces learning rate when metric has stopped improving",
            "parameters": {
                "mode": "min or max (default: min)",
                "factor": "Factor by which learning rate is reduced (default: 0.1)",
                "patience": "Number of epochs with no improvement (default: 10)",
                "threshold": "Threshold for measuring improvement (default: 1e-4)"
            },
            "use_cases": ["Adaptive learning rate", "Plateau detection"]
        },
        "CyclicLR": {
            "description": "Cyclical learning rate policy",
            "parameters": {
                "base_lr": "Lower learning rate boundary",
                "max_lr": "Upper learning rate boundary",
                "step_size_up": "Number of training iterations in increasing half",
                "mode": "triangular, triangular2, or exp_range (default: triangular)"
            },
            "use_cases": ["Cyclical training", "Finding optimal learning rates"]
        }
    }
    
    return scheduler_info.get(name, {"description": "Unknown scheduler"})

# Scheduler recommendations based on training scenario
SCHEDULER_RECOMMENDATIONS = {
    "short_training": ["StepLR", "ExponentialLR"],
    "long_training": ["CosineAnnealingLR", "ReduceLROnPlateau"],
    "fine_tuning": ["CosineAnnealingLR", "ReduceLROnPlateau"],
    "hyperparameter_search": ["CyclicLR"],
    "stable_training": ["StepLR"],
    "adaptive_training": ["ReduceLROnPlateau"],
    "smooth_decay": ["CosineAnnealingLR", "ExponentialLR"]
}

def get_recommended_schedulers(training_scenario):
    """Get recommended schedulers for a training scenario.
    
    Args:
        training_scenario: Description of the training scenario
        
    Returns:
        list: List of recommended scheduler names
    """
    return SCHEDULER_RECOMMENDATIONS.get(training_scenario, [])

# Default scheduler configurations
DEFAULT_SCHEDULER_CONFIGS = {
    "step_lr_default": {
        "name": "StepLR",
        "params": {
            "step_size": 10,
            "gamma": 0.1
        }
    },
    "cosine_annealing_default": {
        "name": "CosineAnnealingLR",
        "params": {
            "T_max": 100,
            "eta_min": 1e-6
        }
    },
    "reduce_on_plateau_default": {
        "name": "ReduceLROnPlateau",
        "params": {
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "threshold": 1e-4
        }
    },
    "exponential_default": {
        "name": "ExponentialLR",
        "params": {
            "gamma": 0.95
        }
    }
}

def get_default_scheduler_config(config_name):
    """Get a default configuration for a scheduler.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_SCHEDULER_CONFIGS.get(config_name, {})

# Training configuration templates
TRAINING_CONFIGS = {
    "screening_basic": {
        "max_epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "scheduler_config": {
            "name": "StepLR",
            "params": {"step_size": 20, "gamma": 0.1}
        },
        "callbacks": ["GradientMonitor"],
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 10,
            "mode": "min"
        }
    },
    "screening_advanced": {
        "max_epochs": 200,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "scheduler_config": {
            "name": "CosineAnnealingLR",
            "params": {"T_max": 200, "eta_min": 1e-6}
        },
        "callbacks": ["GradientMonitor"],
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 20,
            "mode": "min"
        }
    },
    "generative_basic": {
        "max_epochs": 150,
        "batch_size": 32,
        "learning_rate": 0.001,
        "scheduler_config": {
            "name": "ReduceLROnPlateau",
            "params": {"mode": "min", "factor": 0.5, "patience": 10}
        },
        "callbacks": ["GradientMonitor"],
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 15,
            "mode": "min"
        }
    },
    "contrastive_basic": {
        "max_epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.0003,
        "scheduler_config": {
            "name": "CosineAnnealingLR",
            "params": {"T_max": 100, "eta_min": 1e-5}
        },
        "callbacks": ["GradientMonitor"],
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 15,
            "mode": "min"
        }
    }
}

def get_training_config(config_name):
    """Get a complete training configuration.
    
    Args:
        config_name: Name of the training configuration
        
    Returns:
        dict: Complete training configuration
    """
    return TRAINING_CONFIGS.get(config_name, {})

def create_trainer(config_name=None, **kwargs):
    """Create a trainer with specified or default configuration.
    
    Args:
        config_name: Name of predefined configuration (optional)
        **kwargs: Additional trainer parameters
        
    Returns:
        Trainer: Configured trainer instance
    """
    if config_name:
        config = get_training_config(config_name)
        config.update(kwargs)
    else:
        config = kwargs
    
    return Trainer(**config)

def setup_training_environment(model, datamodule, config_name="screening_basic", 
                              **trainer_kwargs):
    """Set up a complete training environment.
    
    Args:
        model: PyTorch Lightning module
        datamodule: PyTorch Lightning data module
        config_name: Name of training configuration
        **trainer_kwargs: Additional trainer arguments
        
    Returns:
        tuple: (trainer, callbacks, scheduler_config)
    """
    # Get training configuration
    config = get_training_config(config_name)
    
    # Create trainer
    trainer_config = {
        "max_epochs": config.get("max_epochs", 100),
        **trainer_kwargs
    }
    trainer = Trainer(**trainer_config)
    
    # Set up callbacks
    callbacks = []
    if "GradientMonitor" in config.get("callbacks", []):
        callbacks.append(GradientMonitor())
    
    # Get scheduler configuration
    scheduler_config = config.get("scheduler_config", {})
    
    return trainer, callbacks, scheduler_config

# Training utilities
def log_training_info(model, datamodule, config):
    """Log information about the training setup.
    
    Args:
        model: PyTorch Lightning module
        datamodule: PyTorch Lightning data module  
        config: Training configuration
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    logger.info("Training Setup Information:")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Data Module: {datamodule.__class__.__name__}")
    logger.info(f"Max Epochs: {config.get('max_epochs', 'Not specified')}")
    logger.info(f"Batch Size: {config.get('batch_size', 'Not specified')}")
    logger.info(f"Learning Rate: {config.get('learning_rate', 'Not specified')}")
    logger.info(f"Scheduler: {config.get('scheduler_config', {}).get('name', 'None')}")

def validate_training_config(config):
    """Validate a training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    required_fields = ["max_epochs", "learning_rate"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate values
    if "max_epochs" in config and config["max_epochs"] <= 0:
        errors.append("max_epochs must be positive")
    
    if "learning_rate" in config and config["learning_rate"] <= 0:
        errors.append("learning_rate must be positive")
    
    if "batch_size" in config and config["batch_size"] <= 0:
        errors.append("batch_size must be positive")
    
    return len(errors) == 0, errors
