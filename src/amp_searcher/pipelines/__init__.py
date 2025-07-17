"""
Pipelines module for AmpSearcher.

This module provides high-level pipelines that combine multiple components
for end-to-end antimicrobial peptide screening and discovery workflows.
"""

# Core pipelines
from .screening_pipeline import ScreeningPipeline
from .search_pipeline import SearchPipeline

__all__ = [
    "ScreeningPipeline",
    "SearchPipeline",
]

# Module-level convenience functions
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

def get_pipeline_info(pipeline_name):
    """Get information about a specific pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        dict: Information about the pipeline
    """
    pipeline_info = {
        "ScreeningPipeline": {
            "description": "Pipeline for screening peptide sequences for antimicrobial activity",
            "use_cases": [
                "Batch prediction of AMP activity",
                "Virtual screening of peptide libraries",
                "Filtering candidate sequences"
            ],
            "components": [
                "Featurizer for sequence encoding",
                "Trained model for prediction",
                "Post-processing for results"
            ],
            "inputs": [
                "List of peptide sequences",
                "FASTA files",
                "CSV files with sequence data"
            ],
            "outputs": [
                "Prediction scores",
                "Binary classifications",
                "Ranked sequence lists"
            ],
            "configuration": {
                "model_config": "Configuration for the prediction model",
                "model_checkpoint_path": "Path to trained model weights",
                "featurizer_config": "Configuration for sequence featurization"
            }
        },
        "SearchPipeline": {
            "description": "Pipeline for discovering novel antimicrobial peptides",
            "use_cases": [
                "De novo peptide design",
                "Lead optimization",
                "Exploration of sequence space"
            ],
            "components": [
                "Generative model for sequence generation",
                "Screening model for activity prediction",
                "Optimizer for guided search",
                "Evaluation metrics"
            ],
            "inputs": [
                "Initial seed sequences (optional)",
                "Design constraints",
                "Optimization objectives"
            ],
            "outputs": [
                "Novel peptide sequences",
                "Predicted activities",
                "Optimization trajectories"
            ],
            "configuration": {
                "generative_model_config": "Configuration for sequence generation",
                "screening_model_config": "Configuration for activity prediction",
                "optimizer_config": "Configuration for optimization algorithm"
            }
        }
    }
    
    return pipeline_info.get(pipeline_name, {"description": "Unknown pipeline"})

# Pipeline recommendations based on use case
PIPELINE_RECOMMENDATIONS = {
    "virtual_screening": {
        "pipeline": "ScreeningPipeline",
        "description": "Screen large libraries of existing peptides",
        "typical_workflow": [
            "Load peptide library",
            "Configure featurizer and model",
            "Run batch predictions",
            "Filter and rank results"
        ]
    },
    "lead_optimization": {
        "pipeline": "SearchPipeline", 
        "description": "Optimize existing peptides for better activity",
        "typical_workflow": [
            "Start with lead compounds",
            "Configure optimization objectives",
            "Run iterative optimization",
            "Validate top candidates"
        ]
    },
    "de_novo_design": {
        "pipeline": "SearchPipeline",
        "description": "Design completely new peptides from scratch",
        "typical_workflow": [
            "Define design constraints",
            "Configure generative model",
            "Run exploration search",
            "Screen generated candidates"
        ]
    },
    "activity_prediction": {
        "pipeline": "ScreeningPipeline",
        "description": "Predict activity of known sequences",
        "typical_workflow": [
            "Prepare sequence data",
            "Load trained model",
            "Generate predictions",
            "Analyze results"
        ]
    }
}

def get_recommended_pipeline(use_case):
    """Get recommended pipeline for a specific use case.
    
    Args:
        use_case: The intended use case
        
    Returns:
        dict: Recommendation including pipeline name and workflow
    """
    return PIPELINE_RECOMMENDATIONS.get(use_case, {})

def get_available_use_cases():
    """Get list of available use cases with pipeline recommendations.
    
    Returns:
        list: List of available use case names
    """
    return list(PIPELINE_RECOMMENDATIONS.keys())

# Default pipeline configurations
DEFAULT_PIPELINE_CONFIGS = {
    "screening_basic": {
        "model_config": {
            "architecture": "FFNN",
            "input_dim": 10,
            "output_dim": 1,
            "hidden_dims": [64, 32]
        },
        "featurizer_config": {
            "name": "PhysicochemicalFeaturizer"
        }
    },
    "screening_advanced": {
        "model_config": {
            "architecture": "TransformerEncoder",
            "vocab_size": 22,
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4
        },
        "featurizer_config": {
            "name": "SimpleSequenceFeaturizer",
            "max_len": 50
        }
    },
    "search_basic": {
        "generative_model_config": {
            "architecture": "VAE",
            "input_dim": 10,
            "latent_dim": 16,
            "hidden_dims": [64, 32]
        },
        "screening_model_config": {
            "architecture": "FFNN",
            "input_dim": 10,
            "output_dim": 1,
            "hidden_dims": [64, 32]
        },
        "optimizer_config": {
            "name": "GeneticAlgorithm",
            "population_size": 100,
            "max_generations": 50
        }
    }
}

def get_default_pipeline_config(config_name):
    """Get a default configuration for a pipeline.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_PIPELINE_CONFIGS.get(config_name, {})

def run_screening_workflow(sequences, model_checkpoint_path, 
                          featurizer_config=None, batch_size=32):
    """Run a complete screening workflow on a list of sequences.
    
    Args:
        sequences: List of peptide sequences to screen
        model_checkpoint_path: Path to trained model checkpoint
        featurizer_config: Configuration for featurizer (optional)
        batch_size: Batch size for prediction
        
    Returns:
        dict: Results including predictions and rankings
    """
    # Use default featurizer if not provided
    if featurizer_config is None:
        featurizer_config = {"name": "PhysicochemicalFeaturizer"}
    
    # Create screening pipeline
    pipeline = ScreeningPipeline(
        model_config={},  # Will be loaded from checkpoint
        model_checkpoint_path=model_checkpoint_path,
        featurizer_config=featurizer_config
    )
    
    # Run predictions
    predictions = pipeline.predict(sequences, batch_size=batch_size)
    
    # Create results with rankings
    sequence_scores = list(zip(sequences, predictions))
    sequence_scores.sort(key=lambda x: x[1], reverse=True)
    
    results = {
        "sequences": sequences,
        "predictions": predictions,
        "ranked_sequences": [seq for seq, score in sequence_scores],
        "ranked_scores": [score for seq, score in sequence_scores],
        "top_10_sequences": [seq for seq, score in sequence_scores[:10]],
        "statistics": {
            "total_sequences": len(sequences),
            "mean_score": sum(predictions) / len(predictions),
            "max_score": max(predictions),
            "min_score": min(predictions)
        }
    }
    
    return results

def run_search_workflow(initial_sequences, generative_model_path, 
                       screening_model_path, optimizer_config, 
                       num_iterations=100):
    """Run a complete search workflow for peptide discovery.
    
    Args:
        initial_sequences: Initial seed sequences (optional)
        generative_model_path: Path to generative model checkpoint
        screening_model_path: Path to screening model checkpoint
        optimizer_config: Configuration for optimizer
        num_iterations: Number of search iterations
        
    Returns:
        dict: Results including discovered sequences and optimization history
    """
    # Create search pipeline
    pipeline = SearchPipeline(
        generative_model_config={},  # Will be loaded from checkpoint
        screening_model_config={},   # Will be loaded from checkpoint
        optimizer_config=optimizer_config,
        generative_model_path=generative_model_path,
        screening_model_path=screening_model_path
    )
    
    # Initialize with seed sequences if provided
    if initial_sequences:
        pipeline.initialize(initial_sequences)
    
    # Run search
    results = pipeline.search(num_iterations=num_iterations)
    
    return results
