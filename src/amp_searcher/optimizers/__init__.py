"""
Optimizers module for AmpSearcher.

This module provides optimization algorithms for peptide design and discovery,
including genetic algorithms, Monte Carlo tree search, and sequential Monte Carlo.
"""

# Base optimizer
from .base import BaseOptimizer

# Factory for creating optimizers
from .optimizer_factory import OptimizerFactory

# Specific optimizers
from .ga import GeneticAlgorithmOptimizer
from .mcts import MCTSOptimizer
from .smc import SMCOptimizer

# Selection strategies
from .selection import (
    NSGA2Selection,
    ElitistSelection,
)

# Import modules to trigger registration
from . import ga, mcts, smc

__all__ = [
    # Base class
    "BaseOptimizer",
    
    # Factory
    "OptimizerFactory",
    
    # Specific optimizers
    "GeneticAlgorithmOptimizer",
    "MCTSOptimizer",
    "SMCOptimizer",
    
    # Selection strategies
    "NSGA2Selection",
    "ElitistSelection",
    
    # Modules (for registration)
    "ga",
    "mcts", 
    "smc",
]

# Module-level convenience functions
def create_optimizer(name, **kwargs):
    """Create an optimizer by name with given parameters.
    
    Args:
        name: Name of the optimizer to create
        **kwargs: Parameters to pass to the optimizer constructor
        
    Returns:
        BaseOptimizer: Configured optimizer instance
    """
    return OptimizerFactory.build_optimizer(name, **kwargs)

def get_available_optimizers():
    """Get a list of available optimizer names.
    
    Returns:
        list: List of available optimizer names
    """
    return list(OptimizerFactory._registry.keys())

def get_optimizer_info(name):
    """Get information about a specific optimizer.
    
    Args:
        name: Name of the optimizer
        
    Returns:
        dict: Information about the optimizer including description and parameters
    """
    optimizer_info = {
        "GeneticAlgorithm": {
            "description": "Genetic algorithm for sequence optimization",
            "use_cases": ["Global optimization", "Multi-objective optimization"],
            "parameters": {
                "population_size": "Size of the population (default: 100)",
                "mutation_rate": "Probability of mutation (default: 0.1)",
                "crossover_rate": "Probability of crossover (default: 0.8)",
                "selection_strategy": "Selection method (default: 'tournament')",
                "elitism_rate": "Fraction of elite individuals to preserve (default: 0.1)",
                "max_generations": "Maximum number of generations (default: 100)"
            },
            "strengths": [
                "Good for global optimization",
                "Handles discrete search spaces well",
                "Can maintain population diversity"
            ],
            "limitations": [
                "Can be slow to converge",
                "Requires tuning of hyperparameters"
            ]
        },
        "MonteCarloTreeSearch": {
            "description": "Monte Carlo tree search for sequential decision making",
            "use_cases": ["Sequential optimization", "Tree-structured search"],
            "parameters": {
                "num_simulations": "Number of MCTS simulations (default: 1000)",
                "exploration_constant": "UCB exploration parameter (default: 1.414)",
                "max_depth": "Maximum tree depth (default: 10)",
                "rollout_policy": "Policy for rollout phase (default: 'random')"
            },
            "strengths": [
                "Good for sequential decision problems",
                "Balances exploration and exploitation",
                "Can handle large search spaces"
            ],
            "limitations": [
                "Requires domain-specific rollout policies",
                "Memory intensive for large trees"
            ]
        },
        "SequentialMonteCarlo": {
            "description": "Sequential Monte Carlo for Bayesian optimization",
            "use_cases": ["Bayesian optimization", "Uncertainty quantification"],
            "parameters": {
                "num_particles": "Number of particles (default: 100)",
                "resampling_threshold": "Threshold for resampling (default: 0.5)",
                "proposal_distribution": "Proposal distribution type (default: 'gaussian')",
                "acquisition_function": "Acquisition function (default: 'expected_improvement')"
            },
            "strengths": [
                "Provides uncertainty estimates",
                "Sample efficient",
                "Good for expensive function evaluations"
            ],
            "limitations": [
                "Requires probabilistic models",
                "Can be computationally intensive"
            ]
        }
    }
    
    return optimizer_info.get(name, {"description": "Unknown optimizer"})

# Optimizer recommendations based on problem characteristics
OPTIMIZER_RECOMMENDATIONS = {
    "small_search_space": ["GeneticAlgorithm"],
    "large_search_space": ["MonteCarloTreeSearch", "SequentialMonteCarlo"],
    "expensive_evaluation": ["SequentialMonteCarlo"],
    "multi_objective": ["GeneticAlgorithm"],
    "sequential_design": ["MonteCarloTreeSearch"],
    "uncertainty_important": ["SequentialMonteCarlo"],
    "discrete_space": ["GeneticAlgorithm", "MonteCarloTreeSearch"],
    "continuous_space": ["SequentialMonteCarlo"],
    "fast_evaluation": ["GeneticAlgorithm", "MonteCarloTreeSearch"]
}

def get_recommended_optimizers(problem_characteristics):
    """Get recommended optimizers based on problem characteristics.
    
    Args:
        problem_characteristics: List of problem characteristics
        
    Returns:
        list: List of recommended optimizer names
    """
    recommendations = set()
    
    for characteristic in problem_characteristics:
        if characteristic in OPTIMIZER_RECOMMENDATIONS:
            recommendations.update(OPTIMIZER_RECOMMENDATIONS[characteristic])
    
    return list(recommendations)

# Default configurations for common use cases
DEFAULT_OPTIMIZER_CONFIGS = {
    "ga_default": {
        "name": "GeneticAlgorithm",
        "params": {
            "population_size": 100,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "selection_strategy": "tournament",
            "elitism_rate": 0.1,
            "max_generations": 100
        }
    },
    "ga_fast": {
        "name": "GeneticAlgorithm", 
        "params": {
            "population_size": 50,
            "mutation_rate": 0.15,
            "crossover_rate": 0.9,
            "selection_strategy": "tournament",
            "elitism_rate": 0.2,
            "max_generations": 50
        }
    },
    "mcts_default": {
        "name": "MonteCarloTreeSearch",
        "params": {
            "num_simulations": 1000,
            "exploration_constant": 1.414,
            "max_depth": 10,
            "rollout_policy": "random"
        }
    },
    "smc_default": {
        "name": "SequentialMonteCarlo",
        "params": {
            "num_particles": 100,
            "resampling_threshold": 0.5,
            "proposal_distribution": "gaussian",
            "acquisition_function": "expected_improvement"
        }
    }
}

def get_default_optimizer_config(config_name):
    """Get a default configuration for an optimizer.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_OPTIMIZER_CONFIGS.get(config_name, {})

def optimize_sequences(sequences, objective_function, optimizer_config, 
                      num_iterations=100):
    """High-level function to optimize sequences using specified optimizer.
    
    Args:
        sequences: Initial sequences to optimize
        objective_function: Function to evaluate sequence quality
        optimizer_config: Optimizer configuration dictionary
        num_iterations: Number of optimization iterations
        
    Returns:
        tuple: (best_sequences, optimization_history)
    """
    # Create optimizer
    optimizer_name = optimizer_config.pop("name")
    optimizer = create_optimizer(optimizer_name, **optimizer_config)
    
    # Initialize with sequences
    optimizer.initialize(sequences)
    
    # Run optimization
    history = []
    for i in range(num_iterations):
        # Get current population/candidates
        candidates = optimizer.get_candidates()
        
        # Evaluate candidates
        scores = [objective_function(seq) for seq in candidates]
        
        # Update optimizer with scores
        optimizer.update(candidates, scores)
        
        # Record best score
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        history.append({
            "iteration": i,
            "best_score": scores[best_idx],
            "best_sequence": candidates[best_idx],
            "mean_score": sum(scores) / len(scores)
        })
    
    # Get final best sequences
    final_candidates = optimizer.get_candidates()
    final_scores = [objective_function(seq) for seq in final_candidates]
    
    # Sort by score and return top sequences
    sorted_pairs = sorted(zip(final_candidates, final_scores), 
                         key=lambda x: x[1], reverse=True)
    best_sequences = [seq for seq, score in sorted_pairs]
    
    return best_sequences, history
