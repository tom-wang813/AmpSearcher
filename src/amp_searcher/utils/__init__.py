"""
Utilities module for AmpSearcher.

This module provides utility functions, constants, and helper classes
used throughout the AmpSearcher package.
"""

# Constants
from . import constants

# Sequence decoders
from .sequence_decoder import BaseSequenceDecoder, SimpleFeatureDecoder
from .sequence_decoder_factory import SequenceDecoderFactory

# Configuration utilities
from .config import Config


# Logging utilities
from .logging_utils import setup_logger

# Performance monitoring
from .performance_monitoring import MemoryTracker

# Oracle utilities
from .oracle import Oracle

__all__ = [
    # Constants
    "constants",
    
    # Sequence decoders
    "BaseSequenceDecoder",
    "SimpleFeatureDecoder", 
    "SequenceDecoderFactory",
    
    # Configuration utilities
    "Config",

    # Logging utilities
    "setup_logger",
    
    # Performance monitoring
    "MemoryTracker",
    
    # Oracle utilities
    "Oracle",
]

# Module-level convenience functions
def create_sequence_decoder(name, **kwargs):
    """Create a sequence decoder by name.
    
    Args:
        name: Name of the decoder to create
        **kwargs: Parameters to pass to the decoder constructor
        
    Returns:
        BaseSequenceDecoder: Configured decoder instance
    """
    return SequenceDecoderFactory.build_decoder(name, **kwargs)

def get_available_decoders():
    """Get a list of available decoder names.
    
    Returns:
        list: List of available decoder names
    """
    return list(SequenceDecoderFactory._registry.keys())

def get_decoder_info(name):
    """Get information about a specific decoder.
    
    Args:
        name: Name of the decoder
        
    Returns:
        dict: Information about the decoder
    """
    decoder_info = {
        "SimpleFeatureDecoder": {
            "description": "Decodes feature vectors back to interpretable representations",
            "use_cases": [
                "Feature interpretation",
                "Reverse engineering features",
                "Debugging feature extraction"
            ],
            "parameters": {
                "feature_names": "List of feature names",
                "feature_ranges": "Expected ranges for each feature"
            }
        }
    }
    
    return decoder_info.get(name, {"description": "Unknown decoder"})

# Utility functions for common tasks
def validate_sequence(sequence, allowed_amino_acids=None):
    """Validate a peptide sequence.
    
    Args:
        sequence: Peptide sequence string
        allowed_amino_acids: Set of allowed amino acids (default: standard 20)
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if allowed_amino_acids is None:
        allowed_amino_acids = set(constants.STANDARD_AMINO_ACIDS)
    
    if not sequence:
        return False, "Empty sequence"
    
    if not isinstance(sequence, str):
        return False, "Sequence must be a string"
    
    # Check for invalid characters
    invalid_chars = set(sequence.upper()) - allowed_amino_acids
    if invalid_chars:
        return False, f"Invalid amino acids: {', '.join(invalid_chars)}"
    
    # Check length constraints
    if len(sequence) < constants.MIN_SEQUENCE_LENGTH:
        return False, f"Sequence too short (minimum: {constants.MIN_SEQUENCE_LENGTH})"
    
    if len(sequence) > constants.MAX_SEQUENCE_LENGTH:
        return False, f"Sequence too long (maximum: {constants.MAX_SEQUENCE_LENGTH})"
    
    return True, "Valid sequence"

def normalize_sequence(sequence):
    """Normalize a peptide sequence.
    
    Args:
        sequence: Peptide sequence string
        
    Returns:
        str: Normalized sequence (uppercase, stripped)
    """
    if not isinstance(sequence, str):
        raise ValueError("Sequence must be a string")
    
    return sequence.strip().upper()

def calculate_sequence_properties(sequence):
    """Calculate basic properties of a peptide sequence.
    
    Args:
        sequence: Peptide sequence string
        
    Returns:
        dict: Dictionary of sequence properties
    """
    sequence = normalize_sequence(sequence)
    
    # Basic properties
    properties = {
        "length": len(sequence),
        "molecular_weight": 0.0,  # Would need BioPython for accurate calculation
        "amino_acid_counts": {},
        "amino_acid_frequencies": {}
    }
    
    # Count amino acids
    for aa in sequence:
        properties["amino_acid_counts"][aa] = properties["amino_acid_counts"].get(aa, 0) + 1
    
    # Calculate frequencies
    for aa, count in properties["amino_acid_counts"].items():
        properties["amino_acid_frequencies"][aa] = count / len(sequence)
    
    return properties

def format_sequence_for_display(sequence, line_length=60):
    """Format a sequence for display with line breaks.
    
    Args:
        sequence: Peptide sequence string
        line_length: Number of characters per line
        
    Returns:
        str: Formatted sequence with line breaks
    """
    sequence = normalize_sequence(sequence)
    
    lines = []
    for i in range(0, len(sequence), line_length):
        lines.append(sequence[i:i + line_length])
    
    return '\n'.join(lines)

def batch_process_sequences(sequences, process_func, batch_size=100, 
                           show_progress=True):
    """Process sequences in batches with optional progress tracking.
    
    Args:
        sequences: List of sequences to process
        process_func: Function to apply to each sequence
        batch_size: Number of sequences per batch
        show_progress: Whether to show progress bar
        
    Returns:
        list: Results from processing each sequence
    """
    results = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_results = [process_func(seq) for seq in batch]
        results.extend(batch_results)
        
        if show_progress:
            batch_num = i // batch_size + 1
            print(f"Processed batch {batch_num}/{total_batches}")
    
    return results

def save_sequences_to_fasta(sequences, output_path, sequence_ids=None):
    """Save sequences to a FASTA file.
    
    Args:
        sequences: List of peptide sequences
        output_path: Path to output FASTA file
        sequence_ids: List of sequence IDs (optional)
    """
    if sequence_ids is None:
        sequence_ids = [f"seq_{i+1}" for i in range(len(sequences))]
    
    if len(sequences) != len(sequence_ids):
        raise ValueError("Number of sequences and IDs must match")
    
    with open(output_path, 'w') as f:
        for seq_id, sequence in zip(sequence_ids, sequences):
            f.write(f">{seq_id}\n")
            f.write(f"{format_sequence_for_display(sequence)}\n")

def load_sequences_from_fasta(fasta_path):
    """Load sequences from a FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        tuple: (sequences, sequence_ids)
    """
    sequences = []
    sequence_ids = []
    current_sequence = ""
    current_id = ""
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_sequence:
                    sequences.append(current_sequence)
                    sequence_ids.append(current_id)
                
                # Start new sequence
                current_id = line[1:]  # Remove '>'
                current_sequence = ""
            else:
                current_sequence += line
        
        # Save last sequence
        if current_sequence:
            sequences.append(current_sequence)
            sequence_ids.append(current_id)
    
    return sequences, sequence_ids

# Performance utilities
def time_function(func):
    """Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        function: Wrapped function that prints execution time
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper

def memory_usage():
    """Get current memory usage.
    
    Returns:
        dict: Memory usage information
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / 1024 / 1024,  # MB
        "vms": memory_info.vms / 1024 / 1024,  # MB
        "percent": process.memory_percent()
    }

# Configuration helpers
def get_default_config():
    """Get default configuration for AmpSearcher.
    
    Returns:
        dict: Default configuration
    """
    return {
        "data": {
            "sequence_column": "sequence",
            "label_column": "label",
            "max_sequence_length": constants.MAX_SEQUENCE_LENGTH,
            "min_sequence_length": constants.MIN_SEQUENCE_LENGTH
        },
        "featurizer": {
            "name": "PhysicochemicalFeaturizer",
            "params": {}
        },
        "model": {
            "architecture": "FFNN",
            "input_dim": 10,
            "output_dim": 1,
            "hidden_dims": [64, 32]
        },
        "training": {
            "max_epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "optimizer": {
            "name": "GeneticAlgorithm",
            "population_size": 100,
            "max_generations": 50
        }
    }

def validate_config(config):
    """Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check required sections
    required_sections = ["data", "model"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate data section
    if "data" in config:
        data_config = config["data"]
        if "sequence_column" not in data_config:
            errors.append("Missing sequence_column in data config")
    
    # Validate model section
    if "model" in config:
        model_config = config["model"]
        if "architecture" not in model_config:
            errors.append("Missing architecture in model config")
    
    return len(errors) == 0, errors
