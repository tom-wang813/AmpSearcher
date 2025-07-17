"""
Data handling module for AmpSearcher.

This module provides tools for loading, processing, and validating peptide sequence data.
It includes dataset classes, data loaders, processors, and validators.
"""

# Core dataset functionality
from .datasets import AmpDataset, load_sequences_from_file, load_data_from_csv

# Data processors
from .processors import (
    BaseProcessor,
    SequenceProcessor,
    ProcessorFactory,
)

# Data validators
from .validators import (
    BaseValidator,
    AminoAcidValidator,
    MissingValueValidator,
    SequenceLengthValidator,
    ValidatorFactory,
)

# Validation schemas
# from .validators.schemas import (
#     SequenceValidationSchema,
#     DataFrameValidationSchema,
# )

__all__ = [
    # Core dataset functionality
    "AmpDataset",
    "load_sequences_from_file", 
    "load_data_from_csv",
    
    # Data processors
    "BaseProcessor",
    "SequenceProcessor",
    "ProcessorFactory",
    
    # Data validators
    "BaseValidator",
    "AminoAcidValidator",
    "MissingValueValidator", 
    "SequenceLengthValidator",
    "ValidatorFactory",
    
    # Validation schemas
    # "SequenceValidationSchema",
    # "DataFrameValidationSchema",
]

# Module-level convenience functions
def create_amp_dataset(data_path, sequence_col="sequence", label_col=None, 
                      processor_config=None, validation_config=None):
    """Create an AmpDataset with optional processing and validation.
    
    Args:
        data_path: Path to the data file
        sequence_col: Name of the sequence column
        label_col: Name of the label column (optional)
        processor_config: Configuration for data processing
        validation_config: Configuration for data validation
        
    Returns:
        AmpDataset: Configured dataset instance
    """
    return AmpDataset(
        data_path=data_path,
        sequence_col=sequence_col,
        label_col=label_col,
        processor_config=processor_config,
        validation_config=validation_config
    )

def validate_sequences(sequences, min_length=5, max_length=50, 
                      allowed_amino_acids="ACDEFGHIKLMNPQRSTVWY"):
    """Validate a list of peptide sequences.
    
    Args:
        sequences: List of peptide sequences
        min_length: Minimum allowed sequence length
        max_length: Maximum allowed sequence length
        allowed_amino_acids: String of allowed amino acid characters
        
    Returns:
        tuple: (valid_sequences, validation_results)
    """
    # Length validator
    length_validator = SequenceLengthValidator(
        min_length=min_length,
        max_length=max_length
    )
    
    # Amino acid validator
    aa_validator = AminoAcidValidator(
        allowed_amino_acids=set(allowed_amino_acids)
    )
    
    valid_sequences = []
    validation_results = []
    
    for seq in sequences:
        length_result = length_validator.validate(seq)
        aa_result = aa_validator.validate(seq)
        
        is_valid = length_result.is_valid and aa_result.is_valid
        validation_results.append({
            "sequence": seq,
            "is_valid": is_valid,
            "length_check": length_result,
            "amino_acid_check": aa_result
        })
        
        if is_valid:
            valid_sequences.append(seq)
    
    return valid_sequences, validation_results
