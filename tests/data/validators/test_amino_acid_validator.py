import pytest
from amp_searcher.data.validators.amino_acid_validator import AminoAcidValidator

def test_amino_acid_validator_valid_sequences():
    validator = AminoAcidValidator()
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "GALA", "PEPTIDE"]
    is_valid, errors = validator.validate(sequences)
    assert is_valid is True
    assert not errors

def test_amino_acid_validator_invalid_sequences():
    validator = AminoAcidValidator()
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "GALAX", "PEPTIDE_INVALID"]
    is_valid, errors = validator.validate(sequences)
    assert is_valid is False
    assert "Sequence at index 1 contains invalid amino acids: GALAX" in errors
    assert "Sequence at index 2 contains invalid amino acids: PEPTIDE_INVALID" in errors

def test_amino_acid_validator_empty_sequence():
    validator = AminoAcidValidator()
    sequences = ["", "ACD"]
    is_valid, errors = validator.validate(sequences)
    assert is_valid is False
    assert "Sequence at index 0 contains invalid amino acids: " in errors

def test_amino_acid_validator_case_insensitivity():
    validator = AminoAcidValidator()
    sequences = ["gala", "peptide"]
    is_valid, errors = validator.validate(sequences)
    assert is_valid is False
    assert "Sequence at index 0 contains invalid amino acids: gala" in errors
    assert "Sequence at index 1 contains invalid amino acids: peptide" in errors

def test_amino_acid_validator_mixed_case_and_invalid():
    validator = AminoAcidValidator()
    sequences = ["ACD", "gAlA", "INV@LID"]
    is_valid, errors = validator.validate(sequences)
    assert is_valid is False
    assert "Sequence at index 1 contains invalid amino acids: gAlA" in errors
    assert "Sequence at index 2 contains invalid amino acids: INV@LID" in errors
