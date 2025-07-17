import pytest
from amp_searcher.data.validators.missing_value_validator import MissingValueValidator

def test_missing_value_validator_no_missing_values():
    validator = MissingValueValidator()
    data = ["seq1", "seq2", "seq3"]
    is_valid, errors = validator.validate(data)
    assert is_valid is True
    assert not errors

def test_missing_value_validator_with_none():
    validator = MissingValueValidator()
    data = ["seq1", None, "seq3"]
    is_valid, errors = validator.validate(data)
    assert is_valid is False
    assert "Sequence at index 1 is missing or empty." in errors

def test_missing_value_validator_with_empty_string():
    validator = MissingValueValidator()
    data = ["seq1", "", "seq3"]
    is_valid, errors = validator.validate(data)
    assert is_valid is False
    assert "Sequence at index 1 is missing or empty." in errors

def test_missing_value_validator_with_whitespace_string():
    validator = MissingValueValidator()
    data = ["seq1", "   ", "seq3"]
    is_valid, errors = validator.validate(data)
    assert is_valid is False
    assert "Sequence at index 1 is missing or empty." in errors

def test_missing_value_validator_mixed_missing_values():
    validator = MissingValueValidator()
    data = ["seq1", None, "", "   ", "seq4"]
    is_valid, errors = validator.validate(data)
    assert is_valid is False
    assert "Sequence at index 1 is missing or empty." in errors
    assert "Sequence at index 2 is missing or empty." in errors
    assert "Sequence at index 3 is missing or empty." in errors

def test_missing_value_validator_empty_list():
    validator = MissingValueValidator()
    data = []
    is_valid, errors = validator.validate(data)
    assert is_valid is True
    assert not errors
