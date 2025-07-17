import pytest
from abc import ABC, abstractmethod
from amp_searcher.data.validators.base_validator import BaseValidator

def test_base_validator_abstract_methods():
    """
    Test that BaseValidator is an abstract class and cannot be instantiated directly.
    Also, ensure that concrete implementations must implement the 'validate' method.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseValidator"):
        BaseValidator()

    class ConcreteValidator(BaseValidator):
        def validate(self, data):
            return True, []

    # Should not raise an error
    validator = ConcreteValidator()
    assert isinstance(validator, BaseValidator)

    # Test the implemented method
    is_valid, errors = validator.validate(["test_data"])
    assert is_valid is True
    assert errors == []

    class IncompleteValidator(BaseValidator):
        # Missing 'validate' method
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteValidator"):
        IncompleteValidator()
