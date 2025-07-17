import pytest
from amp_searcher.data.validators.validator_factory import ValidatorFactory
from amp_searcher.data.validators.base_validator import BaseValidator

# Define a dummy validator for testing registration
class DummyValidator(BaseValidator):
    def __init__(self, param1=None):
        self.param1 = param1

    def validate(self, data):
        return True, []


class AnotherDummyValidator(BaseValidator):
    def validate(self, data):
        return False, ["Another dummy error"]


def test_validator_registration():
    """Test that a validator can be registered and built."""
    ValidatorFactory._validators = {}

    @ValidatorFactory.register("dummy_validator")
    class TempDummyValidator(DummyValidator):
        pass

    assert "dummy_validator" in ValidatorFactory._validators
    validator = ValidatorFactory.build_validator("dummy_validator")
    assert isinstance(validator, TempDummyValidator)
    assert validator.validate(["test"]) == (True, [])


def test_validator_registration_with_params():
    """Test that a validator can be registered and built with parameters."""
    ValidatorFactory._validators = {}

    @ValidatorFactory.register("dummy_validator_with_params")
    class TempDummyValidatorWithParams(DummyValidator):
        pass

    validator = ValidatorFactory.build_validator("dummy_validator_with_params", param1="value1")
    assert isinstance(validator, TempDummyValidatorWithParams)
    assert validator.param1 == "value1"


def test_validator_registration_duplicate_name():
    """Test that registering a validator with a duplicate name raises an error."""
    ValidatorFactory._validators = {}

    @ValidatorFactory.register("duplicate_name")
    class FirstValidator(DummyValidator):
        pass

    with pytest.raises(ValueError, match="Validator with name 'duplicate_name' already registered."):
        @ValidatorFactory.register("duplicate_name")
        class SecondValidator(DummyValidator):
            pass


def test_build_non_existent_validator():
    """Test that building a non-existent validator raises an error."""
    ValidatorFactory._validators = {}
    with pytest.raises(ValueError, match="Unknown validator: non_existent_validator"):
        ValidatorFactory.build_validator("non_existent_validator")


def test_validator_registration_not_subclass_of_base_validator():
    """Test that registering a class not inheriting from BaseValidator raises an error."""
    ValidatorFactory._validators = {}

    with pytest.raises(ValueError, match="Validator class must inherit from BaseValidator"):
        @ValidatorFactory.register("invalid_validator")
        class InvalidValidator:
            pass


def test_validator_factory_multiple_registrations():
    """Test that multiple different validators can be registered."""
    ValidatorFactory._validators = {}

    @ValidatorFactory.register("dummy1")
    class Dummy1(DummyValidator):
        pass

    @ValidatorFactory.register("dummy2")
    class Dummy2(AnotherDummyValidator):
        pass

    assert "dummy1" in ValidatorFactory._validators
    assert "dummy2" in ValidatorFactory._validators

    val1 = ValidatorFactory.build_validator("dummy1")
    val2 = ValidatorFactory.build_validator("dummy2")

    assert isinstance(val1, Dummy1)
    assert isinstance(val2, Dummy2)
    assert val1.validate(["a"]) == (True, [])
    assert val2.validate(["b"]) == (False, ["Another dummy error"])
