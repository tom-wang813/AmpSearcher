from typing import Dict, Type

from amp_searcher.data.validators.base_validator import BaseValidator


class ValidatorFactory:
    _validators: Dict[str, Type[BaseValidator]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(validator_class: Type[BaseValidator]):
            if not issubclass(validator_class, BaseValidator):
                raise ValueError("Validator class must inherit from BaseValidator")
            if name in cls._validators:
                raise ValueError(f"Validator with name '{name}' already registered.")
            cls._validators[name] = validator_class
            return validator_class

        return decorator

    @classmethod
    def build_validator(cls, name: str, **kwargs) -> BaseValidator:
        validator_class = cls._validators.get(name)
        if not validator_class:
            raise ValueError(f"Unknown validator: {name}")
        return validator_class(**kwargs)
