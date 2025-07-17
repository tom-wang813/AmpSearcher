from typing import List, Tuple

from amp_searcher.data.validators.base_validator import BaseValidator
from amp_searcher.data.validators.validator_factory import ValidatorFactory


@ValidatorFactory.register("sequence_length")
class SequenceLengthValidator(BaseValidator):
    def __init__(self, config: dict = None):
        config = config or {}
        self.min_length = config.get("min_length", 1)
        self.max_length = config.get("max_length", 100)

    def validate(self, sequences: List[str]) -> Tuple[bool, List[str]]:
        errors = []
        is_valid = True
        for i, seq in enumerate(sequences):
            if not (self.min_length <= len(seq) <= self.max_length):
                is_valid = False
                errors.append(
                    f"Sequence at index {i} has invalid length ({len(seq)}), expected between {self.min_length} and {self.max_length}: {seq}"
                )
        return is_valid, errors
