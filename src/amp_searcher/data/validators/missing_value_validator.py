from typing import List, Tuple

from amp_searcher.data.validators.base_validator import BaseValidator
from amp_searcher.data.validators.validator_factory import ValidatorFactory


@ValidatorFactory.register("missing_value")
class MissingValueValidator(BaseValidator):
    def validate(self, sequences: List[str]) -> Tuple[bool, List[str]]:
        errors = []
        is_valid = True
        for i, seq in enumerate(sequences):
            if not seq or seq.strip() == "":
                is_valid = False
                errors.append(f"Sequence at index {i} is missing or empty.")
        return is_valid, errors
