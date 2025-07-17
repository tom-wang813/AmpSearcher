from typing import List, Tuple
import re

from amp_searcher.data.validators.base_validator import BaseValidator
from amp_searcher.data.validators.validator_factory import ValidatorFactory


@ValidatorFactory.register("amino_acid")
class AminoAcidValidator(BaseValidator):
    def __init__(self, allowed_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"):
        self.allowed_amino_acids = allowed_amino_acids
        self.pattern = re.compile(f"^[{re.escape(allowed_amino_acids)}]+$")

    def validate(self, sequences: List[str]) -> Tuple[bool, List[str]]:
        errors = []
        is_valid = True
        for i, seq in enumerate(sequences):
            if not self.pattern.match(seq):
                is_valid = False
                errors.append(
                    f"Sequence at index {i} contains invalid amino acids: {seq}"
                )
        return is_valid, errors
