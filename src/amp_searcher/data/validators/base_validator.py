from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, sequences: List[str]) -> Tuple[bool, List[str]]:
        """
        Abstract method to validate a list of sequences.

        Args:
            sequences: A list of peptide sequence strings.

        Returns:
            A tuple containing:
            - bool: True if all sequences are valid, False otherwise.
            - List[str]: A list of error messages for invalid sequences.
        """
        pass
