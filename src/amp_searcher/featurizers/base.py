from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseFeaturizer(ABC):
    """
    Abstract base class for all featurizers.

    Each featurizer must implement the `featurize` method, which takes a
    protein sequence as input and returns a numerical representation.
    """

    @abstractmethod
    def featurize(self, sequence: str) -> np.ndarray | Any:
        """
        Converts a protein sequence into a numerical feature vector.

        Args:
            sequence: A string representing the protein sequence.

        Returns:
            A NumPy array or other numerical representation of the sequence.
        """
        raise NotImplementedError
