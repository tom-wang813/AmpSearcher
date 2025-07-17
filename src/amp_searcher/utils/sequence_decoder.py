from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from .sequence_decoder_factory import SequenceDecoderFactory


class BaseSequenceDecoder(ABC):
    """
    Abstract base class for all sequence decoders.

    A sequence decoder converts a numerical representation (e.g., a feature vector
    or a latent space vector) back into a protein sequence string.
    """

    @abstractmethod
    def decode(self, features: np.ndarray | Any) -> str:
        """
        Converts a numerical representation into a protein sequence string.

        Args:
            features: A numerical representation of the sequence (e.g., a NumPy array).

        Returns:
            A string representing the decoded protein sequence.
        """
        raise NotImplementedError


@SequenceDecoderFactory.register("SimpleFeatureDecoder")
class SimpleFeatureDecoder(BaseSequenceDecoder):
    """
    A simple decoder that converts a feature vector back to a sequence.
    This is a placeholder and would need a more sophisticated implementation
    based on how features relate to sequences.
    """

    def decode(self, features: np.ndarray | Any) -> str:
        # This is a highly simplified placeholder.
        # In a real scenario, this would involve mapping features back to amino acids.
        # For now, it just returns a string representation of the features.
        return f"[Decoded Features: {features[:5]}...]"
