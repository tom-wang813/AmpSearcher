from typing import List

import numpy as np

from .base import BaseFeaturizer
from .featurizer_factory import FeaturizerFactory


@FeaturizerFactory.register("CompositionFeaturizer")
class CompositionFeaturizer(BaseFeaturizer):
    """
    Calculates Amino Acid Composition (AAC) and Dipeptide Composition (DPC).
    """

    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, include_aac: bool = True, include_dpc: bool = False):
        """
        Initializes the featurizer.

        Args:
            include_aac: Whether to include Amino Acid Composition (AAC).
            include_dpc: Whether to include Dipeptide Composition (DPC).
        """
        if not include_aac and not include_dpc:
            raise ValueError("At least one of include_aac or include_dpc must be True.")

        self.include_aac = include_aac
        self.include_dpc = include_dpc
        self.feature_names: List[str] = []

        if self.include_aac:
            self.feature_names.extend([f"AAC_{aa}" for aa in self.AMINO_ACIDS])
        if self.include_dpc:
            self.feature_names.extend(
                [
                    f"DPC_{aa1}{aa2}"
                    for aa1 in self.AMINO_ACIDS
                    for aa2 in self.AMINO_ACIDS
                ]
            )
        self.feature_dim = len(self.feature_names)

    def featurize(self, sequence: str) -> np.ndarray:
        """
        Computes AAC and/or DPC for the given protein sequence.

        Args:
            sequence: The protein sequence string.

        Returns:
            A 1D NumPy array containing the feature values.
        """
        valid_sequence = "".join(
            [aa for aa in sequence.upper() if aa in self.AMINO_ACIDS]
        )
        seq_len = len(valid_sequence)

        if seq_len == 0:
            return np.zeros(len(self.feature_names))

        features = []

        if self.include_aac:
            aac_counts = {aa: 0 for aa in self.AMINO_ACIDS}
            for aa in valid_sequence:
                aac_counts[aa] += 1
            features.extend([aac_counts[aa] / seq_len for aa in self.AMINO_ACIDS])

        if self.include_dpc:
            dpc_counts = {
                f"{aa1}{aa2}": 0 for aa1 in self.AMINO_ACIDS for aa2 in self.AMINO_ACIDS
            }
            if seq_len < 2:
                # If sequence is too short for dipeptides, add zeros for DPC features
                features.extend(np.zeros(len(self.AMINO_ACIDS) ** 2))
            else:
                for i in range(seq_len - 1):
                    dipeptide = valid_sequence[i : i + 2]
                    dpc_counts[dipeptide] += 1
                features.extend([dpc_counts[dp] / (seq_len - 1) for dp in dpc_counts])

        return np.array(features)
