from typing import List

import numpy as np

from amp_searcher.featurizers.base import BaseFeaturizer
from amp_searcher.utils.constants import AMINO_ACIDS, PHYSICOCHEMICAL_PROPERTIES
from .featurizer_factory import FeaturizerFactory


@FeaturizerFactory.register("PseAACFeaturizer")
class PseAACFeaturizer(BaseFeaturizer):
    """
    Calculates Pseudo Amino Acid Composition (PseAAC) for a protein sequence.

    PseAAC combines the amino acid composition with sequence order information
    based on physicochemical properties.

    Reference: Chou, K. C. (2001). Prediction of protein subcellular locations by
    incorporating quasi-amino acid composition into a support vector machine.
    Proteins: Structure, Function, and Bioinformatics, 43(3), 246-255.
    """

    def __init__(
        self,
        lam: int = 5,  # lambda, sequence order correlation factor
        weight: float = 0.1,  # weight factor for sequence order effects
        properties: List[str] | None = None,  # Physicochemical properties to use
    ):
        """
        Initializes the PseAAC featurizer.

        Args:
            lam: The maximum sequence order correlation factor (lambda).
                 Must be less than the sequence length.
            weight: The weight factor (omega) for the sequence order effects.
            properties: A list of physicochemical properties to use for correlation.
                        If None, uses Hydrophobicity, Hydrophilicity, and Side-chain mass.
        """
        if lam < 0:
            raise ValueError("Lambda (lam) must be a non-negative integer.")
        if weight < 0:
            raise ValueError("Weight (omega) must be a non-negative float.")

        self.lam = lam
        self.weight = weight
        self.properties = properties if properties is not None else ["H", "P", "M"]

        # Validate properties
        for prop in self.properties:
            if prop not in PHYSICOCHEMICAL_PROPERTIES:
                raise ValueError(f"Unknown physicochemical property: {prop}")

        self.feature_names: List[str] = []
        # AAC part
        self.feature_names.extend([f"PseAAC_AAC_{aa}" for aa in AMINO_ACIDS])
        # Sequence order correlation part
        for i in range(1, self.lam + 1):
            for prop in self.properties:
                self.feature_names.append(f"PseAAC_Corr_{prop}_{i}")

    def featurize(self, sequence: str) -> np.ndarray:
        """
        Computes the PseAAC features for the given protein sequence.

        Args:
            sequence: The protein sequence string.

        Returns:
            A 1D NumPy array containing the PseAAC feature values.
        """
        valid_sequence = "".join([aa for aa in sequence.upper() if aa in AMINO_ACIDS])
        seq_len = len(valid_sequence)

        if seq_len == 0:
            return np.zeros(len(self.feature_names))

        if self.lam >= seq_len:
            # If lambda is too large, PseAAC is not well-defined or becomes AAC
            # For simplicity, we'll return AAC features and pad the rest with zeros.
            # A more rigorous implementation might raise an error or adjust lambda.
            # Here, we'll just compute AAC and pad.
            aac_counts = {aa: 0 for aa in AMINO_ACIDS}
            for aa in valid_sequence:
                aac_counts[aa] += 1
            aac_features = np.array([aac_counts[aa] / seq_len for aa in AMINO_ACIDS])
            return np.pad(
                aac_features, (0, len(self.feature_names) - len(aac_features))
            )

        # 1. Calculate Amino Acid Composition (AAC)
        aac_counts = {aa: 0 for aa in AMINO_ACIDS}
        for aa in valid_sequence:
            aac_counts[aa] += 1
        aac_frequencies = np.array([aac_counts[aa] / seq_len for aa in AMINO_ACIDS])

        # 2. Calculate sequence order correlation factors
        correlation_factors: List[float] = []
        for j in range(1, self.lam + 1):
            for prop_key in self.properties:
                prop_values = PHYSICOCHEMICAL_PROPERTIES[prop_key]
                correlation_sum = 0.0
                for i in range(seq_len - j):
                    aa1 = valid_sequence[i]
                    aa2 = valid_sequence[i + j]
                    # Normalize property values (mean 0, std 1)
                    # This is a common step in PseAAC to ensure properties are comparable
                    prop_vals = np.array(list(prop_values.values()))
                    mean_prop = np.mean(prop_vals)
                    std_prop = np.std(prop_vals)

                    normalized_val1 = (prop_values[aa1] - mean_prop) / std_prop
                    normalized_val2 = (prop_values[aa2] - mean_prop) / std_prop
                    correlation_sum += (normalized_val1 - normalized_val2) ** 2
                correlation_factors.append(correlation_sum / (seq_len - j))

        np_correlation_factors = np.array(correlation_factors)

        # 3. Combine AAC and correlation factors to form PseAAC
        denominator = 1 + self.weight * np.sum(np_correlation_factors)
        if denominator == 0:
            # Avoid division by zero if correlation_factors sum to -1/weight
            # This is highly unlikely with squared differences, but for robustness
            return np.zeros(len(self.feature_names))

        pse_aac_features = np.zeros(len(self.feature_names))

        # AAC part of PseAAC
        for i, aa in enumerate(AMINO_ACIDS):
            pse_aac_features[i] = aac_frequencies[i] / denominator

        # Correlation part of PseAAC
        start_idx = len(AMINO_ACIDS)
        for i, corr_factor in enumerate(np_correlation_factors):
            pse_aac_features[start_idx + i] = (self.weight * corr_factor) / denominator

        return pse_aac_features
