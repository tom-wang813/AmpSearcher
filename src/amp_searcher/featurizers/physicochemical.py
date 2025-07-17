from typing import Dict, List

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from .base import BaseFeaturizer
from .featurizer_factory import FeaturizerFactory


@FeaturizerFactory.register("PhysicochemicalFeaturizer")
class PhysicochemicalFeaturizer(BaseFeaturizer):
    VERSION = "1.0"  # Add a version for the featurizer
    """
    Calculates a set of physicochemical properties for a protein sequence.
    """

    def __init__(self, custom_features: List[str] | None = None):
        """
        Initializes the featurizer.

        Args:
            custom_features: A list of specific features to calculate.
                             If None, all available features are used.
        """
        self.feature_names = [
            "length",
            "molecular_weight",
            "charge_at_ph_7",
            "isoelectric_point",
            "aromaticity",
            "instability_index",
            "gravy",
            "helix_fraction",
            "turn_fraction",
            "sheet_fraction",
        ]
        if custom_features:
            self.feature_names = [f for f in self.feature_names if f in custom_features]
        self.feature_dim = len(self.feature_names)

    def featurize(self, sequence: str) -> np.ndarray:
        """
        Computes the selected physicochemical features for the sequence.

        Args:
            sequence: The protein sequence string.

        Returns:
            A 1D NumPy array containing the feature values.
        """
        # Remove non-standard amino acids for analysis
        valid_sequence = "".join(
            [aa for aa in sequence.upper() if aa in "ACDEFGHIKLMNPQRSTVWY"]
        )
        if not valid_sequence:
            return np.zeros(len(self.feature_names))

        analyzer = ProteinAnalysis(valid_sequence)

        features: Dict[str, float] = {
            "length": len(valid_sequence),
            "molecular_weight": analyzer.molecular_weight(),
            "aromaticity": analyzer.aromaticity(),
            "instability_index": analyzer.instability_index(),
            "isoelectric_point": analyzer.isoelectric_point(),
            "gravy": analyzer.gravy(),
            "charge_at_ph_7": analyzer.charge_at_pH(7.0),
        }

        # Secondary structure fractions require a valid sequence
        sec_struct = analyzer.secondary_structure_fraction()
        features["helix_fraction"] = sec_struct[0]
        features["turn_fraction"] = sec_struct[1]
        features["sheet_fraction"] = sec_struct[2]

        feature_vector = np.array([features[name] for name in self.feature_names])
        return feature_vector
