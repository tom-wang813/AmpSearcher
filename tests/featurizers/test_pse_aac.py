import numpy as np
import pytest

from amp_searcher.featurizers import PseAACFeaturizer
from amp_searcher.utils.constants import PHYSICOCHEMICAL_PROPERTIES, AMINO_ACIDS


def test_pse_aac_featurizer_basic():
    """Test basic PseAAC calculation with default parameters."""
    featurizer = PseAACFeaturizer()
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    features = featurizer.featurize(sequence)

    # Expected features: 20 (AAC) + lambda * num_properties (5 * 3 = 15) = 35
    assert len(featurizer.feature_names) == 35
    assert features.shape == (35,)
    assert not np.all(features == 0)
    assert np.isclose(np.sum(features), 1.0)  # PseAAC features should sum to 1


def test_pse_aac_featurizer_custom_params():
    """Test PseAAC calculation with custom lambda and properties."""
    featurizer = PseAACFeaturizer(lam=2, properties=["H", "P"])
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    features = featurizer.featurize(sequence)

    # Expected features: 20 (AAC) + lambda * num_properties (2 * 2 = 4) = 24
    assert len(featurizer.feature_names) == 24
    assert features.shape == (24,)
    assert not np.all(features == 0)
    assert np.isclose(np.sum(features), 1.0)


def test_pse_aac_featurizer_empty_sequence():
    """Test handling of empty sequence."""
    featurizer = PseAACFeaturizer()
    sequence = ""
    features = featurizer.featurize(sequence)
    assert features.shape == (35,)
    assert np.all(features == 0)


def test_pse_aac_featurizer_short_sequence():
    """Test handling of sequence shorter than lambda."""
    featurizer = PseAACFeaturizer(lam=5)
    sequence = "ACD"
    features = featurizer.featurize(sequence)

    # Should fall back to AAC and pad with zeros
    assert features.shape == (35,)
    assert np.isclose(np.sum(features[:20]), 1.0)  # AAC part sums to 1
    assert np.all(features[20:] == 0)  # Correlation part should be zero


def test_pse_aac_featurizer_invalid_chars():
    """Test handling of invalid characters in sequence."""
    featurizer = PseAACFeaturizer()
    sequence = "A-C*D!E"
    features = featurizer.featurize(sequence)

    # Should be equivalent to featurizing "ACDE"
    expected_features = featurizer.featurize("ACDE")
    np.testing.assert_allclose(features, expected_features)


def test_pse_aac_featurizer_init_errors():
    """Test that initialization fails with invalid parameters."""
    with pytest.raises(
        ValueError, match=r"Lambda \(lam\) must be a non-negative integer."
    ):
        PseAACFeaturizer(lam=-1)

    with pytest.raises(
        ValueError, match=r"Weight \(omega\) must be a non-negative float."
    ):
        PseAACFeaturizer(weight=-0.5)

    with pytest.raises(ValueError, match="Unknown physicochemical property: XYZ"):
        PseAACFeaturizer(properties=["H", "XYZ"])

def test_pse_aac_featurizer_calculation_logic():
    """
    Test the PseAAC calculation logic against a manual, step-by-step
    implementation within the test itself for a simple case.
    This ensures the featurizer's internal algorithm is correct and avoids
    brittle, pre-computed golden values.
    """
    sequence = "GA"
    lam = 1
    weight = 0.1
    properties = ["H"]  # Using only Hydrophobicity for simplicity

    featurizer = PseAACFeaturizer(lam=lam, weight=weight, properties=properties)
    features = featurizer.featurize(sequence)

    # --- Manual, step-by-step calculation for verification ---
    # This replicates the logic from the featurizer itself.

    # 1. Get property values and normalize them
    prop_values = PHYSICOCHEMICAL_PROPERTIES[properties[0]]
    all_prop_vals = np.array([prop_values[aa] for aa in AMINO_ACIDS])
    mean_prop = np.mean(all_prop_vals)
    std_prop = np.std(all_prop_vals)

    # 2. Calculate correlation factor
    aa1, aa2 = 'G', 'A'
    norm_val1 = (prop_values[aa1] - mean_prop) / std_prop
    norm_val2 = (prop_values[aa2] - mean_prop) / std_prop
    correlation_sum = (norm_val1 - norm_val2) ** 2
    correlation_factor = correlation_sum / (len(sequence) - lam)

    # 3. Calculate PseAAC components
    denominator = 1.0 + weight * correlation_factor
    
    expected_features = {}
    expected_features["PseAAC_AAC_G"] = (0.5) / denominator
    expected_features["PseAAC_AAC_A"] = (0.5) / denominator
    expected_features["PseAAC_Corr_H_1"] = (weight * correlation_factor) / denominator

    # --- Assertions ---
    feature_dict = dict(zip(featurizer.feature_names, features))

    for name, expected_value in expected_features.items():
        assert name in feature_dict
        assert np.isclose(feature_dict[name], expected_value, atol=1e-6)
    
    assert np.isclose(feature_dict["PseAAC_AAC_C"], 0.0)
    assert np.isclose(np.sum(features), 1.0)
