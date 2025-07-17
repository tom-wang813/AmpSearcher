import numpy as np
import pytest

from amp_searcher.featurizers import CompositionFeaturizer


def test_composition_featurizer_aac_only():
    """Test AAC calculation only."""
    featurizer = CompositionFeaturizer(include_aac=True, include_dpc=False)
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    features = featurizer.featurize(sequence)

    assert len(featurizer.feature_names) == 20
    assert features.shape == (20,)
    assert np.isclose(np.sum(features), 1.0)  # Frequencies should sum to 1
    assert np.isclose(features[featurizer.feature_names.index("AAC_A")], 1 / 20)
    assert np.isclose(features[featurizer.feature_names.index("AAC_C")], 1 / 20)


def test_composition_featurizer_dpc_only():
    """Test DPC calculation only."""
    featurizer = CompositionFeaturizer(include_aac=False, include_dpc=True)
    sequence = "AAACCC"
    features = featurizer.featurize(sequence)

    assert len(featurizer.feature_names) == 20 * 20  # 400 dipeptides
    assert features.shape == (400,)
    assert np.isclose(np.sum(features), 1.0)  # Frequencies should sum to 1

    # Sequence: AAACCC (length 6)
    # Dipeptides: AA (2 times), AC (1 time), CC (2 times)
    # Total dipeptides: 5
    assert np.isclose(features[featurizer.feature_names.index("DPC_AA")], 2 / 5)
    assert np.isclose(features[featurizer.feature_names.index("DPC_AC")], 1 / 5)
    assert np.isclose(features[featurizer.feature_names.index("DPC_CC")], 2 / 5)
    assert np.isclose(features[featurizer.feature_names.index("DPC_CA")], 0)


def test_composition_featurizer_aac_and_dpc():
    """Test both AAC and DPC calculation."""
    featurizer = CompositionFeaturizer(include_aac=True, include_dpc=True)
    sequence = "AC"
    features = featurizer.featurize(sequence)

    assert len(featurizer.feature_names) == 20 + 400
    assert features.shape == (420,)

    # AAC for AC (length 2)
    assert np.isclose(features[featurizer.feature_names.index("AAC_A")], 0.5)
    assert np.isclose(features[featurizer.feature_names.index("AAC_C")], 0.5)

    # DPC for AC (length 2, 1 dipeptide)
    assert np.isclose(features[featurizer.feature_names.index("DPC_AC")], 1.0)


def test_composition_featurizer_empty_sequence():
    """Test handling of empty sequence."""
    featurizer = CompositionFeaturizer(include_aac=True, include_dpc=True)
    sequence = ""
    features = featurizer.featurize(sequence)
    assert features.shape == (420,)
    assert np.all(features == 0)


def test_composition_featurizer_short_sequence_for_dpc():
    """Test handling of sequence too short for DPC."""
    featurizer = CompositionFeaturizer(include_aac=False, include_dpc=True)
    sequence = "A"
    features = featurizer.featurize(sequence)
    assert features.shape == (400,)
    assert np.all(features == 0)


def test_composition_featurizer_invalid_chars():
    """Test handling of invalid characters in sequence."""
    featurizer = CompositionFeaturizer(include_aac=True, include_dpc=False)
    sequence = "A-C*D!E"
    features = featurizer.featurize(sequence)
    # Should only consider A, C, D, E
    assert np.isclose(features[featurizer.feature_names.index("AAC_A")], 0.25)
    assert np.isclose(features[featurizer.feature_names.index("AAC_C")], 0.25)
    assert np.isclose(features[featurizer.feature_names.index("AAC_D")], 0.25)
    assert np.isclose(features[featurizer.feature_names.index("AAC_E")], 0.25)


def test_composition_featurizer_init_error():
    """Test that initialization fails if no features are selected."""
    with pytest.raises(
        ValueError, match="At least one of include_aac or include_dpc must be True."
    ):
        CompositionFeaturizer(include_aac=False, include_dpc=False)

def test_composition_featurizer_case_insensitivity():
    """Test that the featurizer is case-insensitive."""
    featurizer = CompositionFeaturizer(include_aac=True, include_dpc=False)
    sequence_upper = "GALA"
    sequence_lower = "gala"
    
    features_upper = featurizer.featurize(sequence_upper)
    features_lower = featurizer.featurize(sequence_lower)
    
    np.testing.assert_array_equal(features_upper, features_lower)
    assert np.isclose(features_upper[featurizer.feature_names.index("AAC_G")], 0.25)
    assert np.isclose(features_upper[featurizer.feature_names.index("AAC_A")], 0.5)
    assert np.isclose(features_upper[featurizer.feature_names.index("AAC_L")], 0.25)

def test_composition_featurizer_fully_invalid_sequence():
    """Test handling of a sequence with only invalid characters."""
    featurizer = CompositionFeaturizer(include_aac=True, include_dpc=True)
    sequence = "X-B*Z"
    features = featurizer.featurize(sequence)
    
    assert features.shape == (420,)
    assert np.all(features == 0)
