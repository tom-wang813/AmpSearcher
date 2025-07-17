import numpy as np
import pytest
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from amp_searcher.featurizers import PhysicochemicalFeaturizer


@pytest.fixture
def featurizer() -> PhysicochemicalFeaturizer:
    return PhysicochemicalFeaturizer()


@pytest.fixture
def standard_sequence() -> str:
    return "GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV"


def test_physicochemical_featurizer_basic(
    featurizer: PhysicochemicalFeaturizer, standard_sequence: str
):
    """Test basic functionality with a standard sequence."""
    features = featurizer.featurize(standard_sequence)

    # Calculate expected values directly using the underlying library
    analyzer = ProteinAnalysis(standard_sequence)
    expected_ip = analyzer.isoelectric_point()
    expected_gravy = analyzer.gravy()

    assert isinstance(features, np.ndarray)
    assert features.shape == (10,)  # Updated number of features
    assert not np.all(features == 0)

    # Assert against library-calculated values
    assert np.isclose(
        features[featurizer.feature_names.index("isoelectric_point")], expected_ip
    )
    assert np.isclose(features[featurizer.feature_names.index("gravy")], expected_gravy)
    assert features[featurizer.feature_names.index("length")] == len(standard_sequence)


def test_physicochemical_featurizer_invalid_chars(
    featurizer: PhysicochemicalFeaturizer, standard_sequence: str
):
    """Test handling of sequences with non-standard amino acids."""
    invalid_sequence = "GLWSKIK-EVGKEAA*KAAAKAAGKAALGAVSEAVX"
    # The featurizer should strip invalid characters and process the rest
    features = featurizer.featurize(invalid_sequence)
    expected_features = featurizer.featurize(standard_sequence)

    assert isinstance(features, np.ndarray)
    assert features.shape == (10,)
    np.testing.assert_allclose(features, expected_features)


def test_physicochemical_featurizer_empty_sequence(
    featurizer: PhysicochemicalFeaturizer,
):
    """Test handling of an empty or fully invalid sequence."""
    sequence = ""
    features = featurizer.featurize(sequence)
    assert features.shape == (10,)
    assert np.all(features == 0)

    sequence_invalid = "X-B*Z"
    features_invalid = featurizer.featurize(sequence_invalid)
    assert features_invalid.shape == (10,)
    assert np.all(features_invalid == 0)


def test_custom_feature_selection(standard_sequence: str):
    """Test selecting a subset of features."""
    custom_features = ["molecular_weight", "aromaticity", "length"]
    featurizer = PhysicochemicalFeaturizer(custom_features=custom_features)
    features = featurizer.featurize(standard_sequence)

    # Calculate expected value
    analyzer = ProteinAnalysis(standard_sequence)
    expected_mw = analyzer.molecular_weight()

    assert set(featurizer.feature_names) == set(custom_features)
    assert isinstance(features, np.ndarray)
    assert features.shape == (3,)
    assert np.isclose(
        features[featurizer.feature_names.index("molecular_weight")], expected_mw
    )
    assert features[featurizer.feature_names.index("length")] == len(standard_sequence)


def test_physicochemical_featurizer_self_consistent(featurizer: PhysicochemicalFeaturizer):
    """
    Test that the featurizer's output is consistent with direct calculation
    using the underlying Bio.SeqUtils.ProtParam library for a simple sequence.
    This avoids brittle 'golden value' tests.
    """
    sequence = "GALA"
    features = featurizer.featurize(sequence)
    feature_dict = dict(zip(featurizer.feature_names, features))

    # Calculate all expected values directly using the library
    analyzer = ProteinAnalysis(sequence)
    sec_struct = analyzer.secondary_structure_fraction()

    expected_values = {
        "length": len(sequence),
        "molecular_weight": analyzer.molecular_weight(),
        "charge_at_ph_7": analyzer.charge_at_pH(7.0),
        "isoelectric_point": analyzer.isoelectric_point(),
        "aromaticity": analyzer.aromaticity(),
        "instability_index": analyzer.instability_index(),
        "gravy": analyzer.gravy(),
        "helix_fraction": sec_struct[0],
        "turn_fraction": sec_struct[1],
        "sheet_fraction": sec_struct[2],
    }

    # Compare the featurizer's output with the directly calculated values
    for name, expected_value in expected_values.items():
        if name in feature_dict:
            assert np.isclose(feature_dict[name], expected_value, atol=1e-2), (
                f"Feature '{name}' mismatch. "
                f"Expected: {expected_value}, Got: {feature_dict[name]}"
            )
