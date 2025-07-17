import numpy as np
import pytest

from amp_searcher.featurizers import BaseFeaturizer


def test_base_featurizer_inheritance():
    """Tests that a concrete implementation must implement featurize."""

    class IncompleteFeaturizer(BaseFeaturizer):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteFeaturizer()

    class CompleteFeaturizer(BaseFeaturizer):
        def featurize(self, sequence: str) -> np.ndarray:
            return np.array([len(sequence)])

    # This should not raise an error
    featurizer = CompleteFeaturizer()
    assert isinstance(featurizer, BaseFeaturizer)

    # Test the implemented method
    result = featurizer.featurize("TEST")
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([4]))
