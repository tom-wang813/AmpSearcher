import pytest
import unittest.mock
from unittest.mock import Mock

from amp_searcher.data.processors.sequence_processor import SequenceProcessor
from amp_searcher.featurizers.base import BaseFeaturizer
from amp_searcher.data.validators.base_validator import BaseValidator

@pytest.fixture
def mock_featurizer():
    mock = Mock(spec=BaseFeaturizer)
    mock.featurize.side_effect = lambda seq: [len(seq)] # Simple featurization
    return mock

@pytest.fixture
def mock_validator():
    mock = Mock(spec=BaseValidator)
    mock.validate.return_value = (True, []) # Always valid by default
    return mock

def test_sequence_processor_init_basic(mock_featurizer):
    # Mock FeaturizerFactory.build_featurizer to return our mock_featurizer
    with unittest.mock.patch('amp_searcher.featurizers.featurizer_factory.FeaturizerFactory.build_featurizer', return_value=mock_featurizer):
        processor = SequenceProcessor(featurizer_config={"name": "MockFeaturizer"})
        assert processor.featurizer == mock_featurizer
        assert not processor.validators

def test_sequence_processor_init_with_validators(mock_featurizer, mock_validator):
    # Mock FeaturizerFactory.build_featurizer and ValidatorFactory.build_validator
    with unittest.mock.patch('amp_searcher.featurizers.featurizer_factory.FeaturizerFactory.build_featurizer', return_value=mock_featurizer):
        with unittest.mock.patch('amp_searcher.data.validators.validator_factory.ValidatorFactory.build_validator', return_value=mock_validator):
            processor = SequenceProcessor(
            featurizer_config={"name": "MockFeaturizer"},
            validator_configs=[{"name": "MockValidator1"}, {"name": "MockValidator2"}]
        )
        assert processor.featurizer == mock_featurizer
        assert len(processor.validators) == 2
        assert processor.validators[0] == mock_validator
        assert processor.validators[1] == mock_validator

def test_sequence_processor_process(mock_featurizer, mock_validator):
    with unittest.mock.patch('amp_searcher.featurizers.featurizer_factory.FeaturizerFactory.build_featurizer', return_value=mock_featurizer):
        with unittest.mock.patch('amp_searcher.data.validators.validator_factory.ValidatorFactory.build_validator', return_value=mock_validator):
            processor = SequenceProcessor(
            featurizer_config={"name": "MockFeaturizer"},
            validator_configs=[{"name": "MockValidator"}]
        )
        sequences = ["SEQ1", "LONGER_SEQ"]
        processed_data = processor.process(sequences)

        mock_validator.validate.assert_called_once_with(sequences)
        mock_featurizer.featurize.assert_any_call("SEQ1")
        mock_featurizer.featurize.assert_any_call("LONGER_SEQ")
        assert processed_data == [[4], [10]] # Based on mock_featurizer.featurize.side_effect

def test_sequence_processor_process_validation_failure(mock_featurizer, mock_validator):
    mock_validator.validate.return_value = (False, ["Invalid sequence found"])
    with unittest.mock.patch('amp_searcher.featurizers.featurizer_factory.FeaturizerFactory.build_featurizer', return_value=mock_featurizer):
        with unittest.mock.patch('amp_searcher.data.validators.validator_factory.ValidatorFactory.build_validator', return_value=mock_validator):
            processor = SequenceProcessor(
            featurizer_config={"name": "MockFeaturizer"},
            validator_configs=[{"name": "MockValidator"}]
        )
        sequences = ["INVALID_SEQ"]
        with pytest.raises(ValueError, match="Data validation failed: Invalid sequence found"):
            processor.process(sequences)

def test_sequence_processor_process_single(mock_featurizer, mock_validator):
    with unittest.mock.patch('amp_searcher.featurizers.featurizer_factory.FeaturizerFactory.build_featurizer', return_value=mock_featurizer):
        with unittest.mock.patch('amp_searcher.data.validators.validator_factory.ValidatorFactory.build_validator', return_value=mock_validator):
            processor = SequenceProcessor(
            featurizer_config={"name": "MockFeaturizer"},
            validator_configs=[{"name": "MockValidator"}]
        )
        sequence = "SINGLE_SEQ"
        processed_data = processor.process_single(sequence)

        mock_validator.validate.assert_called_once_with([sequence])
        mock_featurizer.featurize.assert_called_once_with("SINGLE_SEQ")
        assert processed_data == [len(sequence)]

def test_sequence_processor_process_single_validation_failure(mock_featurizer, mock_validator):
    mock_validator.validate.return_value = (False, ["Single sequence invalid"])
    with unittest.mock.patch('amp_searcher.featurizers.featurizer_factory.FeaturizerFactory.build_featurizer', return_value=mock_featurizer):
        with unittest.mock.patch('amp_searcher.data.validators.validator_factory.ValidatorFactory.build_validator', return_value=mock_validator):
            processor = SequenceProcessor(
            featurizer_config={"name": "MockFeaturizer"},
            validator_configs=[{"name": "MockValidator"}]
        )
        sequence = "INVALID_SINGLE_SEQ"
        with pytest.raises(ValueError, match="Data validation failed for single sequence: Single sequence invalid"):
            processor.process_single(sequence)
