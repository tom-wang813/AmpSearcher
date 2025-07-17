import pytest
import torch
import numpy as np
import os
import shutil
import pandas as pd

from amp_searcher.data.datasets import AmpDataset, load_sequences_from_file, load_data_from_csv
from amp_searcher.data.processors.processor_factory import ProcessorFactory
from amp_searcher.data.processors.sequence_processor import SequenceProcessor
from amp_searcher.data.processors.sequence_processor import SequenceProcessor
from amp_searcher.featurizers.physicochemical import PhysicochemicalFeaturizer

# --- Fixtures ---
@pytest.fixture
def physicochemical_featurizer():
    return PhysicochemicalFeaturizer()

@pytest.fixture
def sequence_processor(physicochemical_featurizer):
    # Create a SequenceProcessor that uses the PhysicochemicalFeaturizer
    return SequenceProcessor(featurizer_config={"name": "PhysicochemicalFeaturizer", "params": {}})

@pytest.fixture
def sample_sequences():
    return ["ACDEF", "GHIJKL", "MNPQRSTVWY"]

@pytest.fixture
def sample_labels():
    return [0.0, 1.0, 0.0]

@pytest.fixture
def temp_dir(tmp_path):
    # pytest's tmp_path fixture provides a unique temporary directory
    return str(tmp_path)

# --- Tests for AmpDataset ---
def test_amp_dataset_init_valid(sample_sequences, sample_labels, sequence_processor):
    dataset = AmpDataset(sample_sequences, sample_labels, sequence_processor)
    assert len(dataset) == len(sample_sequences)
    assert dataset.sequences == sample_sequences
    assert dataset.labels == sample_labels
    assert dataset.processor == sequence_processor
    assert len(dataset.processed_features) == len(sample_sequences)
    assert isinstance(dataset.processed_features[0], np.ndarray)
    assert dataset.processed_features[0].shape == (10,) # PhysicochemicalFeaturizer outputs 10 features

def test_amp_dataset_init_no_labels(sample_sequences, sequence_processor):
    dataset = AmpDataset(sample_sequences, None, sequence_processor)
    assert len(dataset) == len(sample_sequences)
    assert dataset.labels is None

def test_amp_dataset_init_mismatched_lengths(sample_sequences, sequence_processor):
    with pytest.raises(ValueError, match="Number of sequences and labels must be the same."):
        AmpDataset(sample_sequences, [0.0, 1.0], sequence_processor)

def test_amp_dataset_len(sample_sequences, sample_labels, sequence_processor):
    dataset = AmpDataset(sample_sequences, sample_labels, sequence_processor)
    assert len(dataset) == 3

def test_amp_dataset_getitem_with_labels(sample_sequences, sample_labels, sequence_processor):
    dataset = AmpDataset(sample_sequences, sample_labels, sequence_processor)
    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (10,) # PhysicochemicalFeaturizer outputs 10 features
    assert isinstance(label, torch.Tensor)
    assert label.item() == sample_labels[0]

def test_amp_dataset_getitem_no_labels(sample_sequences, sequence_processor):
    dataset = AmpDataset(sample_sequences, None, sequence_processor)
    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (10,)
    assert label is None

def test_amp_dataset_save_features(sample_sequences, sample_labels, sequence_processor, temp_dir):
    dataset = AmpDataset(sample_sequences, sample_labels, sequence_processor)
    output_filename = "test_features.parquet"
    output_path = os.path.join(temp_dir, output_filename)
    
    dataset.save_features(temp_dir, output_filename)
    
    assert os.path.exists(output_path)
    df_loaded = pd.read_parquet(output_path)
    assert len(df_loaded) == len(sample_sequences)
    assert "sequence" in df_loaded.columns
    assert "label" in df_loaded.columns
    assert df_loaded["sequence"].tolist() == sample_sequences
    assert df_loaded["label"].tolist() == sample_labels
    
    # Verify feature columns based on PhysicochemicalFeaturizer's feature_names
    expected_feature_columns = sequence_processor.featurizer.feature_names
    for col in expected_feature_columns:
        assert col in df_loaded.columns
    assert df_loaded.columns.tolist() == expected_feature_columns + ["sequence", "label"]

# --- Tests for load_sequences_from_file ---
def test_load_sequences_from_file_basic(temp_dir, sequence_processor):
    filepath = os.path.join(temp_dir, "sequences.txt")
    with open(filepath, "w") as f:
        f.write("SEQ1\nSEQ2\n\nSEQ3\n")
    
    processor_config = {"name": "sequence_processor", "params": {"featurizer_config": {"name": "PhysicochemicalFeaturizer", "params": {}}}}
    dataset = load_sequences_from_file(filepath, processor_config)
    
    assert isinstance(dataset, AmpDataset)
    assert len(dataset) == 3
    assert dataset.sequences == ["SEQ1", "SEQ2", "SEQ3"]
    assert dataset.labels is None
    assert isinstance(dataset.processor, SequenceProcessor)

def test_load_sequences_from_file_empty(temp_dir, sequence_processor):
    filepath = os.path.join(temp_dir, "empty_sequences.txt")
    with open(filepath, "w") as f:
        f.write("")
    
    processor_config = {"name": "sequence_processor", "params": {"featurizer_config": {"name": "PhysicochemicalFeaturizer", "params": {}}}}
    dataset = load_sequences_from_file(filepath, processor_config)
    
    assert isinstance(dataset, AmpDataset)
    assert len(dataset) == 0
    assert dataset.sequences == []

# --- Tests for load_data_from_csv ---
def test_load_data_from_csv_basic(temp_dir, sequence_processor):
    filepath = os.path.join(temp_dir, "data.csv")
    df = pd.DataFrame({"sequence": ["S1", "S2"], "label": [0.5, 0.8]})
    df.to_csv(filepath, index=False)
    
    processor_config = {"name": "sequence_processor", "params": {"featurizer_config": {"name": "PhysicochemicalFeaturizer", "params": {}}}}
    dataset = load_data_from_csv(filepath, "sequence", "label", processor_config)
    
    assert isinstance(dataset, AmpDataset)
    assert len(dataset) == 2
    assert dataset.sequences == ["S1", "S2"]
    assert dataset.labels == [0.5, 0.8]
    assert isinstance(dataset.processor, SequenceProcessor)

def test_load_data_from_csv_no_labels(temp_dir, sequence_processor):
    filepath = os.path.join(temp_dir, "data_no_labels.csv")
    df = pd.DataFrame({"sequence": ["S1", "S2"]})
    df.to_csv(filepath, index=False)
    
    processor_config = {"name": "sequence_processor", "params": {"featurizer_config": {"name": "PhysicochemicalFeaturizer", "params": {}}}}
    dataset = load_data_from_csv(filepath, "sequence", None, processor_config)
    
    assert isinstance(dataset, AmpDataset)
    assert len(dataset) == 2
    assert dataset.sequences == ["S1", "S2"]
    assert dataset.labels is None

def test_load_data_from_csv_save_features(temp_dir, sequence_processor):
    filepath = os.path.join(temp_dir, "data.csv")
    df = pd.DataFrame({"sequence": ["S1", "S2"], "label": [0.5, 0.8]})
    df.to_csv(filepath, index=False)
    
    processor_config = {"name": "sequence_processor", "params": {"featurizer_config": {"name": "PhysicochemicalFeaturizer", "params": {}}}}
    output_dir = os.path.join(temp_dir, "output_features")
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_data_from_csv(filepath, "sequence", "label", processor_config, output_dir)
    
    expected_filename = f"PhysicochemicalFeaturizer_v{sequence_processor.featurizer.VERSION}_features.parquet"
    expected_output_path = os.path.join(output_dir, expected_filename)
    
    assert os.path.exists(expected_output_path)
    df_loaded = pd.read_parquet(expected_output_path)
    assert len(df_loaded) == 2
    assert "sequence" in df_loaded.columns
    assert "label" in df_loaded.columns
    
    # Verify feature columns based on PhysicochemicalFeaturizer's feature_names
    expected_feature_columns = sequence_processor.featurizer.feature_names
    for col in expected_feature_columns:
        assert col in df_loaded.columns
    assert df_loaded.columns.tolist() == expected_feature_columns + ["sequence", "label"]

def test_load_data_from_csv_missing_processor_config(temp_dir):
    filepath = os.path.join(temp_dir, "data_missing_config.csv")
    df = pd.DataFrame({"sequence": ["S1"], "label": [0.5]})
    df.to_csv(filepath, index=False)
    
    with pytest.raises(ValueError, match="processor_config must be provided when loading data from CSV."):
        load_data_from_csv(filepath, "sequence", "label", None)
