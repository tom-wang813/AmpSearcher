from typing import List, Tuple, Any, Dict

import torch
from torch.utils.data import Dataset
import numpy as np

from amp_searcher.data.processors.base_processor import BaseProcessor
from amp_searcher.data.processors.processor_factory import ProcessorFactory


class AmpDataset(Dataset):
    """
    A PyTorch Dataset for peptide sequences.

    This dataset loads peptide sequences and their corresponding labels (if any),
    and applies a specified processor to convert sequences into numerical features.
    """

    def __init__(
        self,
        sequences: List[str],
        labels: List[float] | None,
        processor: BaseProcessor,
    ):
        """
        Initializes the AmpDataset.

        Args:
            sequences: A list of peptide sequence strings.
            labels: A list of numerical labels corresponding to the sequences.
                    Can be None if no labels are available (e.g., for prediction).
            processor: An initialized BaseProcessor instance to process the sequences.
        """
        if labels is not None and len(sequences) != len(labels):
            raise ValueError("Number of sequences and labels must be the same.")

        # Process all sequences once during initialization
        self.processed_features = processor.process(sequences)

        self.sequences = (
            sequences  # Keep original sequences for potential debugging/logging
        )
        self.labels = labels
        self.processor = processor

    def save_features(self, output_dir: str, filename: str = "features.parquet"):
        """
        Saves the processed features to a Parquet file.
        """
        import pandas as pd
        import os

        # Ensure all features are torch tensors before stacking
        features_tensors = [
            torch.tensor(f, dtype=torch.float32) if isinstance(f, np.ndarray) else f
            for f in self.processed_features
        ]
        features_array = torch.stack(features_tensors).numpy()

        # Create a DataFrame from the features
        # Assuming processor has a way to get feature names
        feature_names = getattr(
            self.processor.featurizer,
            "feature_names",
            [f"feature_{i}" for i in range(features_array.shape[1])],
        )
        df_features = pd.DataFrame(features_array, columns=feature_names)

        # Add sequences and labels if available
        df_features["sequence"] = self.sequences
        if self.labels is not None:
            df_features["label"] = self.labels

        output_path = os.path.join(output_dir, filename)
        df_features.to_parquet(output_path, index=False)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor | None]:
        features = self.processed_features[idx]

        # Convert processed features (e.g., numpy array) to torch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return features_tensor, label_tensor
        else:
            return features_tensor, None


def load_sequences_from_file(
    filepath: str, processor_config: Dict[str, Any]
) -> AmpDataset:
    """
    Loads peptide sequences from a text file (one sequence per line) and returns an AmpDataset.
    """
    sequences = []
    with open(filepath, "r") as f:
        sequences = [line.strip() for line in f if line.strip()]

    processor = ProcessorFactory.build_processor(
        str(processor_config.get("name")), **processor_config.get("params", {})
    )
    return AmpDataset(sequences, None, processor)


def load_data_from_csv(
    filepath: str,
    sequence_col: str,
    label_col: str | None = None,
    processor_config: Dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> AmpDataset:
    """
    Loads peptide sequences and optional labels from a CSV file and returns an AmpDataset.
    """
    import pandas as pd
    from amp_searcher.data.validators.schemas import RawDataSchema, validate_dataframe

    df = pd.read_csv(filepath)

    # Validate the DataFrame against the schema
    # Assuming the CSV has 'sequence' and 'label' columns for validation
    # Adjust RawDataSchema in schemas.py if column names are different
    # validate_dataframe(df, RawDataSchema)  # Temporarily disabled

    sequences = df[sequence_col].tolist()
    labels = df[label_col].tolist() if label_col else None

    if processor_config is None:
        raise ValueError(
            "processor_config must be provided when loading data from CSV."
        )

    processor = ProcessorFactory.build_processor(
        str(processor_config.get("name")), **processor_config.get("params", {})
    )
    dataset = AmpDataset(sequences, labels, processor)

    if output_dir:
        # Construct a filename based on featurizer name and version
        featurizer_name = processor.featurizer.__class__.__name__
        featurizer_version = getattr(
            processor, "VERSION", "1.0"
        )  # Get version from processor if available
        filename = f"{featurizer_name}_v{featurizer_version}_features.parquet"
        dataset.save_features(output_dir, filename)

    return dataset
