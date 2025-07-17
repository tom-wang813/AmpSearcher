import argparse
import yaml

from amp_searcher.pipelines.screening_pipeline import ScreeningPipeline
from amp_searcher.data.datasets import load_data_from_csv


def main(
    model_path: str, model_config_path: str, featurizer_config_path: str, data_path: str, sequence_col: str
):
    # Load model configuration
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Load featurizer configuration
    with open(featurizer_config_path, "r") as f:
        full_config = yaml.safe_load(f)
    featurizer_config = full_config.get("featurizer", {})
    processor_config = {"name": "sequence_processor", "params": {"featurizer_config": featurizer_config}}

    # Load sequences from the data file
    dataset = load_data_from_csv(
        data_path, sequence_col, processor_config=processor_config
    )
    sequences = dataset.sequences

    if not sequences:
        print("No sequences found in the provided data file.")
        return

    # Initialize the ScreeningPipeline
    pipeline = ScreeningPipeline(
        model_config=model_config,
        model_checkpoint_path=model_path,
        featurizer_config=featurizer_config
    )

    # Make predictions
    print(f"Predicting for {len(sequences)} sequences...")
    predictions = pipeline.predict(sequences)

    print("\n--- Predictions ---")
    for i, seq in enumerate(sequences):
        print(f"Sequence: {seq}, Predicted Probability: {predictions[i].item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict AMP activity using a trained model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the model.",
    )
    parser.add_argument(
        "--featurizer_config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the featurizer used during training.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV data file containing sequences.",
    )
    parser.add_argument(
        "--sequence_col",
        type=str,
        required=True,
        help="Name of the column containing peptide sequences.",
    )
    args = parser.parse_args()
    main(args.model_path, args.model_config, args.featurizer_config, args.data_path, args.sequence_col)
