import argparse
import yaml

from amp_searcher.training.trainer import AmpTrainer


def main(config_path: str):
    """Loads configuration, initializes AmpTrainer, and starts training."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize AmpTrainer with the full configuration
    trainer = AmpTrainer(config)

    # The data configuration is expected to be inside the main config file
    data_config = config.get("data")
    if not data_config or not data_config.get("path"):
        raise ValueError("Data configuration with path is missing in the config file.")

    # Start training
    print("Starting training...")
    trainer.train(data_config)
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an AMP model from a config file.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config_path)
