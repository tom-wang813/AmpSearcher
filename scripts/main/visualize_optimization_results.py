import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, Any

from amp_searcher.utils.logging_utils import setup_logger
from amp_searcher.utils.config import Config
from amp_searcher.utils.performance_monitoring import timer, MemoryTracker

logger = setup_logger("visualize_optimization_results")

@timer
def load_data(file_path: str) -> pd.DataFrame:
    with MemoryTracker(f"Loading data from {file_path}"):
        return pd.read_csv(file_path)

@timer
def create_plot(original_df: pd.DataFrame, optimized_df: pd.DataFrame, output_path: str) -> None:
    with MemoryTracker("Creating plot"):
        plt.figure(figsize=(10, 6))
        sns.histplot(
            original_df["score"],
            color="blue",
            label="Original Data",
            kde=True,
            stat="density",
            alpha=0.5,
        )
        sns.histplot(
            optimized_df["score"],
            color="red",
            label="Optimized Sequences",
            kde=True,
            stat="density",
            alpha=0.5,
        )

        plt.title("Score Distribution: Original vs. Optimized Sequences")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(axis="y", alpha=0.75)
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Score distribution plot saved to {output_path}")

@timer
def main(config: Config) -> None:
    try:
        original_data_path = config.get("original_data_path")
        optimized_data_path = config.get("optimized_data_path")
        output_plot_path = config.get("output_plot_path")

        original_df = load_data(original_data_path)
        optimized_df = load_data(optimized_data_path)

        create_plot(original_df, optimized_df, output_plot_path)

    except Exception as e:
        logger.error(f"An error occurred during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize optimization results.")
    parser.add_argument("--config_path", type=str, default="configs/visualization_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = Config(args.config_path)
    main(config)
