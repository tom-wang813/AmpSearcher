import pandas as pd
import random
from Bio import SeqIO
from pathlib import Path
import argparse
from typing import List
from amp_searcher.utils.logging_utils import setup_logger
from amp_searcher.utils.config import Config
from amp_searcher.utils.performance_monitoring import timer, MemoryTracker

logger = setup_logger("prepare_baseline_data")

def generate_random_sequence(min_len: int, max_len: int, amino_acids: str) -> str:
    """Generates a random amino acid sequence."""
    length = random.randint(min_len, max_len)
    return "".join(random.choice(amino_acids) for _ in range(length))

@timer
def prepare_baseline_data(config: Config):
    try:
        fasta_path = Path(config.get("fasta_path"))
        output_path = Path(config.get("output_path"))
        min_len = config.get("min_len", 10)
        max_len = config.get("max_len", 20)
        amino_acids = config.get("amino_acids", "ACDEFGHIKLMNPQRSTVWY")

        logger.info(f"Preparing baseline data from {fasta_path}")

        # Read positive samples from FASTA file
        positive_samples: List[str] = []
        with MemoryTracker("Reading FASTA file"):
            for record in SeqIO.parse(fasta_path, "fasta"):
                positive_samples.append(str(record.seq))

        # Create a DataFrame for positive samples
        df_positive = pd.DataFrame({"sequence": positive_samples, "label": 1})

        # Generate negative samples
        with MemoryTracker("Generating negative samples"):
            num_negative_samples = len(df_positive)
            negative_sequences = [
                generate_random_sequence(min_len, max_len, amino_acids) for _ in range(num_negative_samples)
            ]

        # Create a DataFrame for negative samples
        df_negative = pd.DataFrame({"sequence": negative_sequences, "label": 0})

        # Combine and shuffle the data
        with MemoryTracker("Combining and shuffling data"):
            df_combined = pd.concat([df_positive, df_negative], ignore_index=True)
            df_shuffled = df_combined.sample(frac=1).reset_index(drop=True)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        with MemoryTracker("Saving to CSV"):
            df_shuffled.to_csv(output_path, index=False)

        logger.info(f"Successfully created baseline dataset at: {output_path}")
        logger.info(f"Total samples: {len(df_shuffled)} ({len(df_positive)} positive, {len(df_negative)} negative)")

    except Exception as e:
        logger.error(f"An error occurred while preparing baseline data: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare baseline training data.")
    parser.add_argument("--config_path", type=str, default="configs/prepare_baseline_data_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = Config(args.config_path)
    prepare_baseline_data(config)
