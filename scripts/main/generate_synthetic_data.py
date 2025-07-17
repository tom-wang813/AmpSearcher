import pandas as pd
import random
import argparse
from typing import List, Tuple
from amp_searcher.utils.logging_utils import setup_logger
from amp_searcher.utils.config import Config
from amp_searcher.utils.performance_monitoring import timer, MemoryTracker

logger = setup_logger("generate_synthetic_data")

def generate_peptide_sequence(min_len: int, max_len: int, alphabet: str) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choice(alphabet) for _ in range(length))

def assign_score(sequence: str) -> float:
    score = 0.0
    if "W" in sequence:
        score += 0.5
    if "F" in sequence:
        score += 0.5
    score += len(sequence) / 30.0 * 0.1
    return min(1.0, score)

@timer
def generate_data(num_samples: int, min_len: int, max_len: int, alphabet: str) -> Tuple[List[str], List[int], List[float]]:
    sequences = []
    labels = []
    scores = []

    for _ in range(num_samples):
        seq = generate_peptide_sequence(min_len, max_len, alphabet)
        label = 1 if ("W" in seq and "F" in seq) else 0
        score = assign_score(seq)
        sequences.append(seq)
        labels.append(label)
        scores.append(score)

    return sequences, labels, scores

def main(config: Config):
    try:
        num_samples = config.get("num_samples", 1000)
        output_path = config.get("output_path", "data/synthetic_data.csv")
        min_len = config.get("min_len", 10)
        max_len = config.get("max_len", 30)
        alphabet = config.get("alphabet", "ACDEFGHIKLMNPQRSTVWY")

        with MemoryTracker("Data Generation"):
            sequences, labels, scores = generate_data(num_samples, min_len, max_len, alphabet)

        df = pd.DataFrame({"sequence": sequences, "label": labels, "score": scores})
        df.to_csv(output_path, index=False)
        logger.info(f"Generated {num_samples} synthetic samples and saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic peptide data.")
    parser.add_argument("--config_path", type=str, default="configs/synthetic_data_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = Config(args.config_path)
    main(config)
