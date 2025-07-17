import pandas as pd
from Bio import SeqIO
import argparse
import sys
from typing import List, Dict
from amp_searcher.utils.logging_utils import setup_logger

logger = setup_logger("convert_fasta_to_csv")

def convert_fasta_to_csv(fasta_path: str, output_csv_path: str, label: int = 1) -> None:
    """
    Converts a FASTA file to a CSV file with 'sequence' and 'label' columns.
    Assigns a default label to all sequences.
    """
    try:
        sequences: List[Dict[str, str]] = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append({"sequence": str(record.seq), "label": label})

        df = pd.DataFrame(sequences)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Successfully converted {fasta_path} to {output_csv_path}")
    except Exception as e:
        logger.error(f"Error converting FASTA to CSV: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA to CSV.")
    parser.add_argument(
        "--fasta_path", type=str, required=True, help="Path to the input FASTA file."
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        required=True,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Default label to assign to all sequences (default: 1).",
    )
    args = parser.parse_args()

    convert_fasta_to_csv(args.fasta_path, args.output_csv_path, args.label)
