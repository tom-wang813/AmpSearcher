import os
import glob
import sys
import argparse
from typing import List
from amp_searcher.utils.logging_utils import setup_logger
from amp_searcher.utils.config import Config

logger = setup_logger("find_latest_checkpoint")

def find_latest_checkpoint(log_dir: str, config: Config) -> str:
    # Find all checkpoint files in the log_dir
    checkpoint_pattern = config.get("checkpoint_pattern", "**/*.ckpt")
    checkpoint_files: List[str] = glob.glob(os.path.join(log_dir, checkpoint_pattern), recursive=True)

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    latest_checkpoint = checkpoint_files[0]
    logger.info(f"Latest checkpoint found: {latest_checkpoint}")
    return latest_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the latest checkpoint file.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to search for checkpoint files.")
    parser.add_argument("--config_path", type=str, default="configs/checkpoint_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = Config(args.config_path)

    try:
        latest_checkpoint = find_latest_checkpoint(args.log_dir, config)
        print(latest_checkpoint)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
