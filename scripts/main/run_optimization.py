import argparse
import pandas as pd
import random
from typing import List

from amp_searcher.optimizers.ga import GeneticAlgorithmOptimizer
from amp_searcher.optimizers.mcts import MCTSOptimizer
from amp_searcher.optimizers.smc import SMCOptimizer
from amp_searcher.utils.logging_utils import setup_logger
from amp_searcher.utils.config import Config
from amp_searcher.utils.performance_monitoring import timer, MemoryTracker
from amp_searcher.utils.oracle import Oracle

logger = setup_logger("run_optimization")

@timer
def main(config: Config):
    try:
        model_checkpoint_path = config.get("model_checkpoint_path")

        with MemoryTracker("Oracle Initialization"):
            oracle = Oracle(
                model_config=config.get("model"),
                model_checkpoint_path=model_checkpoint_path,
                featurizer_config=config.get("featurizer"),
            )

        optimizer_name = config.get("optimizer", {}).get("name")
        constraints = config.get("constraints")

        optimizer_class_map = {
            "ga": GeneticAlgorithmOptimizer,
            "mcts": MCTSOptimizer,
            "smc": SMCOptimizer,
        }

        if optimizer_name not in optimizer_class_map:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        optimizer = optimizer_class_map[optimizer_name](
            model=oracle, constraints=constraints, **config.get("optimizer", {})
        )

        # Create a valid initial population based on constraints
        initial_pop = [
            "".join(
                random.choice(constraints["alphabet"])
                for _ in range(constraints["max_length"])
            )
            for _ in range(config.get("optimizer", {}).get("population_size", 1))
        ]

        with MemoryTracker("Optimization"):
            results = optimizer.search(
                initial_population=initial_pop, n_iterations=config.get("n_iterations")
            )

        output_path = config.get("output_path")
        df = pd.DataFrame(results, columns=["sequence", "score"])
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results)} optimized sequences to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during optimization: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sequence optimization.")
    parser.add_argument("--config_path", type=str, default="configs/optimization_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = Config(args.config_path)
    main(config)
