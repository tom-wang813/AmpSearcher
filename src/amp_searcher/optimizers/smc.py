import random
from typing import Any, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

from .base import BaseOptimizer
from .optimizer_factory import OptimizerFactory
from .selection import BaseSelectionStrategy


@OptimizerFactory.register("SMCOptimizer")
class SMCOptimizer(BaseOptimizer):
    """
    Optimizes sequences using Sequential Monte Carlo (SMC), also known as a particle filter.

    This optimizer maintains a population of weighted particles (sequences) that are
    iteratively resampled and propagated to explore the solution space.
    """

    def __init__(
        self,
        model: Any,
        constraints: Dict[str, Any],
        objective_names: List[str],
        selection_strategy: BaseSelectionStrategy,
        population_size: int,
        mutation_rate: float,
        config: Dict[str, Any] | None = None,
    ):
        super().__init__(model, constraints, objective_names, selection_strategy)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.alphabet = list(self.constraints.get("alphabet", "ACDEFGHIKLMNPQRSTVWY"))
        config = config or {}
        self.max_length = config.get("max_length", self.constraints.get("max_length", 30))

    def search(
        self, initial_population: List[str], n_iterations: int
    ) -> List[Tuple[str, float]]:
        """
        Runs the SMC optimization loop.

        Args:
            initial_population: An initial population of sequences.
            n_iterations: The number of SMC steps (resampling and propagation).

        Returns:
            The final population of sequences, sorted by their estimated scores.
        """
        particles = self._initialize_population(initial_population)

        for i in range(n_iterations):
            weights = self._calculate_weights(particles)

            # Resampling step
            indices = np.random.choice(
                len(particles), size=self.population_size, p=weights
            )
            resampled_particles = [particles[i] for i in indices]

            # Propagation/Mutation step
            particles = [self._mutate(p) for p in resampled_particles]

        final_scores = self._evaluate(particles)
        sorted_particles = sorted(
            [(seq, score[0]) for score, seq in zip(final_scores, particles)],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_particles

    def _initialize_population(self, initial_population: List[str]) -> List[str]:
        if len(initial_population) >= self.population_size:
            return initial_population[: self.population_size]

        while len(initial_population) < self.population_size:
            new_sequence = "".join(
                random.choice(self.alphabet) for _ in range(self.max_length)
            )
            initial_population.append(new_sequence)

        return initial_population

    def _calculate_weights(self, particles: List[str]) -> NDArray[np.float32]:
        """Calculates the normalized weights for each particle based on its score."""
        scores = self._evaluate(particles)
        scores_array = np.array(scores)

        # Softmax for numerical stability
        exp_scores = np.exp(scores_array - np.max(scores_array))
        weights = exp_scores / np.sum(exp_scores)

        return weights  # type: ignore

    def _evaluate(self, sequences: List[str]) -> List[List[float]]:
        # Assuming self.model is a PyTorch Lightning module with a predict method
        # that takes a list of sequences and returns a list of (sequence, [score1, score2, ...]) tuples.
        results = self.model.predict(sequences)
        # For SMC, we typically optimize a single objective, so we take the first score and wrap it in a list.
        return [[score[0]] for seq, score in results]

    def _mutate(self, sequence: str) -> str:
        """Applies mutations to a sequence."""
        mutated_sequence = list(sequence)
        for i in range(len(mutated_sequence)):
            if random.random() < self.mutation_rate:
                mutated_sequence[i] = random.choice(self.alphabet)
        return "".join(mutated_sequence)
