import random
from typing import Any, Dict, List, Tuple

from .base import BaseOptimizer, BaseSelectionStrategy
from .optimizer_factory import OptimizerFactory


@OptimizerFactory.register("GeneticAlgorithmOptimizer")
class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Optimizes sequences using a Genetic Algorithm.

    This optimizer evolves a population of sequences over generations
    to maximize a score provided by a scoring model.
    """

    def __init__(
        self,
        model: Any,
        constraints: Dict[str, Any],
        objective_names: List[str],
        selection_strategy: BaseSelectionStrategy,
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        config: Dict[str, Any] | None = None,
    ):
        """
        Initializes the Genetic Algorithm optimizer.

        Args:
            model: The scoring model.
            constraints: Sequence constraints.
            objective_names: Names of the objectives.
            selection_strategy: The strategy for selecting parents and survivors.
            population_size: The number of individuals in the population.
            mutation_rate: The probability of a mutation occurring.
            crossover_rate: The probability of a crossover occurring.
            max_iterations: Maximum number of iterations for the optimization.
        """
        super().__init__(model, constraints, objective_names, selection_strategy)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.config = config or {}
        self.max_iterations = config.get("max_iterations", 100)
        self._current_iteration = 0
        self._population: List[
            Tuple[str, List[float]]
        ] = []  # Stores (sequence, [score1, score2, ...])

    def initialize(
        self, initial_candidates: List[str], initial_scores: List[List[float]]
    ) -> None:
        self._population = list(zip(initial_candidates, initial_scores))
        self._current_iteration = 0

    def propose_candidates(self, num_candidates: int) -> List[str]:
        new_candidates: List[str] = []

        if not self._population:
            # Fallback if population is empty
            alphabet = self.constraints.get("alphabet", "ACDEFGHIKLMNPQRSTVWY")
            seq_len = self.constraints.get("max_length", self.config.get("max_length", 30))
            return [
                "".join(random.choice(alphabet) for _ in range(seq_len))
                for _ in range(num_candidates)
            ]

        parents = self.selection_strategy.select(
            self._population, num_candidates * 2
        )  # Select more parents for crossover

        while len(new_candidates) < num_candidates:
            parent1_seq, _ = random.choice(parents)
            parent2_seq, _ = random.choice(parents)

            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1_seq, parent2_seq)
            else:
                offspring1, offspring2 = parent1_seq, parent2_seq

            new_candidates.append(self._mutate(offspring1))
            if len(new_candidates) < num_candidates:
                new_candidates.append(self._mutate(offspring2))
        return new_candidates[:num_candidates]

    def update_state(
        self, evaluated_candidates: List[str], evaluated_scores: List[List[float]]
    ) -> None:
        new_scored_candidates = list(zip(evaluated_candidates, evaluated_scores))
        self._population.extend(new_scored_candidates)

        # Use the selection strategy to choose the next generation
        self._population = self.selection_strategy.select(
            self._population, self.population_size
        )

        self._current_iteration += 1

    def is_converged(self) -> bool:
        return self._current_iteration >= self.max_iterations

    def get_best_candidates(self, num_best: int) -> List[Tuple[str, List[float]]]:
        return self.selection_strategy.select(self._population, num_best)

    def _crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        if len(parent1) != len(parent2) or len(parent1) < 2:
            return parent1, parent2
        point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[:point]
        return offspring1, offspring2

    def _mutate(self, sequence: str) -> str:
        alphabet = self.constraints.get("alphabet", "ACDEFGHIKLMNPQRSTVWY")
        mutated_sequence = list(sequence)
        for i in range(len(mutated_sequence)):
            if random.random() < self.mutation_rate:
                mutated_sequence[i] = random.choice(alphabet)
        return "".join(mutated_sequence)
