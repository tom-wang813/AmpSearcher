import pytest
from amp_searcher.optimizers.base import BaseOptimizer
from amp_searcher.optimizers.ga import GeneticAlgorithmOptimizer
from amp_searcher.optimizers.selection import ElitistSelection
from typing import List, Tuple


# A mock scoring model for testing purposes
class MockModel:
    def predict(self, sequences: List[str]) -> List[Tuple[str, List[float]]]:
        # Return a list of (sequence, [score1, score2]) tuples
        return [
            (seq, [float(sum(ord(c) for c in seq)) / 1000.0, float(len(seq)) / 10.0])
            for seq in sequences
        ]


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def constraints():
    return {"max_length": 10, "alphabet": "ACGT"}


@pytest.fixture
def objective_names():
    return ["score1", "score2"]


@pytest.fixture
def selection_strategy():
    return ElitistSelection()


# --- Test BaseOptimizer (abstract methods) ---
class ConcreteOptimizer(BaseOptimizer):
    def __init__(self, model, constraints, objective_names, selection_strategy):
        super().__init__(model, constraints, objective_names, selection_strategy)
        self._converged = False
        self._candidates = []

    def initialize(self, initial_candidates, initial_scores):
        self._candidates = list(zip(initial_candidates, initial_scores))

    def propose_candidates(self, num_candidates):
        return [f"SEQ{i}" for i in range(num_candidates)]

    def update_state(self, evaluated_candidates, evaluated_scores):
        for seq, scores in zip(evaluated_candidates, evaluated_scores):
            self._candidates.append((seq, scores))

    def is_converged(self):
        return self._converged

    def get_best_candidates(self, num_best):
        # For testing, just return the first num_best candidates
        return self._candidates[:num_best]


def test_base_optimizer_abstract_methods(
    mock_model, constraints, objective_names, selection_strategy
):
    optimizer = ConcreteOptimizer(
        mock_model, constraints, objective_names, selection_strategy
    )

    initial_candidates = ["AAAA", "CCCC"]
    initial_scores = [[0.1, 0.2], [0.3, 0.4]]
    optimizer.initialize(initial_candidates, initial_scores)
    assert len(optimizer._candidates) == 2

    proposed = optimizer.propose_candidates(3)
    assert len(proposed) == 3

    evaluated_candidates = ["GGGG", "TTTT"]
    evaluated_scores = [[0.5, 0.6], [0.7, 0.8]]
    optimizer.update_state(evaluated_candidates, evaluated_scores)
    assert len(optimizer._candidates) == 4

    assert not optimizer.is_converged()

    best_candidates = optimizer.get_best_candidates(1)
    assert len(best_candidates) == 1
    assert best_candidates[0][0] == "AAAA"

    scores = optimizer._evaluate(["TEST"])
    assert len(scores[0]) == 2  # Check for multi-objective scores


# --- Test GeneticAlgorithmOptimizer ---
# Note: GA, MCTS, SMC will need to be updated to implement the new BaseOptimizer interface
# For now, these tests will likely fail until those optimizers are updated.


def test_ga_optimizer_init(
    mock_model, constraints, objective_names, selection_strategy
):
    optimizer = GeneticAlgorithmOptimizer(
        model=mock_model,
        constraints=constraints,
        objective_names=objective_names,
        selection_strategy=selection_strategy,
        population_size=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
    )
    assert isinstance(optimizer, BaseOptimizer)
    assert optimizer.population_size == 10
    assert optimizer.objective_names == objective_names


def test_ga_optimizer_lifecycle(
    mock_model, constraints, objective_names, selection_strategy
):
    optimizer = GeneticAlgorithmOptimizer(
        model=mock_model,
        constraints=constraints,
        objective_names=objective_names,
        selection_strategy=selection_strategy,
        population_size=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        config={"max_iterations": 2},
    )

    initial_candidates = ["A" * 10, "C" * 10]
    initial_scores = mock_model.predict(initial_candidates)
    optimizer.initialize(
        [seq for seq, _ in initial_scores], [score for _, score in initial_scores]
    )

    assert len(optimizer._population) == 2  # Initial population size
    assert optimizer._current_iteration == 0

    # Simulate one iteration
    proposed = optimizer.propose_candidates(
        optimizer.population_size - len(optimizer._population)
    )  # Propose enough to fill population
    assert len(proposed) == 8  # 10 - 2

    evaluated_scores = mock_model.predict(proposed)
    optimizer.update_state(
        [seq for seq, _ in evaluated_scores], [score for _, score in evaluated_scores]
    )

    assert len(optimizer._population) == 10  # Population size should be maintained
    assert optimizer._current_iteration == 1
    assert not optimizer.is_converged()

    # Simulate another iteration to reach convergence
    proposed = optimizer.propose_candidates(optimizer.population_size)
    evaluated_scores = mock_model.predict(proposed)
    optimizer.update_state(
        [seq for seq, _ in evaluated_scores], [score for _, score in evaluated_scores]
    )

    assert optimizer._current_iteration == 2
    assert optimizer.is_converged()

    best_candidates = optimizer.get_best_candidates(5)
    assert len(best_candidates) == 5
    assert all(
        isinstance(seq, str) and isinstance(scores, list)
        for seq, scores in best_candidates
    )
    assert all(len(scores) == len(objective_names) for seq, scores in best_candidates)


# --- Test MCTSOptimizer ---
# @pytest.mark.skip(reason="Requires MCTS to be updated to new BaseOptimizer interface")
# def test_mcts_optimizer_init(mock_model, constraints, objective_names):
#     optimizer = MCTSOptimizer(model=mock_model, constraints=constraints, objective_names=objective_names, exploration_weight=1.5)
#     assert isinstance(optimizer, BaseOptimizer)
#     assert optimizer.exploration_weight == 1.5

# --- Test SMCOptimizer ---
# @pytest.mark.skip(reason="Requires SMC to be updated to new BaseOptimizer interface")
# def test_smc_optimizer_init(mock_model, constraints, objective_names):
#     optimizer = SMCOptimizer(model=mock_model, constraints=constraints, objective_names=objective_names, population_size=15, mutation_rate=0.05)
#     assert isinstance(optimizer, BaseOptimizer)
#     assert optimizer.population_size == 15
