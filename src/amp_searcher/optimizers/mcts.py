import math
import random
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseOptimizer
from .optimizer_factory import OptimizerFactory
from .selection import BaseSelectionStrategy


class Node:
    """A node in the Monte Carlo Search Tree."""

    def __init__(self, sequence: str, parent: Optional["Node"] = None):
        self.sequence = sequence
        self.parent = parent
        self.children: List["Node"] = []
        self.visits = 0
        self.total_score = 0.0

    def is_fully_expanded(self, alphabet: List[str]) -> bool:
        return len(self.children) == len(alphabet)

    @property
    def average_score(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_score / self.visits


@OptimizerFactory.register("MCTSOptimizer")
class MCTSOptimizer(BaseOptimizer):
    """
    Optimizes sequences using Monte Carlo Tree Search (MCTS).

    This method builds a search tree to intelligently explore the sequence space,
    balancing exploration and exploitation to find high-scoring sequences.
    """

    def __init__(
        self,
        model: Any,
        constraints: Dict[str, Any],
        objective_names: List[str],
        selection_strategy: BaseSelectionStrategy,
        exploration_weight: float = 1.0,
        config: Dict[str, Any] | None = None,
    ):
        super().__init__(model, constraints, objective_names, selection_strategy)
        self.alphabet = list(self.constraints.get("alphabet", "ACDEFGHIKLMNPQRSTVWY"))
        config = config or {}
        self.max_length = config.get("max_length", self.constraints.get("max_length", 30))
        self.exploration_weight = exploration_weight
        self.max_iterations = config.get("max_iterations", 100)
        self._current_iteration = 0
        self._root: Optional[Node] = None
        self._best_candidates: List[
            Tuple[str, List[float]]
        ] = []  # Stores the best candidates found so far

    def initialize(
        self, initial_candidates: List[str], initial_scores: List[List[float]]
    ) -> None:
        if initial_candidates:
            # For MCTS, we typically start with an empty sequence as root
            # but we can use the first initial candidate to seed the search if provided
            self._root = Node(sequence=initial_candidates[0])
            # Evaluate and update the root node with its score
            self._root.total_score = sum(initial_scores[0])  # Sum scores for simplicity
            self._root.visits = 1
            self._update_best_candidates_from_node(self._root)
        else:
            self._root = Node(sequence="")
        self._current_iteration = 0

    def propose_candidates(self, num_candidates: int) -> List[str]:
        # In MCTS, candidates are proposed through the select and expand phases
        # We will run a single MCTS simulation to get one promising candidate
        proposed_sequences = []
        for _ in range(num_candidates):
            if self._root is None:
                raise ValueError("Optimizer not initialized. Call initialize() first.")

            leaf = self._select(self._root)

            node_to_simulate_from = leaf
            if len(leaf.sequence) < self.max_length:
                # If the leaf is not a terminal node, expand it
                child = self._expand(leaf)
                node_to_simulate_from = child

            # Simulate from the chosen node to get a sequence
            simulated_sequence = self._simulate_sequence(node_to_simulate_from.sequence)
            proposed_sequences.append(simulated_sequence)
        return proposed_sequences

    def update_state(
        self, evaluated_candidates: List[str], evaluated_scores: List[List[float]]
    ) -> None:
        # In MCTS, scores are backpropagated to update the tree
        # This method will be called by the pipeline after evaluating proposed candidates
        for seq, scores in zip(evaluated_candidates, evaluated_scores):
            if self._root is None:
                # If root is None, it means the optimizer was not initialized with initial candidates.
                # In this case, we just add the evaluated candidates to the best_candidates list.
                if (seq, scores) not in self._best_candidates:
                    self._best_candidates.append((seq, scores))
                continue

            # Find the node corresponding to the evaluated sequence and backpropagate
            # This is a simplified approach; a full MCTS would track nodes more robustly
            node = self._find_node(self._root, seq)
            if node:
                self._backpropagate(node, sum(scores))  # Sum scores for simplicity
                self._update_best_candidates_from_node(node)
            else:
                # If node not found (e.g., initial candidates not in tree), add it to best candidates
                if (seq, scores) not in self._best_candidates:
                    self._best_candidates.append((seq, scores))
        self._current_iteration += len(evaluated_candidates)

    def is_converged(self) -> bool:
        return self._current_iteration >= self.max_iterations

    def get_best_candidates(self, num_best: int) -> List[Tuple[str, List[float]]]:
        # Return the best candidates found so far, sorted by the sum of their scores
        return sorted(self._best_candidates, key=lambda x: sum(x[1]), reverse=True)[
            :num_best
        ]

    def _update_best_candidates_from_node(self, node: Node):
        # Update the list of best candidates based on the current node's sequence and score
        # This is a simplified approach for multi-objective (sum of scores)
        current_score = node.average_score  # Using average score from MCTS node
        current_sequence = node.sequence

        # Find if this sequence already exists in best_candidates and update if better
        found = False
        for i, (seq, scores) in enumerate(self._best_candidates):
            if seq == current_sequence:
                # Update if the new score (sum) is better
                if sum(scores) < current_score:
                    self._best_candidates[i] = (
                        current_sequence,
                        [current_score] * len(self.objective_names),
                    )  # Placeholder for multi-objective
                found = True
                break
        if not found:
            self._best_candidates.append(
                (current_sequence, [current_score] * len(self.objective_names))
            )  # Placeholder for multi-objective

        # Keep only the top N best candidates (e.g., population_size * 2)
        self._best_candidates.sort(key=lambda x: sum(x[1]), reverse=True)
        self._best_candidates = self._best_candidates[
            : self.population_size * 2 if hasattr(self, "population_size") else 100
        ]  # Use population_size if available, else a default

    def _find_node(self, current_node: Node, sequence: str) -> Optional[Node]:
        # Simple DFS to find a node by sequence
        if current_node.sequence == sequence:
            return current_node
        for child in current_node.children:
            found_node = self._find_node(child, sequence)
            if found_node:
                return found_node
        return None

    def _select(self, node: Node) -> Node:
        """Selects a leaf node from the tree using the UCB1 formula."""
        while node.children:
            if len(node.sequence) >= self.max_length or node.is_fully_expanded(
                self.alphabet
            ):
                node = self._best_child(node)
            else:
                return node  # Return node that is not fully expanded
        return node

    def _expand(self, node: Node) -> Node:
        """
        Expands the tree by adding a new child node.
        """
        if len(node.sequence) >= self.max_length:
            return node

        tried_chars = {child.sequence[len(node.sequence)] for child in node.children}
        untried_chars = [c for c in self.alphabet if c not in tried_chars]

        if not untried_chars:
            # This case should ideally be handled by is_fully_expanded check before calling expand
            return self._best_child(node)  # Fallback

        char = random.choice(untried_chars)
        new_sequence = node.sequence + char
        child_node = Node(sequence=new_sequence, parent=node)
        node.children.append(child_node)
        return child_node

    def _simulate_sequence(self, sequence: str) -> str:
        """
        Simulates a random rollout from a node and returns the full sequence.
        """
        current_sequence = list(sequence)
        while len(current_sequence) < self.max_length:
            current_sequence.append(random.choice(self.alphabet))

        final_sequence = "".join(current_sequence)
        return final_sequence

    def _evaluate_sequence(self, sequence: str) -> List[float]:
        # Use the BaseOptimizer's _evaluate helper method
        return self._evaluate([sequence])[0]

    def _backpropagate(self, node: Optional[Node], score: float):
        """
        Propagates the score back up the tree to the root.
        """
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    def _best_child(self, node: Node) -> Node:
        """
        Selects the best child node based on the UCB1 formula.
        """
        if not node.children:
            return node

        log_total_visits = math.log(node.visits)

        def ucb1(child: Node) -> float:
            if child.visits == 0:
                return float("inf")
            exploitation_term = child.average_score
            exploration_term = self.exploration_weight * math.sqrt(
                log_total_visits / child.visits
            )
            return exploitation_term + exploration_term

        return max(node.children, key=ucb1)
