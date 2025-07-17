from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, Tuple

from .selection import BaseSelectionStrategy


class ScoringModel(Protocol):
    # Updated to return a list of scores for multi-objective optimization
    def predict(self, sequences: List[str]) -> List[Tuple[str, List[float]]]: ...


class BaseOptimizer(ABC):
    """
    Abstract base class for all sequence optimizers supporting multi-objective optimization.

    This class defines the common interface for optimization algorithms.
    Subclasses must implement the abstract methods for initialization, candidate proposal,
    state update, convergence checking, and retrieving best candidates.
    """

    def __init__(
        self,
        model: ScoringModel,
        constraints: Dict[str, Any],
        objective_names: List[str],
        selection_strategy: BaseSelectionStrategy,
        **kwargs: Any,
    ):
        """
        Initializes the optimizer.

        Args:
            model: The trained scoring model (oracle) that takes
                   a sequence or its representation and returns a list of scores.
            constraints: A dictionary defining constraints for the sequences,
                         such as max length, allowed characters, etc.
            objective_names: A list of strings, where each string is the name of an objective.
                             The order of names should correspond to the order of scores returned by the model.
            selection_strategy: The strategy to use for selecting the best candidates.
        """
        self.model: ScoringModel = model
        self.constraints = constraints
        self.objective_names = objective_names
        self.selection_strategy = selection_strategy

    @abstractmethod
    def initialize(
        self, initial_candidates: List[str], initial_scores: List[List[float]]
    ) -> None:
        """
        Initializes the optimizer's internal state with an initial set of candidates and their scores.

        Args:
            initial_candidates: A list of initial peptide sequence strings.
            initial_scores: A list of lists of scores, where each inner list corresponds
                            to the scores for a candidate across all objectives.
        """
        raise NotImplementedError

    @abstractmethod
    def propose_candidates(self, num_candidates: int) -> List[str]:
        """
        Proposes a new set of candidate sequences based on the current optimization state.

        Args:
            num_candidates: The number of new candidates to propose.

        Returns:
            A list of newly proposed peptide sequence strings.
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self, evaluated_candidates: List[str], evaluated_scores: List[List[float]]
    ) -> None:
        """
        Updates the optimizer's internal state with newly evaluated candidates and their scores.

        Args:
            evaluated_candidates: A list of peptide sequence strings that have been evaluated.
            evaluated_scores: A list of lists of scores for the evaluated candidates.
        """
        raise NotImplementedError

    @abstractmethod
    def is_converged(self) -> bool:
        """
        Checks if the optimization process has reached a convergence criterion.

        Returns:
            True if the optimizer has converged, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_candidates(self, num_best: int) -> List[Tuple[str, List[float]]]:
        """
        Retrieves the best candidates found so far by the optimizer.

        Args:
            num_best: The number of best candidates to return.

        Returns:
            A list of tuples, where each tuple contains a peptide sequence string
            and its corresponding list of scores across all objectives.
        """
        raise NotImplementedError

    def _evaluate(self, sequences: List[str]) -> List[List[float]]:
        """
        Helper method to evaluate a list of sequences using the scoring model.

        Args:
            sequences: A list of peptide sequence strings.

        Returns:
            A list of lists of scores, where each inner list corresponds
            to the scores for a sequence across all objectives.
        """
        # Assuming model.predict returns List[Tuple[str, List[float]]]
        results = self.model.predict(sequences)
        return [score for _, score in results]
