from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class BaseSelectionStrategy(ABC):
    """
    Abstract base class for selection strategies in multi-objective optimization.
    """

    @abstractmethod
    def select(
        self, candidates: List[Tuple[str, List[float]]], num_to_select: int
    ) -> List[Tuple[str, List[float]]]:
        """
        Selects the best candidates from a list of candidates with their scores.

        Args:
            candidates: A list of tuples, where each tuple contains a candidate string
                        and its corresponding list of scores.
            num_to_select: The number of candidates to select.

        Returns:
            A list of the selected candidates with their scores.
        """
        pass


class ElitistSelection(BaseSelectionStrategy):
    """
    Selects candidates based on the sum of their scores.
    """

    def select(
        self, candidates: List[Tuple[str, List[float]]], num_to_select: int
    ) -> List[Tuple[str, List[float]]]:
        candidates.sort(key=lambda x: sum(x[1]), reverse=True)
        return candidates[:num_to_select]


class NSGA2Selection(BaseSelectionStrategy):
    """
    Implements the NSGA-II selection strategy, which is based on non-dominated sorting
    and crowding distance.
    """

    def select(
        self, candidates: List[Tuple[str, List[float]]], num_to_select: int
    ) -> List[Tuple[str, List[float]]]:
        fronts = self._non_dominated_sort(candidates)

        next_generation: List[Tuple[str, List[float]]] = []
        for front in fronts:
            if len(next_generation) + len(front) > num_to_select:
                crowding_distances = self._crowding_distance_assignment(front)
                sorted_front = sorted(
                    crowding_distances, key=lambda x: x[1], reverse=True
                )
                remaining = num_to_select - len(next_generation)
                next_generation.extend([c for c, d in sorted_front[:remaining]])
                break
            next_generation.extend(front)

        return next_generation

    def _non_dominated_sort(
        self, candidates: List[Tuple[str, List[float]]]
    ) -> List[List[Tuple[str, List[float]]]]:
        candidate_tuples = [(c[0], tuple(c[1])) for c in candidates]
        dominating_counts = {i: 0 for i in range(len(candidate_tuples))}
        dominated_solutions: Dict[int, List[int]] = {
            i: [] for i in range(len(candidate_tuples))
        }

        for i in range(len(candidate_tuples)):
            for j in range(i + 1, len(candidate_tuples)):
                if self._dominates(candidate_tuples[i][1], candidate_tuples[j][1]):
                    dominated_solutions[i].append(j)
                    dominating_counts[j] += 1
                elif self._dominates(candidate_tuples[j][1], candidate_tuples[i][1]):
                    dominated_solutions[j].append(i)
                    dominating_counts[i] += 1

        fronts = []
        current_front = [i for i, count in dominating_counts.items() if count == 0]

        while current_front:
            fronts.append([candidates[i] for i in current_front])
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    dominating_counts[j] -= 1
                    if dominating_counts[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    def _dominates(
        self, scores1: Tuple[float, ...], scores2: Tuple[float, ...]
    ) -> bool:
        # Assumes maximization for all objectives
        return all(s1 >= s2 for s1, s2 in zip(scores1, scores2)) and any(
            s1 > s2 for s1, s2 in zip(scores1, scores2)
        )

    def _crowding_distance_assignment(
        self, front: List[Tuple[str, List[float]]]
    ) -> List[Tuple[Tuple[str, List[float]], float]]:
        distances = [(c, 0.0) for c in front]
        num_objectives = len(front[0][1])

        for m in range(num_objectives):
            front.sort(key=lambda x: x[1][m])
            distances[0] = (front[0], float("inf"))
            distances[-1] = (front[-1], float("inf"))

            if len(front) > 2:
                min_score = front[0][1][m]
                max_score = front[-1][1][m]
                if max_score == min_score:
                    continue

                for i in range(1, len(front) - 1):
                    distances[i] = (
                        front[i],
                        distances[i][1]
                        + (front[i + 1][1][m] - front[i - 1][1][m])
                        / (max_score - min_score),
                    )

        return distances
