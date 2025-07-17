import pytest
from src.amp_searcher.optimizers.selection import ElitistSelection, NSGA2Selection


@pytest.fixture
def candidates():
    return [
        ("A", [1, 5]),
        ("B", [5, 1]),
        ("C", [3, 3]),
        ("D", [4, 4]),
        ("E", [2, 2]),
    ]


def test_elitist_selection(candidates):
    selector = ElitistSelection()
    selected = selector.select(candidates, 2)
    assert selected[0][0] == "D"
    assert selected[1][0] == "B" or selected[1][0] == "A"


def test_nsga2_selection(candidates):
    selector = NSGA2Selection()
    selected = selector.select(candidates, 3)
    # In this case, D dominates C and E. A and B are on the pareto front.
    # So the first front is [A, B, D]
    selected_names = [s[0] for s in selected]
    assert "A" in selected_names
    assert "B" in selected_names
    assert "D" in selected_names
