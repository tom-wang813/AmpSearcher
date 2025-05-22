import numpy as np
from sklearn.datasets import make_classification

def generate_classification_data(
    n_samples: int = 100,
    n_features: int = 20,
    n_informative: int = 2,
    n_redundant: int = 2,
    n_classes: int = 2,
    class_sep: float = 1.0,
    random_state: int = 42
):
    """
    生成稳定的合成分类数据，用于单元测试模型。
    Returns:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state
    )
    return X, y

def generate_regression_data(
    n_samples: int = 100,
    n_features: int = 10,
    noise: float = 0.1,
    coef=None,
    random_state: int = 42
):
    """
    生成稳定的合成回归数据，用于单元测试模型。
    Returns:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    if coef is None:
        coef = rng.randn(n_features)
    y = X.dot(coef) + noise * rng.randn(n_samples)
    return X, y

def generate_peptides(
    n_samples: int = 100,
    length: int = 10,
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
    random_state: int = 42
):
    """
    生成稳定的合成肽序列，用于单元测试模型。
    Returns:
        peptides: List[str], 每个肽序列的长度为 length
    """
    rng = np.random.RandomState(random_state)
    peptides = [''.join(rng.choice(list(amino_acids), size=length)) for _ in range(n_samples)]
    return peptides
