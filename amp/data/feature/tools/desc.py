from typing import List, Any, Union

import deepchem
from deepchem.feat import CircularFingerprint
from deepchem.data import Dataset, NumpyDataset
import numpy as np

from amp.feature.base import FeatureExtractor

class DeepChemExtractor(FeatureExtractor):
    """
    使用 DeepChem 的 CircularFingerprint（或其他 featurizer）来提取分子特征。
    """
    
    def extract(self, data: Any, **kwargs) -> Any:
        # 如果传入的是 SMILES 列表，先创建 Dataset
        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            featurizer = CircularFingerprint(**kwargs)
            # Create dataset but not needed for featurization
            _ = NumpyDataset(X=data, y=None)
            X = featurizer.featurize(data)
            return X  # 返回 numpy.ndarray
        # 如果已经是 DeepChem Dataset，则直接用 featurizer 转换它的 X
        elif isinstance(data, Dataset):
            # Get the SMILES strings from the dataset
            if hasattr(data, 'ids'):
                smiles = data.ids
            else:
                smiles = data.X  # Assuming X contains SMILES if ids not available
            featurizer = CircularFingerprint(**kwargs)
            X = featurizer.featurize(smiles)
            return X
        else:
            raise ValueError("DeepChemExtractor: unsupported data type, "
                             "expect List[str] of SMILES or deepchem.data.Dataset")