from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModel


class FeatureExtractor(ABC):
    """
    抽象特征提取器接口。子类必须实现 extract() 方法。
    """
    @abstractmethod
    def extract(self, data: Any, **kwargs) -> Any:
        """
        对输入 data 提取特征，并返回特征表示（如 numpy 数组、pandas.DataFrame 等）。
        """
        ...


class LMFeatureExtractor(FeatureExtractor):
    """
    带有 tokenizer + model + device + pooling + max_length 逻辑的基础类，
    供所有 HuggingFace LM 提取器继承。
    """
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        pooling: str = "mean",
        max_length: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def tokenize(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        # 保证空格分隔
        seqs = [s if " " in s else " ".join(s) for s in sequences]
        inputs = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
