from typing import Dict
import torch

from amp_searcher.featurizers.base import BaseFeaturizer
from amp_searcher.featurizers.featurizer_factory import FeaturizerFactory


@FeaturizerFactory.register("SimpleSequenceFeaturizer")
class SimpleSequenceFeaturizer(BaseFeaturizer):
    def __init__(self, max_len: int, vocab: Dict[str, int]):
        self.max_len = max_len
        self.vocab = vocab
        self.char_to_int = vocab
        self.int_to_char = {i: c for c, i in vocab.items()}
        self.feature_dim = max_len

    def featurize(self, sequence: str) -> torch.Tensor:
        encoded = [
            self.char_to_int.get(char, 0) for char in sequence
        ]  # 0 for unknown characters
        # Pad or truncate sequence to max_len
        if len(encoded) < self.max_len:
            encoded.extend([0] * (self.max_len - len(encoded)))
        else:
            encoded = encoded[: self.max_len]
        return torch.tensor(encoded, dtype=torch.long)

    def defragment(self, featurized_sequence: torch.Tensor) -> str:
        # Convert tensor to list of integers, then map back to characters
        chars = [self.int_to_char.get(int(i), "") for i in featurized_sequence.tolist()]
        # Filter out padding (0) or unknown characters ('')
        return "".join(
            [
                char
                for char in chars
                if char != "" and char != self.int_to_char.get(0, "")
            ]
        )
