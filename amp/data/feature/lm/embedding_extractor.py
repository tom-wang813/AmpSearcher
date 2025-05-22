# ...existing code...
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModel

from amp.data.feature.base import LMFeatureExtractor

class EmbeddingExtractor(LMFeatureExtractor):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,  # "cuda" or "cpu"
        pooling: str = "mean",      # "mean" or "cls"
        max_length: Optional[int] = None,
    ):
        """
        model_name: HF model ID (e.g. "Rostlab/prot_bert")
        pooling: "mean" for mean-pooling, "cls" for [CLS] token
        max_length: truncate sequences to this length if set
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # tokenize with padding, truncation
    def extract(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        Convert raw sequences to embeddings.
        Protein models typically expect space-separated amino acids.
        """
        # insert spaces if user passed raw sequence without spaces
        seqs = [seq if " " in seq else " ".join(seq) for seq in sequences]
        # tokenize with padding, truncation
        inputs = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            hs = outputs.last_hidden_state
            if self.pooling == "cls":
                embeddings = hs[:, 0, :]  # [CLS] token
            else:
                embeddings = hs.mean(dim=1)
        return [emb.cpu() for emb in embeddings]
