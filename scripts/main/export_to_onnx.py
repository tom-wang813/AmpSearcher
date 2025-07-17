import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any
from amp_searcher.utils.logging_utils import setup_logger
from amp_searcher.utils.config import Config

logger = setup_logger("export_to_onnx")

def export_to_onnx(model_name: str, output_path: str, config: Config) -> None:
    """Exports a Hugging Face model to ONNX format."""
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Create a dummy input (example SMILES string)
        dummy_smiles = config.get("dummy_smiles", "CCO")
        inputs = tokenizer(dummy_smiles, return_tensors="pt")

        # Move model and inputs to MPS if available
        if torch.backends.mps.is_available():
            model.to(torch.device("mps"))
            inputs = {k: v.to(torch.device("mps")) for k, v in inputs.items()}

        # Export the model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=config.get("opset_version", 17)
        )
        logger.info(f"Model exported to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Hugging Face model to ONNX format.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Hugging Face model (e.g., seyonec/ChemBERTa-zinc-base-v1).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the ONNX model.")
    parser.add_argument("--config_path", type=str, default="configs/export_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = Config(args.config_path)
    export_to_onnx(args.model_name, args.output_path, config)
