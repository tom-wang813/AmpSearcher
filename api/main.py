from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np
from typing import List
from transformers import AutoTokenizer

app = FastAPI()

# Load ONNX model
sess = rt.InferenceSession("models/chemberta.onnx")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

class SequenceInput(BaseModel):
    sequences: List[str] # Expecting SMILES strings

@app.post("/predict")
async def predict(input_data: SequenceInput):
    # Tokenize SMILES strings
    inputs = tokenizer(input_data.sequences, return_tensors="np", padding=True, truncation=True)
    
    # Prepare ONNX inputs
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # Run inference
    onnx_output = sess.run(None, onnx_inputs)
    
    # Return embeddings (last_hidden_state)
    return {"embeddings": onnx_output[0].tolist()}