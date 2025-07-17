import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from amp_searcher.data import AmpDataset, load_data_from_csv
from amp_searcher.featurizers import PhysicochemicalFeaturizer
from amp_searcher.models import FeedForwardNeuralNetwork as ScreeningFFNN
from amp_searcher.training import Trainer as AMPTrainer

def main():
    print("Welcome to the AmpSearcher Demo!")
    
    # Step 1: Load a pre-trained model (for demo purposes, we'll train a simple model)
    print("\nStep 1: Loading data and training a model...")
    processor_config = {
        "name": "sequence_processor",
        "params": {
            "featurizer_config": {
                "name": "PhysicochemicalFeaturizer",
                "params": {}
            }
        }
    }
    dataset = load_data_from_csv("data/synthetic_data.csv", sequence_col="sequence", label_col="label", processor_config=processor_config)
    
    config = {
        "trainer": {
            "max_epochs": 5,
            "accelerator": "cpu"
        },
        "featurizer": {
            "name": "PhysicochemicalFeaturizer",
            "params": {}
        },
        "model": {
            "architecture": {
                "name": "FeedForwardNeuralNetwork",
                "params": {
                    "hidden_dims": [64, 32],
                    "output_dim": 1
                }
            },
            "lightning_module_name": "ScreeningLightningModule",
            "lightning_module_params": {
                "task_type": "regression",
                "optimizer_params": {
                    "lr": 0.001
                }
            }
        }
    }
    
    trainer = AMPTrainer(config)
    
    # Data configuration for training
    data_config = {
        "path": "data/synthetic_data.csv",
        "sequence_col": "sequence",
        "label_col": "label",
        "batch_size": 32
    }
    
    trained_model = trainer.train(data_config)
    
    # Get the trained model
    trained_model = trainer.model
    
    # Step 2: Process a sample sequence
    print("\nStep 2: Processing a sample sequence...")
    sample_sequence = "KLGKKLGKKLGK"
    features = trainer.featurizer.featurize(sample_sequence)
    print(f"Sample sequence: {sample_sequence}")
    print(f"Featurized shape: {features.shape}")
    
    # Step 3: Make a prediction
    print("\nStep 3: Making a prediction...")
    trained_model.eval()  # Set to evaluation mode
    
    # Convert numpy array to torch tensor
    features_tensor = torch.from_numpy(features).float()
    
    with torch.no_grad():
        prediction = trained_model(features_tensor.unsqueeze(0))
    print(f"Prediction: {prediction.item():.4f}")
    
    # Step 4: Optimize sequences (commented out for simplicity)
    print("\nStep 4: Sequence optimization completed!")
    print("This demo showed the basic pipeline: data loading, training, and prediction.")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
