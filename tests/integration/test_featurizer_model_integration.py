import torch
import numpy as np
from amp_searcher.featurizers import PhysicochemicalFeaturizer
from amp_searcher.models.screening.lightning_module import ScreeningLightningModule
from amp_searcher.models.architectures.feed_forward_nn import FeedForwardNeuralNetwork

def test_physicochemical_to_screening_model_integration():
    """
    Tests the integration of PhysicochemicalFeaturizer with ScreeningLightningModule.
    It verifies that the output of the featurizer can be successfully processed
    by the screening model in a forward pass.
    """
    # 1. Setup the Featurizer
    featurizer = PhysicochemicalFeaturizer()
    sequence = "GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV"
    
    # 2. Featurize the sequence
    feature_vector = featurizer.featurize(sequence)
    assert isinstance(feature_vector, np.ndarray)
    assert feature_vector.shape == (10,)
    
    # 3. Setup the Model
    input_dim = featurizer.feature_dim
    model_architecture = FeedForwardNeuralNetwork(input_dim=input_dim, output_dim=1, hidden_dims=[20, 10])
    lightning_model = ScreeningLightningModule(model_architecture=model_architecture, task_type="classification")
    
    # 4. Prepare tensor for the model
    # Convert numpy array to torch tensor and add a batch dimension (B, C) -> (1, 10)
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
    assert feature_tensor.shape == (1, input_dim)
    
    # 5. Perform forward pass
    lightning_model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Deactivate autograd for inference
        output = lightning_model.forward(feature_tensor)
        
    # 6. Assert output properties
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1) # (batch_size, output_dim)
