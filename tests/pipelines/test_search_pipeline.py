import pytest
import os

from amp_searcher.pipelines.search_pipeline import SearchPipeline

# Define the real checkpoint paths
GENERATIVE_CHECKPOINT_PATH = "/Users/wang-workair/cui/AmpSearcher/lightning_logs/generative_vae_physicochemical/version_1/checkpoints/epoch=9-step=1040.ckpt"
SCREENING_CHECKPOINT_PATH = "/Users/wang-workair/cui/AmpSearcher/lightning_logs/amp_training_run/version_4/checkpoints/epoch=4-step=520.ckpt"

# Define the real model configurations
REAL_GENERATIVE_MODEL_CONFIG = {
    "lightning_module_name": "GenerativeLightningModule",
    "architecture": {
        "name": "VAE",
        "params": {
            "input_dim": 10,
            "latent_dim": 5,
            "encoder_hidden_dims": [32],
            "decoder_hidden_dims": [32],
        },
    },
}

REAL_SCREENING_MODEL_CONFIG = {
    "lightning_module_name": "ScreeningLightningModule",
    "architecture": {
        "name": "FeedForwardNeuralNetwork",
        "params": {"input_dim": 10, "output_dim": 1, "hidden_dims": [64, 32]},
    },
    "lightning_module_params": {
        "task_type": "classification",
    },
}


@pytest.fixture
def real_generative_model_config():
    return REAL_GENERATIVE_MODEL_CONFIG


@pytest.fixture
def real_screening_model_config():
    return REAL_SCREENING_MODEL_CONFIG


@pytest.fixture
def real_generative_model_checkpoint():
    if not os.path.exists(GENERATIVE_CHECKPOINT_PATH):
        pytest.skip(
            f"Generative checkpoint not found at {GENERATIVE_CHECKPOINT_PATH}. Please train the model first."
        )
    return GENERATIVE_CHECKPOINT_PATH


@pytest.fixture
def real_screening_model_checkpoint():
    if not os.path.exists(SCREENING_CHECKPOINT_PATH):
        pytest.skip(
            f"Screening checkpoint not found at {SCREENING_CHECKPOINT_PATH}. Please train the model first."
        )
    return SCREENING_CHECKPOINT_PATH


@pytest.fixture
def featurizer_config():
    return {"name": "PhysicochemicalFeaturizer"}


@pytest.fixture
def dummy_sequence_decoder_config():
    # Assuming SimpleFeatureDecoder doesn't need complex params for this test
    return {"name": "SimpleFeatureDecoder", "params": {}}


def test_search_pipeline_init(
    real_generative_model_checkpoint,
    real_screening_model_checkpoint,
    featurizer_config,
    real_generative_model_config,
    real_screening_model_config,
    dummy_sequence_decoder_config,
):
    pipeline = SearchPipeline(
        generative_model_config=real_generative_model_config,
        generative_model_checkpoint_path=real_generative_model_checkpoint,
        screening_model_config=real_screening_model_config,
        screening_model_checkpoint_path=real_screening_model_checkpoint,
        featurizer_config=featurizer_config,
        sequence_decoder_config=dummy_sequence_decoder_config,
        search_config={"num_generations": 1, "top_k": 1},
    )
    assert pipeline.featurizer is not None
    assert pipeline.generative_model is not None
    assert pipeline.screening_model is not None
    assert pipeline.sequence_decoder is not None
    assert pipeline.generative_model.training is False
    assert pipeline.screening_model.training is False


@pytest.mark.skip(
    reason="Skipping search pipeline test due to VAE training instability"
)
def test_search_pipeline_search(
    real_generative_model_checkpoint,
    real_screening_model_checkpoint,
    featurizer_config,
    real_generative_model_config,
    real_screening_model_config,
    dummy_sequence_decoder_config,
):
    pipeline = SearchPipeline(
        generative_model_config=real_generative_model_config,
        generative_model_checkpoint_path=real_generative_model_checkpoint,
        screening_model_config=real_screening_model_config,
        screening_model_checkpoint_path=real_screening_model_checkpoint,
        featurizer_config=featurizer_config,
        sequence_decoder_config=dummy_sequence_decoder_config,
        search_config={"num_generations": 1, "top_k": 1},
    )
    num_generations = 10
    top_k = 3
    results = pipeline.search(num_generations=num_generations, top_k=top_k)

    assert isinstance(results, list)
    assert len(results) == top_k
    for seq, score in results:
        assert isinstance(seq, str)
        assert seq.startswith("[Decoded Features:")
        assert score >= 0 and score <= 1  # Assuming classification probabilities


def test_search_pipeline_unsupported_featurizer(
    real_generative_model_checkpoint,
    real_screening_model_checkpoint,
    real_generative_model_config,
    real_screening_model_config,
    dummy_sequence_decoder_config,
):
    with pytest.raises(
        ValueError, match="No featurizer registered with name 'InvalidFeaturizer'"
    ):
        SearchPipeline(
            generative_model_config=real_generative_model_config,
            generative_model_checkpoint_path=real_generative_model_checkpoint,
            screening_model_config=real_screening_model_config,
            screening_model_checkpoint_path=real_screening_model_checkpoint,
            featurizer_config={"name": "InvalidFeaturizer"},
            sequence_decoder_config=dummy_sequence_decoder_config,
            search_config={"num_generations": 1, "top_k": 1},
        )
