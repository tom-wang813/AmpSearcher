import argparse
import glob
import os
import random
from typing import List, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from tqdm import tqdm

from amp_searcher.featurizers.physicochemical import PhysicochemicalFeaturizer
from amp_searcher.models.architectures.feed_forward_nn import FeedForwardNeuralNetwork
from amp_searcher.models.screening.lightning_module import ScreeningLightningModule
from amp_searcher.utils.logging_utils import setup_logger
from amp_searcher.utils.config import Config
from amp_searcher.utils.performance_monitoring import timer, MemoryTracker

logger = setup_logger("run_ga_visualization")

@timer
def load_model_and_featurizer(checkpoint_path: str, device: torch.device) -> Tuple[ScreeningLightningModule, PhysicochemicalFeaturizer]:
    """Loads the model and featurizer from a checkpoint."""
    featurizer = PhysicochemicalFeaturizer()
    model_architecture = FeedForwardNeuralNetwork(
        input_dim=featurizer.feature_dim,
        hidden_dims=[64, 32],
        output_dim=1
    )
    model = ScreeningLightningModule.load_from_checkpoint(
        checkpoint_path,
        model_architecture=model_architecture,
        task_type="classification"
    )
    model.to(device)
    model.eval()
    return model, featurizer

@timer
def predict_scores(model: ScreeningLightningModule, featurizer: PhysicochemicalFeaturizer, population: List[str], device: torch.device) -> np.ndarray:
    """Predicts scores for a population of sequences."""
    features = np.array([featurizer.featurize(seq) for seq in population])
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(features_tensor)
        scores = torch.sigmoid(logits).cpu().numpy().flatten()
    return scores

def find_latest_checkpoint(log_dir: str) -> str:
    """Finds the latest checkpoint file in a directory."""
    checkpoint_files = glob.glob(os.path.join(log_dir, "**", "*.ckpt"), recursive=True)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]

def generate_initial_population(size: int, min_len: int, max_len: int, alphabet: str) -> List[str]:
    """Generates a random initial population of peptide sequences."""
    return ["".join(random.choice(alphabet) for _ in range(random.randint(min_len, max_len))) for _ in range(size)]

def select_parents(population: List[str], scores: np.ndarray, num_elites: int) -> List[str]:
    """Selects parents using elitism and tournament selection."""
    sorted_indices = np.argsort(scores)[::-1]
    elites = [population[i] for i in sorted_indices[:num_elites]]
    
    parents = elites.copy()
    for _ in range(len(population) - num_elites):
        idx = random.sample(range(len(population)), 5)
        tournament_scores = scores[idx]
        winner_idx = idx[np.argmax(tournament_scores)]
        parents.append(population[winner_idx])
    return parents

def crossover(parent1: str, parent2: str) -> Tuple[str, str]:
    """Performs single-point crossover."""
    min_len = min(len(parent1), len(parent2))
    if min_len < 2:
        return parent1, parent2
    point = random.randint(1, min_len - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(sequence: str, mutation_rate: float, alphabet: str) -> str:
    """Performs random mutation."""
    mutated_sequence = list(sequence)
    for i in range(len(mutated_sequence)):
        if random.random() < mutation_rate:
            mutated_sequence[i] = random.choice(alphabet)
    return "".join(mutated_sequence)

def create_new_generation(parents: List[str], crossover_rate: float, mutation_rate: float, num_elites: int, alphabet: str) -> List[str]:
    """Creates a new generation."""
    next_gen = parents[:num_elites]
    for i in range(num_elites, len(parents), 2):
        p1, p2 = parents[i], parents[i+1] if i+1 < len(parents) else parents[0]
        c1, c2 = crossover(p1, p2) if random.random() < crossover_rate else (p1, p2)
        next_gen.append(mutate(c1, mutation_rate, alphabet))
        if len(next_gen) < len(parents):
            next_gen.append(mutate(c2, mutation_rate, alphabet))
    return next_gen

@timer
def plot_umap(population_features: np.ndarray, scores: np.ndarray, generation: int, output_dir: str):
    """Plots the UMAP visualization for a generation."""
    reducer = umap.UMAP(n_neighbors=min(15, len(population_features) - 1), n_components=2, random_state=42)
    embedding = reducer.fit_transform(population_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=scores, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Fitness Score')
    plt.title(f'UMAP Visualization of AMP Population - Generation {generation}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    frame_path = os.path.join(output_dir, f'generation_{generation:03d}.png')
    plt.savefig(frame_path)
    
    plt.ion()
    plt.show()
    plt.pause(0.1)
    plt.close()

@timer
def main(config: Config):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        output_dir = config.get("output_dir", "results/ga_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint_path = find_latest_checkpoint(config.get("log_dir"))
        logger.info(f"Found latest checkpoint: {checkpoint_path}")
        model, featurizer = load_model_and_featurizer(checkpoint_path, device)

        population = generate_initial_population(
            config.get("population_size"),
            config.get("min_len"),
            config.get("max_len"),
            config.get("alphabet")
        )
        
        for generation in tqdm(range(config.get("generations")), desc="Evolving Population"):
            with MemoryTracker(f"Generation {generation}"):
                scores = predict_scores(model, featurizer, population, device)
                features = np.array([featurizer.featurize(seq) for seq in population])
                plot_umap(features, scores, generation, output_dir)
                
                num_elites = int(config.get("elite_fraction") * config.get("population_size"))
                parents = select_parents(population, scores, num_elites)
                population = create_new_generation(
                    parents,
                    config.get("crossover_rate"),
                    config.get("mutation_rate"),
                    num_elites,
                    config.get("alphabet")
                )
        
        images = [imageio.imread(os.path.join(output_dir, f'generation_{i:03d}.png')) for i in range(config.get("generations"))]
        gif_path = config.get("gif_path", "results/ga_umap_evolution.gif")
        imageio.mimsave(gif_path, images, fps=2)
        logger.info(f"Saved GIF to {gif_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA with UMAP visualization.")
    parser.add_argument("--config_path", type=str, default="configs/ga_visualization_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = Config(args.config_path)
    main(config)
