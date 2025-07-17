# Optimizers in AmpSearcher

## Overview

AmpSearcher provides sophisticated optimization algorithms for discovering and designing novel antimicrobial peptides. These optimizers use trained models to guide the search for peptides with desired properties, employing various strategies from evolutionary algorithms to Monte Carlo methods.

## Base Optimizer

All optimizers inherit from the `BaseOptimizer` abstract class:

```python
from amp_searcher.optimizers.base import BaseOptimizer
from typing import List, Tuple

class CustomOptimizer(BaseOptimizer):
    def __init__(self, scoring_model, **kwargs):
        super().__init__(scoring_model, **kwargs)
    
    def initialize(self, initial_population: List[str], **kwargs):
        # Initialize optimizer state
        pass
    
    def propose_candidates(self, num_candidates: int) -> List[str]:
        # Generate new candidate sequences
        pass
    
    def update_state(self, sequences: List[str], scores: List[List[float]]):
        # Update optimizer based on evaluation results
        pass
    
    def is_converged(self) -> bool:
        # Check convergence criteria
        return False
    
    def get_best_candidates(self, num_best: int) -> List[Tuple[str, List[float]]]:
        # Return best sequences found so far
        pass
```

## Available Optimizers

### 1. Genetic Algorithm (GA)

The Genetic Algorithm optimizer uses evolutionary principles to evolve peptide sequences.

```python
from amp_searcher.optimizers.ga import GeneticAlgorithmOptimizer

# Initialize GA optimizer
ga_optimizer = GeneticAlgorithmOptimizer(
    scoring_model=your_trained_model,
    population_size=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    selection_strategy="elitist",
    amino_acids="ACDEFGHIKLMNPQRSTVWY",
    max_generations=50,
    convergence_threshold=0.001
)

# Initialize with starting population
initial_population = [
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    "FLPIIAKLLSGLL",
    "GIGKFLHSAKKFGKAFVGEIMNS"
]

ga_optimizer.initialize(
    initial_population=initial_population,
    target_length=20
)

# Run optimization
for generation in range(50):
    # Generate new candidates
    candidates = ga_optimizer.propose_candidates(num_candidates=50)
    
    # Evaluate candidates (this would use your scoring model)
    scores = [[model.predict([seq])[0]] for seq in candidates]
    
    # Update optimizer state
    ga_optimizer.update_state(candidates, scores)
    
    # Check convergence
    if ga_optimizer.is_converged():
        break

# Get best results
best_sequences = ga_optimizer.get_best_candidates(num_best=10)
for seq, score in best_sequences:
    print(f"Sequence: {seq}, Score: {score[0]:.4f}")
```

**GA Parameters:**
- `population_size`: Number of sequences in each generation
- `mutation_rate`: Probability of amino acid mutation
- `crossover_rate`: Probability of crossover between parents
- `selection_strategy`: Selection method ("elitist", "tournament", "roulette")
- `max_generations`: Maximum number of generations
- `convergence_threshold`: Threshold for convergence detection

### 2. Monte Carlo Tree Search (MCTS)

MCTS explores the sequence space using a tree-based approach with exploration and exploitation balance.

```python
from amp_searcher.optimizers.mcts import MCTSOptimizer

# Initialize MCTS optimizer
mcts_optimizer = MCTSOptimizer(
    scoring_model=your_trained_model,
    num_simulations=1000,
    exploration_constant=1.4,
    max_sequence_length=30,
    amino_acids="ACDEFGHIKLMNPQRSTVWY"
)

# Initialize with starting sequences
initial_population = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"]
mcts_optimizer.initialize(initial_population=initial_population)

# Run optimization
for iteration in range(100):
    candidates = mcts_optimizer.propose_candidates(num_candidates=10)
    scores = [[model.predict([seq])[0]] for seq in candidates]
    mcts_optimizer.update_state(candidates, scores)
    
    if mcts_optimizer.is_converged():
        break

# Get best results
best_sequences = mcts_optimizer.get_best_candidates(num_best=5)
```

**MCTS Parameters:**
- `num_simulations`: Number of simulations per iteration
- `exploration_constant`: UCB1 exploration parameter (typically √2)
- `max_sequence_length`: Maximum allowed sequence length

### 3. Sequential Monte Carlo (SMC)

SMC uses particle filtering to maintain a population of promising sequences.

```python
from amp_searcher.optimizers.smc import SMCOptimizer

# Initialize SMC optimizer
smc_optimizer = SMCOptimizer(
    scoring_model=your_trained_model,
    num_particles=200,
    resampling_threshold=0.5,
    mutation_rate=0.1,
    amino_acids="ACDEFGHIKLMNPQRSTVWY"
)

# Run SMC search
initial_population = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"] * 10
best_sequences = smc_optimizer.search(
    initial_population=initial_population,
    num_iterations=100,
    target_length=25
)

print("Best sequences found:")
for seq, score in best_sequences[:5]:
    print(f"Sequence: {seq}, Score: {score:.4f}")
```

**SMC Parameters:**
- `num_particles`: Number of particles in the population
- `resampling_threshold`: Threshold for effective sample size
- `mutation_rate`: Rate of sequence mutations

## Optimizer Factory

Use the factory pattern to create optimizers from configuration:

```python
from amp_searcher.optimizers.optimizer_factory import OptimizerFactory

# Register custom optimizer
@OptimizerFactory.register("custom_optimizer")
class CustomOptimizer(BaseOptimizer):
    def __init__(self, scoring_model, custom_param=None, **kwargs):
        super().__init__(scoring_model, **kwargs)
        self.custom_param = custom_param

# Build optimizer from configuration
optimizer_config = {
    "name": "GeneticAlgorithmOptimizer",
    "population_size": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "max_generations": 50
}

optimizer = OptimizerFactory.build_optimizer(
    optimizer_config["name"],
    scoring_model=your_model,
    **{k: v for k, v in optimizer_config.items() if k != "name"}
)
```

## Selection Strategies

AmpSearcher provides various selection strategies for evolutionary algorithms:

### Elitist Selection

```python
from amp_searcher.optimizers.selection import ElitistSelection

selection = ElitistSelection()
selected = selection.select(
    population=sequences,
    scores=scores,
    num_selected=50
)
```

### NSGA-II Selection (Multi-objective)

```python
from amp_searcher.optimizers.selection import NSGA2Selection

# For multi-objective optimization
nsga2_selection = NSGA2Selection()
selected = nsga2_selection.select(
    population=sequences,
    scores=multi_objective_scores,  # List of [score1, score2, ...] for each sequence
    num_selected=50
)
```

## Configuration Examples

### Genetic Algorithm Configuration

```yaml
optimizer:
  name: GeneticAlgorithmOptimizer
  population_size: 100
  mutation_rate: 0.1
  crossover_rate: 0.8
  selection_strategy: elitist
  max_generations: 50
  convergence_threshold: 0.001
  amino_acids: "ACDEFGHIKLMNPQRSTVWY"
```

### MCTS Configuration

```yaml
optimizer:
  name: MCTSOptimizer
  num_simulations: 1000
  exploration_constant: 1.4
  max_sequence_length: 30
  amino_acids: "ACDEFGHIKLMNPQRSTVWY"
```

### SMC Configuration

```yaml
optimizer:
  name: SMCOptimizer
  num_particles: 200
  resampling_threshold: 0.5
  mutation_rate: 0.1
  amino_acids: "ACDEFGHIKLMNPQRSTVWY"
```

## Multi-objective Optimization

Optimize for multiple objectives simultaneously:

```python
# Multi-objective scoring model
class MultiObjectiveModel:
    def __init__(self, amp_model, toxicity_model):
        self.amp_model = amp_model
        self.toxicity_model = toxicity_model
    
    def predict(self, sequences):
        amp_scores = self.amp_model.predict(sequences)
        toxicity_scores = self.toxicity_model.predict(sequences)
        
        # Return list of [amp_score, -toxicity_score] for each sequence
        return [[amp, -tox] for amp, tox in zip(amp_scores, toxicity_scores)]

# Use with NSGA-II selection
multi_model = MultiObjectiveModel(amp_model, toxicity_model)
ga_optimizer = GeneticAlgorithmOptimizer(
    scoring_model=multi_model,
    selection_strategy="nsga2",
    population_size=100
)
```

## Constraint Handling

Add constraints to the optimization process:

```python
class ConstrainedOptimizer(GeneticAlgorithmOptimizer):
    def __init__(self, min_length=5, max_length=50, forbidden_patterns=None, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.forbidden_patterns = forbidden_patterns or []
    
    def _is_valid_sequence(self, sequence):
        # Length constraints
        if not (self.min_length <= len(sequence) <= self.max_length):
            return False
        
        # Pattern constraints
        for pattern in self.forbidden_patterns:
            if pattern in sequence:
                return False
        
        return True
    
    def propose_candidates(self, num_candidates):
        candidates = []
        attempts = 0
        max_attempts = num_candidates * 10
        
        while len(candidates) < num_candidates and attempts < max_attempts:
            candidate = super().propose_candidates(1)[0]
            if self._is_valid_sequence(candidate):
                candidates.append(candidate)
            attempts += 1
        
        return candidates

# Usage
constrained_optimizer = ConstrainedOptimizer(
    scoring_model=your_model,
    min_length=10,
    max_length=30,
    forbidden_patterns=["PP", "CC"],  # Avoid proline-proline and cysteine-cysteine
    population_size=100
)
```

## Advanced Optimization Strategies

### Guided Optimization

Use known active sequences to guide the search:

```python
class GuidedGAOptimizer(GeneticAlgorithmOptimizer):
    def __init__(self, guide_sequences=None, guide_weight=0.3, **kwargs):
        super().__init__(**kwargs)
        self.guide_sequences = guide_sequences or []
        self.guide_weight = guide_weight
    
    def _guided_mutation(self, sequence):
        if not self.guide_sequences:
            return self._mutate(sequence)
        
        # Occasionally use amino acids from guide sequences
        if random.random() < self.guide_weight:
            guide_seq = random.choice(self.guide_sequences)
            position = random.randint(0, len(sequence) - 1)
            if position < len(guide_seq):
                sequence = sequence[:position] + guide_seq[position] + sequence[position+1:]
        
        return self._mutate(sequence)

# Usage with known AMP sequences
known_amps = [
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    "GIGKFLHSAKKFGKAFVGEIMNS",
    "FLPIIAKLLSGLL"
]

guided_optimizer = GuidedGAOptimizer(
    scoring_model=your_model,
    guide_sequences=known_amps,
    guide_weight=0.3,
    population_size=100
)
```

### Adaptive Parameters

Dynamically adjust optimization parameters:

```python
class AdaptiveGAOptimizer(GeneticAlgorithmOptimizer):
    def __init__(self, adaptive_mutation=True, **kwargs):
        super().__init__(**kwargs)
        self.adaptive_mutation = adaptive_mutation
        self.generation = 0
        self.best_scores_history = []
    
    def update_state(self, sequences, scores):
        super().update_state(sequences, scores)
        self.generation += 1
        
        # Track best score
        current_best = max(max(score) for score in scores)
        self.best_scores_history.append(current_best)
        
        # Adapt mutation rate based on progress
        if self.adaptive_mutation and len(self.best_scores_history) > 10:
            recent_improvement = (
                self.best_scores_history[-1] - self.best_scores_history[-10]
            )
            
            if recent_improvement < 0.01:  # Slow progress
                self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
            else:  # Good progress
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)

# Usage
adaptive_optimizer = AdaptiveGAOptimizer(
    scoring_model=your_model,
    adaptive_mutation=True,
    population_size=100
)
```

## Evaluation and Analysis

### Optimization Metrics

```python
def analyze_optimization_run(optimizer, num_generations=50):
    metrics = {
        "best_scores": [],
        "average_scores": [],
        "diversity": [],
        "convergence_generation": None
    }
    
    for generation in range(num_generations):
        candidates = optimizer.propose_candidates(50)
        scores = [[model.predict([seq])[0]] for seq in candidates]
        optimizer.update_state(candidates, scores)
        
        # Calculate metrics
        flat_scores = [score[0] for score in scores]
        metrics["best_scores"].append(max(flat_scores))
        metrics["average_scores"].append(sum(flat_scores) / len(flat_scores))
        
        # Calculate sequence diversity (average pairwise distance)
        diversity = calculate_sequence_diversity(candidates)
        metrics["diversity"].append(diversity)
        
        # Check convergence
        if optimizer.is_converged() and metrics["convergence_generation"] is None:
            metrics["convergence_generation"] = generation
    
    return metrics

def calculate_sequence_diversity(sequences):
    """Calculate average Hamming distance between sequences."""
    if len(sequences) < 2:
        return 0
    
    total_distance = 0
    comparisons = 0
    
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            distance = hamming_distance(sequences[i], sequences[j])
            total_distance += distance
            comparisons += 1
    
    return total_distance / comparisons if comparisons > 0 else 0

def hamming_distance(seq1, seq2):
    """Calculate Hamming distance between two sequences."""
    min_len = min(len(seq1), len(seq2))
    distance = abs(len(seq1) - len(seq2))  # Length difference
    distance += sum(c1 != c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len]))
    return distance
```

### Visualization

```python
import matplotlib.pyplot as plt

def plot_optimization_progress(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Best scores over time
    axes[0, 0].plot(metrics["best_scores"])
    axes[0, 0].set_title("Best Score Over Generations")
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Score")
    
    # Average scores over time
    axes[0, 1].plot(metrics["average_scores"])
    axes[0, 1].set_title("Average Score Over Generations")
    axes[0, 1].set_xlabel("Generation")
    axes[0, 1].set_ylabel("Score")
    
    # Diversity over time
    axes[1, 0].plot(metrics["diversity"])
    axes[1, 0].set_title("Population Diversity Over Generations")
    axes[1, 0].set_xlabel("Generation")
    axes[1, 0].set_ylabel("Average Hamming Distance")
    
    # Score distribution
    axes[1, 1].hist(metrics["best_scores"], bins=20, alpha=0.7)
    axes[1, 1].set_title("Distribution of Best Scores")
    axes[1, 1].set_xlabel("Score")
    axes[1, 1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
```

## Best Practices

1. **Start with diverse initial populations** to avoid local optima
2. **Balance exploration and exploitation** through appropriate parameter tuning
3. **Use multi-objective optimization** for realistic peptide design
4. **Apply domain constraints** (length, composition, patterns)
5. **Monitor convergence** and diversity metrics
6. **Validate results** with experimental data when possible
7. **Use ensemble models** for more robust scoring

## Troubleshooting

### Common Issues

1. **Premature convergence**: Increase mutation rate, population size, or diversity
2. **Slow convergence**: Adjust selection pressure, crossover rate
3. **Poor quality solutions**: Improve scoring model, add constraints
4. **Memory issues**: Reduce population size, use batch evaluation

### Parameter Tuning Guidelines

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| Population Size | 50-500 | Larger = better exploration, slower |
| Mutation Rate | 0.01-0.3 | Higher = more exploration |
| Crossover Rate | 0.6-0.9 | Higher = more recombination |
| Selection Pressure | Low-High | Higher = faster convergence |

## Integration Example

Complete example integrating optimizer with trained model:

```python
import yaml
from amp_searcher.pipelines.screening_pipeline import ScreeningPipeline
from amp_searcher.optimizers.ga import GeneticAlgorithmOptimizer

# Load trained model
with open("configs/main/examples/screening_ffnn_physicochemical.yaml", "r") as f:
    config = yaml.safe_load(f)

pipeline = ScreeningPipeline(
    model_config=config["model"],
    model_checkpoint_path="models/trained_model.ckpt",
    featurizer_config=config["featurizer"]
)

# Create optimizer
optimizer = GeneticAlgorithmOptimizer(
    scoring_model=pipeline,
    population_size=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    max_generations=50
)

# Run optimization
initial_population = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"] * 10
optimizer.initialize(initial_population=initial_population)

for generation in range(50):
    candidates = optimizer.propose_candidates(50)
    scores = [[pipeline.predict([seq])[0].item()] for seq in candidates]
    optimizer.update_state(candidates, scores)
    
    if generation % 10 == 0:
        best = optimizer.get_best_candidates(1)[0]
        print(f"Generation {generation}: Best score = {best[1][0]:.4f}")

# Get final results
best_sequences = optimizer.get_best_candidates(10)
print("\nTop 10 optimized sequences:")
for i, (seq, score) in enumerate(best_sequences, 1):
    print(f"{i:2d}. {seq} (Score: {score[0]:.4f})")
```

This optimization system provides powerful tools for discovering novel antimicrobial peptides with desired properties through intelligent search algorithms.
