
# Self-Correction in Chain-of-Thought Reasoning

Investigating why language models correct back to their original reasoning trajectory when their chain-of-thought is edited mid-generation.

## Overview

Reasoning models like DeepSeek-R1 and o1 generate extended chains-of-thought before producing answers. When you edit these chains mid-generation and regenerate the continuation, models often "self-correct" back toward their original reasoning path. This project characterizes this behavior and attempts to identify the mechanistic substrate. This READme was authored by Claude's Sonnet 4.5.

## Experimental Design

**Model**: DeepSeek-R1-Distill-Qwen-1.5B  
**Dataset**: 25 GSM8K math problems  
**Method**: Generate baseline CoT, apply edit at specific position, regenerate continuation, measure correction via Jaccard token overlap

### Edit Types
- Factual: Introduce incorrect calculations
- Style: Convert to casual speech patterns
- Contradict: Insert explicit doubt/reconsideration
- Random: Insert semantically unrelated text

### Positions
- Early (20% through CoT)
- Mid (50% through CoT)
- Late (80% through CoT)

## Key Results

### Behavioral Findings

**Content-agnostic correction**: Edit type had minimal effect on correction rates (0.34-0.37 across all types). This preliminary finding suggests trajectory alignment rather than semantic content processing, though validation across non-math reasoning tasks is needed.

**Position effects**: Mid-chain edits showed highest correction (0.40), late edits lowest (0.30). This pattern is consistent with trajectory strength increasing with context.

**Soft correction**: 83% of cases showed partial correction (0.25-0.5 overlap) rather than binary accept/reject behavior. This suggests inertial drift rather than explicit error detection.

### Mechanistic Findings

Extracted layer -4 residual stream activations at edit points. Found a direction separating correcting from non-correcting cases with 75% accuracy (mean projections: 31.78 vs -24.70).

Ablation experiment: Removing this direction during generation *increased* correction rates (baseline 0.89 → 0.94 at strength 1.0). This result is opposite the initial hypothesis that the direction drives correction.

**Interpretation**: The finding is consistent with a flexibility mechanism that permits deviation from the established trajectory. When ablated, the model becomes more rigid and returns to its original path more frequently. However, this interpretation remains speculative without control ablations on random directions and other layers.

## Repository Structure

```
├── notebooks/
│   └── self_correction_colab.ipynb    # Main analysis notebook
├── plots/
│   ├── by_edit_type.png               # Correction by edit type
│   ├── by_position.png                # Correction by position
│   ├── correction_types.png           # Distribution of correction types
│   ├── direction_separation.png       # Mean projection comparison
│   ├── ablation_effect.png            # Ablation strength vs correction rate
│   └── combined_results.png           # Combined overview
├── src/
│   ├── model.py                       # Model loading and generation
│   ├── data.py                        # GSM8K dataset handling
│   ├── edits.py                       # Edit type implementations
│   ├── correction.py                  # Correction measurement
│   ├── experiment.py                  # Main experimental loop
│   ├── activations.py                 # Activation extraction
│   ├── directions.py                  # Direction finding and validation
│   ├── interventions.py               # Ablation experiments
│   ├── analysis.py                    # Statistical analysis
│   └── plots.py                       # Visualization
├── config.py                          # Configuration parameters
├── run.py                             # Phase 1 + Phase 2 pipeline
├── run_mechanistic.py                 # Phase 2 standalone
├── results.json                       # Behavioral experiment results
└── mechanistic_results.json           # Mechanistic analysis results
```

## Limitations

**Scale**: Small model (1.5B parameters) and limited sample (25 problems) may not reflect behavior in larger reasoning models or non-math domains.

**Design choices**: Layer -4 selection based on heuristic rather than systematic sweep. Correction likely involves multi-layer circuits. Jaccard similarity threshold (0.5) is arbitrary.

**Missing controls**: No ablation on random/orthogonal directions to confirm specificity. No testing at other layers. No analysis of internal state during ablated generation to validate flexibility interpretation.

**Mechanistic uncertainty**: The flexibility mechanism interpretation is consistent with the ablation result but remains speculative. Alternative explanations include distributed multi-layer circuits, general noise artifacts from intervention, or disruption of context integration bottlenecks.

## Future Directions

**Validation**: Control ablations on random directions and other layers. Bidirectional intervention (test if adding the direction reduces correction). Replication on symbolic logic, creative writing, and code generation tasks.

**Mechanistic depth**: Multi-layer sweep to map correction-related computations. Attribution analysis to identify what drives return-to-path if not this direction. Internal state examination during ablation to distinguish flexibility loss from context disruption.

**Scaling**: Testing on larger models (Qwen 32B, R1-70B, o1-mini) to validate findings at production scale.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run behavioral characterization:
```bash
python run.py
```

Run mechanistic analysis only:
```bash
python run_mechanistic.py
```

## Requirements

- PyTorch
- transformers
- datasets
- numpy
- matplotlib
- tqdm

See `requirements.txt` for full dependencies.

