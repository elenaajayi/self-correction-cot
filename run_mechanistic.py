"""Run only the mechanistic investigation phase on existing results."""

import json
import sys
from pathlib import Path

import numpy as np

import config
from src import (
    load_model,
    collect_contrastive_activations,
    find_correction_direction,
    validate_direction,
    run_ablation_experiment,
)
from src.interventions import compare_ablation_to_baseline


def main():
    results_path = Path(config.RESULTS_FILE)
    if not results_path.exists():
        print(f"Missing {results_path}; run the main experiment first.")
        sys.exit(1)

    with open(results_path, "r") as f:
        results = json.load(f)

    if not isinstance(results, list) or not results:
        print("No results found or invalid format; nothing to process.")
        sys.exit(1)

    print("Loading model/tokenizer...")
    model, tokenizer = load_model()

    print("Collecting contrastive activations...")
    correcting, non_correcting = collect_contrastive_activations(
        model, tokenizer, results
    )

    if correcting.size == 0 or non_correcting.size == 0:
        print("Insufficient activations to compute direction.")
        sys.exit(1)

    print("Finding correction direction...")
    direction = find_correction_direction(correcting, non_correcting)

    # Ensure float16 direction for ablation step
    direction = direction.astype(np.float16)

    print("Validating direction separation...")
    validate_direction(direction, correcting, non_correcting)

    ablation_strengths = [0.5, 1.0, 2.0]
    print(f"Running ablation experiment with strengths {ablation_strengths}...")
    ablation_results = run_ablation_experiment(
        model, tokenizer, results, direction, strengths=ablation_strengths
    )
    summary = compare_ablation_to_baseline(results, ablation_results)

    mech_out = {
        "ablation_results": ablation_results,
        "direction_dim": len(direction),
        "direction_dtype": str(direction.dtype),
        "summary": summary,
    }

    mech_path = Path("mechanistic_results.json")
    with open(mech_path, "w") as f:
        json.dump(mech_out, f, indent=2)

    print(f"Mechanistic results saved to {mech_path}")


if __name__ == "__main__":
    main()

