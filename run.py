"""Main entry point for self-correction experiments."""

import json
import os

import config
from src import (
    load_model,
    generate,
    load_gsm8k,
    generate_cot,
    split_into_sentences,
    apply_edit,
    EDIT_FUNCTIONS,
    measure_correction,
    run_experiment,
    analyze_results,
    print_summary,
    create_all_plots,
)
from src.activations import collect_contrastive_activations
from src.directions import find_correction_direction, validate_direction
from src.interventions import run_ablation_experiment, compare_ablation_to_baseline


def main():
    print("Self-Correction Experiment on R1-Distill")

    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    model, tokenizer = load_model()
    dataset = load_gsm8k(config.N_PROBLEMS)

    results = run_experiment(model, tokenizer, dataset, config.N_PROBLEMS)

    with open(config.RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    summary = analyze_results(results)
    print_summary(results)
    create_all_plots(results, config.PLOTS_DIR)

    # Phase 2: Mechanistic Investigation (if time permits)
    try:
        print("Phase 2: Finding correction direction...")
        correcting, non_correcting = collect_contrastive_activations(model, tokenizer, results)
        direction = find_correction_direction(correcting, non_correcting)
        validate_direction(direction, correcting, non_correcting)

        ablation_strengths = [0.5, 1.0, 2.0]
        ablation_results = run_ablation_experiment(
            model, tokenizer, results, direction, strengths=ablation_strengths
        )
        compare_ablation_to_baseline(results, ablation_results)

        mech_out = {
            "ablation_results": ablation_results,
            "direction_dim": len(direction),
        }
        with open("mechanistic_results.json", "w") as f:
            json.dump(mech_out, f, indent=2)
    except Exception as e:
        print(f"Phase 2 skipped or failed: {e}")

    print(f"Done! Results: {config.RESULTS_FILE}, Plots: {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()

