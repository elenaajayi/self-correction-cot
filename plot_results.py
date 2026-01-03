"""Generate ready plots from self-correction experiment outputs.

Produces:
- ablation_effect.png
- direction_separation.png (if mean projection stats are available)
- combined_results.png
"""

import json
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np


MECH_PATH = Path("mechanistic_results.json")
RESULTS_PATH = Path("results.json")


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def print_structures(mech, results):
    print("=== mechanistic_results.json structure ===")
    print(f"Top-level keys: {list(mech.keys())}")
    if "ablation_results" in mech:
        print("ablation_results keys:", list(mech["ablation_results"].keys()))
    if "summary" in mech:
        print("summary keys:", list(mech["summary"].keys()))
    print("=== results.json structure ===")
    if results:
        print("Example entry keys:", list(results[0].keys()))
        print(f"Total entries: {len(results)}")
    else:
        print("results.json is empty.")


def plot_ablation_effect(mech):
    ablation = mech.get("ablation_results", {})
    summary = mech.get("summary", {})
    baseline = summary.get("baseline", None)
    # Optional extra stats for direction (if present)
    mean_corr = summary.get("mean_projection_correcting")
    mean_non = summary.get("mean_projection_non_correcting")
    acc = summary.get("separation_accuracy")

    labels = []
    values = []

    if baseline is not None:
        labels.append("baseline")
        values.append(baseline)

    for k in sorted(ablation.keys(), key=lambda x: float(x)):
        labels.append(str(k))
        values.append(ablation[k])

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c72b0"] + ["#dd8452"] * (len(labels) - 1) if baseline is not None else ["#dd8452"] * len(labels)
    ax.bar(labels, values, color=colors)

    if baseline is not None:
        ax.axhline(baseline, color="#4c72b0", linestyle="--", linewidth=1.5, label="baseline")
    ax.set_ylabel("Correction rate")
    ax.set_xlabel("Ablation strength")
    ax.set_title("Ablation Effect on Correction Rate\n(ablation unexpectedly increased correction)")
    ax.set_ylim(0, 1.05)
    if baseline is not None:
        ax.legend()
    plt.tight_layout()
    plt.savefig("ablation_effect.png", dpi=200)
    plt.close(fig)


def plot_direction_separation(mech):
    summary = mech.get("summary", {})
    mean_corr = summary.get("mean_projection_correcting")
    mean_non = summary.get("mean_projection_non_correcting")
    acc = summary.get("separation_accuracy")

    if mean_corr is None or mean_non is None or acc is None:
        print("direction_separation: missing projection stats in mechanistic_results.json; skipping plot.")
        return

    labels = ["correcting", "non-correcting"]
    values = [mean_corr, mean_non]
    colors = ["#55a868", "#c44e52"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean projection")
    ax.set_title(f"Direction Separation (accuracy={acc:.3f})")
    plt.tight_layout()
    plt.savefig("direction_separation.png", dpi=200)
    plt.close(fig)


def analyze_by_edit_type(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[row["edit_type"]].append(row["correction_score"])
    return {k: (sum(v) / len(v)) for k, v in grouped.items() if v}


def analyze_by_position(results):
    buckets = {"early": [], "mid": [], "late": []}
    for row in results:
        frac = row["position_fraction"]
        if frac < 0.33:
            buckets["early"].append(row["correction_score"])
        elif frac <= 0.66:
            buckets["mid"].append(row["correction_score"])
        else:
            buckets["late"].append(row["correction_score"])
    return {k: (sum(v) / len(v)) for k, v in buckets.items() if v}


def analyze_correction_types(results):
    counter = Counter(r["correction_type"] for r in results)
    return dict(counter)


def plot_combined_results(results):
    by_edit = analyze_by_edit_type(results)
    by_pos = analyze_by_position(results)
    type_counts = analyze_correction_types(results)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # consistent palette
    palette = ["#4c72b0", "#55a868", "#c44e52", "#dd8452", "#8172b3", "#937860"]

    # Edit type
    ax = axes[0]
    labels = list(by_edit.keys())
    vals = [by_edit[k] for k in labels]
    ax.bar(labels, vals, color=palette[: len(labels)])
    ax.set_title("Correction by Edit Type")
    ax.set_ylabel("Mean correction score")
    ax.set_ylim(0, 1.0)

    # Position
    ax = axes[1]
    pos_labels = ["early", "mid", "late"]
    pos_vals = [by_pos.get(k, 0.0) for k in pos_labels]
    ax.bar(pos_labels, pos_vals, color=palette[2:2 + len(pos_labels)])
    ax.set_title("Correction by Position")
    ax.set_ylim(0, 1.0)

    # Correction type distribution
    ax = axes[2]
    type_labels = list(type_counts.keys())
    type_vals = [type_counts[k] for k in type_labels]
    ax.bar(type_labels, type_vals, color=palette[4:4 + len(type_labels)])
    ax.set_title("Correction Type Distribution")
    ax.set_ylabel("Count")

    for ax in axes:
        ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig("combined_results.png", dpi=200)
    plt.close(fig)


def main():
    if not MECH_PATH.exists():
        raise FileNotFoundError(f"{MECH_PATH} not found. Run mechanistic phase first.")
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found. Run experiment first.")

    mech = load_json(MECH_PATH)
    results = load_json(RESULTS_PATH)

    print_structures(mech, results)

    # Set global style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })

    plot_ablation_effect(mech)
    plot_direction_separation(mech)
    plot_combined_results(results)
    print("Saved: ablation_effect.png, direction_separation.png (if stats exist), combined_results.png")


if __name__ == "__main__":
    main()

