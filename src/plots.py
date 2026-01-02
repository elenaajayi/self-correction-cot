"""Plotting utilities for self-correction experiments."""

import os
from typing import List

import matplotlib.pyplot as plt

from .analysis import analyze_by_edit_type, analyze_by_position, analyze_correction_types


def plot_by_edit_type(results: List[dict], save_path: str) -> None:
    data = analyze_by_edit_type(results)
    labels = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="skyblue")
    plt.ylabel("Mean correction score")
    plt.title("Correction by Edit Type")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_by_position(results: List[dict], save_path: str) -> None:
    data = analyze_by_position(results)
    labels = ["early", "mid", "late"]
    values = [data.get(k, 0.0) for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="salmon")
    plt.ylabel("Mean correction score")
    plt.title("Correction by Position")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_correction_types(results: List[dict], save_path: str) -> None:
    data = analyze_correction_types(results)
    labels = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="seagreen")
    plt.ylabel("Count")
    plt.title("Correction Types")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_all_plots(results: List[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    plot_by_edit_type(results, os.path.join(output_dir, "by_edit_type.png"))
    plot_by_position(results, os.path.join(output_dir, "by_position.png"))
    plot_correction_types(results, os.path.join(output_dir, "correction_types.png"))

    print(f"Saved plots to {output_dir}")

