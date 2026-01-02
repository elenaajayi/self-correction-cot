"""Utilities for deriving correction directions and probes from activations."""

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def find_correction_direction(correcting_acts: np.ndarray, non_correcting_acts: np.ndarray) -> np.ndarray:
    """Compute a unit vector pointing from non-correcting to correcting activations."""
    if correcting_acts.size == 0 or non_correcting_acts.size == 0:
        raise ValueError("Both correcting and non-correcting activations are required.")

    mean_correcting = correcting_acts.mean(axis=0)
    mean_non_correcting = non_correcting_acts.mean(axis=0)
    direction = mean_correcting - mean_non_correcting

    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Direction norm is zero; activations may be identical.")
    return direction / norm


def train_edit_probe(edited_acts: np.ndarray, normal_acts: np.ndarray) -> Tuple[LogisticRegression, float]:
    """Train a logistic regression probe to distinguish edited from normal activations."""
    if edited_acts.size == 0 or normal_acts.size == 0:
        raise ValueError("Both edited and normal activations are required.")

    X = np.concatenate([edited_acts, normal_acts], axis=0)
    y = np.concatenate([np.ones(len(edited_acts)), np.zeros(len(normal_acts))], axis=0)

    if len(np.unique(y)) < 2:
        raise ValueError("Need both edited and normal samples for training.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return probe, acc


def validate_direction(direction: np.ndarray, correcting_acts: np.ndarray, non_correcting_acts: np.ndarray) -> float:
    """Project activations onto direction and compute separation accuracy."""
    if direction.ndim != 1:
        raise ValueError("Direction must be a 1D vector.")
    if correcting_acts.size == 0 or non_correcting_acts.size == 0:
        raise ValueError("Both correcting and non-correcting activations are required.")

    dir_norm = np.linalg.norm(direction)
    if dir_norm == 0:
        raise ValueError("Direction norm must be non-zero.")
    unit_dir = direction / dir_norm

    corr_proj = correcting_acts @ unit_dir
    non_proj = non_correcting_acts @ unit_dir

    mean_corr = corr_proj.mean()
    mean_non = non_proj.mean()
    threshold = 0.5 * (mean_corr + mean_non)

    corr_correct = (corr_proj > threshold).sum()
    corr_total = len(corr_proj)
    non_correct = (non_proj <= threshold).sum()
    non_total = len(non_proj)

    total_correct = corr_correct + non_correct
    total = corr_total + non_total
    accuracy = total_correct / total if total else 0.0

    print(f"Mean projection correcting: {mean_corr:.4f}")
    print(f"Mean projection non-correcting: {mean_non:.4f}")
    print(f"Separation accuracy: {accuracy:.4f}")
    return accuracy

