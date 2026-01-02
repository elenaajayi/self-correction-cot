"""Convenient imports for self-correction experiment modules."""

from .model import load_model, generate
from .data import load_gsm8k, generate_cot, split_into_sentences
from .edits import apply_edit, EDIT_FUNCTIONS
from .correction import measure_correction
from .experiment import run_experiment
from .analysis import analyze_results, print_summary
from .plots import create_all_plots
from .activations import ActivationCache, collect_contrastive_activations
from .directions import find_correction_direction, validate_direction
from .interventions import run_ablation_experiment

__all__ = [
    "load_model",
    "generate",
    "load_gsm8k",
    "generate_cot",
    "split_into_sentences",
    "apply_edit",
    "EDIT_FUNCTIONS",
    "measure_correction",
    "run_experiment",
    "analyze_results",
    "print_summary",
    "create_all_plots",
    "ActivationCache",
    "collect_contrastive_activations",
    "find_correction_direction",
    "validate_direction",
    "run_ablation_experiment",
]

