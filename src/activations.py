"""Activation extraction utilities for self-correction experiments."""

from typing import List, Tuple

import numpy as np
import torch

from .data import regenerate_from_prefix  # noqa: F401 (may be useful to callers)


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Could not find transformer layers on the model.")


class ActivationCache:
    """Capture activations from a specified transformer block."""

    def __init__(self, model, layer_idx: int = -4):
        self.model = model
        layers = _get_layers(model)
        self.layer_idx = layer_idx
        self.layer = layers[layer_idx]
        self.cached_activation = None
        self.hook = self.layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        # Many HF blocks return (hidden_states,) or hidden_states directly
        hidden = output[0] if isinstance(output, tuple) else output
        self.cached_activation = hidden.detach()

    def get_last_token_activation(self):
        if self.cached_activation is None:
            raise RuntimeError("No activation captured. Run a forward pass first.")
        return self.cached_activation[:, -1, :]

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def __del__(self):
        self.remove()


def get_activations_after_edit(model, tokenizer, problem: str, edited_prefix: str, layer_idx: int = -4) -> np.ndarray:
    """Forward the edited prompt and return activation at the token right after the edit."""
    prompt = f"Problem: {problem}\n\nSolution:{edited_prefix}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    cache = ActivationCache(model, layer_idx=layer_idx)
    with torch.no_grad():
        model(**inputs)
    acts = cache.get_last_token_activation()
    cache.remove()
    return acts.squeeze(0).cpu().numpy()


def collect_contrastive_activations(model, tokenizer, experiment_results: List[dict], layer_idx: int = -4) -> Tuple[np.ndarray, np.ndarray]:
    """Collect correcting vs non-correcting activations based on experiment results."""
    correcting: List[np.ndarray] = []
    non_correcting: List[np.ndarray] = []

    for row in experiment_results:
        problem = row.get("problem")
        edited_prefix = row.get("edited_prefix")
        if problem is None or edited_prefix is None:
            continue  # insufficient info to reconstruct prompt

        try:
            act = get_activations_after_edit(model, tokenizer, problem, edited_prefix, layer_idx=layer_idx)
        except Exception:
            continue  # skip problematic examples

        correction_type = row.get("correction_type", "")
        is_correcting = correction_type != "accepted_edit"

        if is_correcting:
            correcting.append(act)
        else:
            non_correcting.append(act)

    correcting_arr = np.stack(correcting) if correcting else np.empty((0,))
    non_correcting_arr = np.stack(non_correcting) if non_correcting else np.empty((0,))
    return correcting_arr, non_correcting_arr


def smoke_test_activation(model, tokenizer, layer_idx: int = -4) -> Tuple[int, ...]:
    """Lightweight smoke test to ensure activation capture works."""
    prompt = "Problem: 2 + 2\n\nSolution: It's 4."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    cache = ActivationCache(model, layer_idx=layer_idx)
    with torch.no_grad():
        model(**inputs)
    acts = cache.get_last_token_activation()
    cache.remove()

    shape = tuple(acts.shape)
    print(f"Activation captured with shape: {shape}")
    return shape

