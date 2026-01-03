"""Interventions and ablation utilities for self-correction experiments."""

from typing import Dict, List

import numpy as np
import torch

import config
from .correction import measure_correction, classify_correction_type
from .data import regenerate_from_prefix


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Could not find transformer layers on the model.")


def generate_with_ablation(model, tokenizer, prompt: str, direction: np.ndarray, layer_idx: int, strength: float, max_new_tokens: int = config.MAX_CONTINUATION_TOKENS) -> str:
    """Generate text while ablating along a correction direction on a chosen layer."""
    layers = _get_layers(model)
    target_layer = layers[layer_idx]

    direction_tensor = torch.tensor(direction, dtype=torch.float16, device=model.device)
    direction_tensor = direction_tensor / (torch.linalg.norm(direction_tensor) + 1e-8)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Match direction dtype to hidden dtype to avoid type mismatch
        dir_t = direction_tensor.to(hidden.dtype)
        # projection shape: [batch, seq]
        projection = torch.matmul(hidden, dir_t)
        hidden = hidden - strength * projection.unsqueeze(-1) * dir_t
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    hook = target_layer.register_forward_hook(hook_fn)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=config.TEMPERATURE,
                do_sample=True,
                pad_token_id=pad_token_id,
            )[0]
        gen_ids = output_ids[input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    finally:
        hook.remove()
    return text


def run_ablation_experiment(model, tokenizer, edited_examples: List[Dict], correction_direction: np.ndarray, strengths: List[float] = None) -> Dict[float, float]:
    """For each strength, ablate and measure how often corrections still occur."""
    if strengths is None:
        strengths = [0.5, 1.0, 2.0]

    results: Dict[float, float] = {}

    for strength in strengths:
        corrections = 0
        total = 0
        for ex in edited_examples:
            problem = ex.get("problem")
            edited_prefix = ex.get("edited_prefix")
            original_continuation = ex.get("original_continuation", "")
            if problem is None or edited_prefix is None:
                continue

            prompt = f"Problem: {problem}\n\nSolution:{edited_prefix}"
            new_continuation = generate_with_ablation(
                model,
                tokenizer,
                prompt,
                correction_direction,
                ex.get("layer_idx", -4),
                strength,
                max_new_tokens=config.MAX_CONTINUATION_TOKENS,
            )

            correction = measure_correction(tokenizer, original_continuation, new_continuation)
            correction_type = classify_correction_type(correction["token_overlap"], correction["explicit_correction"])
            is_correcting = correction_type != "accepted_edit"
            corrections += int(is_correcting)
            total += 1

        results[strength] = corrections / total if total else 0.0

    return results


def compare_ablation_to_baseline(baseline_results: List[Dict], ablation_results: Dict[float, float]) -> Dict[str, object]:
    """Print a simple comparison table of correction rates and return summary."""
    # Baseline correction rate: proportion not classified as accepted_edit
    if baseline_results:
        baseline_corr = sum(1 for r in baseline_results if r.get("correction_type") != "accepted_edit") / len(baseline_results)
    else:
        baseline_corr = 0.0

    print("Strength | Correction Rate")
    print("--------------------------")
    print(f"baseline | {baseline_corr:.3f}")
    for strength, rate in sorted(ablation_results.items()):
        print(f"{strength:8.2f} | {rate:.3f}")

    return {"baseline": baseline_corr, "ablation": ablation_results}

