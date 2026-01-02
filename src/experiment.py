"""Run self-correction experiments across edited reasoning traces."""

from typing import List, Dict, Any
import json
import os
import torch

from tqdm import tqdm

import config
from .data import generate_cot, split_into_sentences, regenerate_from_prefix
from .edits import apply_edit
from .correction import measure_correction, classify_correction_type


def _select_position_index(sentences: List[str], position_fraction: float) -> int:
    """Map a fraction to a valid sentence index."""
    if not sentences:
        return 0
    idx = int(position_fraction * len(sentences))
    return max(0, min(len(sentences) - 1, idx))


def run_experiment(model, tokenizer, dataset, n_problems: int) -> List[Dict[str, Any]]:
    """Run edits and measure corrections over a subset of the dataset."""
    results: List[Dict[str, Any]] = []
    total = min(n_problems, len(dataset))

    def _checkpoint():
        if not results:
            return
        try:
            with open(config.CHECKPOINT_PATH, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Checkpoint write failed: {e}")

    for problem_idx in tqdm(range(total), desc="Self-correction"):
        row = dataset[problem_idx]
        problem = row.get("question", str(row))

        try:
            cot = generate_cot(model, tokenizer, problem)
        except RuntimeError as e:
            # Commonly OOM; attempt to clear and skip
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Skipping problem {problem_idx} due to generation error: {e}")
            continue

        sentences = split_into_sentences(cot)
        if len(sentences) < 4:
            continue

        for position_fraction in config.POSITION_FRACTIONS:
            pos_idx = _select_position_index(sentences, position_fraction)
            original_sentence = sentences[pos_idx]

            for edit_type in config.EDIT_TYPES:
                edited_sentence = apply_edit(original_sentence, edit_type)

                prefix_sentences = sentences[:pos_idx] + [edited_sentence]
                prefix_text = " ".join(prefix_sentences)

                try:
                    new_continuation = regenerate_from_prefix(
                        model,
                        tokenizer,
                        problem,
                        prefix_text,
                    )
                except RuntimeError as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"Skipping continuation for problem {problem_idx}, edit {edit_type} due to error: {e}")
                    continue

                original_continuation = " ".join(sentences[pos_idx + 1 :])
                correction = measure_correction(
                    tokenizer, original_continuation, new_continuation
                )
                correction_type = classify_correction_type(
                    correction["token_overlap"], correction["explicit_correction"]
                )

                results.append(
                    {
                        "problem": problem,
                        "problem_idx": problem_idx,
                        "position": pos_idx,
                        "position_fraction": position_fraction,
                        "edit_type": edit_type,
                        "original_sentence": original_sentence,
                        "edited_sentence": edited_sentence,
                        "edited_prefix": prefix_text,
                        "original_continuation": original_continuation,
                        "correction_score": correction["token_overlap"],
                        "explicit_correction": correction["explicit_correction"],
                        "correction_phrase_found": correction["correction_phrase_found"],
                        "correction_type": correction_type,
                    }
                )

        if config.CHECKPOINT_EVERY and (problem_idx + 1) % config.CHECKPOINT_EVERY == 0:
            _checkpoint()

    print(f"Processed {len(results)} edited continuations.")
    return results

