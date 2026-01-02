"""Correction measurement utilities."""

from typing import Dict, Optional

import config


def _jaccard_similarity(a_tokens: set, b_tokens: set) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a_tokens and not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union else 0.0


def measure_correction(tokenizer, original_continuation: str, new_continuation: str) -> Dict[str, object]:
    """Measure how much the new continuation corrects the original."""
    original_tokens = set(tokenizer.tokenize(original_continuation))
    new_tokens = set(tokenizer.tokenize(new_continuation))

    token_overlap = _jaccard_similarity(original_tokens, new_tokens)

    lower_new = new_continuation.lower()
    phrase_found: Optional[str] = None
    for phrase in config.EXPLICIT_CORRECTION_PHRASES:
        if phrase.lower() in lower_new:
            phrase_found = phrase
            break

    return {
        "token_overlap": token_overlap,
        "explicit_correction": phrase_found is not None,
        "correction_phrase_found": phrase_found,
    }


def classify_correction_type(overlap_score: float, explicit: bool) -> str:
    """Classify the correction based on overlap and explicit cues."""
    if overlap_score > 0.5 and explicit:
        return "strong_correction"
    if overlap_score > 0.5:
        return "implicit_correction"
    if 0.25 <= overlap_score <= 0.5:
        return "partial_correction"
    return "accepted_edit"

