"""Data loading and generation helpers for self-correction experiments."""

from typing import List
import re

from datasets import load_dataset  # type: ignore[import-not-found]

import config
from .model import generate


def load_gsm8k(n_samples: int):
    """Load the GSM8K test split and return the first n_samples rows."""
    dataset = load_dataset(config.DATASET_NAME, "main", split="test")
    return dataset.select(range(min(n_samples, len(dataset))))


def generate_cot(model, tokenizer, problem: str) -> str:
    """Generate chain-of-thought for a problem using the model's native reasoning."""
    prompt = f"Problem: {problem}\n\nSolution:"
    return generate(model, tokenizer, prompt, max_tokens=config.MAX_COT_TOKENS)


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\(\[])")


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while avoiding splits on decimals like 3.14."""
    sentences = _SENTENCE_SPLIT_PATTERN.split(text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned


def regenerate_from_prefix(model, tokenizer, problem: str, prefix: str) -> str:
    """Regenerate continuation from a given prefix, returning only the new part."""
    prompt = f"Problem: {problem}\n\nSolution:{prefix}"
    return generate(model, tokenizer, prompt, max_tokens=config.MAX_CONTINUATION_TOKENS)

