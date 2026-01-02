"""Edit functions for perturbing sentences in self-correction experiments."""

from typing import Callable, Dict
import re


def edit_factual(sentence: str) -> str:
    """Perturb numeric facts by adding 7 to the first number found."""
    match = re.search(r"\d+(\.\d+)?", sentence)
    if not match:
        return f"{sentence} which equals 999"

    number_text = match.group(0)
    number = float(number_text) if "." in number_text else int(number_text)
    new_value = number + 7
    return re.sub(r"\d+(\.\d+)?", str(new_value), sentence, count=1)


def edit_style(sentence: str) -> str:
    """Make the sentence casual in tone."""
    return f"so like {sentence.lower()} or whatever"


def edit_contradict(sentence: str) -> str:
    """Introduce a contradiction cue."""
    return "Wait, that doesn't seem right. Let me reconsider this step."


def edit_random(sentence: str) -> str:
    """Insert a random, unrelated statement."""
    return "Purple elephants dance magnificently on Tuesdays."


EDIT_FUNCTIONS: Dict[str, Callable[[str], str]] = {
    "factual": edit_factual,
    "style": edit_style,
    "contradict": edit_contradict,
    "random": edit_random,
}


def apply_edit(sentence: str, edit_type: str) -> str:
    """Apply the requested edit type to a sentence."""
    if edit_type not in EDIT_FUNCTIONS:
        raise ValueError(f"Unknown edit type: {edit_type}")
    return EDIT_FUNCTIONS[edit_type](sentence)

