"""Model loading and generation utilities."""

from typing import Tuple

import torch  # type: ignore[import-not-found]
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

import config


def load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the DeepSeek R1 distilled Qwen model and tokenizer."""
    model_name = config.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure padding is defined for generation.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def generate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_tokens: int) -> str:
    """Generate continuation text, returning only newly generated tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=config.TEMPERATURE,
        do_sample=True,
        pad_token_id=pad_token_id,
    )[0]

    generated_ids = output_ids[input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()

