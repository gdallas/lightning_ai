from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def configure_torch_threads(num_threads: int | None = None) -> None:
    threads = num_threads or max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(threads)


@lru_cache(maxsize=1)
def _threads_configured() -> bool:
    configure_torch_threads()
    return True


def load_model(name: str, *, trust_remote_code: bool = False) -> tuple[Any, Any]:
    _threads_configured()
    tokenizer = load_tokenizer(name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def load_tokenizer(name: str, *, trust_remote_code: bool = False) -> Any:
    return AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)


def format_prompt(tokenizer: Any, model_name: str, prompt: str) -> str:
    if "qwen" not in model_name.lower() or not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
