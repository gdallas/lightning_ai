from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def configure_torch_threads(num_threads: int | None = None) -> None:
    threads = num_threads or max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(threads)


configure_torch_threads()


def load_model(
    name: str,
    *,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    tokenizer = load_tokenizer(
        name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tokenizer


def load_tokenizer(
    name: str,
    *,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> Any:
    return AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )


def format_prompt(tokenizer: Any, model_name: str, prompt: str) -> str:
    if "qwen" not in model_name.lower() or not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
