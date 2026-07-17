from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def single_token_answer_spaces(
    categories: dict[str, list[str]],
    tokenizer: Any,
    *,
    leading_space: bool = True,
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    filtered: dict[str, list[str]] = {}
    report: dict[str, dict[str, int]] = {}

    for category, words in categories.items():
        kept: list[str] = []
        for word in words:
            text = f" {word}" if leading_space else word
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                kept.append(word)
        filtered[category] = kept
        report[category] = {"input": len(words), "kept": len(kept)}

    return filtered, report


def filter_category_file(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: Any,
) -> dict[str, dict[str, int]]:
    with Path(input_path).open("r", encoding="utf-8") as handle:
        categories = json.load(handle)

    filtered, report = single_token_answer_spaces(categories, tokenizer)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return report

