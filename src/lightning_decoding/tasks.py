from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Task(Protocol):
    name: str

    def prompts(self) -> list[dict[str, str]]:
        ...

    def is_valid(self, key: str, answer: str) -> bool:
        ...

    def normalize(self, answer: str) -> str:
        ...

    def answer_space_size(self, key: str) -> int:
        ...


def normalize_single_word(answer: str) -> str:
    text = answer.strip().lower()
    if not text:
        return ""
    first = text.split()[0]
    first = first.strip(string.punctuation + "\"'`")
    singular_s_endings = ("ss", "is", "us")
    if len(first) > 3 and first.endswith("s") and not first.endswith(singular_s_endings):
        first = first[:-1]
    return first


@dataclass
class CategoryTask:
    categories: dict[str, list[str]]
    prompt_template: str = "The most common {category} is the"
    name: str = "category"

    @classmethod
    def from_json(cls, path: str | Path, prompt_template: str | None = None) -> "CategoryTask":
        with Path(path).open("r", encoding="utf-8") as handle:
            categories = json.load(handle)
        kwargs = {"categories": categories}
        if prompt_template:
            kwargs["prompt_template"] = prompt_template
        return cls(**kwargs)

    def prompts(self) -> list[dict[str, str]]:
        return [
            {
                "prompt_id": f"category:{category}",
                "prompt_text": self.prompt_template.format(category=category),
                "key": category,
            }
            for category in sorted(self.categories)
        ]

    def normalize(self, answer: str) -> str:
        return normalize_single_word(answer)

    def is_valid(self, key: str, answer: str) -> bool:
        return self.normalize(answer) in set(self.categories[key])

    def answer_space_size(self, key: str) -> int:
        return len(self.categories[key])


@dataclass
class RhymeTask:
    seed_words: list[str]
    prompt_template: str = "Q: Say a word that rhymes with {word}.\nA: A word that rhymes with {word} is"
    name: str = "rhyme"

    @classmethod
    def from_json(cls, path: str | Path, prompt_template: str | None = None) -> "RhymeTask":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        kwargs = {"seed_words": payload["seed_words"]}
        if prompt_template:
            kwargs["prompt_template"] = prompt_template
        return cls(**kwargs)

    def prompts(self) -> list[dict[str, str]]:
        return [
            {
                "prompt_id": f"rhyme:{word}",
                "prompt_text": self.prompt_template.format(word=word),
                "key": word,
            }
            for word in self.seed_words
        ]

    def normalize(self, answer: str) -> str:
        return normalize_single_word(answer)

    def is_valid(self, key: str, answer: str) -> bool:
        import pronouncing

        return self.normalize(answer) in set(pronouncing.rhymes(key))

    def answer_space_size(self, key: str) -> int:
        import pronouncing

        return len(pronouncing.rhymes(key))


def load_task(cfg: dict) -> Task:
    task_type = cfg["type"]
    prompt_template = cfg.get("prompt_template")
    if task_type == "category":
        return CategoryTask.from_json(cfg["path"], prompt_template)
    if task_type == "rhyme":
        return RhymeTask.from_json(cfg["path"], prompt_template)
    raise ValueError(f"Unknown task type: {task_type}")
