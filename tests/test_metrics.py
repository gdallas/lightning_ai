import pytest

pytest.importorskip("torch")

from lightning_decoding.metrics import distinct_valid_per_prompt, mode_share, summarize


def test_metrics_summary() -> None:
    rows = [
        {"prompt_id": "p1", "normalized": "cat", "valid": True},
        {"prompt_id": "p1", "normalized": "dog", "valid": True},
        {"prompt_id": "p1", "normalized": "cat", "valid": True},
        {"prompt_id": "p2", "normalized": "red", "valid": True},
        {"prompt_id": "p2", "normalized": "zzz", "valid": False},
    ]

    assert distinct_valid_per_prompt(rows) == 1.5
    assert mode_share(rows) == 0.75
    summary = summarize(rows, {"p1": 10, "p2": 10})
    assert summary["validity_rate"] == 0.8
    assert summary["distinct_valid_per_prompt"] == 1.5

