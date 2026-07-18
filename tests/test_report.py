import pytest


def test_comparison_bar_with_ci_writes_file(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from lightning_decoding.report import save_comparison_bar

    rows = [
        {"label": "greedy", "distinct_valid_per_prompt": 0.5, "distinct_valid_ci_low": 0.3, "distinct_valid_ci_high": 0.7},
        {"label": "gap_sampler", "distinct_valid_per_prompt": 1.5, "distinct_valid_ci_low": 1.0, "distinct_valid_ci_high": 2.0},
    ]
    out = tmp_path / "bar.png"
    save_comparison_bar(
        rows,
        "distinct_valid_per_prompt",
        out,
        ci_keys=("distinct_valid_ci_low", "distinct_valid_ci_high"),
    )
    assert out.exists() and out.stat().st_size > 0


def test_comparison_bar_without_ci_writes_file(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from lightning_decoding.report import save_comparison_bar

    out = tmp_path / "bar.png"
    save_comparison_bar([{"label": "greedy", "validity_rate": 0.8}], "validity_rate", out)
    assert out.exists() and out.stat().st_size > 0


def test_answer_breakdown_counts_and_sorts() -> None:
    from lightning_decoding.report import build_answer_breakdown

    prompts = [{"prompt_id": "category:color", "key": "color", "prompt_text": "x"}]
    method_records = {
        "greedy": [
            {"prompt_id": "category:color", "normalized": "red", "token_str": " red", "valid": True},
        ],
        "nucleus": [
            {"prompt_id": "category:color", "normalized": "blue", "token_str": " blue", "valid": True},
            {"prompt_id": "category:color", "normalized": "blue", "token_str": " blue", "valid": True},
            {"prompt_id": "category:color", "normalized": "red", "token_str": " red", "valid": True},
            {"prompt_id": "category:color", "normalized": "same", "token_str": " same", "valid": False},
        ],
    }
    breakdown = build_answer_breakdown(method_records, prompts)
    assert len(breakdown) == 1 and breakdown[0]["key"] == "color"
    assert breakdown[0]["methods"]["greedy"] == [{"answer": "red", "count": 1, "valid": True}]
    nucleus = breakdown[0]["methods"]["nucleus"]
    assert nucleus[0] == {"answer": "blue", "count": 2, "valid": True}  # most frequent first
    assert {a["answer"] for a in nucleus} == {"blue", "red", "same"}


def test_answers_markdown_writes_table(tmp_path) -> None:
    from lightning_decoding.report import save_answers_markdown

    breakdown = [
        {
            "prompt_id": "category:color",
            "key": "color",
            "methods": {
                "greedy": [{"answer": "red", "count": 20, "valid": True}],
                "nucleus": [
                    {"answer": "blue", "count": 5, "valid": True},
                    {"answer": "same", "count": 3, "valid": False},
                ],
            },
        }
    ]
    out = tmp_path / "answers.md"
    save_answers_markdown(breakdown, ["greedy", "nucleus"], out, task_name="category")
    text = out.read_text(encoding="utf-8")
    assert "| color |" in text and "red✓(20)" in text and "same✗(3)" in text
