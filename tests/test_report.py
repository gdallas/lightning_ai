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
