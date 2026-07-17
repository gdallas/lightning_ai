import pytest

from lightning_decoding.analysis import (
    classify_valid_trials,
    compare_depths,
    depth_groups,
    modal_token_id,
)

LENS = {
    "p1": {"prompt_id": "p1", "num_layers": 4, "lens_argmax_per_layer": [10, 20, 30, 7]},
}


def test_modal_token_id_is_final_layer_argmax() -> None:
    assert modal_token_id(LENS["p1"]) == 7


def test_classify_labels_and_filters() -> None:
    trials = [
        {"prompt_id": "p1", "token_id": 7, "valid": True, "commitment_depth": 1, "token_str": "a"},
        {"prompt_id": "p1", "token_id": 9, "valid": True, "commitment_depth": 3, "token_str": "b"},
        {"prompt_id": "p1", "token_id": 5, "valid": False, "commitment_depth": 2},  # invalid
        {"prompt_id": "p1", "token_id": 8, "valid": True},  # no depth
        {"prompt_id": "p2", "token_id": 1, "valid": True, "commitment_depth": 0},  # no lens record
    ]
    classified = classify_valid_trials(trials, LENS)
    labels = {(row["token_id"], row["label"]) for row in classified}
    assert labels == {(7, "modal"), (9, "novel")}


def test_depth_groups_split() -> None:
    classified = [
        {"label": "modal", "commitment_depth": 1},
        {"label": "novel", "commitment_depth": 3},
        {"label": "modal", "commitment_depth": 2},
    ]
    assert depth_groups(classified) == ([1, 2], [3])


def test_compare_depths_reports_gap_and_pvalue() -> None:
    result = compare_depths([1, 1, 2], [3, 4, 5])
    assert result["modal"]["n"] == 3
    assert result["novel"]["n"] == 3
    assert result["p_value"] is not None
    assert result["mean_depth_gap"] > 0  # novel commits later


def test_compare_depths_handles_empty_group() -> None:
    result = compare_depths([1, 2], [])
    assert result["novel"]["n"] == 0
    assert result["p_value"] is None
    assert result["mean_depth_gap"] is None


def test_histogram_writes_file(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from lightning_decoding.analysis import save_commitment_histogram

    out = tmp_path / "hist.png"
    save_commitment_histogram([1, 2, 2], [3, 4], out, num_layers=6)
    assert out.exists() and out.stat().st_size > 0
