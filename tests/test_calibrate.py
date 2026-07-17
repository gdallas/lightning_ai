from lightning_decoding.calibrate import knob_combinations, select_best


def test_knob_combinations_single_axis() -> None:
    assert knob_combinations({"T": [0.7, 1.0]}) == [{"T": 0.7}, {"T": 1.0}]


def test_knob_combinations_multi_axis() -> None:
    combos = knob_combinations({"a": [1, 2], "b": [3]})
    assert combos == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def _row(knobs, validity, distinct):
    return {"knobs": knobs, "validity_rate": validity, "distinct_valid_per_prompt": distinct}


def test_select_best_prefers_distinct_above_floor() -> None:
    rows = [
        _row({"p": 0.90}, 0.95, 1.2),
        _row({"p": 0.98}, 0.92, 1.8),  # highest distinct while clearing the floor
        _row({"p": 0.995}, 0.80, 3.0),  # more distinct but below the validity floor
    ]
    assert select_best(rows, validity_floor=0.9)["knobs"] == {"p": 0.98}


def test_select_best_falls_back_when_nothing_meets_floor() -> None:
    rows = [
        _row({"p": 0.9}, 0.60, 1.0),
        _row({"p": 0.98}, 0.70, 2.5),
    ]
    assert select_best(rows, validity_floor=0.9)["knobs"] == {"p": 0.98}


def test_select_best_breaks_ties_on_validity() -> None:
    rows = [
        _row({"delta": 1}, 0.91, 2.0),
        _row({"delta": 2}, 0.97, 2.0),
    ]
    assert select_best(rows, validity_floor=0.9)["knobs"] == {"delta": 2}
