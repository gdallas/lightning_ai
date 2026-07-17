from lightning_decoding.tasks import CategoryTask, normalize_single_word


def test_normalize_single_word_strips_punctuation_and_plural_s() -> None:
    assert normalize_single_word(" Cats, and dogs") == "cat"
    assert normalize_single_word("glass") == "glass"
    assert normalize_single_word("  Paris.") == "paris"


def test_category_task_validity() -> None:
    task = CategoryTask({"animal": ["cat", "dog"]})
    assert task.is_valid("animal", " cats")
    assert not task.is_valid("animal", "table")

