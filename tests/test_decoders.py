import pytest

torch = pytest.importorskip("torch")

from lightning_decoding import decoders


class FakeTokenizer:
    def __call__(self, prompt: str, return_tensors: str):
        return {"input_ids": torch.tensor([[0]])}

    def decode(self, ids):
        return f"tok{ids[0]}"


def test_gap_sampler_excludes_argmax_and_samples_within_delta(monkeypatch) -> None:
    logits = torch.tensor([10.0, 9.5, 8.0, 3.0])
    monkeypatch.setattr(decoders, "last_logits", lambda model, tokenizer, prompt: logits)

    result = decoders.gap_sampler(
        model=object(),
        tokenizer=FakeTokenizer(),
        prompt="prompt",
        cfg={"delta": 2.0},
        rng=torch.Generator().manual_seed(0),
    )

    assert result.token_id in {1, 2}
    assert result.token_id != 0
    assert result.meta["candidate_set_size"] == 2
    assert result.meta["fallback"] is False


def test_gap_sampler_falls_back_when_no_candidate(monkeypatch) -> None:
    logits = torch.tensor([10.0, 1.0, 0.0])
    monkeypatch.setattr(decoders, "last_logits", lambda model, tokenizer, prompt: logits)

    result = decoders.gap_sampler(
        model=object(),
        tokenizer=FakeTokenizer(),
        prompt="prompt",
        cfg={"delta": 2.0},
        rng=torch.Generator().manual_seed(0),
    )

    assert result.token_id == 0
    assert result.meta["fallback"] is True

