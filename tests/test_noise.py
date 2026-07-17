import pytest

torch = pytest.importorskip("torch")

from lightning_decoding.noise import NoiseState, perturb_, stable_hash, unperturb_


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(4, 4, bias=False)
        self.norm = torch.nn.LayerNorm(4)


def test_stable_hash_is_deterministic() -> None:
    assert stable_hash("layers.0.dense.weight") == stable_hash("layers.0.dense.weight")


def test_noise_round_trip_restores_weights() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    state = NoiseState(model)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    perturb_(model, sigma=0.02, seed=123, state=state)
    assert not torch.equal(before["dense.weight"], model.dense.weight)

    unperturb_(model, sigma=0.02, seed=123, state=state)
    assert torch.allclose(before["dense.weight"], model.dense.weight, atol=1e-6, rtol=0)
    assert torch.equal(before["norm.weight"], model.norm.weight)

