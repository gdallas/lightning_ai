import pytest

torch = pytest.importorskip("torch")

from lightning_decoding.noise import (
    NoiseState,
    component_of,
    default_noise_filter,
    layer_index,
    make_noise_filter,
    noise_filter_from_scope,
    parse_layer_spec,
    perturb_,
    stable_hash,
    unperturb_,
)


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


def _w() -> "torch.nn.Parameter":
    return torch.nn.Parameter(torch.zeros(4, 4))


def test_layer_index_and_component() -> None:
    assert layer_index("gpt_neox.layers.5.attention.query_key_value.weight") == 5
    assert layer_index("gpt_neox.embed_out.weight") is None
    assert component_of("gpt_neox.layers.5.attention.query_key_value.weight") == "attention"
    assert component_of("gpt_neox.layers.5.mlp.dense_h_to_4h.weight") == "mlp"
    assert component_of("model.layers.2.self_attn.q_proj.weight") == "attention"


def test_parse_layer_spec() -> None:
    assert parse_layer_spec("0-3") == {0, 1, 2, 3}
    assert parse_layer_spec(5) == {5}
    assert parse_layer_spec([1, 2]) == {1, 2}
    assert parse_layer_spec(None) is None


def test_make_noise_filter_component_and_layer() -> None:
    attn_only = make_noise_filter(components=["attention"])
    assert attn_only("gpt_neox.layers.0.attention.query_key_value.weight", _w())
    assert not attn_only("gpt_neox.layers.0.mlp.dense_h_to_4h.weight", _w())

    late_only = make_noise_filter(layers="8-11")
    assert late_only("gpt_neox.layers.9.attention.dense.weight", _w())
    assert not late_only("gpt_neox.layers.3.attention.dense.weight", _w())


def test_noise_filter_from_scope_default_passthrough() -> None:
    assert noise_filter_from_scope(None) is default_noise_filter
    assert noise_filter_from_scope({}) is default_noise_filter

