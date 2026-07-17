import pytest

torch = pytest.importorskip("torch")

from lightning_decoding.lens import (
    CaptureHiddenStates,
    commitment_depth_from_argmaxes,
    transformer_layers,
)


class Block(torch.nn.Module):
    def __init__(self, dim: int, as_tuple: bool = False) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim)
        self.as_tuple = as_tuple

    def forward(self, x):
        out = self.lin(x)
        return (out,) if self.as_tuple else out


class TinyStack(torch.nn.Module):
    def __init__(self, dim: int, num_layers: int, as_tuple: bool = False) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(dim, as_tuple) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return x


def test_transformer_layers_resolves_layers_attr() -> None:
    model = TinyStack(4, 3)
    assert len(transformer_layers(model)) == 3


def test_capture_hidden_states_records_each_layer() -> None:
    model = TinyStack(4, 3)
    x = torch.randn(1, 5, 4)
    with CaptureHiddenStates(model) as capture:
        model(x)
    assert len(capture.hidden_states) == 3
    assert capture.hidden_states[0].shape == (1, 5, 4)
    assert capture._handles == []


def test_capture_handles_tuple_layer_output() -> None:
    model = TinyStack(4, 2, as_tuple=True)
    with CaptureHiddenStates(model) as capture:
        model(torch.randn(1, 3, 4))
    assert len(capture.hidden_states) == 2
    assert all(isinstance(h, torch.Tensor) for h in capture.hidden_states)


def test_capture_stops_after_context_exit() -> None:
    model = TinyStack(4, 2)
    x = torch.randn(1, 3, 4)
    with CaptureHiddenStates(model) as capture:
        model(x)
    model(x)  # no hooks registered anymore
    assert len(capture.hidden_states) == 2


def test_commitment_depth_first_sustained_layer() -> None:
    assert commitment_depth_from_argmaxes([5, 2, 7, 7, 7], 7) == 2
    assert commitment_depth_from_argmaxes([7, 7, 7], 7) == 0
    assert commitment_depth_from_argmaxes([7, 2, 7], 7) == 2


def test_commitment_depth_never_commits() -> None:
    assert commitment_depth_from_argmaxes([1, 2, 3], 7) == 3
    assert commitment_depth_from_argmaxes([], 7) == 0
