from __future__ import annotations

from types import TracebackType
from typing import Any

import torch


def transformer_layers(model: Any) -> torch.nn.ModuleList:
    """Return the ordered list of transformer blocks for supported architectures."""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise TypeError("Unsupported model architecture for hidden-state capture")


class CaptureHiddenStates:
    """Context manager that records each transformer block's output hidden state.

    Register a forward hook on every block; while the context is open, running a
    forward pass populates ``hidden_states`` with one tensor per layer, ordered
    from the earliest block to the last. Each tensor has shape ``[batch, seq, hidden]``.
    Hooks are always removed on exit.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.layers = transformer_layers(model)
        self.hidden_states: list[torch.Tensor] = []
        self._handles: list[Any] = []

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self.hidden_states.append(hidden.detach())

    def __enter__(self) -> "CaptureHiddenStates":
        self.hidden_states = []
        self._handles = [layer.register_forward_hook(self._hook) for layer in self.layers]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False


def lens_logits(model: Any, hidden_state: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        normalized = model.gpt_neox.final_layer_norm(hidden_state)
        return model.embed_out(normalized)

    if hasattr(model, "model") and hasattr(model.model, "norm") and hasattr(model, "lm_head"):
        normalized = model.model.norm(hidden_state)
        return model.lm_head(normalized)

    raise TypeError("Unsupported model architecture for logit lens")


def lens_argmax_per_layer(model: Any, per_layer_hidden_states: list[torch.Tensor]) -> list[int]:
    """Logit-lens argmax token id at the last position, for each captured layer."""
    return [
        int(torch.argmax(lens_logits(model, hidden[:, -1, :]), dim=-1).item())
        for hidden in per_layer_hidden_states
    ]


def commitment_depth_from_argmaxes(argmaxes: list[int], final_token_id: int) -> int:
    """First layer index from which the lens argmax equals ``final_token_id`` and never changes.

    Returns ``len(argmaxes)`` when the final token is never a sustained argmax
    (i.e. the model never commits to it under the lens).
    """
    for layer_idx in range(len(argmaxes)):
        if all(tid == final_token_id for tid in argmaxes[layer_idx:]):
            return layer_idx
    return len(argmaxes)


def commitment_depth(
    per_layer_hidden_states: list[torch.Tensor], final_token_id: int, model: Any
) -> int:
    argmaxes = lens_argmax_per_layer(model, per_layer_hidden_states)
    return commitment_depth_from_argmaxes(argmaxes, final_token_id)
