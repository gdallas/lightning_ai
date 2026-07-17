from __future__ import annotations

from typing import Any

import torch


def lens_logits(model: Any, hidden_state: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        normalized = model.gpt_neox.final_layer_norm(hidden_state)
        return model.embed_out(normalized)

    if hasattr(model, "model") and hasattr(model.model, "norm") and hasattr(model, "lm_head"):
        normalized = model.model.norm(hidden_state)
        return model.lm_head(normalized)

    raise TypeError("Unsupported model architecture for logit lens")


def commitment_depth(per_layer_hidden_states: list[torch.Tensor], final_token_id: int, model: Any) -> int:
    argmaxes = [
        int(torch.argmax(lens_logits(model, hidden[:, -1, :]), dim=-1).item())
        for hidden in per_layer_hidden_states
    ]

    for layer_idx, token_id in enumerate(argmaxes):
        if token_id == final_token_id and all(tid == final_token_id for tid in argmaxes[layer_idx:]):
            return layer_idx
    return len(per_layer_hidden_states)

