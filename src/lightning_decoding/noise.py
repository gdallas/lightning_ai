from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

ParamFilter = Callable[[str, torch.nn.Parameter], bool]


def stable_hash(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def default_noise_filter(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim != 2:
        return False

    lowered = name.lower()
    excluded = ("embed", "norm", "lm_head", "embed_out", "bias")
    if any(part in lowered for part in excluded):
        return False

    included = (
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    return any(part in lowered for part in included)


def layer_index(name: str) -> int | None:
    """Transformer block index parsed from a parameter name, or None."""
    match = re.search(r"\.(?:layers|h)\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def component_of(name: str) -> str | None:
    """Classify a weight as ``attention`` or ``mlp`` (works for GPTNeoX and Llama/Qwen names)."""
    lowered = name.lower()
    if "attention" in lowered or "attn" in lowered:
        return "attention"
    if "mlp" in lowered:
        return "mlp"
    return None


def parse_layer_spec(spec: Any) -> set[int] | None:
    """Accept ``"0-3"``, ``5``, or ``[0, 1, 2]`` -> a set of layer indices (None = all)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        if "-" in spec:
            lo, hi = spec.split("-", 1)
            return set(range(int(lo), int(hi) + 1))
        return {int(spec)}
    if isinstance(spec, int):
        return {spec}
    return {int(x) for x in spec}


def make_noise_filter(
    components: list[str] | None = None, layers: Any = None
) -> ParamFilter:
    """Restrict the default filter to given components (attention/mlp) and/or layer indices."""
    comp_set = {c.lower() for c in components} if components else None
    layer_set = parse_layer_spec(layers)

    def _filter(name: str, param: torch.nn.Parameter) -> bool:
        if not default_noise_filter(name, param):
            return False
        if comp_set is not None and component_of(name) not in comp_set:
            return False
        if layer_set is not None:
            idx = layer_index(name)
            if idx is None or idx not in layer_set:
                return False
        return True

    return _filter


def noise_filter_from_scope(scope: dict[str, Any] | None) -> ParamFilter:
    """Build a filter from a config ``noise_scope`` block, e.g.
    ``{components: [attention], layers: "8-11"}``. Empty/None -> the default (all)."""
    if not scope:
        return default_noise_filter
    return make_noise_filter(components=scope.get("components"), layers=scope.get("layers"))


@dataclass
class NoiseState:
    model: torch.nn.Module
    filter_fn: ParamFilter = default_noise_filter
    std_cache: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if self.filter_fn(name, param):
                    self.std_cache[name] = param.detach().float().std().to(param.device)

    def selected_names(self) -> list[str]:
        return sorted(self.std_cache)


def _noise_like(param: torch.Tensor, sigma: float, seed: int, name: str, std: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device=param.device)
    generator.manual_seed(seed + stable_hash(name))
    noise = torch.randn(param.shape, generator=generator, device=param.device, dtype=param.dtype)
    return noise * (sigma * std.to(dtype=param.dtype, device=param.device))


def perturb_(model: torch.nn.Module, sigma: float, seed: int, state: NoiseState) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            std = state.std_cache.get(name)
            if std is not None:
                param.add_(_noise_like(param, sigma, seed, name, std))


def unperturb_(model: torch.nn.Module, sigma: float, seed: int, state: NoiseState) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            std = state.std_cache.get(name)
            if std is not None:
                param.sub_(_noise_like(param, sigma, seed, name, std))

