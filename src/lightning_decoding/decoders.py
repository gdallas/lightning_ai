from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from lightning_decoding.noise import NoiseState, perturb_, unperturb_


@dataclass
class DecodeResult:
    token_id: int
    token_str: str
    logits_top20: list[tuple[int, float]]
    meta: dict[str, Any]


def _encode(tokenizer: Any, prompt: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value for key, value in encoded.items()}


def last_logits(model: Any, tokenizer: Any, prompt: str) -> torch.Tensor:
    inputs = _encode(tokenizer, prompt)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[0, -1].detach().float()


def top_logits(logits: torch.Tensor, k: int = 20) -> list[tuple[int, float]]:
    values, indices = torch.topk(logits, k=min(k, logits.numel()))
    return [(int(idx), float(value)) for idx, value in zip(indices, values, strict=True)]


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id])


def greedy(model: Any, tokenizer: Any, prompt: str, cfg: dict, rng: torch.Generator) -> DecodeResult:
    start = time.perf_counter()
    logits = last_logits(model, tokenizer, prompt)
    token_id = int(torch.argmax(logits).item())
    return DecodeResult(
        token_id=token_id,
        token_str=token_text(tokenizer, token_id),
        logits_top20=top_logits(logits),
        meta={"latency_ms": (time.perf_counter() - start) * 1000},
    )


def temperature(model: Any, tokenizer: Any, prompt: str, cfg: dict, rng: torch.Generator) -> DecodeResult:
    start = time.perf_counter()
    logits = last_logits(model, tokenizer, prompt)
    temp = float(cfg.get("T", cfg.get("temperature", 1.0)))
    probs = torch.softmax(logits / temp, dim=-1)
    token_id = int(torch.multinomial(probs, 1, generator=rng).item())
    return DecodeResult(
        token_id=token_id,
        token_str=token_text(tokenizer, token_id),
        logits_top20=top_logits(logits),
        meta={"T": temp, "latency_ms": (time.perf_counter() - start) * 1000},
    )


def nucleus(model: Any, tokenizer: Any, prompt: str, cfg: dict, rng: torch.Generator) -> DecodeResult:
    start = time.perf_counter()
    logits = last_logits(model, tokenizer, prompt)
    p = float(cfg.get("p", 0.95))
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= p
    keep[0] = True
    kept_probs = sorted_probs[keep]
    kept_ids = sorted_ids[keep]
    kept_probs = kept_probs / kept_probs.sum()
    sample_idx = int(torch.multinomial(kept_probs, 1, generator=rng).item())
    token_id = int(kept_ids[sample_idx].item())
    return DecodeResult(
        token_id=token_id,
        token_str=token_text(tokenizer, token_id),
        logits_top20=top_logits(logits),
        meta={"p": p, "candidate_set_size": int(kept_ids.numel()), "latency_ms": (time.perf_counter() - start) * 1000},
    )


def gap_sampler(model: Any, tokenizer: Any, prompt: str, cfg: dict, rng: torch.Generator) -> DecodeResult:
    start = time.perf_counter()
    logits = last_logits(model, tokenizer, prompt)
    delta = float(cfg.get("delta", 2.0))
    max_logit = torch.max(logits)
    argmax = int(torch.argmax(logits).item())
    candidates = torch.nonzero(logits >= max_logit - delta, as_tuple=False).flatten()
    candidates = candidates[candidates != argmax]

    fallback = candidates.numel() == 0
    if fallback:
        token_id = argmax
    else:
        choice = int(torch.randint(candidates.numel(), (1,), generator=rng).item())
        token_id = int(candidates[choice].item())

    return DecodeResult(
        token_id=token_id,
        token_str=token_text(tokenizer, token_id),
        logits_top20=top_logits(logits),
        meta={
            "delta": delta,
            "candidate_set_size": int(candidates.numel()),
            "fallback": fallback,
            "latency_ms": (time.perf_counter() - start) * 1000,
        },
    )


def ensemble_minority(model: Any, tokenizer: Any, prompt: str, cfg: dict, rng: torch.Generator) -> DecodeResult:
    start = time.perf_counter()
    clean_logits = last_logits(model, tokenizer, prompt)
    clean_argmax = int(torch.argmax(clean_logits).item())
    state = cfg.get("_noise_state") or NoiseState(model)
    sigma = float(cfg.get("sigma", 0.02))
    n = int(cfg.get("N", 10))
    k = int(cfg.get("k", 3))
    global_seed = int(cfg.get("global_seed", 0))

    counts: Counter[int] = Counter()
    per_run_latency: list[float] = []

    for run_idx in range(n):
        seed = global_seed * 1000 + run_idx
        run_start = time.perf_counter()
        perturb_(model, sigma, seed, state)
        try:
            logits = last_logits(model, tokenizer, prompt)
            counts[int(torch.argmax(logits).item())] += 1
        finally:
            unperturb_(model, sigma, seed, state)
        per_run_latency.append((time.perf_counter() - run_start) * 1000)

    candidate_ids = [token_id for token_id, count in counts.items() if count >= k and token_id != clean_argmax]
    fallback = len(candidate_ids) == 0
    if fallback:
        token_id = clean_argmax
    else:
        weights = torch.tensor([counts[token_id] for token_id in candidate_ids], dtype=torch.float32)
        idx = int(torch.multinomial(weights / weights.sum(), 1, generator=rng).item())
        token_id = candidate_ids[idx]

    clean_max = float(torch.max(clean_logits))
    clean_gap = float(clean_max - clean_logits[token_id])
    return DecodeResult(
        token_id=token_id,
        token_str=token_text(tokenizer, token_id),
        logits_top20=top_logits(clean_logits),
        meta={
            "sigma": sigma,
            "N": n,
            "k": k,
            "clean_argmax": clean_argmax,
            "counts": {str(key): value for key, value in sorted(counts.items())},
            "fallback": fallback,
            "clean_logit_gap": clean_gap,
            "per_run_latency_ms": per_run_latency,
            "latency_ms": (time.perf_counter() - start) * 1000,
        },
    )


DECODERS = {
    "greedy": greedy,
    "temperature": temperature,
    "nucleus": nucleus,
    "gap_sampler": gap_sampler,
    "ensemble_minority": ensemble_minority,
}


def decode(model: Any, tokenizer: Any, prompt: str, cfg: dict, rng: torch.Generator) -> DecodeResult:
    method = cfg["method"]
    try:
        fn = DECODERS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown decoder method: {method}") from exc
    return fn(model, tokenizer, prompt, cfg, rng)

