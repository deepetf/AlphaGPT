"""
Shared signal cleaning and top-k selection helpers.
"""

from typing import List, Optional, Tuple

import torch


NEG_INF = float("-inf")


def default_min_valid_count(
    top_k: int,
    override: Optional[int] = None,
    floor: Optional[int] = None,
) -> int:
    if override is not None and int(override) > 0:
        return int(override)
    min_floor = int(floor) if floor is not None else 0
    return max(min_floor, int(top_k) * 2)


def preprocess_signal_row(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    enabled: bool = True,
    winsor_q: float = 0.01,
    clip_value: float = 5.0,
    rank_output: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Clean one cross-sectional signal row.

    Steps:
    1) tradable + finite filter
    2) winsorize
    3) clip
    4) optional rank transform to [0, 1]
    """
    if values.dim() != 1 or valid_mask.dim() != 1:
        raise ValueError("values and valid_mask must be 1D tensors")
    if values.numel() != valid_mask.numel():
        raise ValueError("values and valid_mask length mismatch")

    out_dtype = values.dtype if torch.is_floating_point(values) else torch.float32
    cleaned = torch.full(values.shape, NEG_INF, device=values.device, dtype=out_dtype)

    tradable_mask = valid_mask.to(device=values.device, dtype=torch.bool)
    finite_mask = torch.isfinite(values)
    effective_mask = tradable_mask & finite_mask
    valid_count = int(effective_mask.sum().item())
    if valid_count == 0:
        return cleaned, effective_mask, 0

    work = values[effective_mask].to(out_dtype)

    if enabled:
        q = max(0.0, min(float(winsor_q), 0.49))
        if q > 0.0 and work.numel() >= 4:
            low = torch.quantile(work, q)
            high = torch.quantile(work, 1.0 - q)
            work = torch.clamp(work, low, high)

        if clip_value is not None and float(clip_value) > 0:
            bound = float(clip_value)
            work = torch.clamp(work, -bound, bound)

        if rank_output:
            n = work.numel()
            if n == 1:
                work = torch.zeros_like(work)
            else:
                order = torch.argsort(work)
                ranked = torch.empty_like(work)
                ranked[order] = torch.linspace(
                    0.0, 1.0, n, device=work.device, dtype=work.dtype
                )
                work = ranked

    cleaned[effective_mask] = work
    return cleaned, effective_mask, valid_count


def select_top_k_indices(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_k: int,
    min_valid_count: int,
    clean_enabled: bool = True,
    winsor_q: float = 0.01,
    clip_value: float = 5.0,
    rank_output: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    cleaned, _, valid_count = preprocess_signal_row(
        values=values,
        valid_mask=valid_mask,
        enabled=clean_enabled,
        winsor_q=winsor_q,
        clip_value=clip_value,
        rank_output=rank_output,
    )
    if valid_count < int(min_valid_count):
        empty = torch.empty(0, device=values.device, dtype=torch.long)
        empty_score = torch.empty(0, device=values.device, dtype=cleaned.dtype)
        return empty, empty_score, valid_count

    k = min(int(top_k), valid_count)
    if k <= 0:
        empty = torch.empty(0, device=values.device, dtype=torch.long)
        empty_score = torch.empty(0, device=values.device, dtype=cleaned.dtype)
        return empty, empty_score, valid_count

    scores, indices = torch.topk(cleaned, k, largest=True)
    return indices, scores, valid_count


def build_topk_weights(
    factors: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_k: int,
    min_valid_count: int,
    clean_enabled: bool = True,
    winsor_q: float = 0.01,
    clip_value: float = 5.0,
    rank_output: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
    if factors.dim() != 2 or valid_mask.dim() != 2:
        raise ValueError("factors and valid_mask must be 2D tensors")
    if factors.shape != valid_mask.shape:
        raise ValueError("factors and valid_mask shape mismatch")

    t_count, n_count = factors.shape
    weights = torch.zeros(t_count, n_count, device=factors.device, dtype=factors.dtype)
    valid_trading_day = torch.zeros(t_count, device=factors.device, dtype=torch.bool)
    daily_valid_count = torch.zeros(t_count, device=factors.device, dtype=torch.long)
    daily_holdings: List[List[int]] = []

    for t in range(t_count):
        indices, _, valid_count = select_top_k_indices(
            values=factors[t],
            valid_mask=valid_mask[t],
            top_k=top_k,
            min_valid_count=min_valid_count,
            clean_enabled=clean_enabled,
            winsor_q=winsor_q,
            clip_value=clip_value,
            rank_output=rank_output,
        )
        daily_valid_count[t] = int(valid_count)
        if indices.numel() == 0:
            daily_holdings.append([])
            continue

        k = int(indices.numel())
        weights[t, indices] = 1.0 / float(k)
        valid_trading_day[t] = True
        daily_holdings.append(indices.tolist())

    return weights, valid_trading_day, daily_valid_count, daily_holdings
