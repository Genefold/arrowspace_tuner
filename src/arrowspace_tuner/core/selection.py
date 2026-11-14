"""Pure, deterministic refinement scheduling and quality-first selection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from optuna.trial import FrozenTrial, TrialState

from .config import StudyConfig


def trial_score(trial: FrozenTrial) -> float:
    """Read a finite completed score after candidate validation."""
    assert trial.value is not None and math.isfinite(trial.value)
    return trial.value


def valid_trials(trials: Sequence[FrozenTrial], cfg: StudyConfig) -> list[FrozenTrial]:
    """Exclude incomplete, unsuccessful, nonfinite, and out-of-bounds evaluations."""
    valid = []
    for trial in trials:
        eps, k = trial.params.get("eps"), trial.params.get("k")
        if (
            trial.state == TrialState.COMPLETE
            and trial.user_attrs.get("evaluation_success") is True
            and trial.value is not None
            and math.isfinite(trial.value)
            and type(k) is int
            and cfg.k_low <= k <= cfg.k_high
            and isinstance(eps, (int, float))
            and not isinstance(eps, bool)
            and cfg.eps_low <= eps <= cfg.eps_high
        ):
            valid.append(trial)
    return valid


def select_trial(trials: Sequence[FrozenTrial], cfg: StudyConfig) -> FrozenTrial:
    """Select a complete parameter tuple, comparing unrounded objective values."""
    candidates = valid_trials(trials, cfg)
    if not candidates:
        raise RuntimeError("No valid successfully evaluated trials")
    best = max(trial_score(t) for t in candidates)
    if cfg.selection_policy == "best_score":
        return min(candidates, key=lambda t: (-trial_score(t), t.number))
    eligible = [t for t in candidates if trial_score(t) >= best - cfg.score_tolerance]
    return min(eligible, key=lambda t: (t.params["k"], -trial_score(t), t.number))


def refinement_schedule(
    cfg: StudyConfig, *, k_anchor: int, eps_anchor: float
) -> list[dict[str, float | int]]:
    """Precompute bounded low-k points; the first is the low-k/high-eps corner."""
    n = cfg.refinement_trials
    if n == 0 or k_anchor <= cfg.k_low:
        return []
    rng = np.random.default_rng(cfg.seed)
    width = k_anchor - cfg.k_low
    # Stratify integer k before repeating. Start at the smallest legal k.
    ks: list[int] = []
    while len(ks) < n:
        count = min(width, n - len(ks))
        if not ks:
            block = [cfg.k_low]
            if count > 1:
                edges = np.linspace(cfg.k_low + 1, k_anchor, count)
                block += [
                    int(rng.integers(int(a), max(int(a) + 1, int(b))))
                    for a, b in zip(edges[:-1], edges[1:], strict=True)
                ]
        else:
            edges = np.linspace(cfg.k_low, k_anchor, count + 1)
            block = [
                int(rng.integers(int(a), max(int(a) + 1, int(b))))
                for a, b in zip(edges[:-1], edges[1:], strict=True)
            ]
        ks.extend(block)
    targeted = 1 if n == 1 else max(1, min(n - 1, int(0.75 * n)))
    if eps_anchor == cfg.eps_high:
        targeted = 1
    # Spread full-range slots through the schedule.
    targeted_slots = set(np.linspace(0, n - 1, targeted, dtype=int).tolist())
    targeted_slots.add(0)
    values = [cfg.eps_high] * n
    for slots, low in [
        (sorted(targeted_slots - {0}), eps_anchor),
        ([i for i in range(n) if i not in targeted_slots], cfg.eps_low),
    ]:
        count = len(slots)
        if count:
            fractions = (np.arange(count) + rng.random(count)) / count
            rng.shuffle(fractions)
            for slot, fraction in zip(slots, fractions, strict=True):
                values[slot] = float(
                    np.exp(np.log(low) + fraction * (np.log(cfg.eps_high) - np.log(low)))
                )
    return [{"k": k, "eps": eps} for k, eps in zip(ks, values, strict=True)]
