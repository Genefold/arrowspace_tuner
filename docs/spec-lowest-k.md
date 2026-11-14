# eps/k tuning with targeted low-k refinement

Approved implementation contract for T-001. This records the reviewed decisions
from the implementation specification; no additional product scope is introduced.

## Objective

Optimize eps and k only and optionally choose the smallest evaluated k within an
explicit quality tolerance. Deliberately explore low-k/high-eps configurations
that a stochastic main search can miss. Do not assume an inverse eps/k equation
or guarantee a global minimum k.

## Configuration and evaluation

- Keep the existing composite score (MRR proxy, Fiedler, lambda variance).
- Evaluate retrieval at fixed internal temperature 0.5, never optimize or return it.
- `selection_policy`: `best_score` (default) or `lowest_k`.
- `score_tolerance`: finite, nonnegative absolute score difference; default zero.
- `refinement_trials`: nonnegative integer extra attempt budget; default zero.
- `lowest_k` requires a positive explicit refinement budget. `best_score`
  requires zero refinement budget and zero tolerance.
- Expose these settings through Python, CLI, and MCP; reject obsolete tau bounds.
- Main phase runs `n_trials` attempts. Both phases share the same fixed corpus
  subsample and probe anchors.
- Only COMPLETE, successful, finite, in-bounds evaluations can win. Failed
  evaluations must not become zero-score successful candidates.

## Refinement

Freeze the raw-score winner after the main phase as the anchor. If it already
has `k_low`, skip refinement and record the reason. Otherwise:

- Explicitly schedule `refinement_trials` points below the anchor's k.
- Include `k_low` first; stratify other integer k values and cover distinct
  values before repetition when the budget permits.
- Target floor(75% of budget) toward `[anchor_eps, eps_high]`, with the first
  point at `(k_low, eps_high)`. A one-attempt budget uses this corner.
- Allocate the remainder to full-range log-scale eps exploration. With two or
  more attempts, retain at least one full-range point and interleave the regions.
- If the anchor is already at `eps_high`, retain one boundary point and allocate
  other targeted points to full-range exploration.
- Precompute the seed-controlled schedule before parallel execution.
- Bypass statistical median pruning in refinement, but retain invalid-graph,
  disconnected/flat-spectrum, retrieval, and nonfinite-score guards.
- Failed/pruned attempts consume budget; no unbounded retries or bound expansion.
- If every refinement attempt fails, fall back to valid main-phase candidates.
- MCP caps apply to the combined main and refinement budgets.

## Selection and outputs

For `best_score`, maximize raw score, breaking exact ties by trial number.
For `lowest_k`, compute the global best valid raw score across both phases,
retain candidates within tolerance, then order by k ascending, score descending,
and trial number ascending. Use full precision, not rounded display metrics.

Return a single selected trial's builder-ready dictionary:

```python
{
    "eps": selected.params["eps"],
    "k": selected.params["k"],
    "topk": max(1, selected.params["k"] // 2),
    "p": 2.0,
    "sigma": None,
}
```

Do not add selection metadata or tau fields to graph parameters. Remove best_tau
from public attributes, API/CLI/MCP results, build instructions, and reports.
The selected parameters and quality metrics must agree across fit, result
envelopes, saved reports, and reload. Optuna's native best_trial remains the
raw-score winner. Report both identities, score gap, policy, tolerance, and
requested/actual refinement attempts outside graph_params.

Any graph cache reuse must match selected trial identity; do not retain every
trial graph. The current fit returns parameters only and releases its build cache.
Older saved graph parameters remain loadable without re-emitting tau.

## Persistence and concurrency

Version the internal evaluation contract and verify data/settings before adding
trials on resume. Reject legacy tau-optimized and incompatible studies; never
rewrite historical objective values. Compatible historical candidates participate
in selection. A new invocation adds its own explicit budgets and records phase
and invocation identity. Compatible completed studies may change policy/tolerance.
Preserve per-result/report selection snapshots. Deterministic selection applies
to a fixed trial set; parallel optimization can produce different trial sets.

## Structure, style, and verification

Core configuration/evaluation/selection live under `src/arrowspace_tuner/core`;
orchestration in `tuner.py`, interface contracts in `models.py`/`service.py`,
adapters in `cli.py`/`mcp_server.py`, and reports in `reporting/reporter.py`.
Use typed Python 3.12+, snake_case, existing error conventions, 100-character
lines, pytest, Ruff, and strict mypy. No new runtime dependencies.

Verify deterministic schedule edge cases, quality thresholds/ties, failed trial
exclusion, statistical-pruning bypass, shared evaluation data, real ArrowSpace
builder compatibility, report/reload identity, incompatible/compatible resumes,
CLI JSON integrity, and MCP combined budget enforcement.

```sh
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/arrowspace_tuner
uv build --wheel --out-dir dist
```

Always validate inputs and respect bounds/budgets. Ask before changing scoring,
adding dependencies, migrating legacy studies, or changing builder keys.
Never silently exceed tolerance, mix parameters from different trials, return
tau, rewrite historical scores, or weaken tests to obtain a green result.
