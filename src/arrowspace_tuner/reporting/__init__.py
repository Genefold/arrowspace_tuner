"""
arrowspace_tuner.reporting — optional result persistence.

Requires the [report] extra:
    pip install arrowspace-tuner[report]

Usage:
    from arrowspace_tuner.reporting import save_results
    run_dir = save_results(tuner.study, out_dir="results")
"""
from .reporter import save_results

__all__ = ["save_results"]