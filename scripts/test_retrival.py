#!/usr/bin/env python3
"""
run_all_params.py
=================
Sweeps the full test_17_CVE_neurips_v2 experiment across:
  - The baseline graph params
  - All unique (k, eps) combos dynamically loaded from best_params.json runs

Key changes:
  - Corpus and query embeddings are loaded directly from .npy caches;
    no model or encoding step is ever executed.
  - Dynamically sweeps 'results/cve_arrowspace_fstar' for configurations.
  - Only uses `k` and `eps` from the tuning runs (p=1.8, sigma=0.535 are fixed).
  - De-duplicates identical (k, eps) runs.
  - Zero lines of test_17 logic are modified; all functions are imported.
  - Every run writes its full set of test_17 outputs to its own sub-directory.
  - A master sweep_all_runs_summary.csv is updated after every run.

Usage:
    python run_all_params.py \
        --dataset    /path/to/cvelistV5 \
        --corpus-emb /path/to/cve_embeddings_cache.npy \
        --query-emb  /path/to/cve_query_embs_cache.npy \
        --sweep-dir  results/cve_param_sweep
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from random import shuffle

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Import every public symbol from test_17 unchanged ──────────────────────
import importlib.util, sys

_HERE = Path(__file__).parent
# Ensure this points to the correct location of test_17_CVE_neurips_v2.py
_T17  = _HERE / "test_17_CVE_neurips_v2.py"
assert _T17.exists(), f"Cannot find test_17 at {_T17}"

spec = importlib.util.spec_from_file_location("test_17", _T17)
t17  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t17)

# Pull every symbol we call directly
ArrowSpaceBuilder               = t17.ArrowSpaceBuilder
set_debug                       = t17.set_debug
iter_cve_json                   = t17.iter_cve_json
extract_text                    = t17.extract_text
compute_ranking_metrics         = t17.compute_ranking_metrics
compute_ndcg                    = t17.compute_ndcg
analyze_tail_distribution       = t17.analyze_tail_distribution
compute_all_recall_metrics      = t17.compute_all_recall_metrics
run_headk_sweep                 = t17.run_headk_sweep
save_search_results_to_csv      = t17.save_search_results_to_csv
save_metrics_to_csv             = t17.save_metrics_to_csv
save_tail_metrics_to_csv        = t17.save_tail_metrics_to_csv
save_semantic_recall_to_csv     = t17.save_semantic_recall_to_csv
save_summary_to_csv             = t17.save_summary_to_csv
save_headk_sweep_to_csv         = t17.save_headk_sweep_to_csv
save_query_comparison           = t17.save_query_comparison
save_run_metadata               = t17.save_run_metadata
plot_comparison                 = t17.plot_comparison
plot_tail_comparison            = t17.plot_tail_comparison
plot_semantic_recall_comparison = t17.plot_semantic_recall_comparison
plot_metric_deltas              = t17.plot_metric_deltas
plot_win_loss_heatmap           = t17.plot_win_loss_heatmap
plot_pareto_tradeoff            = t17.plot_pareto_tradeoff
plot_headk_sweep                = t17.plot_headk_sweep

TAU_COSINE   = t17.TAU_COSINE
TAU_HYBRID   = t17.TAU_HYBRID
TAU_TAUMODE  = t17.TAU_TAUMODE
TAU_DISPLAY  = t17.TAU_DISPLAY
NDCG_K       = t17.NDCG_K
RESULTS_K    = t17.RESULTS_K
HEAD_K       = t17.HEAD_K
HEAD_K_SWEEP = t17.HEAD_K_SWEEP

set_debug(True)

# ── Fixed queries (identical to test_17 main()) ────────────────────────────
QUERIES = [
    "authenticated arbitrary file read path traversal",
    "remote code execution in ERP web component",
    "SQL injection in login endpoint",
    "stored cross-site scripting XSS in user profile page",
    "server-side request forgery SSRF in URL preview feature",
    "XML external entity XXE injection in SOAP parser",
    "insecure direct object reference IDOR in invoice download",
    "heap buffer overflow in image processing library",
    "local privilege escalation via race condition in kernel",
    "use-after-free vulnerability in browser rendering engine",
    "integer overflow leading to heap corruption in video codec",
    "authentication bypass via JWT token manipulation",
    "unsafe deserialization in Java RMI service",
    "improper access control in REST API DELETE method",
    "command injection in router web administration interface",
    "hardcoded credentials in firmware update mechanism",
    "denial of service via malformed network packets",
    "sensitive information disclosure in cloud metadata service",
    "directory traversal in backup restore endpoint",
    "cross-site request forgery CSRF in admin password change form",
    "out-of-bounds read in PDF rendering engine",
    "format string vulnerability in logging daemon",
    "prototype pollution in Node.js configuration merge utility",
    "arbitrary file upload leading to remote code execution",
    "open redirect in single sign-on logout flow",
    "improper certificate validation in mobile banking application",
    "sandbox escape in browser extension messaging interface",
    "null pointer dereference in Bluetooth service",
    "double free vulnerability in archive extraction library",
    "buffer underflow in font parsing component",
    "OS command injection in diagnostic CGI script",
    "LDAP injection in enterprise directory search feature",
    "NoSQL injection in user search API",
    "path canonicalization bypass in file download servlet",
    "improper neutralization of CRLF in mail gateway",
    "privilege escalation via writable system service configuration",
    "race condition in temporary file creation",
    "authorization bypass in GraphQL mutation resolver",
    "memory disclosure via uninitialized stack buffer",
    "heap use-after-free in media streaming server",
    "stack-based buffer overflow in DHCP client",
    "improper signature verification in software update channel",
    "symlink attack in installer cleanup routine",
    "cross-tenant data exposure in multi-tenant storage service",
    "request smuggling in reverse proxy parser",
    "host header injection in password reset workflow",
    "business logic bypass in coupon redemption API",
    "improper input validation in SAML assertion processing",
    "authentication brute force due to missing rate limiting",
    "insecure default permissions in container runtime socket",
]

# ── Dynamic Parameter Loading ──────────────────────────────────────────────
BASELINE_P     = 1.8
BASELINE_SIGMA = 0.535
BASELINE_TOPK  = 25

def get_param_configs(results_dir):
    """
    Loads configs dynamically from best_params.json runs.
    Extracts only `k` and `eps`, applying deduping so identical params aren't rerun.
    """
    configs = []
    
    # 1. Add baseline
    configs.append({
        "tag": "baseline", "k": 30, "eps": 1.31,
        "p": BASELINE_P, "sigma": BASELINE_SIGMA, "topk": BASELINE_TOPK
    })
    seen_params = {(30, 1.31)}

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Warning: Results dir {results_path} not found. Running baseline only.")
        return configs

    # 2. Iterate through best_params.json dynamically
    for bp_path in sorted(results_path.glob("*/best_params.json")):
        run_id = bp_path.parent.name
        with open(bp_path) as f:
            d = json.load(f)
            
        params = d.get("params", {})
        k = params.get("k")
        eps = params.get("eps")
        
        if k is None or eps is None:
            continue
            
        # Deduplicate based on (k, eps)
        if (k, eps) not in seen_params:
            seen_params.add((k, eps))
            
            # topk logic: 25 unless k < 25, then k // 2
            topk = BASELINE_TOPK if k >= 25 else k // 2
            
            configs.append({
                "tag": f"{run_id}_k{k}_eps{eps:.3f}", 
                "k": k, "eps": eps, 
                "p": BASELINE_P, "sigma": BASELINE_SIGMA, "topk": topk
            })
            
    print(f"Loaded {len(configs)} unique configurations (including baseline).")
    return configs


# ── Per-run experiment ──────────────────────────────────────────────────────

def run_experiment(cfg, ids, titles, docs, corpus_emb, queries, query_emb,
                   sweep_dir, dataset_root):
    tag     = cfg["tag"]
    out_dir = sweep_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_params = {
        "eps":   cfg["eps"],
        "k":     cfg["k"],
        "topk":  cfg["topk"],
        "p":     cfg["p"],
        "sigma": cfg["sigma"],
    }
    print(f"\n{'#'*70}")
    print(f"  RUN : {tag}")
    print(f"  params: {graph_params}")
    print(f"{'#'*70}")

    # patch t17.OUTPUT_DIR so all save helpers route to out_dir
    t17.OUTPUT_DIR = out_dir

    # ── build graph ────────────────────────────────────────────────────────
    t_build = time.perf_counter()
    aspace, gl = (
        ArrowSpaceBuilder()
        .with_seed(42)
        .with_dims_reduction(enabled=False, eps=None)
        .with_sampling("simple", 1.0)
    ).build(graph_params, corpus_emb)
    build_time = time.perf_counter() - t_build
    print(f"  Build time: {build_time:.2f}s")

    tau_labels = [
        TAU_DISPLAY["Cosine"],
        TAU_DISPLAY["Hybrid"],
        TAU_DISPLAY["Taumode"],
    ]

    all_results        = []
    comparison_metrics = []

    for qi, q in enumerate(queries):
        results_cosine  = aspace.search(query_emb[qi], gl, tau=TAU_COSINE)
        results_hybrid  = aspace.search(query_emb[qi], gl, tau=TAU_HYBRID)
        results_taumode = aspace.search(query_emb[qi], gl, tau=TAU_TAUMODE)

        min_len = min(len(results_cosine), len(results_hybrid), len(results_taumode))
        results_cosine  = results_cosine[:min_len]
        results_hybrid  = results_hybrid[:min_len]
        results_taumode = results_taumode[:min_len]

        all_results.append((results_cosine, results_hybrid, results_taumode))

        spear_c_h, kendall_c_h = compute_ranking_metrics(results_cosine, results_hybrid)
        spear_c_t, kendall_c_t = compute_ranking_metrics(results_cosine, results_taumode)
        spear_h_t, kendall_h_t = compute_ranking_metrics(results_hybrid, results_taumode)

        k_ndcg   = min(NDCG_K, min_len)
        ndcg_h_c = compute_ndcg(results_hybrid,  results_cosine, k=k_ndcg)
        ndcg_t_c = compute_ndcg(results_taumode, results_cosine, k=k_ndcg)
        ndcg_t_h = compute_ndcg(results_taumode, results_hybrid, k=k_ndcg)

        tail_metrics = analyze_tail_distribution(
            [results_cosine, results_hybrid, results_taumode],
            tau_labels,
            k_head=HEAD_K,
            k_tail=graph_params["topk"],
        )

        comparison_metrics.append({
            "query":        q,
            "min_length":   min_len,
            "spearman":     (spear_c_h, spear_c_t, spear_h_t),
            "kendall":      (kendall_c_h, kendall_c_t, kendall_h_t),
            "ndcg":         (ndcg_h_c, ndcg_t_c, ndcg_t_h),
            "tail_metrics": tail_metrics,
        })

        gt_ids    = [idx for idx, _ in results_cosine]
        gt_scores = [sc  for _, sc  in results_cosine]

        recall_metrics_per_tau = {}
        for tau_key, ret_results in [
            ("Cosine",  results_cosine),
            ("Hybrid",  results_hybrid),
            ("Taumode", results_taumode),
        ]:
            ret_ids    = [idx for idx, _ in ret_results]
            ret_scores = [sc  for _, sc  in ret_results]
            recall_metrics_per_tau[tau_key] = compute_all_recall_metrics(
                retrieved_ids=ret_ids,
                retrieved_scores=ret_scores,
                ground_truth_ids=gt_ids,
                ground_truth_scores=gt_scores,
            )
        comparison_metrics[-1]["recall_metrics"] = recall_metrics_per_tau

    # ── Save all standard test_17 outputs ──────────────────────────────────
    save_search_results_to_csv(queries, all_results, ids, titles,
                               out_dir / "cve_search_results.csv")
    save_metrics_to_csv(comparison_metrics,
                        out_dir / "cve_comparison_metrics.csv")
    save_tail_metrics_to_csv(comparison_metrics,
                             out_dir / "cve_tail_metrics.csv")
    save_semantic_recall_to_csv(comparison_metrics,
                                out_dir / "cve_semantic_recall_metrics.csv")
    save_summary_to_csv(comparison_metrics,
                        out_dir / "cve_summary.csv")
    save_query_comparison(queries, all_results, titles, docs,
                          out_dir / "query_comparison.txt")

    headk_sweep_rows = run_headk_sweep(queries, all_results, HEAD_K_SWEEP)
    save_headk_sweep_to_csv(headk_sweep_rows,
                            out_dir / "cve_headk_sweep.csv")

    save_run_metadata(
        out_dir / "cve_run_metadata.json",
        dataset_root=str(dataset_root),
        n_docs=len(docs),
        embedding_model_path="<precomputed .npy — no model used>",
        query_count=len(queries),
        comparison_metrics=comparison_metrics,
    )

    # ── Plots ──────────────────────────────────────────────────────────────
    for fn, kwargs in [
        (plot_comparison,
         dict(queries=queries, all_results=all_results, ids=ids, titles=titles,
              output_file=out_dir / "cve_top25_comparison.png")),
        (plot_tail_comparison,
         dict(queries=queries, all_results=all_results, ids=ids, titles=titles,
              output_file=out_dir / "cve_tail_analysis.png")),
        (plot_semantic_recall_comparison,
         dict(comparison_metrics=comparison_metrics,
              output_file=out_dir / "cve_semantic_recall_comparison.png")),
        (plot_metric_deltas,
         dict(comparison_metrics=comparison_metrics,
              output_file=out_dir / "cve_metric_deltas.png")),
        (plot_win_loss_heatmap,
         dict(comparison_metrics=comparison_metrics,
              output_file=out_dir / "cve_win_loss_heatmap.png")),
        (plot_pareto_tradeoff,
         dict(comparison_metrics=comparison_metrics,
              output_file=out_dir / "cve_pareto_tradeoff.png")),
        (plot_headk_sweep,
         dict(headk_sweep_rows=headk_sweep_rows,
              output_file=out_dir / "cve_headk_sweep.png")),
    ]:
        try:
            fn(**kwargs)
        except Exception as e:
            print(f"  [warn] {fn.__name__} skipped: {e}")

    # ── Aggregate scalars for master CSV ───────────────────────────────────
    avg_ndcg_h_c   = float(np.mean([m["ndcg"][0] for m in comparison_metrics]))
    avg_ndcg_t_c   = float(np.mean([m["ndcg"][1] for m in comparison_metrics]))
    avg_ndcg_t_h   = float(np.mean([m["ndcg"][2] for m in comparison_metrics]))
    avg_spear_c_h  = float(np.mean([m["spearman"][0] for m in comparison_metrics]))
    avg_spear_c_t  = float(np.mean([m["spearman"][1] for m in comparison_metrics]))
    avg_kend_c_h   = float(np.mean([m["kendall"][0]  for m in comparison_metrics]))
    avg_kend_c_t   = float(np.mean([m["kendall"][1]  for m in comparison_metrics]))

    valid_tail = [m for m in comparison_metrics if m["tail_metrics"]]
    th_ratios  = {lbl: [] for lbl in tau_labels}
    for m in valid_tail:
        for lbl in tau_labels:
            if lbl in m["tail_metrics"]:
                th_ratios[lbl].append(m["tail_metrics"][lbl]["tail_to_head_ratio"])

    recall_avgs = {}
    for tau_key in ["Cosine", "Hybrid", "Taumode"]:
        trad_v, sem_v, tol_v = [], [], []
        for m in comparison_metrics:
            rm = m.get("recall_metrics", {}).get(tau_key)
            if rm:
                trad_v.append(rm["traditional_recall"])
                s = rm["semantic_recall"]
                if not (isinstance(s, float) and math.isnan(s)):
                    sem_v.append(s)
                tol_v.append(rm["tolerant_recall"])
        recall_avgs[tau_key] = {
            "trad": float(np.mean(trad_v)) if trad_v else float("nan"),
            "sem":  float(np.mean(sem_v))  if sem_v  else float("nan"),
            "tol":  float(np.mean(tol_v))  if tol_v  else float("nan"),
        }

    def _th(label):
        vals = th_ratios.get(label, [])
        return round(float(np.mean(vals)), 6) if vals else float("nan")

    def _r(tau_key, metric):
        v = recall_avgs[tau_key][metric]
        return round(v, 6) if not (isinstance(v, float) and math.isnan(v)) else float("nan")

    summary_row = {
        "tag":                          tag,
        "k":                            cfg["k"],
        "eps":                          cfg["eps"],
        "topk":                         cfg["topk"],
        "p":                            cfg["p"],
        "sigma":                        cfg["sigma"],
        "build_time_s":                 round(build_time, 3),
        "n_queries":                    len(queries),
        "avg_ndcg_hybrid_vs_cosine":    round(avg_ndcg_h_c, 6),
        "avg_ndcg_taumode_vs_cosine":   round(avg_ndcg_t_c, 6),
        "avg_ndcg_taumode_vs_hybrid":   round(avg_ndcg_t_h, 6),
        "avg_spearman_cosine_hybrid":   round(avg_spear_c_h, 6),
        "avg_spearman_cosine_taumode":  round(avg_spear_c_t, 6),
        "avg_kendall_cosine_hybrid":    round(avg_kend_c_h, 6),
        "avg_kendall_cosine_taumode":   round(avg_kend_c_t, 6),
        "th_ratio_cosine_mean":         _th(TAU_DISPLAY["Cosine"]),
        "th_ratio_hybrid_mean":         _th(TAU_DISPLAY["Hybrid"]),
        "th_ratio_taumode_mean":        _th(TAU_DISPLAY["Taumode"]),
        "trad_recall_cosine":           _r("Cosine",  "trad"),
        "sem_recall_cosine":            _r("Cosine",  "sem"),
        "tol_recall_cosine":            _r("Cosine",  "tol"),
        "trad_recall_hybrid":           _r("Hybrid",  "trad"),
        "sem_recall_hybrid":            _r("Hybrid",  "sem"),
        "tol_recall_hybrid":            _r("Hybrid",  "tol"),
        "trad_recall_taumode":          _r("Taumode", "trad"),
        "sem_recall_taumode":           _r("Taumode", "sem"),
        "tol_recall_taumode":           _r("Taumode", "tol"),
        "output_dir":                   str(out_dir),
    }
    return summary_row


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep test_17 experiment over all tuned param configs dynamically"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to CVE dataset root (cvelistV5) — used for ids/titles/docs metadata only"
    )
    parser.add_argument(
        "--corpus-emb",
        default=None,
        help="Path to precomputed corpus embeddings .npy "
             "(default: ROOT/data/cve/cve_embeddings_cache.npy)"
    )
    parser.add_argument(
        "--query-emb",
        default=None,
        help="Path to precomputed query embeddings .npy  "
             "(default: ROOT/data/cve/cve_query_embs_cache.npy)"
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Path where best_params.json runs are stored "
             "(default: ROOT/results/cve_arrowspace_fstar)"
    )
    parser.add_argument(
        "--sweep-dir", default="results/cve_param_sweep",
        help="Output root for all runs  (default: results/cve_param_sweep)"
    )
    args = parser.parse_args()

    sweep_dir    = Path(args.sweep_dir)
    dataset_root = Path(args.dataset)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    ROOT = _HERE.parent
    
    # Resolve paths (fall back to defaults if not provided)
    corpus_emb_path = Path(args.corpus_emb) if args.corpus_emb else \
                      ROOT / "data" / "cve" / "cve_embeddings_cache.npy"
    query_emb_path  = Path(args.query_emb)  if args.query_emb  else \
                      ROOT / "data" / "cve" / "cve_query_embs_cache.npy"
    results_dir_path = Path(args.results_dir) if args.results_dir else \
                       ROOT / "results" / "cve_arrowspace_fstar"

    assert corpus_emb_path.exists(), f"Corpus embeddings not found: {corpus_emb_path}"
    assert query_emb_path.exists(),  f"Query embeddings not found:  {query_emb_path}"

    # ── Load metadata (ids / titles / docs) from CVE JSONs ─────────────────
    print("\nScanning CVE JSON files for ids / titles / docs metadata...")
    ids, titles, docs = [], [], []
    for _, j in tqdm(iter_cve_json(str(dataset_root))):
        cve_id, title, text = extract_text(j)
        ids.append(cve_id)
        titles.append(title)
        docs.append(text)

    if not ids:
        raise SystemExit("No CVE JSON files found under --dataset")
    print(f"Loaded metadata for {len(ids)} CVEs")

    # ── Load precomputed embeddings directly ────────────────────────────────
    print(f"\nLoading corpus embeddings from {corpus_emb_path}")
    corpus_emb = np.load(str(corpus_emb_path))
    print(f"  corpus_emb shape: {corpus_emb.shape}")

    if len(corpus_emb) != len(ids):
        print(
            f"[warn] Embedding count ({len(corpus_emb)}) != CVE count ({len(ids)}). "
            "Padding/truncating metadata with empty strings so the pipeline continues."
        )
        n = len(corpus_emb)
        if len(ids) < n:
            pad = n - len(ids)
            ids.extend([""] * pad)
            titles.extend([""] * pad)
            docs.extend([""] * pad)
        else:
            ids = ids[:n]
            titles = titles[:n]
            docs = docs[:n]

    print(f"\nLoading query embeddings from {query_emb_path}")
    query_emb = np.load(str(query_emb_path))
    print(f"  query_emb shape: {query_emb.shape}")

    # Use the same fixed query list as test_17; shuffle once for consistency
    queries = list(QUERIES)


    if len(query_emb) != len(queries):
        raise SystemExit(
            f"Query embedding count ({len(query_emb)}) != query list length ({len(queries)}). "
            "Regenerate query embeddings from the fixed 50-query list in this script."
        )

    # ── Dynamic Parameter Generation ────────────────────────────────────────
    param_configs = get_param_configs(results_dir_path)

    # ── Parameter sweep ─────────────────────────────────────────────────────
    all_summary_rows = []
    master_csv       = sweep_dir / "sweep_all_runs_summary.csv"

    for cfg in param_configs:
        row = run_experiment(
            cfg, ids, titles, docs,
            corpus_emb, queries, query_emb,
            sweep_dir, dataset_root,
        )
        all_summary_rows.append(row)

        # persist after every run so partial results survive interruption
        pd.DataFrame(all_summary_rows).to_csv(master_csv, index=False)
        print(f"  [sweep] master summary updated → {master_csv}")

    print(f"\n{'='*70}")
    print(f"All {len(all_summary_rows)} runs complete.")
    print(f"Master summary : {master_csv}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()