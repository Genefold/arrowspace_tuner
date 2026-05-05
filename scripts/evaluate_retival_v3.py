#!/usr/bin/env python3
"""
evaluate_retrieval_semantic.py

CVE spectral eval with 3 additional semantic/novelty diagnostics:
A) BERTScore query -> retrieved TEXT using an external model (independent from retrieval cosine)
B) Cross-config agreement / overlap diagnostics (RRF-style overlap + Jaccard@k)
D) Novelty diagnostics vs cosine (Jaccard novelty@k)
"""
from __future__ import annotations

import argparse, csv, json, logging, time, gc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from bert_score import BERTScorer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TAU_COSINE = 1.0
TAU_HYBRID = 0.75
TAU_TAUMODE = 0.42
K_TAIL_MAX = 30
K_HEAD = 5
CORPUS_SCALE = 12.0
SEMANTIC_TOPK = 10
OVERLAP_K = 10
BASELINE_GRAPH = {"eps": 1.31, "k": K_TAIL_MAX, "topk": int(K_TAIL_MAX/2), "p": 1.8, "sigma": 0.535}
graph_params = {'eps': 4.0, 'k': 40, 'topk': 19, 'p': 2.0, 'sigma': None}


def load_corpus_parquet(path):
    import pyarrow.parquet as pq
    logger.info("Loading parquet file into memory...")
    df = pq.read_table(path).to_pandas()
    if df["row_id"].tolist() != list(range(len(df))):
        raise ValueError("parquet row_id non contiguo")
    
    # Extract data
    corpus = np.stack(df["embedding"].values)
    cve_ids = df["cve_id"].tolist()
    titles = df["title"].tolist() if "title" in df.columns else cve_ids
    descriptions = df["description"].tolist() if "description" in df.columns else titles
    
    # MEMORY FIX: Destroy the massive pandas dataframe immediately
    del df
    gc.collect() 
    logger.info("Parquet loaded and DataFrame memory freed.")
    
    return corpus, cve_ids, titles, descriptions


def load_corpus_npy(npy_path, mapping_path):
    corpus = np.load(npy_path)
    df = pd.read_csv(mapping_path, dtype={"row_id": int, "cve_id": str})
    if df["row_id"].tolist() != list(range(len(df))):
        raise ValueError("mapping CSV row_id non contiguo")
    
    cve_ids = df["cve_id"].tolist()
    titles = df["title"].tolist() if "title" in df.columns else cve_ids
    descriptions = df["description"].tolist() if "description" in df.columns else titles
    
    # MEMORY FIX: Free pandas dataframe
    del df
    gc.collect()
    
    return corpus, cve_ids, titles, descriptions


def build_graph(corpus_raw, graph_params, scale, label):
    from arrowspace import ArrowSpaceBuilder
    if isinstance(graph_params, dict):
        params = graph_params
    else:
        params = {k: v for k, v in graph_params.items() if v is not None}
    
    corpus_f64 = np.ascontiguousarray(corpus_raw.astype(np.float64) * scale)
    logger.info("[%s] Build — params=%s scale=×%.1f", label, params, scale)
    t0 = time.perf_counter()
    aspace, gl = ArrowSpaceBuilder().build(params, corpus_f64)
    logger.info("[%s] Grafo costruito in %.1fs", label, time.perf_counter() - t0)
    
    # MEMORY FIX: Delete the temporary f64 copy
    del corpus_f64
    gc.collect()
    
    return aspace, gl


def ranking_corr(ra, rb):
    ia = [i for i, _ in ra]
    ib = [i for i, _ in rb]
    shared = list(set(ia) & set(ib))
    if len(shared) < 2:
        return 0.0, 0.0
    rank_a = [ia.index(x) for x in shared]
    rank_b = [ib.index(x) for x in shared]
    rho, _ = spearmanr(rank_a, rank_b)
    kt, _ = kendalltau(rank_a, rank_b)
    return float(0 if np.isnan(rho) else rho), float(0 if np.isnan(kt) else kt)


def synth_ndcg(pred, ref, k=10):
    ref_idx = [i for i, _ in ref[:k]]
    rel_map = {i: k - r for r, i in enumerate(ref_idx)}
    pred_idx = [i for i, _ in pred[:k]]
    true_rel = [rel_map.get(i, 0) for i in pred_idx]
    if sum(true_rel) == 0:
        return 0.0
    scores = np.array([s for _, s in pred[:k]], dtype=float)
    if scores.max() > 0:
        scores = scores / scores.max()
    try:
        return float(ndcg_score(np.array([true_rel]), np.array([scores]), k=k))
    except Exception:
        return 0.0


def tail_metrics(results_list, labels, k_head=K_HEAD, k_tail=K_TAIL_MAX):
    min_len = min(len(r) for r in results_list)
    if min_len <= k_head:
        return {}
    kt = min(k_tail, min_len)
    out = {}
    for res, lbl in zip(results_list, labels):
        seg = res[:kt]
        head = [s for _, s in seg[:k_head]]
        tail = [s for _, s in seg[k_head:kt]]
        if not head or not tail:
            continue
        hm = float(np.mean(head))
        tm = float(np.mean(tail))
        ts = float(np.std(tail))
        out[lbl] = {
            "head_mean": hm,
            "tail_mean": tm,
            "tail_std": ts,
            "tail_to_head_ratio": tm / hm if hm > 1e-10 else 0.0,
            "tail_cv": ts / tm if tm > 1e-10 else 0.0,
            "tail_decay_rate": (tail[0] - tail[-1]) / len(tail) if len(tail) > 1 else 0.0,
            "n_tail": len(tail),
            "k_total": kt,
        }
    return out


def topk_ids(res, k=10):
    return [idx for idx, _ in res[:k]]


def jaccard_at_k(a, b, k=10):
    sa = set(topk_ids(a, k))
    sb = set(topk_ids(b, k))
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return float(len(sa & sb) / union)


def novelty_at_k(a, b, k=10):
    return 1.0 - jaccard_at_k(a, b, k)


def rrf_score(indices, k=60):
    return {doc_id: 1.0 / (k + rank + 1) for rank, doc_id in enumerate(indices)}


def weighted_rrf_overlap(res_a, res_b, k=10, c=60):
    ia = topk_ids(res_a, k)
    ib = topk_ids(res_b, k)
    ra = rrf_score(ia, c)
    rb = rrf_score(ib, c)
    shared = set(ra) & set(rb)
    if not shared:
        return 0.0
    numer = sum(ra[d] + rb[d] for d in shared)
    denom = sum(ra.values()) + sum(rb.values())
    return float(numer / denom)


def borda_from_metrics(metric_values, higher_better=True):
    ordered = sorted(metric_values.items(), key=lambda kv: kv[1], reverse=higher_better)
    return {cfg: rank + 1 for rank, (cfg, _) in enumerate(ordered)}


def compute_borda_winner(per_cfg_metrics):
    ranks_tail = borda_from_metrics({k: v["tail_to_head_ratio"] for k, v in per_cfg_metrics.items()}, True)
    ranks_cv = borda_from_metrics({k: v["tail_cv"] for k, v in per_cfg_metrics.items()}, False)
    ranks_top1 = borda_from_metrics({k: v["top1_score"] for k, v in per_cfg_metrics.items()}, True)
    avg_rank = {
        cfg: float(np.mean([ranks_tail[cfg], ranks_cv[cfg], ranks_top1[cfg]]))
        for cfg in per_cfg_metrics.keys()
    }
    winner = min(avg_rank, key=avg_rank.get)
    return winner, avg_rank, ranks_tail, ranks_cv, ranks_top1


def build_configs(best_eps, best_k):
    optuna_graph = {"eps": best_eps, "k": best_k, "topk": int(best_k/2), "p": 2.0}
    configs = []
    for graph_label, graph_param_dict in [("baseline", BASELINE_GRAPH), ("optuna", optuna_graph)]:
        for tau_label, tau_val in [("cosine", TAU_COSINE), ("hybrid", TAU_HYBRID), ("taumode", TAU_TAUMODE)]:
            configs.append({
                "cfg_label": f"{graph_label}_{tau_label}",
                "graph_label": graph_label,
                "graph_params": graph_param_dict,
                "tau_label": tau_label,
                "tau": tau_val,
            })
    return configs, optuna_graph


def run_experiment(corpus_raw, cve_ids, titles, descriptions, q_embs, queries, best_eps, best_k, scale, bert_model):
    configs, optuna_graph = build_configs(best_eps, best_k)
    
    # 1. Build graphs
    aspace_bl, gl_bl = build_graph(corpus_raw, BASELINE_GRAPH, scale, "baseline")
    aspace_op, gl_op = build_graph(corpus_raw, optuna_graph, scale, "optuna")
    graph_map = {"baseline": (aspace_bl, gl_bl), "optuna": (aspace_op, gl_op)}
    
    # MEMORY FIX: Destroy corpus_raw immediately after graphs are built to free ~1GB RAM
    del corpus_raw
    gc.collect()
    logger.info("Raw corpus memory freed after graph construction.")

    q_f64 = np.ascontiguousarray(q_embs.astype(np.float64) * scale)

    # 2. Load BERT
    logger.info("Initializing external BERTScorer model=%s", bert_model)
    scorer = BERTScorer(
        model_type=bert_model, 
        lang="en", 
        rescale_with_baseline=True,
        device="cpu",  
        batch_size=4
    )
    scorer._model.half()
    scorer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer._model.to(scorer.device)
    
    all_query_results, all_query_metrics, winner_rows = [], [], []
    cfg_labels = [c["cfg_label"] for c in configs]

    for qi, q in enumerate(tqdm(queries, desc="Searching & Scoring")):
        q_results = {}
        for cfg in configs:
            aspace, gl = graph_map[cfg["graph_label"]]
            q_results[cfg["cfg_label"]] = aspace.search(q_f64[qi], gl, cfg["tau"])

        min_len = min(len(r) for r in q_results.values())
        q_results = {k: v[:min_len] for k, v in q_results.items()}
        all_query_results.append(q_results)

        qm = {"query": q, "min_len": min_len}
        per_cfg_metrics = {}

        for graph_label in ["baseline", "optuna"]:
            cos_key = f"{graph_label}_cosine"
            hyb_key = f"{graph_label}_hybrid"
            tau_key = f"{graph_label}_taumode"

            res_cos, res_hyb, res_tau = q_results[cos_key], q_results[hyb_key], q_results[tau_key]
            k_ndcg = min(10, min_len)

            rho_h, kt_h = ranking_corr(res_hyb, res_cos)
            rho_t, kt_t = ranking_corr(res_tau, res_cos)

            qm[f"{graph_label}_spearman_hybrid"] = rho_h
            qm[f"{graph_label}_spearman_taumode"] = rho_t
            qm[f"{graph_label}_kendall_hybrid"] = kt_h
            qm[f"{graph_label}_kendall_taumode"] = kt_t
            qm[f"{graph_label}_ndcg_hybrid"] = synth_ndcg(res_hyb, res_cos, k_ndcg)
            qm[f"{graph_label}_ndcg_taumode"] = synth_ndcg(res_tau, res_cos, k_ndcg)

            tm = tail_metrics([res_cos, res_hyb, res_tau], [cos_key, hyb_key, tau_key], K_HEAD, K_TAIL_MAX)
            qm[f"{graph_label}_tail_metrics"] = tm

            for cfg_key in [cos_key, hyb_key, tau_key]:
                idxs = topk_ids(q_results[cfg_key], SEMANTIC_TOPK)
                docs = [descriptions[i] if isinstance(descriptions[i], str) and descriptions[i].strip() else titles[i] for i in idxs]
                if docs:
                    refs = [q] * len(docs)
                    with torch.no_grad(): # Ensure no memory leaks during scoring
                        _, _, F1 = scorer.score(docs, refs)
                    mean_sem = float(F1.mean())
                else:
                    mean_sem = 0.0
                qm[f"{cfg_key}_semantic_bertscore"] = mean_sem

            qm[f"{graph_label}_jaccard_hybrid_vs_cosine@{OVERLAP_K}"] = jaccard_at_k(res_hyb, res_cos, OVERLAP_K)
            qm[f"{graph_label}_jaccard_taumode_vs_cosine@{OVERLAP_K}"] = jaccard_at_k(res_tau, res_cos, OVERLAP_K)
            qm[f"{graph_label}_rrf_overlap_hybrid_vs_cosine@{OVERLAP_K}"] = weighted_rrf_overlap(res_hyb, res_cos, OVERLAP_K)
            qm[f"{graph_label}_rrf_overlap_taumode_vs_cosine@{OVERLAP_K}"] = weighted_rrf_overlap(res_tau, res_cos, OVERLAP_K)

            qm[f"{graph_label}_novelty_hybrid_vs_cosine@{OVERLAP_K}"] = novelty_at_k(res_hyb, res_cos, OVERLAP_K)
            qm[f"{graph_label}_novelty_taumode_vs_cosine@{OVERLAP_K}"] = novelty_at_k(res_tau, res_cos, OVERLAP_K)

            for cfg_key in [cos_key, hyb_key, tau_key]:
                per_cfg_metrics[cfg_key] = {
                    "tail_to_head_ratio": tm[cfg_key]["tail_to_head_ratio"] if cfg_key in tm else 0,
                    "tail_cv": tm[cfg_key]["tail_cv"] if cfg_key in tm else 0,
                    "top1_score": q_results[cfg_key][0][1] if q_results[cfg_key] else 0.0,
                }

        winner, avg_rank, ranks_tail, ranks_cv, ranks_top1 = compute_borda_winner(per_cfg_metrics)
        qm["borda_avg_rank"] = avg_rank
        all_query_metrics.append(qm)

        row = {"query_id": qi + 1, "query_text": q, "winner": winner, "winner_avg_rank": f"{avg_rank[winner]:.4f}"}
        for cfg in cfg_labels:
            row[f"avg_{cfg}"] = f"{avg_rank[cfg]:.4f}"
            row[f"tail_rank_{cfg}"] = ranks_tail[cfg]
            row[f"cv_rank_{cfg}"] = ranks_cv[cfg]
            row[f"top1_rank_{cfg}"] = ranks_top1[cfg]
        winner_rows.append(row)

        # MEMORY FIX: Force PyTorch to release cached VRAM periodically
        if qi % 10 == 0:
            torch.cuda.empty_cache()

    return all_query_results, all_query_metrics, winner_rows, cfg_labels


def save_search_results(queries, all_results, cve_ids, out_path, cfg_labels):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "query_text", "config", "rank", "cve_id", "score"])
        w.writeheader()
        for qi, q in enumerate(queries):
            for cfg in cfg_labels:
                for rank, (idx, score) in enumerate(all_results[qi][cfg][:20], 1):
                    w.writerow({"query_id": qi + 1, "query_text": q, "config": cfg, "rank": rank, "cve_id": cve_ids[idx], "score": f"{score:.6f}"})


def save_metrics(all_metrics, out_path, cfg_labels):
    fields = [
        "query_id", "query_text", "min_len",
        "baseline_spearman_hybrid", "baseline_spearman_taumode", "baseline_kendall_hybrid", "baseline_kendall_taumode", "baseline_ndcg_hybrid", "baseline_ndcg_taumode",
        "optuna_spearman_hybrid", "optuna_spearman_taumode", "optuna_kendall_hybrid", "optuna_kendall_taumode", "optuna_ndcg_hybrid", "optuna_ndcg_taumode",
        f"baseline_jaccard_hybrid_vs_cosine@{OVERLAP_K}", f"baseline_jaccard_taumode_vs_cosine@{OVERLAP_K}",
        f"optuna_jaccard_hybrid_vs_cosine@{OVERLAP_K}", f"optuna_jaccard_taumode_vs_cosine@{OVERLAP_K}",
        f"baseline_rrf_overlap_hybrid_vs_cosine@{OVERLAP_K}", f"baseline_rrf_overlap_taumode_vs_cosine@{OVERLAP_K}",
        f"optuna_rrf_overlap_hybrid_vs_cosine@{OVERLAP_K}", f"optuna_rrf_overlap_taumode_vs_cosine@{OVERLAP_K}",
        f"baseline_novelty_hybrid_vs_cosine@{OVERLAP_K}", f"baseline_novelty_taumode_vs_cosine@{OVERLAP_K}",
        f"optuna_novelty_hybrid_vs_cosine@{OVERLAP_K}", f"optuna_novelty_taumode_vs_cosine@{OVERLAP_K}",
    ]
    for cfg in cfg_labels:
        fields.append(f"{cfg}_semantic_bertscore")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for qi, m in enumerate(all_metrics):
            row = {"query_id": qi + 1, "query_text": m["query"], "min_len": m["min_len"]}
            for k in fields[3:]:
                row[k] = f"{m.get(k, 0):.6f}"
            w.writerow(row)


def save_tail_csv(all_metrics, out_path):
    fields = ["query_id", "query_text", "graph", "tau_mode", "head_mean", "tail_mean", "tail_std", "tail_to_head_ratio", "tail_cv", "tail_decay_rate"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for qi, m in enumerate(all_metrics):
            for graph in ["baseline", "optuna"]:
                tm = m.get(f"{graph}_tail_metrics", {})
                for cfg_key, tv in tm.items():
                    tau_part = cfg_key.split("_", 1)[1]
                    w.writerow({
                        "query_id": qi + 1, "query_text": m["query"], "graph": graph, "tau_mode": tau_part,
                        "head_mean": f"{tv['head_mean']:.6f}", "tail_mean": f"{tv['tail_mean']:.6f}", "tail_std": f"{tv['tail_std']:.6f}",
                        "tail_to_head_ratio": f"{tv['tail_to_head_ratio']:.6f}", "tail_cv": f"{tv['tail_cv']:.6f}", "tail_decay_rate": f"{tv['tail_decay_rate']:.6f}"
                    })


def save_borda_winners(winner_rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(winner_rows[0].keys()))
        w.writeheader()
        w.writerows(winner_rows)


def save_summary(all_metrics, winner_rows, cfg_labels, out_path):
    rows = []
    for graph in ["baseline", "optuna"]:
        for tau in ["cosine", "hybrid", "taumode"]:
            cfg = f"{graph}_{tau}"
            tail_vals = [m.get(f"{graph}_tail_metrics", {}).get(cfg, {}).get("tail_to_head_ratio", np.nan) for m in all_metrics]
            cv_vals = [m.get(f"{graph}_tail_metrics", {}).get(cfg, {}).get("tail_cv", np.nan) for m in all_metrics]
            bert_vals = [m.get(f"{cfg}_semantic_bertscore", np.nan) for m in all_metrics]
            avg_rank_vals = [float(r[f"avg_{cfg}"]) for r in winner_rows]
            win_count = sum(1 for r in winner_rows if r["winner"] == cfg)
            rows.append({
                "config": cfg,
                "mean_borda_rank": f"{np.mean(avg_rank_vals):.6f}",
                "wins": win_count,
                "win_rate": f"{win_count/len(winner_rows):.6f}",
                "mean_tail_head": f"{np.nanmean(tail_vals):.6f}",
                "mean_tail_cv": f"{np.nanmean(cv_vals):.6f}",
                "mean_semantic_bertscore": f"{np.nanmean(bert_vals):.6f}",
            })

    rows.append({
        "config": f"baseline_DIAG@{OVERLAP_K}",
        "mean_borda_rank": "",
        "wins": "",
        "win_rate": "",
        "mean_tail_head": f"J_hyb={np.nanmean([m.get(f'baseline_jaccard_hybrid_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}; J_tau={np.nanmean([m.get(f'baseline_jaccard_taumode_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}",
        "mean_tail_cv": f"RRF_hyb={np.nanmean([m.get(f'baseline_rrf_overlap_hybrid_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}; RRF_tau={np.nanmean([m.get(f'baseline_rrf_overlap_taumode_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}",
        "mean_semantic_bertscore": f"N_hyb={np.nanmean([m.get(f'baseline_novelty_hybrid_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}; N_tau={np.nanmean([m.get(f'baseline_novelty_taumode_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}",
    })
    rows.append({
        "config": f"optuna_DIAG@{OVERLAP_K}",
        "mean_borda_rank": "",
        "wins": "",
        "win_rate": "",
        "mean_tail_head": f"J_hyb={np.nanmean([m.get(f'optuna_jaccard_hybrid_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}; J_tau={np.nanmean([m.get(f'optuna_jaccard_taumode_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}",
        "mean_tail_cv": f"RRF_hyb={np.nanmean([m.get(f'optuna_rrf_overlap_hybrid_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}; RRF_tau={np.nanmean([m.get(f'optuna_rrf_overlap_taumode_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}",
        "mean_semantic_bertscore": f"N_hyb={np.nanmean([m.get(f'optuna_novelty_hybrid_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}; N_tau={np.nanmean([m.get(f'optuna_novelty_taumode_vs_cosine@{OVERLAP_K}', np.nan) for m in all_metrics]):.6f}",
    })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_borda_summary(winner_rows, cfg_labels, out_path):
    means = [np.mean([float(r[f"avg_{cfg}"]) for r in winner_rows]) for cfg in cfg_labels]
    wins = [sum(1 for r in winner_rows if r["winner"] == cfg) for cfg in cfg_labels]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1f77b4", "#6baed6", "#08519c", "#d62728", "#fb6a4a", "#a50f15"]
    x = np.arange(len(cfg_labels))
    axes[0].bar(x, means, color=colors, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace("_", "\n") for c in cfg_labels], fontsize=9)
    axes[0].set_title("Mean Borda rank\n(lower = better)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, wins, color=colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace("_", "\n") for c in cfg_labels], fontsize=9)
    axes[1].set_title("Per-query wins\n(higher = better)")
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="CVE spectral eval with semantic BERTScore + overlap + novelty diagnostics")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--corpus-parquet")
    g.add_argument("--corpus-npy")
    p.add_argument("--corpus-mapping")
    p.add_argument("--queries-npy", required=True)
    p.add_argument("--queries-csv", required=True)
    p.add_argument("--best-params", required=False)
    p.add_argument("--out-dir", default="results/cve_spectral_semantic")
    p.add_argument("--corpus-scale", type=float, default=CORPUS_SCALE)
    p.add_argument("--n-queries", type=int, default=150)
    p.add_argument("--bert-model", default="microsoft/deberta-xlarge-mnli")
    p.add_argument("--log", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log.upper(), logging.INFO))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.corpus_parquet:
        corpus, cve_ids, titles, descriptions = load_corpus_parquet(args.corpus_parquet)
    else:
        if not args.corpus_mapping:
            raise SystemExit("--corpus-mapping obbligatorio con --corpus-npy")
        corpus, cve_ids, titles, descriptions = load_corpus_npy(args.corpus_npy, args.corpus_mapping)

    q_embs = np.load(args.queries_npy).astype(np.float64)
    df_q = pd.read_csv(args.queries_csv).reset_index(drop=True)
    if len(df_q) != q_embs.shape[0]:
        raise ValueError(f"CSV query ({len(df_q)}) ≠ npy ({q_embs.shape[0]})")
    n = min(args.n_queries, len(df_q))
    queries = df_q["query_text"].tolist()[:n]
    q_embs = q_embs[:n]
    
    if args.best_params:
        logger.info("Loading graph params from %s", args.best_params)
        with open(args.best_params) as f:
            bp = json.load(f)["params"]
        best_eps = float(bp["eps"])
        best_k = int(bp["k"])
    else:
        logger.info("No --best-params provided, falling back to hardcoded graph_params")
        best_eps = float(graph_params["eps"])
        best_k = int(graph_params["k"])

    all_results, all_metrics, winner_rows, cfg_labels = run_experiment(
        corpus, cve_ids, titles, descriptions, q_embs, queries, best_eps, best_k, args.corpus_scale, args.bert_model
    )

    save_search_results(queries, all_results, cve_ids, out_dir / "spectral_search_results.csv", cfg_labels)
    save_metrics(all_metrics, out_dir / "spectral_metrics.csv", cfg_labels)
    save_tail_csv(all_metrics, out_dir / "spectral_tail_metrics.csv")
    save_borda_winners(winner_rows, out_dir / "spectral_borda_winners.csv")
    save_summary(all_metrics, winner_rows, cfg_labels, out_dir / "spectral_summary.csv")
    plot_borda_summary(winner_rows, cfg_labels, out_dir / "spectral_borda_summary.png")

    print("\n" + "="*100)
    print("SUMMARY (Borda = tail + cv + top1 | semantic BERTScore + overlap + novelty are diagnostics)")
    print("="*100)
    rows = pd.read_csv(out_dir / "spectral_summary.csv")
    print(rows.to_string(index=False))
    print(f"\nRisultati salvati in {out_dir}")

if __name__ == "__main__":
    main()