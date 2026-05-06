#!/usr/bin/env python3
"""
prepare_evaluation_dataset.py

Builds the ArrowSpace evaluation dataset:
  1. Loads queries_cwe.csv (query_id, query_cve_id, primary_cwe, row_id, year, query_text)
  2. Encodes query_text with BAAI/bge-small-en-v1.5 (no normalization, matching corpus)
  3. Loads cwe_buckets.json and builds relevant_cves per query
  4. Outputs:
       data/arrowspace_eval_queries.jsonl       query metadata + relevant CVE ids
       data/arrowspace_eval_queries_emb.npy     freshly-encoded query embeddings (N x 384)
       data/arrowspace_eval_summary.txt         sanity stats

Usage:
    uv run python scripts/build_ground_truth/prepare_evaluation_dataset.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


ROOT      = Path("/content/drive/MyDrive/")
DATA      = ROOT / "query_cve"
QUERY_CSV = DATA / "queries_cwe.csv"
BUCKETS   = DATA / "cwe_buckets.json"
OUT_JSONL = DATA / "arrowspace_eval_queries.jsonl"
OUT_EMB   = DATA / "arrowspace_eval_queries_emb.npy"
OUT_STATS = DATA / "arrowspace_eval_summary.txt"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 256


def main():
    # ── 1. Load queries ──────────────────────────────────────────────────────
    print(f"Loading {QUERY_CSV} ...")
    qdf = pd.read_csv(QUERY_CSV, dtype={
        "query_id": int, "query_cve_id": str,
        "primary_cwe": str, "row_id": int,
        "year": int, "query_text": str,
    })
    print(f"  Queries loaded : {len(qdf):,}")
    print(f"  Columns        : {qdf.columns.tolist()}")

    for col in ("query_id", "query_cve_id", "primary_cwe", "row_id", "year", "query_text"):
        assert col in qdf.columns, f"Missing required column: '{col}'"

    # ── 2. Encode query texts with the corpus model ───────────────────────────
    print(f"\nLoading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    query_texts = qdf["query_text"].fillna("").tolist()
    print(f"Encoding {len(query_texts):,} queries in batches of {BATCH_SIZE} ...")
    emb = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=False,   # match corpus encoding (no l2-norm baked in)
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  Embedding shape: {emb.shape}  dtype={emb.dtype}")

    # ── 3. Load CWE buckets ───────────────────────────────────────────────────
    print(f"\nLoading {BUCKETS} ...")
    buckets: dict[str, list[str]] = json.loads(BUCKETS.read_text())
    print(f"  CWE buckets    : {len(buckets):,}")

    # ── 4. Build relevant CVE list per query ──────────────────────────────────
    #    Relevant = all CVEs in same CWE bucket, excluding the query CVE itself
    def get_relevant(row):
        bucket = buckets.get(row["primary_cwe"], [])
        return [c for c in bucket if c != row["query_cve_id"]]

    qdf["relevant_cves"] = qdf.apply(get_relevant, axis=1)
    qdf["n_relevant"]    = qdf["relevant_cves"].apply(len)

    zero_rel = (qdf["n_relevant"] == 0).sum()
    if zero_rel:
        print(f"  WARNING: {zero_rel} queries have 0 relevant CVEs — dropping them.")
        keep_mask = qdf["n_relevant"] > 0
        qdf  = qdf[keep_mask].reset_index(drop=True)
        emb  = emb[keep_mask.to_numpy()]

    print(f"  Final queries  : {len(qdf):,}")
    print(f"  Avg relevant   : {qdf['n_relevant'].mean():.1f}")
    assert emb.shape[0] == len(qdf), "Embedding/query count mismatch after filtering"

    # ── 5. Save JSONL ─────────────────────────────────────────────────────────
    records = [
        {
            "query_id":      int(row["query_id"]),
            "query_cve_id":  row["query_cve_id"],
            "primary_cwe":   row["primary_cwe"],
            "row_id":        int(row["row_id"]),
            "year":          int(row["year"]),
            "query_text":    row["query_text"],
            "relevant_cves": row["relevant_cves"],
            "n_relevant":    int(row["n_relevant"]),
        }
        for _, row in qdf.iterrows()
    ]

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved {OUT_JSONL}  ({len(records):,} records)")

    # ── 6. Save embeddings ────────────────────────────────────────────────────
    np.save(OUT_EMB, emb)
    print(f"Saved {OUT_EMB}  shape={emb.shape}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    top_cwes  = qdf.groupby("primary_cwe")["query_id"].count().sort_values(ascending=False).head(10)
    year_dist = qdf.groupby("year")["query_id"].count().sort_values()

    lines = [
        "=== ArrowSpace Evaluation Dataset Summary ===",
        f"Model              : {MODEL_NAME}",
        f"normalize_embeddings: False",
        f"Total queries      : {len(qdf):,}",
        f"Distinct CWEs      : {qdf['primary_cwe'].nunique():,}",
        f"Embedding shape    : {emb.shape}",
        f"Avg relevant/query : {qdf['n_relevant'].mean():.1f}",
        f"Min relevant/query : {qdf['n_relevant'].min()}",
        f"Max relevant/query : {qdf['n_relevant'].max()}",
        "",
        "Top 10 CWEs by query count:",
    ] + [f"  {cwe:15s}  {cnt:4d}" for cwe, cnt in top_cwes.items()] + [
        "",
        "Year distribution (last 8 years):",
    ] + [f"  {yr}  {cnt:,}" for yr, cnt in year_dist.tail(8).items()] + [
        "",
        "Output files:",
        f"  {OUT_JSONL}",
        f"  {OUT_EMB}",
    ]

    summary = "\n".join(lines)
    print("\n" + summary)
    OUT_STATS.write_text(summary + "\n")
    print(f"\nSaved {OUT_STATS}")


if __name__ == "__main__":
    main()