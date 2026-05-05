#!/usr/bin/env python3
"""
Stage D — sample queries from CWE buckets.
Outputs:
  data/queries_cwe.csv            (query_id, query_cve_id, primary_cwe, row_id, year, query_text)
  data/queries_cwe_rels.json      [{query_id, query_cve_id, primary_cwe, relevant_cve_ids}]

Run from repo root:
    python scripts/build_ground_truth/stage_d_sample_queries.py
"""
import csv, json, random, sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

BUCKETS_PATH     = Path("data/cwe_buckets.json")
FILTERED_MAP     = Path("data/cve_cwe_map_filtered.csv")
PARQUET_PATH     = Path("data/cve_embs/cve_corpus.parquet")
OUT_QUERIES_CSV  = Path("data/queries_cwe.csv")
OUT_RELS_JSON    = Path("data/queries_cwe_rels.json")

MAX_QUERIES_PER_CWE = 5
RANDOM_SEED         = 42


def main():
    random.seed(RANDOM_SEED)

    # --- load buckets ---
    print(f"Loading {BUCKETS_PATH} ...")
    with open(BUCKETS_PATH, encoding="utf-8") as f:
        buckets: dict[str, list[str]] = json.load(f)
    print(f"  CWE buckets: {len(buckets):,}")

    # --- load filtered map for row_id / year lookup ---
    df_map = pd.read_csv(FILTERED_MAP, dtype={"cve_id": str, "primary_cwe": str,
                                               "row_id": int, "year": int})
    cve_meta = df_map.set_index("cve_id").to_dict("index")
    # cve_meta[cve_id] = {"primary_cwe": ..., "row_id": ..., "year": ...}

    # --- load text from parquet ---
    print(f"Loading text column from {PARQUET_PATH} ...")
    df_text = pq.read_table(PARQUET_PATH, columns=["cve_id", "text"]).to_pandas()
    cve_text = dict(zip(df_text["cve_id"], df_text["text"]))

    # --- sample queries ---
    queries_csv_rows = []
    rels_rows        = []
    query_id         = 1

    for cwe, bucket_ids in sorted(buckets.items()):
        # how many queries to take from this bucket
        n_take = min(MAX_QUERIES_PER_CWE, len(bucket_ids))
        chosen = random.sample(bucket_ids, n_take)

        for cve_q in chosen:
            meta = cve_meta.get(cve_q)
            text = cve_text.get(cve_q, "")
            if meta is None or not text:
                continue  # skip if somehow missing

            relevant = [c for c in bucket_ids if c != cve_q]

            queries_csv_rows.append({
                "query_id":      query_id,
                "query_cve_id":  cve_q,
                "primary_cwe":   cwe,
                "row_id":        meta["row_id"],
                "year":          meta["year"],
                "query_text":    text,
            })
            rels_rows.append({
                "query_id":         query_id,
                "query_cve_id":     cve_q,
                "primary_cwe":      cwe,
                "relevant_cve_ids": relevant,
            })
            query_id += 1

    total_queries = len(queries_csv_rows)
    print(f"  Total queries sampled: {total_queries:,}")

    # --- cwe coverage ---
    cwes_covered = len(set(r["primary_cwe"] for r in queries_csv_rows))
    print(f"  CWEs covered: {cwes_covered:,}")

    # --- avg relevant per query ---
    avg_rel = sum(len(r["relevant_cve_ids"]) for r in rels_rows) / len(rels_rows)
    print(f"  Avg relevant docs per query: {avg_rel:.1f}")

    # --- save queries CSV ---
    with open(OUT_QUERIES_CSV, "w", newline="", encoding="utf-8") as f:
        fields = ["query_id", "query_cve_id", "primary_cwe", "row_id", "year", "query_text"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(queries_csv_rows)
    print(f"\nSaved {OUT_QUERIES_CSV}")

    # --- save rels JSON ---
    with open(OUT_RELS_JSON, "w", encoding="utf-8") as f:
        json.dump(rels_rows, f, indent=2)
    print(f"Saved {OUT_RELS_JSON}")


if __name__ == "__main__":
    main()
