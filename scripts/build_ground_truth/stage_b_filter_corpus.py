#!/usr/bin/env python3
"""
Stage B — join cve_cwe_map.csv onto the mapping CSV (row_id, cve_id, year, text).

We do NOT use data/cve1999-2025.parquet here because it's an old embedding-only
dump with anonymous col_0..col_383. The clean canonical metadata is:

    data/cve_embs/cve1999-2025.mapping.csv   (row_id, cve_id, year, text)
    data/cve_embs/cve1999-2025.npy           (embeddings, shape=(N, 384))

Output:
    data/cve_cwe_map_filtered.csv
        columns: cve_id, primary_cwe, row_id, year
"""

import sys
from pathlib import Path

import pandas as pd

CWE_MAP_PATH      = Path("data/cve_cwe_map.csv")
MAPPING_CSV_PATH  = Path("data/cve_embs/cve1999-2025.mapping.csv")
OUT_PATH          = Path("data/cve_cwe_map_filtered.csv")


def main():
    # --- load CWE labels ---
    print(f"Loading {CWE_MAP_PATH} ...")
    df_cwe = pd.read_csv(CWE_MAP_PATH, dtype=str)
    print(f"  CWE map rows: {len(df_cwe):,}")

    # --- load corpus metadata (row_id, cve_id, year, text) ---
    print(f"Loading mapping CSV {MAPPING_CSV_PATH} ...")
    df_map = pd.read_csv(
        MAPPING_CSV_PATH,
        dtype={"row_id": int, "cve_id": str, "year": int, "text": str},
    )
    print(f"  Mapping rows (corpus size): {len(df_map):,}")
    print(f"  Sample columns: {list(df_map.columns)}")

    # --- inner join on cve_id ---
    df = df_map.merge(df_cwe, on="cve_id", how="inner")
    print(f"  Matched (have both embedding + CWE): {len(df):,}")
    print(f"  Distinct CWEs: {df['primary_cwe'].nunique():,}")

    # --- year distribution of labeled CVEs ---
    year_counts = df.groupby("year").size()
    print("\nLabeled CVEs by year (tail):")
    print(year_counts.tail(10).to_string())

    # --- save ---
    df_out = df[["cve_id", "primary_cwe", "row_id", "year"]].sort_values("row_id")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}  ({len(df_out):,} rows)")


if __name__ == "__main__":
    main()