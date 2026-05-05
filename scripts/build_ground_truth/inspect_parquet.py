#!/usr/bin/env python3
"""
Diagnostic — find all data files in data/ and inspect the non-parquet ones.
Run from repo root:
    uv run python scripts/build_ground_truth/inspect_data_dir.py
"""
import os
import json
from pathlib import Path

DATA_DIR = Path("data")

print("=== ALL FILES IN data/ (top level + cve_embs/) ===")
for p in sorted(DATA_DIR.iterdir()):
    if p.is_file():
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:<40s}  {size_mb:8.1f} MB")
    elif p.is_dir() and p.name != "cvelistV5":
        print(f"  {p.name}/")
        for sub in sorted(p.iterdir()):
            size_mb = sub.stat().st_size / 1e6
            print(f"    {sub.name:<40s}  {size_mb:8.1f} MB")

# Inspect CSV/TSV files that might carry the metadata
print("\n=== INSPECTING CSV/TSV FILES ===")
for p in sorted(DATA_DIR.rglob("*.csv")) + sorted(DATA_DIR.rglob("*.tsv")):
    if "cvelistV5" in str(p):
        continue
    print(f"\n--- {p} ---")
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(f"  {line.rstrip()}")
            if i >= 4:
                break

# Inspect .npy shapes
print("\n=== NPY FILE SHAPES ===")
import numpy as np
for p in sorted(DATA_DIR.rglob("*.npy")):
    arr = np.load(p, allow_pickle=True)
    print(f"  {p.name:<45s}  shape={arr.shape}  dtype={arr.dtype}")

# check README
readme = DATA_DIR / "README.md"
if readme.exists():
    print("\n=== data/README.md ===")
    print(readme.read_text()[:2000])