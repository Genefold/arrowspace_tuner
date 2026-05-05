#!/usr/bin/env python3
"""
Stage C — build CWE buckets filtered by size.
Output: data/cwe_buckets.json  {cwe: [cve_id, ...]}

Run from repo root:
    python scripts/build_ground_truth/stage_c_build_buckets.py
"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict, Counter

FILTERED_MAP_PATH = Path("data/cve_cwe_map_filtered.csv")
OUT_PATH          = Path("data/cwe_buckets.json")

MIN_BUCKET_SIZE = 5
MAX_BUCKET_SIZE = 200


def main():
    print(f"Loading {FILTERED_MAP_PATH} ...")
    buckets: dict[str, list[str]] = defaultdict(list)

    with open(FILTERED_MAP_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            buckets[row["primary_cwe"]].append(row["cve_id"])

    print(f"  Total CWEs before filtering: {len(buckets):,}")
    print(f"  Total CVEs before filtering: {sum(len(v) for v in buckets.values()):,}")

    # --- filter ---
    filtered = {
        cwe: ids
        for cwe, ids in buckets.items()
        if MIN_BUCKET_SIZE <= len(ids) <= MAX_BUCKET_SIZE
    }

    too_small = sum(1 for ids in buckets.values() if len(ids) < MIN_BUCKET_SIZE)
    too_large = sum(1 for ids in buckets.values() if len(ids) > MAX_BUCKET_SIZE)

    print(f"  Dropped (< {MIN_BUCKET_SIZE}): {too_small:,}")
    print(f"  Dropped (> {MAX_BUCKET_SIZE}): {too_large:,}")
    print(f"  Kept CWEs: {len(filtered):,}")
    print(f"  Total CVEs in kept buckets: {sum(len(v) for v in filtered.values()):,}")

    # --- size distribution ---
    sizes = sorted(len(v) for v in filtered.values())
    print(f"\nBucket size stats:")
    print(f"  min={sizes[0]}  median={sizes[len(sizes)//2]}  max={sizes[-1]}")

    top_cwes = sorted(filtered.items(), key=lambda kv: len(kv[1]), reverse=True)[:15]
    print("\nTop 15 CWEs by bucket size:")
    for cwe, ids in top_cwes:
        print(f"  {cwe:15s}  {len(ids):4d} CVEs")

    # --- save ---
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
