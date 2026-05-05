"""
Stage A — scan cvelistV5 JSON files and extract primary CWE per CVE.
Output: data/cve_cwe_map.csv  (cve_id, primary_cwe)

Run from repo root:
    uv run python scripts/build_ground_truth/stage_a_extract_cwe.py
"""
import csv, json, re, sys
from pathlib import Path
from collections import Counter

CVELIST_ROOT = Path("data/cvelistV5/cves")
OUT_PATH     = Path("data/cve_cwe_map.csv")

PLACEHOLDERS = {"NVD-CWE-noinfo", "NVD-CWE-Other"}
CWE_RE       = re.compile(r"\bCWE-\d+\b")


def extract_primary_cwe(record: dict) -> str | None:
    """
    Returns the first usable CWE-<number> from containers.cna.problemTypes.
    Priority:
      1. type=="CWE" AND cweId field present  (modern CVE JSON 5.x)
      2. regex fallback on description text    (older enriched records)
    Returns None if nothing usable found.
    """
    problem_types = (
        record.get("containers", {})
              .get("cna", {})
              .get("problemTypes", [])
    )
    for pt in problem_types:
        for desc in pt.get("descriptions", []):
            # path 1: structured cweId field (preferred, modern records)
            if desc.get("type") == "CWE" and "cweId" in desc:
                cwe = desc["cweId"].strip()
                if cwe not in PLACEHOLDERS and re.match(r"^CWE-\d+$", cwe):
                    return cwe
            # path 2: regex on description text (fallback for older enriched records)
            text = desc.get("description", "")
            m = CWE_RE.search(text)
            if m:
                cwe = m.group(0)
                if cwe not in PLACEHOLDERS:
                    return cwe
    return None


def iter_records(jf: Path):
    """
    Yield individual CVE record dicts from a JSON file.
    Handles:
      - single dict  (one record per file, most common in cvelistV5)
      - list of dicts (bulk/delta files that pack multiple records)
    """
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def main():
    if not CVELIST_ROOT.exists():
        sys.exit(f"ERROR: {CVELIST_ROOT} not found. Check path.")

    json_files = sorted(CVELIST_ROOT.rglob("*.json"))
    print(f"Scanning {len(json_files):,} JSON files in {CVELIST_ROOT} ...")

    results    = {}
    unlabeled  = 0
    total_recs = 0

    for jf in json_files:
        for rec in iter_records(jf):
            total_recs += 1
            cve_id = rec.get("cveMetadata", {}).get("cveId")
            if not cve_id:
                continue

            cwe = extract_primary_cwe(rec)
            if cwe:
                results[cve_id] = cwe
            else:
                unlabeled += 1

    print(f"  Total records parsed : {total_recs:,}")
    print(f"  Labeled (have CWE)   : {len(results):,}")
    print(f"  Unlabeled (no CWE)   : {unlabeled:,}")
    label_rate = len(results) / total_recs * 100 if total_recs else 0
    print(f"  Label rate           : {label_rate:.1f}%")

    top_cwes = Counter(results.values()).most_common(15)
    print("\nTop 15 CWEs:")
    for cwe, cnt in top_cwes:
        print(f"  {cwe:15s}  {cnt:6,}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cve_id", "primary_cwe"])
        for cve_id, cwe in sorted(results.items()):
            w.writerow([cve_id, cwe])

    print(f"\nSaved {OUT_PATH}  ({len(results):,} rows)")


if __name__ == "__main__":
    main()