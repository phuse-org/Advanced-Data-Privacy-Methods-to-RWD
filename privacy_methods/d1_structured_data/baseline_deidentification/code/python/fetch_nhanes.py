"""Download NHANES data from CDC and merge into a single CSV.

Downloads demographics, examination, laboratory, and questionnaire XPT files
for a given NHANES cycle, merges them on SEQN, and writes one CSV.

Usage (CLI):
    python fetch_nhanes.py
    python fetch_nhanes.py --cycle 2017-2018
    python fetch_nhanes.py --tables DEMO BMX GLU

Usage (Jupyter):
    from fetch_nhanes import fetch_nhanes
    df = fetch_nhanes()

Data source: https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
"""

from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# NHANES cycle definitions
# ---------------------------------------------------------------------------

_BASE_2021 = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
_BASE_2017 = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

CYCLES: dict[str, dict] = {
    "2021-2023": {
        "description": "August 2021–August 2023",
        "url": _BASE_2021,
        "tables": {
            "DEMO":   {"file": "DEMO_L.XPT",   "desc": "Demographics"},
            "BMX":    {"file": "BMX_L.XPT",    "desc": "Body measures"},
            "BPXO":   {"file": "BPXO_L.XPT",   "desc": "Blood pressure"},
            "TCHOL":  {"file": "TCHOL_L.XPT",  "desc": "Total cholesterol"},
            "HDL":    {"file": "HDL_L.XPT",    "desc": "HDL cholesterol"},
            "GLU":    {"file": "GLU_L.XPT",    "desc": "Fasting glucose"},
            "GHB":    {"file": "GHB_L.XPT",    "desc": "HbA1c"},
            "TRIGLY": {"file": "TRIGLY_L.XPT", "desc": "LDL & triglycerides"},
            "SMQ":    {"file": "SMQ_L.XPT",    "desc": "Smoking"},
            "DIQ":    {"file": "DIQ_L.XPT",    "desc": "Diabetes"},
            "BPQ":    {"file": "BPQ_L.XPT",    "desc": "BP & cholesterol Qs"},
            "MCQ":    {"file": "MCQ_L.XPT",    "desc": "Medical conditions"},
        },
    },
    "2017-2020": {
        "description": "2017–March 2020 pre-pandemic",
        "url": _BASE_2017,
        "tables": {
            "DEMO":  {"file": "P_DEMO.XPT",  "desc": "Demographics"},
            "BMX":   {"file": "P_BMX.XPT",   "desc": "Body measures"},
            "BPXO":  {"file": "P_BPXO.XPT",  "desc": "Blood pressure"},
            "TCHOL": {"file": "P_TCHOL.XPT", "desc": "Total cholesterol"},
            "HDL":   {"file": "P_HDL.XPT",   "desc": "HDL cholesterol"},
            "GLU":   {"file": "P_GLU.XPT",   "desc": "Fasting glucose"},
            "GHB":   {"file": "P_GHB.XPT",   "desc": "HbA1c"},
            "SMQ":   {"file": "P_SMQ.XPT",   "desc": "Smoking"},
            "DIQ":   {"file": "P_DIQ.XPT",   "desc": "Diabetes"},
            "BPQ":   {"file": "P_BPQ.XPT",   "desc": "BP & cholesterol Qs"},
            "MCQ":   {"file": "P_MCQ.XPT",   "desc": "Medical conditions"},
        },
    },
    "2017-2018": {
        "description": "2017-2018",
        "url": _BASE_2017,
        "tables": {
            "DEMO":  {"file": "DEMO_J.XPT",  "desc": "Demographics"},
            "BMX":   {"file": "BMX_J.XPT",   "desc": "Body measures"},
            "BPX":   {"file": "BPX_J.XPT",   "desc": "Blood pressure"},
            "TCHOL": {"file": "TCHOL_J.XPT", "desc": "Total cholesterol"},
            "HDL":   {"file": "HDL_J.XPT",   "desc": "HDL cholesterol"},
            "GLU":   {"file": "GLU_J.XPT",   "desc": "Fasting glucose"},
            "GHB":   {"file": "GHB_J.XPT",   "desc": "HbA1c"},
            "SMQ":   {"file": "SMQ_J.XPT",   "desc": "Smoking"},
            "DIQ":   {"file": "DIQ_J.XPT",   "desc": "Diabetes"},
            "BPQ":   {"file": "BPQ_J.XPT",   "desc": "BP & cholesterol Qs"},
            "MCQ":   {"file": "MCQ_J.XPT",   "desc": "Medical conditions"},
        },
    },
}


def _download_xpt(url: str, dest: Path) -> bool:
    """Download a single XPT file. Return True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NHANES-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
        print(f"    OK ({dest.stat().st_size / 1024:.0f} KB)")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"    FAILED — {e}")
        return False


def fetch_nhanes(
    cycle: str = "2021-2023",
    tables: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download NHANES XPT files and return a merged DataFrame.

    Args:
        cycle: "2021-2023", "2017-2020", or "2017-2018".
        tables: Subset of table names (e.g. ["DEMO", "BMX"]). None = all.
        output_dir: Where to save the merged CSV. Defaults to ./output.
    """
    if cycle not in CYCLES:
        raise ValueError(f"Unknown cycle '{cycle}'. Choose from: {list(CYCLES.keys())}")

    cycle_def = CYCLES[cycle]
    base_url = cycle_def["url"]
    all_tables = cycle_def["tables"]

    default_dir = Path(os.environ.get("PHUSE_DATA_DIR", Path.home() / "Documents" / "Data"))
    out_path = Path(output_dir) if output_dir else default_dir
    cache_dir = out_path / "raw" / cycle
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Select tables (always include DEMO)
    if tables:
        selected = {t.upper(): all_tables[t.upper()] for t in tables if t.upper() in all_tables}
        if "DEMO" not in selected:
            selected = {"DEMO": all_tables["DEMO"], **selected}
    else:
        selected = all_tables

    print(f"NHANES {cycle} — downloading {len(selected)} tables\n")

    dfs: dict[str, pd.DataFrame] = {}
    for name, info in selected.items():
        filename = info["file"]
        xpt_path = cache_dir / filename
        print(f"  [{name}] {info['desc']} ({filename})")

        if xpt_path.exists() and xpt_path.stat().st_size > 0:
            print(f"    cached ({xpt_path.stat().st_size / 1024:.0f} KB)")
        else:
            if not _download_xpt(f"{base_url}/{filename}", xpt_path):
                continue

        try:
            dfs[name] = pd.read_sas(xpt_path, format="xport")
            print(f"    {len(dfs[name]):,} rows, {len(dfs[name].columns)} cols")
        except Exception as e:
            print(f"    READ ERROR — {e}")

    if "DEMO" not in dfs:
        raise RuntimeError("DEMO table is required but could not be loaded.")

    # Merge on SEQN
    merged = dfs.pop("DEMO")
    for name, df in dfs.items():
        overlap = set(merged.columns) & set(df.columns) - {"SEQN"}
        if overlap:
            df = df.drop(columns=list(overlap))
        merged = merged.merge(df, on="SEQN", how="left")

    out_csv = out_path / f"nhanes_{cycle.replace('-', '_')}.csv"
    merged.to_csv(out_csv, index=False)

    print(f"\nSaved {len(merged):,} rows x {len(merged.columns)} cols -> {out_csv}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NHANES data from CDC")
    parser.add_argument("--cycle", default="2021-2023", choices=list(CYCLES.keys()))
    parser.add_argument("--tables", nargs="*", default=None, help="Tables to download (default: all)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: ./output)")
    # parse_known_args ignores unrecognized args (e.g. Jupyter's --f=kernel-xxx.json)
    args, _ = parser.parse_known_args()

    fetch_nhanes(cycle=args.cycle, tables=args.tables, output_dir=args.output_dir)
