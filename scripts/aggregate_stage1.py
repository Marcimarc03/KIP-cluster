#!/usr/bin/env python
"""Aggregate Stage-1 runs into mean +/- std per (model, aug, tag, split-fingerprint).

Liest DIREKT aus den metrics.json jedes Laufs -- robust gegen eine kaputte/
gemergte summary.csv. Gruppen mit abweichendem Fingerprint (test_sha, n_test,
workers) werden als INKONSISTENT markiert und NICHT als eine Zahl aggregiert.

Usage:  python scripts/aggregate_stage1.py
"""
from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "results" / "component_benchmark"


def load_runs():
    runs = []
    for f in sorted(glob.glob(str(BASE / "*" / "metrics.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("stage") != 1 or d.get("smoke"):
            continue
        m = d.get("metrics", {}) or {}
        fp = (d.get("split") or {}).get("fingerprint", {}) or {}
        env = d.get("environment", {}) or {}
        runs.append({
            "model": d.get("model"),
            "aug": "on" if d.get("augmentation") else "off",
            "tag": d.get("tag") or "-",
            "seed": d.get("seed"),
            "sha": fp.get("test_json_sha256", "") or "",
            "n_test": (d.get("dataset") or {}).get("n_test"),
            "workers": env.get("kip_workers", ""),
            "segm50": m.get("segm_map50"),
            "segm5095": m.get("segm_map50_95"),
            "bbox50": m.get("bbox_map50"),
        })
    return runs


def _stat(rs, key):
    vals = [r[key] for r in rs if isinstance(r[key], (int, float))]
    if not vals:
        return "  --  "
    if len(vals) == 1:
        return f"{vals[0]:.4f} (n=1)"
    return f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}"


def main() -> None:
    runs = load_runs()
    if not runs:
        print(f"Keine Stage-1 metrics.json unter {BASE}")
        return

    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["aug"], r["tag"], (r["sha"] or "nofp")[:8])].append(r)

    print(f"{'model':12}{'aug':4}{'tag':13}{'n':>2}  "
          f"{'segm50 (mean+-std)':>20}{'segm50-95':>16}{'bbox50':>14}   Fingerprint")
    print("-" * 114)
    for key in sorted(groups):
        model, aug, tag, _ = key
        rs = groups[key]
        shas = sorted({r["sha"] for r in rs})
        nts = sorted({str(r["n_test"]) for r in rs})
        ws = sorted({str(r["workers"]) for r in rs})
        consistent = len(shas) <= 1 and len(nts) <= 1 and len(ws) <= 1
        flag = "" if consistent else "  <<< INKONSISTENT -> NICHT aggregieren"
        print(f"{model:12}{aug:4}{tag[:12]:13}{len(rs):>2}  "
              f"{_stat(rs, 'segm50'):>20}{_stat(rs, 'segm5095'):>16}{_stat(rs, 'bbox50'):>14}   "
              f"sha={shas} n={nts} w={ws}{flag}")
        pairs = ", ".join(
            f"{r['seed']}:{r['segm50']:.4f}"
            for r in sorted(rs, key=lambda x: str(x["seed"]))
            if isinstance(r["segm50"], (int, float))
        )
        print(f"    seeds/segm50: {pairs}")

    print("\nQuelle: metrics.json je Lauf. Modellabstand nur behaupten, wenn die")
    print("Seed-Baender disjunkt sind (bei uns Primaerbeleg statt Bootstrap).")


if __name__ == "__main__":
    main()
