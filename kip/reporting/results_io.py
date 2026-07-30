"""Run directory management and schema-conformant results persistence.

Schema (§3 of BUILD_PLAN):
    <base>/<run>/
        config.yaml
        metrics.json
        hardware.json
        [manifest.csv]          (optional)
        curves/
        predictions/
        figures/
    <base>/summary.csv          (append-only, one row per run)

Hard safety constraint: output base must be under
    results/component_benchmark  OR  results/defect_detection
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_BASE_DIRS = {"component_benchmark", "defect_detection"}

METRICS_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _assert_safe_base(base: Path) -> None:
    """Raise ValueError if base is not under an allowed output directory."""
    parts = set(base.parts)
    if not (parts & ALLOWED_BASE_DIRS):
        # Also accept if one of the allowed names is a direct parent component
        resolved = base.resolve()
        ok = any(
            allowed in resolved.parts
            for allowed in ALLOWED_BASE_DIRS
        )
        if not ok:
            raise ValueError(
                f"Output base '{base}' is not under an allowed directory "
                f"({ALLOWED_BASE_DIRS}). "
                f"Never write to results/results/ or results/stage2_bgad/."
            )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Run dir
# ---------------------------------------------------------------------------

def create_run_dir(base: str | Path, run_name: str) -> Path:
    """Create and return <base>/<run_name>_<YYYYmmdd_HHMMSS>/.

    Also creates sub-dirs: curves/, predictions/, figures/.
    """
    base = Path(base)
    _assert_safe_base(base)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("curves", "predictions", "figures"):
        (run_dir / sub).mkdir(exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Save run artifacts
# ---------------------------------------------------------------------------

def save_run(
    run_dir: Path,
    config,
    metrics: dict,
    manifest_path: Optional[Path] = None,
    curves: Optional[dict] = None,
    hardware: Optional[dict] = None,
) -> None:
    """Write all schema-conformant run artifacts to run_dir.

    Parameters
    ----------
    run_dir:    Directory returned by create_run_dir.
    config:     Config dataclass or dict — serialised to config.yaml.
    metrics:    metrics.json payload (must include 'schema_version', 'run_id', 'stage').
    manifest_path: If provided, copy a reference (not the file) to manifest.csv link.
    curves:     Optional dict of curve data -> written to curves/*.json.
    hardware:   Dict from hardware_info() -> written to hardware.json.
    """
    run_dir = Path(run_dir)

    # --- config.yaml ---
    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict
        cfg_dict = asdict(config)
    elif isinstance(config, dict):
        cfg_dict = config
    else:
        cfg_dict = {"config": str(config)}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)

    # --- metrics.json ---
    if "schema_version" not in metrics:
        metrics = {"schema_version": METRICS_SCHEMA_VERSION, **metrics}
    if "git_commit" not in metrics:
        metrics["git_commit"] = _git_commit()
    if "timestamp_utc" not in metrics:
        metrics["timestamp_utc"] = datetime.now(tz=timezone.utc).isoformat()
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- hardware.json ---
    if hardware is not None:
        (run_dir / "hardware.json").write_text(json.dumps(hardware, indent=2))

    # --- manifest reference ---
    if manifest_path is not None:
        ref_path = run_dir / "manifest.csv"
        ref_path.write_text(f"# manifest path: {manifest_path}\n")

    # --- curves ---
    if curves:
        curves_dir = run_dir / "curves"
        curves_dir.mkdir(exist_ok=True)
        for name, data in curves.items():
            (curves_dir / f"{name}.json").write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

_SUMMARY_STABLE_COLS = [
    "run_id", "stage", "model", "augmentation", "seed",
    "smoke", "split_scheme", "device",
]


def append_summary(base: str | Path, flat_row: dict) -> None:
    """Append one flat row to <base>/summary.csv (stable column order).

    Metric columns are named `metric.<key>` and are appended after the
    stable base columns. New metric columns are added to the header when
    first seen.
    """
    base = Path(base)
    _assert_safe_base(base)
    base.mkdir(parents=True, exist_ok=True)
    summary_path = base / "summary.csv"

    # Separate stable cols from metric cols
    row_base = {k: flat_row.get(k, "") for k in _SUMMARY_STABLE_COLS}
    row_metrics = {
        k: v for k, v in flat_row.items() if k not in _SUMMARY_STABLE_COLS
    }

    # Prefix metric keys
    prefixed_metrics = {
        (k if k.startswith("metric.") else f"metric.{k}"): v
        for k, v in row_metrics.items()
    }

    full_row = {**row_base, **prefixed_metrics}

    if summary_path.exists():
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_cols = list(reader.fieldnames or [])
    else:
        existing_cols = []

    # Merge columns
    new_cols = [c for c in full_row if c not in existing_cols]
    all_cols = existing_cols + new_cols if existing_cols else list(full_row.keys())

    # Read existing rows
    existing_rows: list[dict] = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    all_cols_set = set(all_cols)
    all_cols_ordered = all_cols + [c for c in full_row if c not in all_cols_set]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols_ordered, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k, "") for k in all_cols_ordered})
        writer.writerow({k: full_row.get(k, "") for k in all_cols_ordered})
