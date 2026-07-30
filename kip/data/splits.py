"""Tool-based data splits with hard leakage guards.

Splits operate on the manifest (full images) strictly BEFORE tiling;
tiles inherit their fold via image_id (BUILD_PLAN 2.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Fold:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    held_out_tools: list[str] = field(default_factory=list)


def leave_one_tool_out(manifest: pd.DataFrame) -> list[Fold]:
    """One fold per tool: that tool's images are the test set."""
    folds: list[Fold] = []
    idx = np.arange(len(manifest))
    for tool in sorted(manifest["tool_id"].unique()):
        is_test = (manifest["tool_id"] == tool).to_numpy()
        folds.append(Fold(
            name=f"loto_{tool}",
            train_idx=idx[~is_test],
            test_idx=idx[is_test],
            held_out_tools=[tool],
        ))
    return folds


def group_kfold(manifest: pd.DataFrame, n_splits: int, seed: int = 42) -> list[Fold]:
    """K folds with tool-disjoint train/test (grouped by tool_id, shuffled by seed)."""
    tools = sorted(manifest["tool_id"].unique())
    if n_splits > len(tools):
        raise ValueError(
            f"n_splits={n_splits} exceeds number of tools ({len(tools)})"
        )
    rng = np.random.RandomState(seed)
    shuffled = list(tools)
    rng.shuffle(shuffled)
    groups = np.array_split(np.array(shuffled, dtype=object), n_splits)

    folds: list[Fold] = []
    idx = np.arange(len(manifest))
    for k, group in enumerate(groups):
        held = sorted(str(t) for t in group)
        is_test = manifest["tool_id"].isin(held).to_numpy()
        folds.append(Fold(
            name=f"gkf_{k}",
            train_idx=idx[~is_test],
            test_idx=idx[is_test],
            held_out_tools=held,
        ))
    return folds


def fixed_split(manifest: pd.DataFrame) -> Fold:
    """Use the delivered train/val directory split (secondary protocol).

    NOTE: BGAD's fixed split has the SAME tools in train and val by design,
    so ``assert_no_tool_leakage`` is expected to raise for this fold.
    """
    idx = np.arange(len(manifest))
    is_train = (manifest["split"] == "train").to_numpy()
    is_val = (manifest["split"] == "val").to_numpy()
    return Fold(
        name="fixed",
        train_idx=idx[is_train],
        test_idx=idx[is_val],
        held_out_tools=[],
    )


def assert_no_tool_leakage(fold: Fold, manifest: pd.DataFrame) -> None:
    """Raise ValueError if any tool_id appears in both train and test."""
    train_tools = set(manifest.iloc[fold.train_idx]["tool_id"])
    test_tools = set(manifest.iloc[fold.test_idx]["tool_id"])
    overlap = train_tools & test_tools
    if overlap:
        raise ValueError(
            f"Tool leakage detected in fold '{fold.name}': "
            f"tools {sorted(overlap)} appear in both train and test."
        )
