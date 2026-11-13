"""Build stratified, group-exclusive train/validation folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.manifest import build_manifest


def assign_folds(manifest: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """Assign every row exactly one stratified fold without splitting groups."""
    if n_splits < 2:
        raise ValueError("The number of folds must be at least 2.")
    if manifest.empty:
        raise ValueError("Cannot create folds from an empty manifest.")
    if manifest["group_id"].isna().any():
        raise ValueError("Group IDs cannot be empty.")

    labels = manifest["label"].astype(int)
    groups = manifest["group_id"].astype(str)
    group_counts = manifest.assign(label=labels, group_id=groups).groupby("label")[
        "group_id"
    ].nunique()
    if set(group_counts.index) != {0, 1}:
        raise ValueError("The dataset must contain both real (0) and fake (1) images.")
    insufficient = group_counts[group_counts < n_splits]
    if not insufficient.empty:
        details = ", ".join(
            f"{'real' if label == 0 else 'fake'}: {count} groups"
            for label, count in insufficient.items()
        )
        raise ValueError(
            f"{n_splits} folds require at least {n_splits} distinct groups per "
            f"class; found {details}."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    result = manifest.copy()
    result["label"] = labels
    result["group_id"] = groups
    result["fold"] = -1

    for fold, (_, validation_indices) in enumerate(
        splitter.split(result, labels, groups)
    ):
        result.loc[result.index[validation_indices], "fold"] = fold

    if (result["fold"] < 0).any():
        raise RuntimeError("Internal error: one or more images were not assigned a fold.")

    group_fold_counts = result.groupby("group_id")["fold"].nunique()
    if (group_fold_counts != 1).any():
        raise RuntimeError("Internal error: a group was assigned to multiple folds.")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Raw image directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--metadata", type=Path, help="Optional input manifest CSV.")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default="group_id")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_manifest(
        args.data,
        metadata_path=args.metadata,
        path_column=args.path_column,
        label_column=args.label_column,
        group_column=args.group_column,
    )
    folds = assign_folds(manifest, args.folds, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.out, index=False)
    print(f"Wrote {len(folds)} images across {args.folds} folds to {args.out}")
    for fold, rows in folds.groupby("fold"):
        counts = rows["label"].value_counts().reindex([0, 1], fill_value=0)
        print(
            f"  fold {fold}: {len(rows)} images "
            f"({counts[0]} real, {counts[1]} fake; "
            f"{rows['group_id'].nunique()} groups)"
        )


if __name__ == "__main__":
    main()
