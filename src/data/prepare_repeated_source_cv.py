"""Prepare repeated grouped source-level cross-validation manifests for B7.

Each source thin section is assigned wholly to train, validation, or test in every
outer fold. All frames (and therefore all registered views) of a source remain in
the same partition. The script creates deterministic folds using stable SHA-256
ordering, writes one per-image manifest per outer fold, and records a full run plan.

The default is 2 repeats x 5 outer folds = 10 held-out fold estimates. Test folds
within a repeat are disjoint; repeats use different deterministic source orderings.
Validation sources are selected only from outer-training sources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def stable_order(source_ids, token):
    return sorted(source_ids, key=lambda s: hashlib.sha256(f"{token}|{s}".encode()).hexdigest())


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Normalized per-image manifest with source_id and image/mask paths")
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.folds < 2 or args.repeats < 1 or not (0 < args.validation_fraction < 0.5):
        raise ValueError("Use folds >= 2, repeats >= 1, and 0 < validation_fraction < 0.5.")

    with Path(args.manifest).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "source_id" not in rows[0]:
        raise ValueError("Manifest must be non-empty and include source_id.")
    fields = list(rows[0])
    source_to_rows = {}
    for row in rows: source_to_rows.setdefault(row["source_id"], []).append(row)
    sources = sorted(source_to_rows)
    expected_frames = {len(v) for v in source_to_rows.values()}
    if expected_frames != {8}:
        raise ValueError(f"Expected exactly eight frames per source; found counts {sorted(expected_frames)}")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plan = []
    all_assignments = []
    for repeat in range(args.repeats):
        ordered = stable_order(sources, f"outer-{args.seed}-{repeat}")
        fold_to_test = {fold: set(ordered[fold::args.folds]) for fold in range(args.folds)}
        for fold, test_sources in fold_to_test.items():
            outer_train_sources = set(sources) - test_sources
            val_count = max(1, math.ceil(len(outer_train_sources) * args.validation_fraction))
            val_order = stable_order(outer_train_sources, f"validation-{args.seed}-{repeat}-{fold}")
            val_sources = set(val_order[:val_count])
            train_sources = outer_train_sources - val_sources
            if train_sources & val_sources or train_sources & test_sources or val_sources & test_sources:
                raise RuntimeError("Source leakage generated during fold construction")
            fold_rows = []
            for source in sources:
                split = "test" if source in test_sources else "val" if source in val_sources else "train"
                for original in source_to_rows[source]:
                    record = dict(original); record["split"] = split; record["cv_repeat"] = str(repeat); record["cv_fold"] = str(fold)
                    fold_rows.append(record)
                    all_assignments.append({"repeat": repeat, "fold": fold, "source_id": source, "split": split})
            fold_dir = out / f"repeat_{repeat + 1:02d}" / f"fold_{fold + 1:02d}"
            manifest_path = fold_dir / "manifest.csv"
            write_csv(manifest_path, fold_rows, fields + ["cv_repeat", "cv_fold"])
            counts = Counter(r["split"] for r in fold_rows)
            source_counts = {"train": len(train_sources), "val": len(val_sources), "test": len(test_sources)}
            plan.append({
                "repeat": repeat + 1, "fold": fold + 1, "manifest": str(manifest_path),
                "output_dir": str(out / "results" / f"repeat_{repeat + 1:02d}" / f"fold_{fold + 1:02d}"),
                "train_sources": source_counts["train"], "val_sources": source_counts["val"], "test_sources": source_counts["test"],
                "train_frames": counts["train"], "val_frames": counts["val"], "test_frames": counts["test"],
            })
    write_csv(out / "cv_run_plan.csv", plan, list(plan[0]))
    write_csv(out / "source_assignments.csv", all_assignments, ["repeat", "fold", "source_id", "split"])
    summary = {
        "input_frames": len(rows), "sources": len(sources), "frames_per_source": 8,
        "folds": args.folds, "repeats": args.repeats, "total_runs": len(plan),
        "validation_fraction_of_outer_training_sources": args.validation_fraction,
        "run_plan": str(out / "cv_run_plan.csv"), "safeguard": "all frames of each source remain in exactly one split in every fold",
    }
    (out / "cv_design_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
