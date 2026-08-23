"""Build leakage-safe image/mask manifests for the registered angular mineral dataset.

The archive contains multiple angular images per original thin section and one
semantic mask per source. This script groups by mask stem/source ID and assigns
all angular views from a source to exactly one split. It never splits individual
frames from the same source across train/validation/test.

Run from geocore_ws:
  & $Python scripts\\prepare_mineral_manifest.py `
    --dataset-root '..\\registered_angular_mineral_segmentation' `
    --output data\\mineral_manifests
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".tif", ".tiff"}


def source_id_from_image(path: Path, mask_stems: set[str]) -> str | None:
    """Find the longest mask stem that is a prefix of the angular image stem."""
    stem = path.stem
    candidates = [candidate for candidate in mask_stems if stem == candidate or stem.startswith(candidate + "_")]
    return max(candidates, key=len) if candidates else None


def stable_bucket(source_id: str) -> float:
    value = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return int(value, 16) / float(16**12 - 1)


def split_for_source(source_id: str, train_fraction: float, val_fraction: float) -> str:
    bucket = stable_bucket(source_id)
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + val_fraction:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default="data/mineral_manifests")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.train_fraction <= 0 or args.val_fraction <= 0 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must be positive and leave a non-empty test split")

    root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    mask_files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS and path.parent.name.lower() == "mask")
    image_files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.parent.name.lower() == "image")
    mask_by_source = {}
    for path in mask_files:
        # Validation masks are named source_1.png although source_1.jpg through
        # source_8.jpg are the eight registered views of that same thin section.
        # Training masks use source.png. Normalize only a trailing underscore-one.
        source = re.sub(r"_1$", "", path.stem)
        if source in mask_by_source:
            raise RuntimeError(f"Duplicate normalized mask source: {source}")
        mask_by_source[source] = path
    if not mask_by_source:
        raise RuntimeError("No files found under a mask directory")
    if not image_files:
        raise RuntimeError("No files found under an image directory")

    rows: list[dict[str, str | int]] = []
    unmatched: list[str] = []
    for image in image_files:
        source = source_id_from_image(image, set(mask_by_source))
        if source is None:
            unmatched.append(str(image))
            continue
        mask = mask_by_source[source]
        split = split_for_source(source, args.train_fraction, args.val_fraction)
        rows.append({"image_path": str(image), "mask_path": str(mask), "source_id": source, "split": split, "image_name": image.name, "mask_name": mask.name})

    by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_source[str(row["source_id"])].add(str(row["split"]))
    leakage_sources = {source: sorted(splits) for source, splits in by_source.items() if len(splits) != 1}
    if leakage_sources:
        raise RuntimeError(f"Source leakage detected: {leakage_sources}")

    out.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "mask_path", "source_id", "split", "image_name", "mask_name"]
    manifest_path = out / "mineral_segmentation_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    source_rows = [{"source_id": source, "split": next(iter(splits)), "frame_count": sum(1 for row in rows if row["source_id"] == source), "mask_path": str(mask_by_source[source])} for source, splits in sorted(by_source.items())]
    with (out / "source_split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "split", "frame_count", "mask_path"])
        writer.writeheader(); writer.writerows(source_rows)
    summary = {"dataset_root": str(root), "images": len(image_files), "matched_images": len(rows), "unmatched_images": len(unmatched), "masks": len(mask_files), "sources": len(by_source), "sources_by_split": Counter(row["split"] for row in source_rows), "frames_by_split": Counter(row["split"] for row in rows), "leakage_sources": leakage_sources, "manifest": str(manifest_path), "unmatched": unmatched[:50], "split_rule": "stable SHA-256 source-level split; all angular frames from one source remain in one split"}
    (out / "mineral_manifest_summary.json").write_text(json.dumps(summary, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=dict))


if __name__ == "__main__":
    main()
