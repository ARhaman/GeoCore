"""Audit registered angular mineral image/mask pairs before GPU training.

Run from geocore_ws:
  & $Python scripts\\audit_mineral_masks.py `
    --manifest data\\mineral_manifests\\mineral_segmentation_manifest.csv `
    --output data\\mineral_manifests\\mask_audit
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/mineral_manifests/mask_audit")
    args = parser.parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    unique_pairs = {}
    for row in rows:
        unique_pairs[row["mask_path"]] = row
    class_pixels = Counter()
    dimensions = Counter()
    mask_dimensions = Counter()
    mismatches = []
    unreadable = []
    sources = defaultdict(set)
    for row in rows:
        image_path, mask_path = Path(row["image_path"]), Path(row["mask_path"])
        sources[row["source_id"]].add(row["split"])
        try:
            with Image.open(image_path) as image:
                image_size = image.size
                image_mode = image.mode
            with Image.open(mask_path) as mask:
                mask_size = mask.size
                mask_mode = mask.mode
                values = Counter(mask.convert("L").getdata())
        except Exception as exc:
            unreadable.append({"image": str(image_path), "mask": str(mask_path), "error": repr(exc)})
            continue
        dimensions[str(image_size)] += 1
        mask_dimensions[str(mask_size)] += 1
        class_pixels.update(values)
        if image_size != mask_size:
            mismatches.append({"image": str(image_path), "mask": str(mask_path), "image_size": image_size, "mask_size": mask_size, "image_mode": image_mode, "mask_mode": mask_mode})
    leakage = {source: sorted(splits) for source, splits in sources.items() if len(splits) != 1}
    summary = {"manifest_rows": len(rows), "unique_masks": len(unique_pairs), "image_dimensions": dimensions, "mask_dimensions": mask_dimensions, "mask_pixel_values": class_pixels, "dimension_mismatches": len(mismatches), "unreadable": len(unreadable), "source_leakage": leakage, "scientific_checks": {"all_images_have_matching_mask_shape": not mismatches, "all_masks_readable": not unreadable, "no_source_leakage": not leakage}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "mask_audit_summary.json").write_text(json.dumps(summary, indent=2, default=dict), encoding="utf-8")
    (output / "dimension_mismatches.json").write_text(json.dumps(mismatches, indent=2), encoding="utf-8")
    (output / "unreadable.json").write_text(json.dumps(unreadable, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=dict))
    if mismatches or unreadable or leakage:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
