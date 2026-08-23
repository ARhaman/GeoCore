"""Normalize registered angular mineral masks to explicit class-index PNGs.

The source PNG files are mode P (palette) and their raw indices are already
0..8. This script preserves the raw indices, writes mode-L masks, and adds the
explicit class map used by the trainer. It never uses grayscale luminance.

Run from geocore_ws:
  & $Python scripts\\normalize_mineral_masks.py `
    --manifest data\\mineral_manifests\\mineral_segmentation_manifest.csv `
    --output data\\mineral_normalized
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_MAP = {
    0: "background",
    1: "olivine",
    2: "pyroxene",
    3: "plagioclase",
    4: "alkali_feldspar",
    5: "quartz",
    6: "biotite",
    7: "muscovite",
    8: "hornblende",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/mineral_normalized")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    unique_masks = {}
    for row in rows:
        unique_masks[row["mask_path"]] = row
    output.mkdir(parents=True, exist_ok=True)
    normalized = {}
    values_seen = set()
    for raw_path, row in unique_masks.items():
        raw_path = Path(raw_path)
        with Image.open(raw_path) as image:
            array = np.asarray(image)
            if image.mode == "P":
                array = np.asarray(image, dtype=np.uint8)
            elif array.ndim == 3:
                raise ValueError(f"Expected indexed or single-channel mask, got {image.mode}: {raw_path}")
            array = array.astype(np.uint8)
        values_seen.update(int(value) for value in np.unique(array))
        invalid = sorted(values_seen.difference(CLASS_MAP))
        if invalid:
            raise ValueError(f"Unexpected raw class values {invalid} in {raw_path}")
        out_path = output / (raw_path.stem + ".png")
        Image.fromarray(array, mode="L").save(out_path)
        normalized[str(raw_path)] = str(out_path)
    out_rows = []
    for row in rows:
        updated = dict(row)
        updated["mask_path"] = normalized[row["mask_path"]]
        updated["mask_encoding"] = "explicit_class_index_0_background_1_to_8_minerals"
        out_rows.append(updated)
    out_manifest = output / "mineral_segmentation_manifest_normalized.csv"
    fields = list(out_rows[0]) if out_rows else []
    with out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(out_rows)
    metadata = {"class_map": CLASS_MAP, "raw_values_seen": sorted(values_seen), "unique_masks": len(unique_masks), "normalized_manifest": str(out_manifest), "encoding": "PNG mode L with exact source palette indices; no grayscale luminance conversion"}
    (output / "class_map.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
