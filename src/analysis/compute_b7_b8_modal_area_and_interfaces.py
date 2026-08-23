"""Compute source-level 2D mineral modal-area proxies and semantic phase-interface frequencies.

This script uses the validated B7–B8 probability ensemble on a single manifest partition
(default: original source-held test partition). It reports one aggregate observation per
thin-section source by averaging the eight registered-view measurements. It does not treat
the eight views as independent petrographic samples.

Scientific scope:
- Mineral area fractions are 2D modal-area proxies, not 3D modal mineralogy.
- Interface frequencies are pixel-edge semantic interface frequencies, not 3D contact areas
  or individual-grain contact networks.
- Background must never be interpreted as porosity.
- Physical units are not reported because the manifest does not contain verified pixel scale.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50

# Reuse the exact B7 architecture, normalization constants, names, and class count from
# the validated native-figure pipeline.
from generate_b7_native_900dpi_figures import PretrainedResUNet, MEAN, STD, NAMES, C

MINERAL_IDS = list(range(1, C))


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = [int(x) for x in value.lower().split("x")]
    except Exception as exc:
        raise argparse.ArgumentTypeError("--inference-size must be WIDTHxHEIGHT, e.g. 640x512") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Inference dimensions must be positive.")
    return width, height


def load_rows(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_id", "image_path", "mask_path", "split"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        available = sorted({row["split"] for row in rows})
        raise ValueError(f"No rows with split={split!r}. Available values: {available}")
    return selected


def build_deeplab() -> torch.nn.Module:
    return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=C, aux_loss=False)


def load_models(b7_checkpoint: Path, b8_checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    b7 = PretrainedResUNet().to(device)
    b8 = build_deeplab().to(device)
    b7_payload = torch.load(b7_checkpoint, map_location=device, weights_only=False)
    b8_payload = torch.load(b8_checkpoint, map_location=device, weights_only=False)
    b7.load_state_dict(b7_payload["model"])
    b8.load_state_dict(b8_payload["model"])
    b7.eval(); b8.eval()
    return b7, b8


def read_image_and_mask(row: dict[str, str], size: tuple[int, int]) -> tuple[torch.Tensor, np.ndarray]:
    width, height = size
    with Image.open(row["image_path"]) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    with Image.open(row["mask_path"]) as mask:
        label = np.asarray(mask.convert("L").resize((width, height), Image.Resampling.NEAREST), dtype=np.int64)
    if label.min() < 0 or label.max() >= C:
        raise ValueError(f"Unexpected label values for {row['source_id']}: {np.unique(label).tolist()}")
    tensor = torch.from_numpy(image_np)
    tensor = (tensor - MEAN.squeeze(0)) / STD.squeeze(0)
    return tensor, label


def predict(b7: torch.nn.Module, b8: torch.nn.Module, x: torch.Tensor, weight: float, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        batch = x.unsqueeze(0).to(device, non_blocking=True)
        p7 = b7(batch).softmax(1)
        p8 = b8(batch)["out"].softmax(1)
        result = (weight * p7 + (1.0 - weight) * p8).argmax(1)[0].cpu().numpy().astype(np.int64)
    return result


def fractions(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels.reshape(-1), minlength=C).astype(np.float64) / labels.size


def interface_counts(labels: np.ndarray, mineral_only: bool) -> tuple[np.ndarray, int]:
    """Return unordered pixel-edge interface counts in the upper triangle and total changed edges.

    Each horizontal and vertical four-neighbour edge is evaluated once. The returned matrix stores
    only [lower_class, higher_class], so counts are not double-counted.
    """
    matrix = np.zeros((C, C), dtype=np.int64)
    pairs = ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :]))
    for left, right in pairs:
        changed = left != right
        if mineral_only:
            changed &= (left > 0) & (right > 0)
        if not np.any(changed):
            continue
        lo = np.minimum(left[changed], right[changed])
        hi = np.maximum(left[changed], right[changed])
        packed = lo * C + hi
        counts = np.bincount(packed, minlength=C * C).reshape(C, C)
        matrix += counts
    return matrix, int(matrix.sum())


def source_summary(source_id: str, records: list[dict]) -> tuple[dict, list[dict]]:
    """Average per-view composition and interface frequencies into one source-level observation."""
    gt_stack = np.stack([record["gt_frac"] for record in records])
    pred_stack = np.stack([record["pred_frac"] for record in records])
    gt = gt_stack.mean(axis=0)
    pred = pred_stack.mean(axis=0)
    signed = pred - gt
    absolute = np.abs(signed)

    summary = {
        "source_id": source_id,
        "registered_views": len(records),
        "mineral_area_mae": float(absolute[MINERAL_IDS].mean()),
        "mineral_composition_l1": float(absolute[MINERAL_IDS].sum()),
        "gt_mineral_coverage": float(gt[MINERAL_IDS].sum()),
        "pred_mineral_coverage": float(pred[MINERAL_IDS].sum()),
    }
    for class_id, name in enumerate(NAMES):
        token = name.replace(" ", "_")
        summary[f"gt_area_{token}"] = float(gt[class_id])
        summary[f"pred_area_{token}"] = float(pred[class_id])
        summary[f"signed_error_{token}"] = float(signed[class_id])
        summary[f"absolute_error_{token}"] = float(absolute[class_id])

    interface_rows: list[dict] = []
    for scope, key in (("all_nonidentical_classes", "all_interface"), ("mineral_to_mineral_only", "mineral_interface")):
        for map_type, prefix in (("ground_truth", "gt"), ("ensemble_prediction", "pred")):
            per_view_counts = [record[f"{prefix}_{key}_counts"] for record in records]
            per_view_totals = np.asarray([record[f"{prefix}_{key}_total"] for record in records], dtype=np.float64)
            per_view_frequencies = []
            for counts, total in zip(per_view_counts, per_view_totals):
                per_view_frequencies.append(counts / total if total > 0 else np.zeros_like(counts, dtype=np.float64))
            mean_frequency = np.mean(np.stack(per_view_frequencies), axis=0)
            total_edges = int(sum(per_view_totals))
            total_counts = np.sum(np.stack(per_view_counts), axis=0)
            for a in range(C):
                for b in range(a + 1, C):
                    interface_rows.append({
                        "source_id": source_id,
                        "registered_views": len(records),
                        "map_type": map_type,
                        "scope": scope,
                        "class_a": NAMES[a],
                        "class_b": NAMES[b],
                        "edge_count_across_views": int(total_counts[a, b]),
                        "mean_view_normalized_frequency": float(mean_frequency[a, b]),
                        "total_candidate_interface_edges": total_edges,
                    })
    return summary, interface_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def numeric_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def aggregate_modal(source_rows: list[dict]) -> list[dict]:
    result = []
    for class_id, name in enumerate(NAMES):
        token = name.replace(" ", "_")
        gt = np.asarray([row[f"gt_area_{token}"] for row in source_rows], dtype=float)
        pred = np.asarray([row[f"pred_area_{token}"] for row in source_rows], dtype=float)
        signed = np.asarray([row[f"signed_error_{token}"] for row in source_rows], dtype=float)
        absolute = np.asarray([row[f"absolute_error_{token}"] for row in source_rows], dtype=float)
        result.append({
            "class": name,
            "source_count": len(source_rows),
            "mean_gt_area_fraction": numeric_stats(gt)["mean"],
            "mean_pred_area_fraction": numeric_stats(pred)["mean"],
            "mean_signed_area_error": numeric_stats(signed)["mean"],
            "median_signed_area_error": numeric_stats(signed)["median"],
            "mean_absolute_area_error": numeric_stats(absolute)["mean"],
            "median_absolute_area_error": numeric_stats(absolute)["median"],
            "absolute_error_sd": numeric_stats(absolute)["sd"],
        })
    return result


def aggregate_interfaces(interface_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in interface_rows:
        grouped[(row["map_type"], row["scope"], row["class_a"], row["class_b"])].append(row)
    result = []
    for (map_type, scope, a, b), rows in sorted(grouped.items()):
        frequencies = np.asarray([row["mean_view_normalized_frequency"] for row in rows], dtype=float)
        counts = np.asarray([row["edge_count_across_views"] for row in rows], dtype=float)
        result.append({
            "map_type": map_type,
            "scope": scope,
            "class_a": a,
            "class_b": b,
            "source_count": len(rows),
            "mean_source_normalized_frequency": numeric_stats(frequencies)["mean"],
            "sd_source_normalized_frequency": numeric_stats(frequencies)["sd"],
            "median_source_normalized_frequency": numeric_stats(frequencies)["median"],
            "total_edge_count_across_sources_and_views": int(counts.sum()),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Normalized source-level mineral manifest CSV.")
    parser.add_argument("--b7-checkpoint", required=True, help="Validated B7 checkpoint.")
    parser.add_argument("--b8-checkpoint", required=True, help="Validated B8 checkpoint.")
    parser.add_argument("--output", required=True, help="Output directory for source-level CSV and JSON files.")
    parser.add_argument("--split", default="test", help="Manifest partition to analyse; default is the untouched original test split.")
    parser.add_argument("--b7-weight", type=float, default=0.6, help="Fixed B7 probability weight; default is original validation-calibrated 0.6.")
    parser.add_argument("--inference-size", type=parse_size, default=(640, 512), help="WIDTHxHEIGHT. Default 640x512 matches original test evaluation.")
    args = parser.parse_args()

    if not 0.0 <= args.b7_weight <= 1.0:
        raise ValueError("--b7-weight must lie in [0, 1].")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ensemble analysis. Activate the project GPU environment.")

    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(manifest, args.split)
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows: by_source[row["source_id"]].append(row)
    view_counts = {source: len(values) for source, values in by_source.items()}
    unexpected = {source: count for source, count in view_counts.items() if count != 8}
    if unexpected:
        raise ValueError(f"Expected exactly 8 registered views per analysed source; found {unexpected}")

    device = torch.device("cuda")
    b7, b8 = load_models(Path(args.b7_checkpoint), Path(args.b8_checkpoint), device)
    source_records: dict[str, list[dict]] = defaultdict(list)
    frame_rows: list[dict] = []

    for index, row in enumerate(rows, start=1):
        tensor, gt_map = read_image_and_mask(row, args.inference_size)
        pred_map = predict(b7, b8, tensor, args.b7_weight, device)
        pred_all_counts, pred_all_total = interface_counts(pred_map, mineral_only=False)
        pred_mineral_counts, pred_mineral_total = interface_counts(pred_map, mineral_only=True)
        gt_all_counts, gt_all_total = interface_counts(gt_map, mineral_only=False)
        gt_mineral_counts, gt_mineral_total = interface_counts(gt_map, mineral_only=True)
        record = {
            "gt_frac": fractions(gt_map),
            "pred_frac": fractions(pred_map),
            "pred_all_interface_counts": pred_all_counts,
            "pred_all_interface_total": pred_all_total,
            "pred_mineral_interface_counts": pred_mineral_counts,
            "pred_mineral_interface_total": pred_mineral_total,
            "gt_all_interface_counts": gt_all_counts,
            "gt_all_interface_total": gt_all_total,
            "gt_mineral_interface_counts": gt_mineral_counts,
            "gt_mineral_interface_total": gt_mineral_total,
        }
        source_records[row["source_id"]].append(record)
        frame_rows.append({
            "source_id": row["source_id"],
            "image_path": row["image_path"],
            "split": row["split"],
            "pixel_width": args.inference_size[0],
            "pixel_height": args.inference_size[1],
            "pred_all_nonidentical_interface_edges": pred_all_total,
            "pred_mineral_to_mineral_interface_edges": pred_mineral_total,
            "gt_all_nonidentical_interface_edges": gt_all_total,
            "gt_mineral_to_mineral_interface_edges": gt_mineral_total,
        })
        print(f"[{index}/{len(rows)}] {row['source_id']}")

    source_rows: list[dict] = []
    interface_rows: list[dict] = []
    for source_id in sorted(source_records):
        summary, records = source_summary(source_id, source_records[source_id])
        source_rows.append(summary); interface_rows.extend(records)

    modal_summary = aggregate_modal(source_rows)
    interface_summary = aggregate_interfaces(interface_rows)
    write_csv(output / "modal_area_proxy_by_source.csv", source_rows)
    write_csv(output / "modal_area_proxy_summary_by_mineral.csv", modal_summary)
    write_csv(output / "phase_interface_frequency_by_source.csv", interface_rows)
    write_csv(output / "phase_interface_frequency_summary.csv", interface_summary)
    write_csv(output / "processed_registered_views.csv", frame_rows)

    provenance = {
        "analysis": "Source-level 2D mineral modal-area proxies and semantic phase-interface frequencies",
        "scientific_scope": {
            "area_fraction": "Two-dimensional modal-area proxy; not 3D modal mineralogy.",
            "interface_frequency": "Four-neighbour pixel-edge semantic interface frequency for both ensemble predictions and ground-truth masks; not individual-grain contact network or 3D contact area.",
            "background": "Background is retained for quality-control interfaces but is not interpreted as porosity.",
            "physical_units": "No physical length or area units are reported because no verified pixel scale is supplied in the manifest.",
        },
        "source_level_rule": "Eight registered views are averaged to one source-level observation; frames are not treated as independent petrographic samples.",
        "manifest": str(manifest),
        "split": args.split,
        "sources": len(source_rows),
        "registered_views": len(rows),
        "views_per_source": 8,
        "inference_size_width_height": list(args.inference_size),
        "b7_checkpoint": str(Path(args.b7_checkpoint).resolve()),
        "b8_checkpoint": str(Path(args.b8_checkpoint).resolve()),
        "b7_weight": args.b7_weight,
        "b8_weight": 1.0 - args.b7_weight,
        "classes": NAMES,
        "outputs": {
            "source_modal_area_proxies": "modal_area_proxy_by_source.csv",
            "mineral_modal_area_summary": "modal_area_proxy_summary_by_mineral.csv",
            "source_interface_frequencies": "phase_interface_frequency_by_source.csv",
            "interface_frequency_summary": "phase_interface_frequency_summary.csv",
            "processed_views": "processed_registered_views.csv",
        },
    }
    (output / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(output)
    print("Completed source-level modal-area and semantic-interface analysis.")


if __name__ == "__main__":
    main()
