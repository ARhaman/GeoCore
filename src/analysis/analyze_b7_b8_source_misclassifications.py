"""Detailed source-specific B7–B8 error analysis for registered thin-section sequences.

Outputs are aggregated at the thin-section source level after inference on its eight registered
views. This reports composition bias, per-class precision/recall/IoU, confusion pathways, and
semantic interface distortion for specified sources. It does not treat individual angular views as
independent facies samples.
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

from generate_b7_native_900dpi_figures import PretrainedResUNet, MEAN, STD, NAMES, C


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except Exception as exc:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, for example 640x512.") from exc
    return width, height


def read_rows(path: Path, split: str, wanted: set[str]) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == split and row.get("source_id") in wanted:
            grouped[row["source_id"]].append(row)
    missing = wanted - set(grouped)
    if missing:
        raise ValueError(f"Requested sources not found in split={split!r}: {sorted(missing)}")
    invalid = {key: len(value) for key, value in grouped.items() if len(value) != 8}
    if invalid:
        raise ValueError(f"Each analysed source must have eight registered views: {invalid}")
    return grouped


def deeplab() -> torch.nn.Module:
    return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=C, aux_loss=False)


def load_models(b7_path: Path, b8_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    b7 = PretrainedResUNet().to(device); b8 = deeplab().to(device)
    b7.load_state_dict(torch.load(b7_path, map_location=device, weights_only=False)["model"])
    b8.load_state_dict(torch.load(b8_path, map_location=device, weights_only=False)["model"])
    b7.eval(); b8.eval()
    return b7, b8


def infer(row: dict[str, str], b7: torch.nn.Module, b8: torch.nn.Module, device: torch.device, weight: float, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(row["image_path"]) as image:
        rgb = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255.0
    with Image.open(row["mask_path"]) as mask:
        gt = np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST), dtype=np.int64)
    x = torch.from_numpy(rgb)
    x = (x - MEAN.squeeze(0)) / STD.squeeze(0)
    with torch.no_grad():
        batch = x.unsqueeze(0).to(device, non_blocking=True)
        p7 = b7(batch).softmax(1); p8 = b8(batch)["out"].softmax(1)
        pred = (weight * p7 + (1.0 - weight) * p8).argmax(1)[0].cpu().numpy().astype(np.int64)
    return gt, pred


def confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(gt.reshape(-1) * C + pred.reshape(-1), minlength=C * C).reshape(C, C).astype(np.int64)


def interfaces(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((C, C), dtype=np.int64)
    for first, second in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        changed = (first != second) & (first > 0) & (second > 0)
        if np.any(changed):
            lo = np.minimum(first[changed], second[changed]); hi = np.maximum(first[changed], second[changed])
            result += np.bincount(lo * C + hi, minlength=C * C).reshape(C, C)
    return result


def per_class_rows(matrix: np.ndarray, source: str) -> list[dict]:
    rows = []
    for c, name in enumerate(NAMES):
        tp = int(matrix[c, c]); actual = int(matrix[c, :].sum()); predicted = int(matrix[:, c].sum())
        fp = predicted - tp; fn = actual - tp
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        union = tp + fp + fn
        iou = tp / union if union else 0.0
        rows.append({
            "source_id": source, "class": name, "gt_pixels": actual, "pred_pixels": predicted,
            "true_positive_pixels": tp, "false_positive_pixels": fp, "false_negative_pixels": fn,
            "precision": precision, "recall": recall, "iou": iou,
            "gt_area_fraction": actual / matrix.sum(), "pred_area_fraction": predicted / matrix.sum(),
            "signed_area_error": (predicted - actual) / matrix.sum(),
        })
    return rows


def misclassification_rows(matrix: np.ndarray, source: str) -> list[dict]:
    rows = []
    for gt_id in range(C):
        actual = matrix[gt_id, :].sum()
        for pred_id in range(C):
            if gt_id == pred_id or matrix[gt_id, pred_id] == 0:
                continue
            count = int(matrix[gt_id, pred_id])
            rows.append({
                "source_id": source, "ground_truth_class": NAMES[gt_id], "predicted_class": NAMES[pred_id],
                "misclassified_pixels": count, "fraction_of_gt_class": count / actual if actual else 0.0,
                "fraction_of_all_pixels": count / matrix.sum(),
            })
    return sorted(rows, key=lambda r: r["misclassified_pixels"], reverse=True)


def interface_rows(gt: np.ndarray, pred: np.ndarray, source: str) -> list[dict]:
    total_gt = int(gt.sum()); total_pred = int(pred.sum())
    rows = []
    for a in range(1, C):
        for b in range(a + 1, C):
            gt_count = int(gt[a, b]); pred_count = int(pred[a, b])
            gt_frequency = gt_count / total_gt if total_gt else 0.0
            pred_frequency = pred_count / total_pred if total_pred else 0.0
            rows.append({
                "source_id": source, "class_a": NAMES[a], "class_b": NAMES[b],
                "gt_edge_count": gt_count, "pred_edge_count": pred_count,
                "gt_normalized_frequency": gt_frequency, "pred_normalized_frequency": pred_frequency,
                "frequency_delta_prediction_minus_gt": pred_frequency - gt_frequency,
            })
    return sorted(rows, key=lambda r: abs(r["frequency_delta_prediction_minus_gt"]), reverse=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_report(path: Path, source_summaries: list[dict], errors: list[dict], interfaces_all: list[dict]) -> None:
    lines = ["# Detailed B7–B8 source-level misclassification analysis", ""]
    lines += ["## Scope", "", "The analysis aggregates exactly eight registered angular views into one observation per source. It compares two source/lithology cases—Muscovite–Schist–3 and Pyroxene–Peridotite–1—rather than treating views as independent facies samples. Pixel errors are reported at the validated 640 × 512 inference resolution.", ""]
    for summary in source_summaries:
        source = summary["source_id"]
        lines += [f"## {source}", "", f"- Mineral mIoU: **{summary['mineral_miou']:.4f}**.", f"- Mineral composition L1 error: **{summary['mineral_composition_l1']:.4f}**.", f"- Mineral area MAE: **{summary['mineral_area_mae']:.4f}**.", "", "### Largest pixel-level confusion pathways", "", "| Ground truth | Predicted as | Pixels | Share of source GT class |", "|---|---|---:|---:|"]
        source_errors = [row for row in errors if row["source_id"] == source][:8]
        for row in source_errors:
            lines.append(f"| {row['ground_truth_class']} | {row['predicted_class']} | {row['misclassified_pixels']:,} | {row['fraction_of_gt_class']:.3f} |")
        lines += ["", "### Largest semantic-interface distortions", "", "| Interface | Ground truth frequency | Prediction frequency | Delta |", "|---|---:|---:|---:|"]
        source_interfaces = [row for row in interfaces_all if row["source_id"] == source][:8]
        for row in source_interfaces:
            lines.append(f"| {row['class_a']}–{row['class_b']} | {row['gt_normalized_frequency']:.4f} | {row['pred_normalized_frequency']:.4f} | {row['frequency_delta_prediction_minus_gt']:+.4f} |")
        lines.append("")
    lines += ["## Interpretation guardrails", "", "- Mineral area fractions are two-dimensional modal-area proxies, not three-dimensional modal mineralogy.", "- Semantic interfaces are pixel-edge class transitions, not individual-grain or 3D contact networks.", "- Differences between the two sources describe composition- and texture-dependent model errors; they do not prove a unique optical or petrogenetic cause without independent microscopy/chemical validation."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--b7-checkpoint", required=True); p.add_argument("--b8-checkpoint", required=True)
    p.add_argument("--output", required=True); p.add_argument("--split", default="test"); p.add_argument("--b7-weight", type=float, default=0.6)
    p.add_argument("--inference-size", type=parse_size, default=(640, 512)); p.add_argument("--sources", nargs="+", default=["Muscovite-Schist-3", "Pyroxene-Peridotite-1"])
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    if not 0 <= args.b7_weight <= 1: raise ValueError("B7 weight must be in [0,1].")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    grouped = read_rows(Path(args.manifest), args.split, set(args.sources))
    device = torch.device("cuda"); b7, b8 = load_models(Path(args.b7_checkpoint), Path(args.b8_checkpoint), device)

    class_rows = []; error_rows = []; interface_all = []; source_summaries = []; view_rows = []
    for source in sorted(grouped):
        cm = np.zeros((C, C), dtype=np.int64); gt_interfaces = np.zeros((C, C), dtype=np.int64); pred_interfaces = np.zeros((C, C), dtype=np.int64)
        for index, row in enumerate(grouped[source], start=1):
            gt, pred = infer(row, b7, b8, device, args.b7_weight, args.inference_size)
            current = confusion(gt, pred); cm += current; gt_interfaces += interfaces(gt); pred_interfaces += interfaces(pred)
            mineral_iou = []
            for c in range(1, C):
                tp = current[c,c]; union = current[c,:].sum()+current[:,c].sum()-tp
                mineral_iou.append(float(tp/union) if union else 0.0)
            view_rows.append({"source_id":source,"view_index":index,"image_path":row['image_path'],"mineral_miou":float(np.mean(mineral_iou))})
            print(f"{source}: view {index}/8")
        metrics = per_class_rows(cm, source); class_rows.extend(metrics); errors = misclassification_rows(cm, source); error_rows.extend(errors); interface_all.extend(interface_rows(gt_interfaces, pred_interfaces, source))
        mineral = [r for r in metrics if r['class'].lower() != 'background']
        source_summaries.append({
            "source_id": source, "registered_views": len(grouped[source]),
            "mineral_miou": float(np.mean([r['iou'] for r in mineral])),
            "mineral_area_mae": float(np.mean([abs(r['signed_area_error']) for r in mineral])),
            "mineral_composition_l1": float(np.sum([abs(r['signed_area_error']) for r in mineral])),
        })
    write_csv(output/'per_source_class_metrics.csv', class_rows); write_csv(output/'misclassification_pathways.csv', error_rows); write_csv(output/'semantic_interface_distortion.csv', interface_all); write_csv(output/'per_registered_view_miou.csv', view_rows); write_csv(output/'source_error_summary.csv', source_summaries)
    write_report(output/'source_specific_error_analysis.md', source_summaries, error_rows, interface_all)
    provenance = {"sources":args.sources,"split":args.split,"views_per_source":8,"inference_size":list(args.inference_size),"b7_weight":args.b7_weight,"b8_weight":1-args.b7_weight,"scope":"Source-level semantic segmentation error analysis; no independent facies claim is made for correlated angular views."}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(output)

if __name__ == '__main__': main()
