# Mineral Semantic Segmentation in Registered Polarized-Light Thin Sections

This repository contains the reproducibility code for a pixel-level, eight-mineral semantic-segmentation study using the **Registered Angular Mineral Segmentation Dataset**. The final model is a validation-calibrated probability ensemble of a pretrained ResNet-34 U-Net (**B7**) and pretrained DeepLabV3–ResNet50 (**B8**). The central methodological safeguard is that all eight registered angular views of one thin-section source remain in the same partition.

> **Primary result.** Across two repeats of five grouped source-level outer folds (10 held-out evaluations), the B7–B8 ensemble achieved mineral mIoU **0.3294 ± 0.0459** and mineral Dice **0.4738 ± 0.0545**. The separate original source-held test produced mineral mIoU **0.2440** and mineral Dice **0.3732**. These estimates answer different questions and must not be merged.

## Scope and scientific safeguards

The repository provides dataset audit, direct palette-index mask normalization, deterministic source-level splitting, B7 and B8 training, validation-only probability fusion, repeated grouped cross-validation, native-resolution figure generation, and supplementary quantitative-petrography diagnostics. It does **not** ship raw microscopy images, semantic masks, trained weights, or generated output tables. Users must obtain the source dataset and train or provide their own checkpoints.

The model’s map-derived mineral fractions are reported as **two-dimensional modal-area proxies**. Pixel-edge phase-interface values are **semantic interface frequencies**. Neither quantity is bulk three-dimensional modal mineralogy, grain-contact topology, porosity, permeability, or a physical-scale measurement.

## Data access

Obtain the Registered Angular Mineral Segmentation Dataset from the official record: Zhu and Yang, *Registered Angular Mineral Segmentation Dataset*, IEEE DataPort, DOI: https://doi.org/10.21227/sa8g-wr95. Respect the dataset licence and cite the dataset in derivative work.

Expected local structure:

```text
registered_angular_mineral_segmentation/
  train/
    image/
    mask/
```

## Environment

The validated workstation used Python 3.11, PyTorch with CUDA, `segmentation-models-pytorch`, `albumentations`, `numpy`, `pandas`, `Pillow`, `matplotlib`, and `python-pptx`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision segmentation-models-pytorch albumentations numpy pandas pillow matplotlib python-pptx openpyxl
```

## Reproduction workflow

All commands below are written for Windows PowerShell. From the repository root, define:

```powershell
$Python = '.\.venv\Scripts\python.exe'
$Dataset = 'E:\path\to\registered_angular_mineral_segmentation'
```

### 1. Audit and normalize indexed semantic masks

```powershell
& $Python .\src\data\prepare_mineral_manifest.py --dataset-root $Dataset --output-dir .\data\mineral_manifests
& $Python .\src\data\normalize_mineral_masks.py --manifest .\data\mineral_manifests\mineral_segmentation_manifest.csv --output-dir .\data\mineral_normalized
& $Python .\src\data\audit_mineral_masks.py --manifest .\data\mineral_normalized\mineral_segmentation_manifest_normalized.csv
```

### 2. Train the two complementary models

```powershell
& $Python .\src\models\train_mineral_segmentation_pretrained_rareclass.py --manifest .\data\mineral_normalized\mineral_segmentation_manifest_normalized.csv --output .\artifacts\mineral_segmentation_pretrained_rareclass
& $Python .\src\models\train_mineral_segmentation_deeplabv3.py --manifest .\data\mineral_normalized\mineral_segmentation_manifest_normalized.csv --output .\artifacts\mineral_segmentation_deeplabv3
```

### 3. Evaluate the validation-calibrated B7–B8 ensemble

```powershell
& $Python .\src\models\evaluate_b7_b8_ensemble.py --manifest .\data\mineral_normalized\mineral_segmentation_manifest_normalized.csv --b7-checkpoint .\artifacts\mineral_segmentation_pretrained_rareclass\best.pt --b8-checkpoint .\artifacts\mineral_segmentation_deeplabv3\best.pt --b7-weight 0.6 --split test --output .\artifacts\mineral_segmentation_b7_b8_ensemble
```

### 4. Repeated source-level cross-validation

```powershell
& $Python .\src\data\prepare_repeated_source_cv.py --manifest .\data\mineral_normalized\mineral_segmentation_manifest_normalized.csv --output-dir .\data\repeated_source_cv
& $Python .\src\models\run_b7_b8_repeated_cv.py --cv-root .\data\repeated_source_cv --output-root .\artifacts\b7_b8_repeated_cv
& $Python .\src\models\aggregate_b7_b8_repeated_cv.py --results-root .\artifacts\b7_b8_repeated_cv --output-dir .\artifacts\b7_b8_repeated_cv\summary
```

### 5. Supplementary quantitative petrography

```powershell
& $Python .\src\analysis\compute_b7_b8_modal_area_and_interfaces.py --manifest .\data\mineral_normalized\mineral_segmentation_manifest_normalized.csv --b7-checkpoint .\artifacts\mineral_segmentation_pretrained_rareclass\best.pt --b8-checkpoint .\artifacts\mineral_segmentation_deeplabv3\best.pt --split test --b7-weight 0.6 --output .\artifacts\mineral_segmentation_b7_b8_ensemble\quantitative_petrography_test

& $Python .\src\analysis\analyze_b7_b8_source_misclassifications.py --manifest .\data\mineral_normalized\mineral_segmentation_manifest_normalized.csv --b7-checkpoint .\artifacts\mineral_segmentation_pretrained_rareclass\best.pt --b8-checkpoint .\artifacts\mineral_segmentation_deeplabv3\best.pt --split test --b7-weight 0.6 --output .\artifacts\mineral_segmentation_b7_b8_ensemble\source_specific_error_analysis
```

## Repository organisation

| Path | Purpose |
|---|---|
| `src/data` | Manifest, indexed-mask normalization, leakage audit, grouped-fold preparation. |
| `src/models` | B7/B8 training, ensemble evaluation, repeated grouped CV. |
| `src/analysis` | Source-level modal-area, semantic-interface, and source-error analyses. |
| `src/figures` | Native 900-DPI scientific raster figures and editable PowerPoint figures. |
| `src/manuscript` | Direct-edit scripts for the editable Word manuscript; user edits remain authoritative. |
| `docs` | Metric definitions, results interpretation, limitations, and a citation list. |

## Reproducibility conditions

Use a source-level split. Do not randomly split the eight registered angular frames. Select any ensemble coefficient only with validation sources. Report the original fixed test and repeated grouped-CV result separately. Treat all eight angular views of a source as correlated views, not independent geological replicates.

## Citation

If using or adapting this code, cite the associated manuscript and the Registered Angular Mineral Segmentation Dataset. A BibTeX citation placeholder is provided in `CITATION.cff` and must be updated with the final chapter DOI after publication.

## Licence

No code licence is granted by this staging repository until the authors select and add a licence. Dataset terms are governed separately by the dataset provider.
