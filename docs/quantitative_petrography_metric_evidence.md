# Quantitative Petrography Metrics: Verified Evidence and Constraints

## Evidence directly relevant to the existing B7–B8 semantic maps

Saxena et al. (2021), *Application of deep learning for semantic segmentation of sandstone thin sections*, state that whole-scene semantic segmentation can provide spatial mineralogy, grain-size-related information, and inputs relevant to petrophysical evaluation. They also emphasize that pixelwise labels can support geometry assessment, grain-size analysis, textural measures, contact length, nearest neighbours, geomechanical models, and rock-physics applications when appropriate post-processing and validation are performed. Their work supports use of semantic maps as a starting point, not automatic proof of downstream physical properties. URL: https://doi.org/10.1016/j.cageo.2021.104778.

Acevedo Zamora and Kamber (2023) discuss modal-mineralogy estimates from segmented multi-angle polarized imagery and limitations induced by optical anisotropy, fine grains, and optically indistinct phases. This supports conservative use of pixel fractions and angle-consistency reporting. URL: https://doi.org/10.3390/min13020156.

Dabek et al. (2023), *Superpixel-Based Grain Segmentation in Sandstone Thin-Section*, distinguish mineral/phase segmentation from grain-boundary detection. They show that quantitative composition requires boundaries between rock components, while individual-grain counting requires a dedicated grain-segmentation process. This supports the conclusion that grain size, shape, perimeter, and roundness are not defensibly available from the current semantic maps alone. URL: https://doi.org/10.3390/min13020219.

Azzam, Blaise, and Brigaud (2024), *Automated petrographic image analysis by supervised and unsupervised machine learning methods*, show that an instance/grain-detection stage plus physical image scale enables grain area, perimeter, circularity, aspect ratio, and Feret diameters. They separately use segmentation for mineral associations and porosity exploration. This supports the need for grain-instance masks and a micrometre-per-pixel calibration before reporting physical grain morphometrics. URL: https://doi.org/10.57035/journals/sdk.2024.e22.1594.

Dong et al. (2025), *Hybrid Architecture for Tight Sandstone: Automated Mineral Identification and Quantitative Petrology*, supports the potential of segmentation for grain-scale properties, contact relationships, and pore-network analysis, but is a different tight-sandstone task with different data and outputs. It supports future-work framing only. URL: https://doi.org/10.3390/min15090962.

## Metric classification to preserve scientific validity

### Directly computable now from semantic maps

- 2D phase areal fraction (per mineral), called a *modal-area proxy* rather than 3D modal mineralogy.
- Predicted-versus-reference phase-fraction error on the existing held-out masks.
- Class-presence richness and evenness/diversity indices from phase fractions.
- Pixel-edge phase-contact/adjacency matrix, normalized by total class-boundary length; report it as 2D semantic interface frequency, not physical contact area.
- Phase-domain fragmentation: connected-component count, largest-component fraction, and component-area distribution, explicitly for semantic domains rather than individual grains.
- Registered-view prediction stability: pairwise agreement or consensus agreement across the eight angular predictions of one source. This measures orientation/observation robustness, not correctness.
- Per-source uncertainty and heterogeneity: source-level mIoU, Dice, and phase-fraction error distributions across existing held-out sources.

### Possible after dedicated post-processing and calibration

- Grain count, grain size distribution, Feret diameter, perimeter, circularity, aspect ratio, and orientation: require an instance/grain-boundary pipeline, validation against expert delineations, and micrometre-per-pixel calibration.
- Grain-contact network topology: requires validated grain-instance boundaries. Class-domain adjacency alone must not be called a grain-contact network.
- Mineral orientation/fabric: requires orientation-resolved grain domains or crystallographic/orientation measurements; current connected regions are insufficient.

### Not defensible from the present maps

- Porosity/permeability, pore-throat size, or reservoir quality: background cannot be assumed to be pore space in this dataset and no pore class or calibrated fluid-flow model was trained.
- 3D modal mineralogy or bulk-volume fractions: one 2D section needs stereological assumptions and external validation.
- Mineral chemistry, alteration intensity, grade, or phase composition beyond the eight label classes: require chemical/mineralogical reference data such as SEM-EDS, EPMA, XRD, or hyperspectral measurements.

## Critical implementation constraint

The dataset documentation and current study files establish image dimensions but not a verified physical pixel scale. Therefore current metrics should remain in pixels, pixels squared, unitless proportions, or normalized boundary frequencies. Do not report micrometres, millimetres, or physical areas until a valid acquisition-scale calibration is supplied.
