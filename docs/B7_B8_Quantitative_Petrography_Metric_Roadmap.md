# Quantitative Petrography Roadmap for the Existing B7–B8 Semantic Maps

## Bottom line

**Yes—several meaningful quantitative petrography metrics can be derived now from the existing B7–B8 prediction maps and their paired ground-truth masks.** The most defensible near-term outputs are **two-dimensional phase areal fractions, mineral-composition error, class richness/evenness, phase-domain fragmentation, class-interface adjacency, registered-view stability, and source-level uncertainty summaries**.

The current maps do **not** yet justify grain-size distributions, grain-shape statistics, physical contact networks, porosity/permeability estimates, or bulk 3D modal mineralogy. Those require a dedicated grain-instance stage, a physical scale calibration, and in some cases external chemical or petrophysical reference data.

## What can be calculated now

| Metric family | Exact quantity | Scientific interpretation | Required input | Reporting constraint |
|---|---|---|---|---|
| **Phase areal fraction** | \(A_c=N_c/N\) for each mineral class \(c\) | Two-dimensional mineral-area proportion; a **modal-area proxy** | Predicted map and ground-truth map | Do not call it 3D modal mineralogy without stereological/external validation. |
| **Phase-fraction error** | \(\Delta A_c=A_c^{pred}-A_c^{GT}\); MAE across classes/sources | Bias in predicted mineral proportions | Predicted and ground-truth maps | Report signed bias and absolute error by mineral, not only a pooled score. |
| **Mineral richness** | Number of mineral classes with nonzero area | Observed mineral-class presence within the field | Semantic map | Presence is label-scheme dependent; small predicted regions should be confidence-screened. |
| **Mineral evenness/diversity** | Shannon entropy \(H=-\sum p_c\log p_c\), normalized entropy, Simpson index | Heterogeneity of the labelled mineral assemblage | Semantic map | Describes mapped class composition, not mineralogical complexity beyond eight labels. |
| **Phase-domain fragmentation** | Connected-component count, component-area distribution, largest-component fraction per class | Spatial continuity or fragmentation of *semantic phase domains* | Semantic map | Do not interpret components as individual grains. Same-mineral neighbouring grains can merge. |
| **Semantic interface adjacency** | Four-neighbour pixel-edge counts between classes; normalized interface matrix \(I_{ab}\) | Frequency of two-dimensional labelled phase contacts | Semantic map | Report as semantic interface frequency or boundary length in pixels—not a physical 3D contact-area network. |
| **Registered-view stability** | Pairwise prediction agreement or consensus agreement across eight registered views of one source | Sensitivity of predictions to angular/polarization state | Eight predicted maps per source | A robustness measure, not ground-truth accuracy. Pair with IoU/Dice. |
| **Source-level performance heterogeneity** | Per-source mIoU, Dice, phase-fraction MAE, diversity error | Which source assemblages cause degradation | Current held-out source predictions and masks | Use sources, not 80 correlated frames, as the reporting unit. |

## Recommended immediate quantitative supplement

The following four analyses can be added to the current study without claiming a new model or changing the B7–B8 result.

### 1. Mineral modal-area proxy and composition error

For every held-out source \(s\) and mineral \(c\), calculate

\[
A_{s,c}=\frac{1}{N_s}\sum_{i=1}^{N_s}\mathbb{1}(y_i=c),
\qquad
\Delta A_{s,c}=A^{pred}_{s,c}-A^{GT}_{s,c}.
\]

Report the median signed bias, median absolute error, and source-level interquartile range for each mineral. This is more interpretable for petrographers than mIoU alone because it asks whether mineral proportions are systematically over- or under-estimated.

### 2. Mineral assemblage diversity and compositional similarity

Calculate class richness and normalized Shannon diversity from ground truth and predictions for each source. Then calculate a compositional distance, such as the \(L_1\) distance between the eight mineral-area vectors. This provides a source-level answer to: **does the model preserve the mapped mineral assemblage even when pixel boundaries are imperfect?**

### 3. Semantic phase-interface matrix

Count horizontal and vertical class transitions in every predicted map and compare the normalized predicted matrix with the ground-truth matrix. For example, report whether quartz–feldspar, olivine–pyroxene, or plagioclase–alkali-feldspar interfaces are over- or under-represented. This is a useful textural descriptor, but it must be named a **semantic interface frequency** because the current maps do not separate all individual grains.

### 4. Multi-angle stability of the registered data

The registered angular structure is a special advantage of this dataset. For each source, create the eight B7–B8 predictions and compute pairwise agreement across angles, or compare each view with a source-level majority-vote consensus. Report both the angular-stability score and the source mIoU. A source could be internally stable but consistently wrong, or unstable but partially correct; both outcomes are informative.

## What requires a new validated post-processing stage

| Desired metric | Why current semantic maps are insufficient | Minimum additional work |
|---|---|---|
| Individual grain count and grain size | One semantic connected component can contain multiple adjacent grains of the same mineral. | Grain-boundary or instance segmentation; validation against expert grain outlines. |
| Grain area, perimeter, Feret diameter, aspect ratio, circularity | These require separate objects and a physical spatial scale. | Instance masks plus micrometre-per-pixel calibration. |
| Grain-contact network | Class interfaces do not identify contacts between individual grains of the same or different minerals. | Validated grain-instance graph extraction; boundary-review protocol. |
| Preferred orientation/fabric | Connected semantic patches are not crystallographic orientations. | Grain-instance orientations, EBSD, or a validated optical orientation workflow. |
| Porosity, permeability, pore-throat network | The background label is not a validated pore class. | Pore-resolved annotations, calibrated imaging, and a separately validated rock-physics/flow model. |
| 3D modal mineralogy or bulk volume | One 2D section does not establish 3D volume fraction without assumptions. | Stereological design, serial sections or 3D imaging, and external calibration. |

## Essential uncertainty rules

1. **Use source-level summaries.** Eight angular frames of the same thin section are correlated and must not be treated as eight independent petrological samples.
2. **Pair every predicted metric with its ground-truth analogue** on the current held-out data. For example, report \(\Delta A_c\), not only \(A_c^{pred}\).
3. **Report class-specific results.** Biotite is the weakest grouped-CV mineral class; any biotite-derived composition or interface metric must be flagged as low confidence.
4. **Stay in pixel or normalized units.** The present study verifies 1280 × 1024 dimensions but does not establish a physical pixel scale. Do not report \(\mu m\), \(mm^2\), or physical boundary length without image-scale metadata.
5. **Separate robustness from correctness.** Registered-view agreement measures angular robustness; mIoU/Dice and phase-fraction error measure agreement with the reference mask.

## Scientific value for this paper

The strongest addition is **not** a poorly validated grain-size or reservoir-property claim. It is a source-level mineral-composition analysis that demonstrates whether the final ensemble preserves the mapped mineral assemblage, then complements that analysis with angular stability and semantic-interface results. This extension would deepen the paper’s quantitative petrography relevance while remaining consistent with the current semantic-label task and honest about its limitations.

## Key literature

[1] Saxena N, Day-Stirrat RJ, Hows A, Hofmann R. Application of deep learning for semantic segmentation of sandstone thin sections. *Computers & Geosciences*. 2021;152:104778. https://doi.org/10.1016/j.cageo.2021.104778.

[2] Acevedo Zamora MA, Kamber BS. Petrographic microscopy with ray tracing and segmentation from multi-angle polarisation whole-slide images. *Minerals*. 2023;13(2):156. https://doi.org/10.3390/min13020156.

[3] Dabek P, Chudy K, Nowak I, Zimroz R. Superpixel-Based Grain Segmentation in Sandstone Thin-Section. *Minerals*. 2023;13(2):219. https://doi.org/10.3390/min13020219.

[4] Azzam F, Blaise T, Brigaud B. Automated petrographic image analysis by supervised and unsupervised machine learning methods. *Swiss Journal of Geosciences*. 2024;2(2):e22. https://doi.org/10.57035/journals/sdk.2024.e22.1594.

[5] Dong L, Sun C, Yu X, Zhang X, Chen M, Xu M. Hybrid architecture for tight sandstone: Automated mineral identification and quantitative petrology. *Minerals*. 2025;15(9):962. https://doi.org/10.3390/min15090962.
