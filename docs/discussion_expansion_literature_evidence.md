# Discussion Expansion: Verified Literature Evidence

## Existing literature already supports the following claims

Saxena et al. (2021) show that semantic segmentation can provide whole-scene pixel-scale mineralogical information from 2D transmission-light sandstone images. Their abstract and introduction describe potential outputs relevant to mineral spatial distribution, grain size, petrophysical evaluation, and integration with 3D imaging. They also state that model effectiveness is strongly related to the amount of labelled data for the class concerned. This supports a cautious comparison and the statement that semantic maps can be an intermediate input to quantitative petrography, rather than a direct substitute for independently calibrated petrophysical measurements.

Acevedo Zamora and Kamber (2023) document the angle- and polarization-dependent appearance of anisotropic minerals and describe pixel-based semantic analysis limitations near fine-grained or optically indistinct phases. This supports the present interpretation of source-dependent mineral uncertainty and the rationale for future multiview design.

## New literature for expanded Discussion

Fan et al. (2025) describe a foundation-model workflow for rock thin-section analysis based on a large multi-region image cohort and four external validation cohorts. The paper supports the need for external/multi-centre validation, but it should not be used as a direct numerical comparator because it addresses different data, classes, model task components, and performance measures.

Dong et al. (2025) describe a hybrid system for automated mineral identification and quantitative petrology in tight sandstone and discuss downstream analysis of grain-scale metrics, contact relationships, and pore networks. It supports a carefully qualified statement that segmentation maps can be a prerequisite for geometry- and topology-based petrophysical analysis. It does not validate this present ensemble for reservoir-property estimation.

## Recommended new citations

[9] Fan J, Yu X, Di Y, Lv T, Zhang R, Bao J, Liu Y, Li L, Pan X. A foundation model for rock thin-section images analysis. *Communications Engineering*. 2025;5:9. https://doi.org/10.1038/s44172-025-00565-5.

[10] Dong L, Sun C, Yu X, Zhang X, Chen M, Xu M. Hybrid architecture for tight sandstone: Automated mineral identification and quantitative petrology. *Minerals*. 2025;15(9):962. https://doi.org/10.3390/min15090962.

## Editorial guardrails

- Do not numerically compare the present 97-source eight-mineral B7–B8 semantic model with Fan et al. or Dong et al.; task definitions, source populations, annotations, modalities, and metrics are non-equivalent.
- Explain that modal mineralogy, grain-size, contact topology, and pore–mineral analysis are **downstream hypotheses or future validated applications**. They have not been estimated or benchmarked in the present study.
- State that any quantitative use must propagate segmentation uncertainty and be checked against source-appropriate petrographic and/or chemical reference measurements.
- Use the external-validation design of Fan et al. to motivate—not claim completion of—the next stage of this research.
