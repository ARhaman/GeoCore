# Quantitative Petrography Supplement: B7–B8 Source-Held Analysis

## Verified analysis scope

The validated B7–B8 ensemble was evaluated on the original held-out partition comprising **10 independent thin-section sources** and **80 registered views**. Each source contributed exactly eight angular views, and all view-level values were aggregated to a single source-level observation before summary. Therefore, none of the results below treats correlated angular frames as independent petrographic samples.

The analysis calculates two types of map-derived quantity. Mineral area fractions are reported as **two-dimensional modal-area proxies**, not as bulk three-dimensional modal mineralogy. Mineral-to-mineral contacts are reported as **four-neighbour semantic pixel-edge interface frequencies**, not as individual-grain contact topology or three-dimensional contact area. Background is retained only as a dataset class and is not interpreted as porosity.

## Results: mineral modal-area proxies

Across the 10 held-out sources, the mean source-level mineral area mean absolute error was **0.0412**, with a median of **0.0307** and a range of **0.0114–0.0856**. The corresponding mineral-composition \(L_1\) error was **0.3294** on average, with a median of **0.2452** and a range of **0.0908–0.6848**. This composition statistic is a sum of mineral-area discrepancies and should not be confused with the independently calculated repeated-cross-validation mIoU value, which happens to have the same numerical value when rounded to four decimals.

| Mineral | Mean ground-truth 2D area fraction | Mean predicted 2D area fraction | Mean signed area error | Mean absolute area error | Interpretation |
|---|---:|---:|---:|---:|---|
| Olivine | 0.0256 | 0.0688 | +0.0432 | 0.0432 | Over-represented on average; the error is source dependent. |
| Pyroxene | 0.0540 | 0.0470 | −0.0071 | 0.0390 | Near-zero mean signed bias masks sizeable source-to-source error. |
| Plagioclase | 0.1258 | 0.1281 | +0.0023 | 0.0539 | Mean composition is close, but local/source-specific area error remains material. |
| Alkali feldspar | 0.0742 | 0.1442 | +0.0701 | 0.0835 | Largest systematic mineral-area over-representation. |
| Quartz | 0.1358 | 0.0812 | −0.0546 | 0.0562 | Systematically under-represented on average. |
| Biotite | 0.0028 | 0.0180 | +0.0152 | 0.0187 | Over-represented despite low mapped ground-truth abundance; interpretation requires particular caution. |
| Muscovite | 0.0336 | 0.0143 | −0.0193 | 0.0211 | Under-represented on average, with errors concentrated in selected sources. |
| Hornblende | 0.0073 | 0.0096 | +0.0023 | 0.0138 | Small positive mean bias with substantial relative uncertainty due to low abundance. |

The largest source-level composition discrepancies were observed for **Muscovite–Schist–3** (mineral-composition \(L_1=0.6848\)), **Pyroxene–Peridotite–1** (0.5369), **Muscovite–Granite–3** (0.4474), and **Granite–1** (0.4459). These differences underline why a single mean metric is insufficient for petrographic use: a model may preserve total mineral coverage approximately while substantially redistributing area between specific mineral classes.

### Suggested manuscript text

> To assess whether pixel-level segmentation preserved the mapped mineral assemblage, we computed source-level two-dimensional mineral modal-area proxies from the original held-out test partition. Each of the 10 thin sections contributed eight registered views, which were averaged within source before summary. The ensemble achieved a mean source-level mineral area mean absolute error of 0.0412 (median 0.0307; range 0.0114–0.0856) and a mean mineral-composition \(L_1\) error of 0.3294. Alkali feldspar was systematically over-represented by 0.0701 mean area fraction, whereas quartz was under-represented by 0.0546. Plagioclase had near-zero mean signed composition bias (+0.0023) but non-negligible absolute error across sources (0.0539). These results indicate that the ensemble can support exploratory source-level compositional mapping, but class-specific area bias must be reported before using the maps as quantitative modal-mineral proxies.

## Results: semantic mineral-interface frequencies

The interface analysis compares the normalized mineral-to-mineral pixel-edge frequency matrix of the ensemble prediction with that of the reference semantic masks. It quantifies whether the **types of labelled phase boundaries** are reproduced; it does not validate individual grain contacts.

The most over-represented predicted interfaces were pyroxene–alkali feldspar (+0.1057), plagioclase–alkali feldspar (+0.0556), olivine–plagioclase (+0.0520), biotite–hornblende (+0.0471), pyroxene–biotite (+0.0456), and pyroxene–hornblende (+0.0375). The most under-represented interfaces were plagioclase–hornblende (−0.0979), quartz–muscovite (−0.0878), alkali feldspar–hornblende (−0.0836), pyroxene–plagioclase (−0.0555), plagioclase–biotite (−0.0134), and olivine–pyroxene (−0.0097).

These differences should be interpreted as **semantic-class redistribution at mineral boundaries**, not as proof of physical contact changes. Interfaces with zero ground-truth mean frequency but nonzero predicted frequency are particularly clear indicators of spurious predicted label adjacencies. Conversely, strong ground-truth interfaces that are under-represented in the prediction indicate that one or both participating mineral labels are missing, fragmented, or reassigned.

### Suggested manuscript text

> We additionally evaluated mineral-to-mineral semantic interface frequencies to examine whether the ensemble preserved the mapped pattern of labelled phase adjacencies. The analysis counted horizontal and vertical boundaries between unlike mineral labels in each view, normalized the interface matrix within view, and then aggregated the eight registered views within each source. The largest over-represented predicted interfaces were pyroxene–alkali feldspar (+0.1057) and plagioclase–alkali feldspar (+0.0556), while plagioclase–hornblende (−0.0979), quartz–muscovite (−0.0878), and alkali feldspar–hornblende (−0.0836) were under-represented. These patterns are consistent with mineral-specific area biases and demonstrate that an apparently plausible segment map may still distort compositional and adjacency information. The measure is intentionally reported as a semantic pixel-edge frequency rather than a grain-contact network because the current labels do not delineate all individual grains.

## Publication recommendation

The modal-area analysis is suitable for a supplementary results subsection and a supplementary table. The interface matrix is valuable as a **supplementary diagnostic**, but a complete 36-pair table is too dense for the main text. The main manuscript should report the aggregate area-error findings and cite the interface analysis as a class-boundary diagnostic, with the complete predicted-versus-reference matrix placed in a supplement.

Before elevating these metrics to a primary claim, repeat the same analysis within the completed 2 × 5 grouped source-level cross-validation folds. The original 10-source held-out analysis is valid and informative, but repeated grouped-CV estimates are needed to express between-source uncertainty for derived petrographic quantities.
