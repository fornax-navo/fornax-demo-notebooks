# Time Domain

In this set of Use Case Scenario we work towards creating multi-band light curves from multiple archival and publication resources at scale and classifying and analyzing them with machine learning tools. Tutorials included in this folder are:

1. light_curve_generator.md This notebook automatically retrieves target positions from the literature and then queries archival sources for light curves of those targets. This notebook is intended to be run on a small number of sources (<~ few hundred)
2. scale_up.md This notebook uses the same functions as light_curve_generator (above) but is able to generate light curves for large number of sources (~1000 -> millions?) and provides additional monitoring options.
3. light_curve_classifier.md This notebook takes output from light_curve generator and trains a ML classifier to be able to differentiate amongst the samples based on their light curves.
4. ML_AGNzoo.md This notebook takes output from the light_curve_generator (above) and visualizes/compares different labelled samples on a reduced dimension grid.

```{toctree}
---
maxdepth: 1
---
light_curve_generator
lc_classifier

```
