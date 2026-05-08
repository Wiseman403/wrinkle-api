"""Core algorithm modules.

Each module owns one stage of the pipeline:

- ``landmarks``: index sets and anchor constants.
- ``masks``: face mask, skin mask, and 14 zone masks.
- ``preprocess``: L* + CLAHE + bilateral + unsharp.
- ``ridges``: Frangi multi-scale + hysteresis + skeletonize + branch-split.
- ``measure``: per-zone metrics and per-wrinkle entries.
- ``severity``: cosmetic grading.

Algorithm constants tuned by hand on real selfies (Frangi sigmas, hysteresis
floors, exclusion margins, severity thresholds) live next to the code that
uses them and are NOT exposed via ``app.config``. Treat them as part of the
algorithm definition.
"""
