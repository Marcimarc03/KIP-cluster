"""Data handling for the KIP benchmark: manifest, tiling, splits, conversion.

RECONSTRUCTED 2026-07-07: the original ``kip/data/`` was never committed
because the repo's ``.gitignore`` pattern ``data/`` (no leading slash) also
matched ``kip/data/``. This reconstruction follows the interfaces specified
in ``docs/BUILD_PLAN.md`` (section 2) and is validated by the committed test
suite (``tests/``) plus the committed reference manifest under
``results/defect_detection/manifest/``.
"""
