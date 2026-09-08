"""RecGen paper evaluations (HB / ArtVIP) — standalone release.

Reproduces the RecGen paper's shape/pose evaluation table on the
HomebrewedDB (HB) and ArtVIP benchmarks with the released checkpoints.
See EVALUATION.md for install, data download, and run instructions.

Entry points:
    python -m recgen_eval.run        # run an evaluation (inference + metrics)
    python -m recgen_eval.aggregate  # (re-)aggregate per-sample metrics.json
    python -m recgen_eval.diameter   # DRE@0.05 diameter-error metric
    python -m recgen_eval.table      # summary vs the published numbers
"""

from .poseval6d.evaluation.evaluator import Evaluator

__all__ = ["Evaluator"]
