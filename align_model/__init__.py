"""Gap-aware alignment model: evidence, time map, diagnostics."""

from align_model.decompose import AlignmentReport, decompose
from align_model.pipeline import map_stem

__all__ = ["AlignmentReport", "decompose", "map_stem"]
