"""
Palantir: Pseudotime and lineage commitment inference
======================================================

This module implements the Palantir algorithm for inferring pseudotime
and cell fate probabilities from single-cell RNA-seq data.

Reference: Setty et al., Nature Biotechnology 2019
https://www.nature.com/articles/s41587-019-0068-4

The implementation is equivalent to scanpy.external.tl.palantir
"""

from .core import (
    palantir,
    palantir_results,
)
from .utils import (
    early_cell,
    find_terminal_states,
)

__all__ = [
    "palantir",
    "palantir_results",
    "early_cell",
    "find_terminal_states",
]
