"""Rich (weighted) FRASE triplet index.

Self-contained: PDB parsing (:mod:`frase_rich.pdb_io`) and the paper
fragmentation rule (:mod:`frase_rich.fragmentation`) are vendored from the
companion ``fragenv`` tree.

See ``docs/WEIGHTED_INDEX.md`` for the design and the borrow from the
attention-score decomposition (content + pair bias + RBF potential).
"""

from .index import WeightedTripletIndex

__all__ = ["WeightedTripletIndex"]
