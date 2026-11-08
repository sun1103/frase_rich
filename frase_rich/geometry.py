"""Residue encoding, triplet enumeration, and pairwise bias terms.

The 11-bit residue code and the 33-bit triplet key are kept byte-compatible
with the existing FRASE index (``fragenv/src/triplet.py``, transcribed there
from ``kireevlab/FRASE-bot-RDKit``). Keeping the key identical lets the old and
new indexes be compared and merged without a remapping.

The new pieces are the continuous terms borrowed from the attention score:

* ``rbf`` / ``side_lengths``  -- term 3, the distance potential.
* ``pair_bias``               -- term 2, the residue-residue relation.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

# ── paper-exact 11-bit residue encoding ─────────────────────────────────────
N_BITS = 11

_IONPOS = {"ARG", "LYS", "HIS", "HIP", "HID", "HIE"}
_IONNEG = {"ASP", "GLU"}
_HBA = {"ASP", "GLU", "SER", "THR", "ASN", "GLN", "TYR", "HIS", "HIP", "HID", "HIE"}
_HBD = {"ARG", "LYS", "SER", "THR", "ASN", "GLN", "TYR", "HIS", "HIP", "HID", "HIE",
        "CYS", "TRP"}
_AR = {"PHE", "TYR", "HIS", "HIP", "HID", "HIE", "TRP"}

_HY = {
    "GLY": 0b000,
    "ALA": 0b100, "PRO": 0b100,
    "VAL": 0b110, "ILE": 0b110, "LEU": 0b110, "THR": 0b110, "CYS": 0b110,
    "MET": 0b111, "PHE": 0b111, "TYR": 0b111, "TRP": 0b111,
}
_VOL = {
    "GLY": 0b000,
    "ALA": 0b100, "PRO": 0b100,
    "VAL": 0b110, "ILE": 0b110, "LEU": 0b110, "THR": 0b110, "CYS": 0b110,
    "SER": 0b110, "ASN": 0b110, "ASP": 0b110, "HIS": 0b110, "HIP": 0b110,
    "HID": 0b110, "HIE": 0b110,
    "MET": 0b111, "PHE": 0b111, "TYR": 0b111, "TRP": 0b111, "ARG": 0b111,
    "LYS": 0b111, "GLU": 0b111, "GLN": 0b111,
}


def residue_code(res_name: str) -> int:
    """Encode one residue as an 11-bit integer; unknown residues -> 0."""
    rn = res_name.strip().upper()
    code = 0
    if rn in _IONPOS:
        code |= 1 << 0
    if rn in _IONNEG:
        code |= 1 << 1
    if rn in _HBA:
        code |= 1 << 2
    if rn in _HBD:
        code |= 1 << 3
    if rn in _AR:
        code |= 1 << 4
    code |= _HY.get(rn, 0b000) << 5
    code |= _VOL.get(rn, 0b000) << 8
    return code


# ── triplet enumeration and fingerprint ────────────────────────────────────

def enumerate_triplet_idxs(
    pts: np.ndarray,
    *,
    d_lo: float = 8.0,
    d_hi: float = 12.0,
) -> list[tuple[int, int, int]]:
    """Index triples whose three pairwise sides all lie in [d_lo, d_hi]."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return []
    diff = pts[:, None, :] - pts[None, :, :]
    dmat = np.sqrt((diff * diff).sum(axis=-1))
    tri = np.array(list(itertools.combinations(range(n), 3)), dtype=np.intp)
    if tri.size == 0:
        return []
    i, j, k = tri[:, 0], tri[:, 1], tri[:, 2]
    dij, dik, djk = dmat[i, j], dmat[i, k], dmat[j, k]
    ok = ((dij >= d_lo) & (dij <= d_hi)
          & (dik >= d_lo) & (dik <= d_hi)
          & (djk >= d_lo) & (djk <= d_hi))
    return [tuple(t) for t in tri[ok].tolist()]


def canonical_triplet_fp(codes: tuple[int, int, int]) -> int:
    """Order-invariant 33-bit key: sorted 11-bit codes packed together."""
    a, b, c = sorted(codes, reverse=True)
    return (a << (2 * N_BITS)) | (b << N_BITS) | c


def side_lengths(pts: np.ndarray, tri: tuple[int, int, int]) -> np.ndarray:
    i, j, k = tri
    return np.array([
        np.linalg.norm(pts[i] - pts[j]),
        np.linalg.norm(pts[i] - pts[k]),
        np.linalg.norm(pts[j] - pts[k]),
    ], dtype=np.float64)


def rbf(value: float | np.ndarray, *, center: float, sigma: float) -> float | np.ndarray:
    """Gaussian radial basis function; the smooth analog of a distance bin."""
    return np.exp(-0.5 * ((np.asarray(value, dtype=np.float64) - center) / sigma) ** 2)


def seq_number(res_seq: str) -> int:
    """Residue number from strings like ``93A`` (insertion codes ignored)."""
    digits = re.sub(r"\D", "", str(res_seq))
    return int(digits) if digits else 0


def pair_bias(
    chain_i: str,
    seq_i: int,
    chain_j: str,
    seq_j: int,
    distance: float,
    *,
    distance_center: float = 10.0,
    distance_sigma: float = 2.0,
) -> float:
    """Residue-residue relation bias (term 2).

    Two ingredients, both cheap and query-free:

    * topology: residues far apart in sequence (or on different chains) are
      more distinctive than an i,i+1 pair from the same helix;
    * geometry: a soft preference for the middle of the 8-12 A band.

    The topology factor stays in [0.6, 1.0] rather than zeroing local pairs --
    this is a bias, not a cutoff.
    """
    if chain_i != chain_j:
        seq_w = 1.0
    else:
        sep = abs(int(seq_i) - int(seq_j))
        seq_w = 0.6 + 0.4 * (1.0 - np.exp(-sep / 3.0))
    return float(seq_w * rbf(distance, center=distance_center, sigma=distance_sigma))


def triplet_bias(
    pts: np.ndarray,
    tri: tuple[int, int, int],
    chains: list[str],
    seqs: list[int],
) -> float:
    """Mean pair bias over the three sides of a triplet."""
    i, j, k = tri
    b_ij = pair_bias(chains[i], seqs[i], chains[j], seqs[j],
                     float(np.linalg.norm(pts[i] - pts[j])))
    b_ik = pair_bias(chains[i], seqs[i], chains[k], seqs[k],
                     float(np.linalg.norm(pts[i] - pts[k])))
    b_jk = pair_bias(chains[j], seqs[j], chains[k], seqs[k],
                     float(np.linalg.norm(pts[j] - pts[k])))
    return (b_ij + b_ik + b_jk) / 3.0
