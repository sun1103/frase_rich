"""PDB IO, geometry helpers, atom typing — no BioPython dependency.

Vendored from the companion ``fragenv`` tree (``src/utils.py``) so this
repository is standalone. Only the PDB parsing and atom-typing helpers are
used here; the file is kept intact to avoid a second source of truth.
"""

from __future__ import annotations

import re
import gzip
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from scipy.spatial import KDTree

# ── constants ──────────────────────────────────────────────────────────────

# three-letter codes for standard amino acids
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# Eisenberg normalized hydrophobicity scale (0–1, higher = more hydrophobic)
HYDROPHOBICITY = {
    "ALA": 0.62, "ARG": 0.00, "ASN": 0.06, "ASP": 0.00, "CYS": 0.68, "GLN": 0.06,
    "GLU": 0.00, "GLY": 0.50, "HIS": 0.17, "ILE": 1.00, "LEU": 0.97, "LYS": 0.00,
    "MET": 0.74, "PHE": 1.00, "PRO": 0.29, "SER": 0.06, "THR": 0.14, "TRP": 0.81,
    "TYR": 0.63, "VAL": 0.91,
}

# residue → single-letter code (needed for some applications)
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# atom name → element (heuristic fallback)
def _guess_element(atom_name: str) -> str:
    """Guess element from PDB atom name."""
    name = atom_name.strip()
    if not name:
        return "C"
    # Two-char elements (FE, CL, BR, MG, MN, ZN, etc.)
    if len(name) >= 2 and name[:2].upper() in {
        "FE", "CL", "BR", "MG", "MN", "ZN", "CA", "NA", "K ",
        "CU", "NI", "CO", "CD", "HG", "PT", "AU", "SE", "LI",
    }:
        return name[:2].upper()
    # Single-char element
    c = name[0].upper()
    if c in "CNOPSFIH":
        return c
    return c if c.isalpha() else "C"


# ── lightweight PDB parser ─────────────────────────────────────────────────

@dataclass
class ProteinAtom:
    serial: int
    name: str          # atom name (e.g. "CA", "NE2")
    res_name: str      # residue name (e.g. "ALA", "HEM")
    chain: str
    res_seq: str       # residue sequence + insertion code (e.g. "93A")
    x: float
    y: float
    z: float
    element: str
    is_hetero: bool

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)


def parse_pdb(filepath: str, *, model: int = 1) -> list[ProteinAtom]:
    """Parse a PDB (or .pdb.gz) file, returning all ATOM/HETATM records.

    Handles multi-model files by extracting *model* (1-indexed).
    Handles alternate conformations by taking the 'A' (or first) altLoc.
    """
    atoms: list[ProteinAtom] = []
    open_fn = gzip.open if str(filepath).endswith(".gz") else open
    current_model = 1
    in_target_model = True

    with open_fn(filepath, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("MODEL "):
                current_model = int(line[10:14])
                in_target_model = (current_model == model)
                if current_model > model:
                    break
                continue
            if line.startswith("ENDMDL"):
                in_target_model = False
                continue
            if not in_target_model:
                continue
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue

            # Skip alternate conformations that are not A / blank / 1
            alt_loc = line[16]
            if alt_loc not in {" ", "A", "1"}:
                continue

            is_hetero = line.startswith("HETATM")
            serial = int(line[6:11])
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21]
            res_seq = line[22:26].strip() + line[26].strip()  # resSeq + iCode
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() or _guess_element(atom_name)

            atoms.append(ProteinAtom(
                serial=serial,
                name=atom_name,
                res_name=res_name,
                chain=chain,
                res_seq=res_seq,
                x=x,
                y=y,
                z=z,
                element=element,
                is_hetero=is_hetero,
            ))
    return atoms


def atoms_to_array(atoms: list[ProteinAtom]) -> np.ndarray:
    """Return (N, 3) float32 coordinate array."""
    return np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)


# ── PDB line formatting ────────────────────────────────────────────────────

def _pdb_element_slot(element: str) -> str:
    e = element.strip()
    if not e:
        return "  "  # two blanks
    if len(e) == 1:
        return f" {e}"
    return e[:2].upper()


def format_atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain: str,
    res_seq: str,
    x: float,
    y: float,
    z: float,
    element: str = "",
    *,
    is_hetero: bool = False,
    occupancy: float = 1.0,
    temp_factor: float = 0.0,
) -> str:
    """Format a PDB v3 ATOM/HETATM record."""
    rec = "HETATM" if is_hetero else "ATOM  "
    res_seq_num = res_seq[:4]  # first 4 chars = residue number
    res_seq_icode = res_seq[4] if len(res_seq) > 4 else " "

    line = (
        f"{rec}{serial:>5d}  {atom_name:<3s}"
        f" {res_name:>3s} {chain}{res_seq_num:>4s}{res_seq_icode}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{temp_factor:6.2f}          "
        f"{_pdb_element_slot(element):>2s}"
    )
    assert len(line) <= 80, f"PDB line too long ({len(line)}): {line}"
    return line


def write_pdb(
    filepath: str,
    fragment_atoms: list[dict],
    shell_atoms: dict[str, list[ProteinAtom]],
    metadata: dict,
):
    """Write a composite PDB with fragment + shell atoms.

    Parameters
    ----------
    fragment_atoms : list of dict with keys serial, name, x, y, z, element
        3D coordinates of BRICS fragment atoms.
    shell_atoms : dict
        Keys "A" (0-4Å), "B" (4-8Å), "C" (8-12Å); values are lists of ProteinAtom.
    metadata : dict
        Written as REMARK lines (JSON string).
    """
    with open(filepath, "w") as fh:
        # Header remarks
        fh.write("REMARK  Fragment-Microenvironment pair\n")
        for key, val in metadata.items():
            fh.write(f"REMARK  {key}: {val}\n")
        fh.write(
            "REMARK  Chains: F=fragment, A=0-4A shell, B=4-8A shell, C=8-12A shell\n"
        )

        serial = 1
        # Fragment atoms — chain F, residue FRG
        for i, fatom in enumerate(fragment_atoms):
            line = format_atom_line(
                serial=serial,
                atom_name=fatom.get("name", f"C{i+1:02d}"),
                res_name="FRG",
                chain="F",
                res_seq="1",
                x=fatom["x"],
                y=fatom["y"],
                z=fatom["z"],
                element=fatom.get("element", "C"),
                is_hetero=True,
            )
            fh.write(line + "\n")
            serial += 1

        # Shell atoms — chains A, B, C
        for chain_id in ("A", "B", "C"):
            for atom in shell_atoms.get(chain_id, []):
                line = format_atom_line(
                    serial=serial,
                    atom_name=atom.name,
                    res_name=atom.res_name,
                    chain=chain_id,
                    res_seq=atom.res_seq,
                    x=atom.x,
                    y=atom.y,
                    z=atom.z,
                    element=atom.element,
                    is_hetero=atom.is_hetero,
                )
                fh.write(line + "\n")
                serial += 1

        fh.write("END\n")


# ── geometry helpers ───────────────────────────────────────────────────────

def centroid(coords: np.ndarray) -> np.ndarray:
    """Centroid of (N, 3) array."""
    return coords.mean(axis=0)


def query_shells(
    query_points: np.ndarray,
    protein_coords: np.ndarray,
    protein_atoms: list[ProteinAtom],
    radii: tuple[float, float, float] = (4.0, 8.0, 12.0),
) -> dict[str, list[ProteinAtom]]:
    """Extract protein atoms in concentric shells around query_points.

    Returns dict with keys "A", "B", "C" for the three shell layers.
    Atoms in inner shells are excluded from outer shells.
    """
    tree = KDTree(protein_coords)
    results: dict[str, list[ProteinAtom]] = {"A": [], "B": [], "C": []}
    chain_keys = ("A", "B", "C")

    prev_indices: set[int] = set()
    cumulative: set[int] = set()

    for i, r in enumerate(radii):
        idx_list = tree.query_ball_point(query_points, r)
        all_in_radius: set[int] = set()
        for idxs in idx_list:
            all_in_radius.update(idxs)
        shell_set = all_in_radius - cumulative
        results[chain_keys[i]] = [protein_atoms[j] for j in sorted(shell_set)]
        cumulative.update(shell_set)

    return results


def pca_eigenvalues(coords: np.ndarray) -> np.ndarray:
    """Return sorted eigenvalues (λ1, λ2, λ3) of the coordinate covariance matrix.

    Returns zeros if fewer than 3 points. The ratio λ1/λ2/λ3 encodes local shape:
      λ1≈λ2≈λ3 → spherical pocket
      λ1≈λ2≫λ3 → flat surface / shallow cleft
      λ1≫λ2≈λ3 → narrow channel / groove
    """
    n = coords.shape[0]
    if n < 3:
        return np.zeros(3, dtype=np.float32)
    centered = coords - coords.mean(axis=0)
    cov = centered.T @ centered / (n - 1)
    eigvals = np.linalg.eigvalsh(cov)
    # Sort descending
    return np.sort(eigvals)[::-1].astype(np.float32)


# ── residue typing ─────────────────────────────────────────────────────────

def classify_residue(res_name: str) -> str:
    """Classify a residue into one of: AA code, 'HOH', 'METAL', 'HET'."""
    name = res_name.strip().upper()
    if name in STANDARD_AA:
        return name
    if name == "HOH":
        return "HOH"
    # Common metal ions and cofactors
    if name in {
        "ZN", "MG", "MN", "CA", "FE", "CU", "NI", "CO", "CD", "NA",
        "K ", "ZN2", "MG2", "CA2", "FE2", "HEM", "HEC", "FAD", "NAD",
        "NAP", "ATP", "ADP", "GDP", "GTP", "SAM", "COA",
    }:
        return "METAL" if len(name) <= 3 else "HET"
    return "HET"


def classify_atom_type(element: str) -> int:
    """Map element to integer class for atom-type counting."""
    e = element.strip().upper()[:2]  # e.g. "FE", "CL", "BR"
    if e in {"C", "CA"}:
        return 0
    if e in {"N", "NA"}:
        return 1
    if e in {"O", "OX"}:
        return 2
    if e == "S":
        return 3
    if e == "P":
        return 4
    if e in {"FE", "ZN", "MG", "MN", "CA", "CU", "NI", "CO", "CD",
             "NA", "K ", "PT", "AU", "HG", "LI"}:
        return 5
    if e in {"F", "CL", "BR", "I ", "SE"}:
        return 6
    return 7  # other


# ── Dipole moment utilities ─────────────────────────────────────────────────

def compute_dipole_vector(mol, conf_id=-1, assign_charges=True):
    """Compute dipole moment vector (dx, dy, dz) from Gasteiger charges + 3D coords.

    RDKit < 2024.03 lacks AllChem.ComputeMolDipole, so we compute it manually:
    dipole = Σ q_i * (r_i - r_center) where q_i are Gasteiger partial charges
    and r_center is the geometric center of heavy atoms.

    Returns (dx, dy, dz) as np.ndarray (e·Å units), or None on failure.
    """
    from rdkit.Chem import AllChem
    try:
        if assign_charges:
            AllChem.ComputeGasteigerCharges(mol)
        conf = mol.GetConformer(conf_id)

        # Collect positions and charges of heavy atoms
        positions = []
        charges = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # skip dummy atoms
                continue
            idx = atom.GetIdx()
            try:
                q = float(atom.GetProp('_GasteigerCharge'))
            except KeyError:
                continue
            pos = conf.GetAtomPosition(idx)
            positions.append([pos.x, pos.y, pos.z])
            charges.append(q)

        if not charges:
            return None

        positions = np.array(positions)
        charges = np.array(charges)

        # Center at geometric mean to remove spurious net-charge dipole
        center = positions.mean(axis=0)
        rel = positions - center
        dip = np.dot(charges, rel)  # shape (3,)
        return dip
    except Exception:
        return None


def dipole_alignment(dip1, dip2):
    """Cosine similarity between two dipole vectors.

    Returns a float in [-1, 1]. 1.0 = perfectly aligned, 0.0 = orthogonal,
    -1.0 = perfectly anti-aligned. Returns 0.0 if either vector is near-zero
    (no meaningful dipole).
    """
    if dip1 is None or dip2 is None:
        return 0.0
    mag1 = float(np.linalg.norm(dip1))
    mag2 = float(np.linalg.norm(dip2))
    if mag1 < 1e-4 or mag2 < 1e-4:
        return 0.0
    cos_sim = float(np.dot(dip1, dip2) / (mag1 * mag2))
    return max(-1.0, min(1.0, cos_sim))
