"""Fragment chemistry: explicit hydrogens and EEM partial charges.

WHY hydrogens are part of the package: EEM does not keep the formal charge on
the ionized atom. On a quaternary ammonium it puts the same +1 on the N-H
hydrogens (N about -0.97, each H about +0.37), which is the physical partial
charge distribution. An energy that only sees heavy atoms therefore scores the
salt bridge backwards, so hydrogen coordinates and charges must travel
together.

WHY obabel: EEM is cheap (no QM) and already installed. The batch helper keeps
one subprocess per complex rather than per fragment. Charges are mapped back
to RDKit atom order by nearest 3D coordinate, because neither obabel nor RDKit
guarantees a shared atom ordering between formats.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

DEFAULT_OBABEL = os.environ.get("OBABEL", "/opt/homebrew/bin/obabel")


def with_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Add hydrogens with coordinates from the existing 3D conformer."""
    if mol.GetNumConformers() == 0:
        raise ValueError("molecule has no 3D conformer")
    return Chem.AddHs(mol, addCoords=True)


def formal_net_charge(mol: Chem.Mol) -> int:
    return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))


def _split_mol2(text: str) -> list[str]:
    """Split a multi-record MOL2 file into per-molecule blocks."""
    marker = "@<TRIPOS>MOLECULE"
    chunks = []
    for part in text.split(marker):
        if not part.strip():
            continue
        chunks.append(marker + part)
    return chunks


def _map_by_coordinate(src: np.ndarray, dst: np.ndarray, tol: float = 1e-3):
    """Return ``dst`` index for every ``src`` row; require a bijection."""
    if src.shape != dst.shape:
        raise ValueError(f"coordinate shape mismatch: {src.shape} vs {dst.shape}")
    d = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=-1)
    mapping = np.argmin(d, axis=1)
    picked = d[np.arange(len(src)), mapping]
    if float(picked.max()) > tol:
        raise ValueError(
            f"coordinate mapping failed (max displacement {picked.max():.2e})")
    if len(set(mapping.tolist())) != len(mapping):
        raise ValueError("coordinate mapping is not one-to-one")
    return mapping


def _read_mol2_charges(chunk: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords, partial charges) for one MOL2 record."""
    mol = Chem.MolFromMol2Block(chunk, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("RDKit could not parse the MOL2 record")
    if mol.GetNumConformers() == 0:
        raise ValueError("MOL2 record has no conformer")
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    charges = []
    for atom in mol.GetAtoms():
        if atom.HasProp("_TriposPartialCharge"):
            charges.append(atom.GetDoubleProp("_TriposPartialCharge"))
        else:
            raise ValueError("MOL2 record has no partial charges")
    return xyz, np.asarray(charges, dtype=np.float64)


def eem_charges_batch(
    mols: list[Chem.Mol],
    *,
    obabel: str = DEFAULT_OBABEL,
    timeout: float = 180.0,
) -> list[np.ndarray]:
    """EEM partial charges for several molecules in one obabel call.

    Each molecule must have explicit hydrogens and a conformer. Charges are
    returned in the input atom order and offset so that every molecule sums to
    its formal net charge; obabel's SDF -> MOL2 round trip is otherwise neutral
    about rounding.
    """
    if not mols:
        return []
    for mol in mols:
        if mol.GetNumConformers() == 0:
            raise ValueError("every molecule needs a 3D conformer")

    with tempfile.TemporaryDirectory() as tmp:
        sdf = os.path.join(tmp, "batch.sdf")
        mol2 = os.path.join(tmp, "batch.mol2")
        writer = Chem.SDWriter(sdf)
        for mol in mols:
            writer.write(mol)
        writer.close()

        proc = subprocess.run(
            [obabel, sdf, "-O", mol2, "--partialcharge", "eem"],
            capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not os.path.isfile(mol2):
            raise RuntimeError(f"obabel EEM failed: {proc.stderr.strip()}")
        with open(mol2) as handle:
            chunks = _split_mol2(handle.read())

    if len(chunks) != len(mols):
        raise RuntimeError(
            f"obabel returned {len(chunks)} molecules for {len(mols)} inputs")

    out = []
    for mol, chunk in zip(mols, chunks):
        conf = mol.GetConformer()
        src_xyz = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        dst_xyz, dst_q = _read_mol2_charges(chunk)
        order = _map_by_coordinate(src_xyz, dst_xyz)
        q = dst_q[order].copy()
        q += (formal_net_charge(mol) - q.sum()) / len(q)
        out.append(q)
    return out


def gasteiger_charges(mol: Chem.Mol) -> np.ndarray:
    """Gasteiger fallback, offset to the formal net charge."""
    work = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(work)
    q = np.array([
        work.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge")
        for i in range(work.GetNumAtoms())
    ], dtype=np.float64)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q += (formal_net_charge(mol) - q.sum()) / len(q)
    return q


def charges_for(
    mol: Chem.Mol,
    *,
    method: str = "eem",
    strict: bool = False,
    obabel: str = DEFAULT_OBABEL,
) -> tuple[np.ndarray, str]:
    """Return ``(charges, method_used)`` in the molecule's atom order.

    ``method`` is ``eem``, ``gasteiger`` or ``formal``. EEM failure falls back
    to Gasteiger unless ``strict``; the caller records the method actually
    used so a mixed build is visible rather than silent.
    """
    if method == "formal":
        return (np.array([float(a.GetFormalCharge()) for a in mol.GetAtoms()]),
                "formal")
    if method == "gasteiger":
        return gasteiger_charges(mol), "gasteiger"
    if method != "eem":
        raise ValueError(f"unknown charge method: {method}")

    try:
        return eem_charges_batch([mol], obabel=obabel)[0], "eem"
    except Exception:
        if strict:
            raise
        return gasteiger_charges(mol), "gasteiger"
