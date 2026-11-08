"""Protein residues with explicit hydrogens, protonation and HIS tautomers.

The builder's original parser (``build._group_residues``) is atom-only: it has
no bonds, no hydrogens and no protonation state, so it cannot support
angle-aware H-bonds or backbone donors/acceptors. This module uses RDKit once
per complex to add those missing pieces:

* pH-7 formal charges before ``AddHs`` (RDKit otherwise leaves GLU as COOH and
  LYS as NH2 -- verified on 1ax9);
* added hydrogens carry their parent's residue identity;
* histidine is not forced into RDKit's default tautomer. RDKit puts the H on
  NE2 (HIE) for every HIS, which is arbitrary. Both neutral tautomers are
  produced and the energy stage keeps the better one.

Everything here is optional: ``build.py`` falls back to the original heavy-atom
parser if RDKit cannot parse a structure.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
from rdkit import Chem

from .geometry import residue_code


@dataclass
class AtomH:
    """Minimal atom record: enough for features and for H-bond geometry."""

    name: str
    element: str
    x: float
    y: float
    z: float
    is_hetero: bool = False
    is_hydrogen: bool = False
    parent: int = -1
    neighbors: tuple[int, ...] = ()


@dataclass
class ProteinResidue:
    res_name: str
    chain: str
    res_seq: str
    coords: np.ndarray
    code: int
    atoms: list[AtomH]
    variants: list[list[AtomH]]


def _residue_key(info: Chem.AtomPDBResidueInfo) -> tuple[str, int, str, str]:
    return (
        info.GetChainId().strip(),
        int(info.GetResidueNumber()),
        info.GetInsertionCode().strip(),
        info.GetResidueName().strip(),
    )


def _assign_protonation(mol: Chem.Mol) -> Chem.Mol:
    """Apply pH-7 formal charges to charged sidechains before adding H.

    Only the states that change the *number* of hydrogens matter here: a
    deprotonated carboxylate must not receive an O-H, and protonated lysine /
    arginine must receive their extra N-H. The function is deliberately
    conservative: if sanitization after the edits fails, the original neutral
    molecule is returned and H placement proceeds without the correction.
    """
    rw = Chem.RWMol(mol)
    by_res: "OrderedDict[tuple, dict[str, int]]" = OrderedDict()
    for atom in rw.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetIsHeteroAtom():
            continue
        by_res.setdefault(_residue_key(info), {})[
            info.GetName().strip()] = atom.GetIdx()

    for (chain, num, icode, res_name), names in by_res.items():
        if res_name in {"ASP", "GLU"}:
            oxygen_names = ("OD1", "OD2") if res_name == "ASP" else ("OE1", "OE2")
            carbon_name = "CG" if res_name == "ASP" else "CD"
            o_idxs = [names[n] for n in oxygen_names if n in names]
            target = next(
                (i for i in o_idxs
                 if rw.GetAtomWithIdx(i).GetTotalNumHs() > 0), None)
            if target is None and carbon_name in names:
                for i in o_idxs:
                    bond = rw.GetBondBetweenAtoms(names[carbon_name], i)
                    if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                        target = i
                        break
            if target is not None:
                rw.GetAtomWithIdx(target).SetFormalCharge(-1)

        elif res_name == "LYS" and "NZ" in names:
            rw.GetAtomWithIdx(names["NZ"]).SetFormalCharge(1)

        elif res_name == "ARG":
            terminal = [(rw.GetAtomWithIdx(names[n]).GetTotalNumHs(), n,
                         names[n])
                        for n in ("NH1", "NH2") if n in names]
            if terminal:
                # The imine N is the one already carrying a single H; giving
                # it +1 lets RDKit add the protonation H without turning the
                # other terminal N into an ammonium.
                terminal.sort(key=lambda item: item[0])
                rw.GetAtomWithIdx(terminal[0][2]).SetFormalCharge(1)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return mol
    return out


def _place_tautomer_h(xyz: np.ndarray, n_idx: int,
                      neighbors: tuple[int, ...], bond: float = 1.01) -> np.ndarray:
    """H position on a planar pyrrole-like N: external bisector of its bonds."""
    n = xyz[n_idx]
    vectors = []
    for j in neighbors:
        v = xyz[j] - n
        norm = float(np.linalg.norm(v))
        if norm > 0:
            vectors.append(v / norm)
    direction = -np.sum(vectors, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.array([0.0, 0.0, 1.0])
        norm = 1.0
    return n + bond * direction / norm


def _his_variants(res_name: str, atoms: list[AtomH]) -> list[list[AtomH]]:
    """Both neutral histidine tautomers, or ``[atoms]`` for other residues.

    HID puts the H on ND1 and leaves NE2 as the acceptor; HIE is the reverse.
    A single placeholder H is moved between the two ring nitrogens, so the two
    atom lists differ in exactly one entry.
    """
    if res_name != "HIS":
        return [atoms]
    by_name = {a.name: i for i, a in enumerate(atoms)}
    if "ND1" not in by_name or "NE2" not in by_name:
        return [atoms]
    nd1, ne2 = by_name["ND1"], by_name["NE2"]

    h_slot = next(
        (i for i, a in enumerate(atoms)
         if a.is_hydrogen and a.parent in {nd1, ne2}), None)
    base = list(atoms)
    if h_slot is None:
        base.append(AtomH(name="H", element="H", x=0.0, y=0.0, z=0.0,
                          is_hydrogen=True, parent=nd1))
        h_slot = len(base) - 1

    xyz = np.array([[a.x, a.y, a.z] for a in base], dtype=np.float64)
    variants = []
    for n_idx in (nd1, ne2):
        pos = _place_tautomer_h(xyz, n_idx, base[n_idx].neighbors)
        variant = list(base)
        variant[h_slot] = replace(base[h_slot], parent=n_idx,
                                  x=float(pos[0]), y=float(pos[1]),
                                  z=float(pos[2]))
        variants.append(variant)
    return variants


def protein_residues(block: str) -> list[ProteinResidue] | None:
    """Parse standard residues with hydrogens; ``None`` if RDKit cannot."""
    try:
        mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        mol = _assign_protonation(mol)
    except Exception:
        pass
    try:
        molh = Chem.AddHs(mol, addCoords=True)
    except Exception:
        return None

    heavy: "OrderedDict[tuple, list[int]]" = OrderedDict()
    for atom in molh.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetIsHeteroAtom():
            continue
        heavy.setdefault(_residue_key(info), []).append(atom.GetIdx())

    hydrogens: dict[tuple, list[tuple[int, int]]] = {}
    for atom in molh.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        nbrs = atom.GetNeighbors()
        if len(nbrs) != 1 or nbrs[0].GetAtomicNum() == 1:
            continue
        parent = nbrs[0]
        info = parent.GetPDBResidueInfo()
        if info is None or info.GetIsHeteroAtom():
            continue
        hydrogens.setdefault(_residue_key(info), []).append(
            (atom.GetIdx(), parent.GetIdx()))

    conformer = molh.GetConformer()
    residues: list[ProteinResidue] = []
    for key, heavy_idxs in heavy.items():
        chain, _num, _icode, res_name = key
        res_seq = f"{key[1]}{key[2]}"
        global_to_local = {g: i for i, g in enumerate(heavy_idxs)}

        atoms: list[AtomH] = []
        for g in heavy_idxs:
            atom = molh.GetAtomWithIdx(g)
            pos = conformer.GetAtomPosition(g)
            atoms.append(AtomH(
                name=atom.GetPDBResidueInfo().GetName().strip(),
                element=atom.GetSymbol(),
                x=float(pos.x), y=float(pos.y), z=float(pos.z),
            ))
        for local, g in enumerate(heavy_idxs):
            atom = molh.GetAtomWithIdx(g)
            nb = tuple(global_to_local[n.GetIdx()]
                       for n in atom.GetNeighbors()
                       if n.GetAtomicNum() > 1 and n.GetIdx() in global_to_local)
            atoms[local] = replace(atoms[local], neighbors=nb)

        for g, parent_g in hydrogens.get(key, ()):
            pos = conformer.GetAtomPosition(g)
            parent_local = global_to_local.get(parent_g, -1)
            if parent_local < 0:
                continue
            atoms.append(AtomH(
                name="H", element="H",
                x=float(pos.x), y=float(pos.y), z=float(pos.z),
                is_hydrogen=True, parent=parent_local,
            ))

        ca = next((a for a in atoms if a.name == "CA" and not a.is_hydrogen),
                  None)
        if ca is not None:
            coords = np.array([ca.x, ca.y, ca.z], dtype=np.float32)
        else:
            heavy_xyz = np.array(
                [[a.x, a.y, a.z] for a in atoms if not a.is_hydrogen],
                dtype=np.float32)
            coords = heavy_xyz.mean(axis=0)

        variants = _his_variants(res_name, atoms)
        residues.append(ProteinResidue(
            res_name=res_name, chain=chain, res_seq=res_seq,
            coords=coords, code=residue_code(res_name),
            atoms=variants[0], variants=variants,
        ))
    return residues
