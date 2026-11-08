"""FRASE-bot ligand fragmentation (Nat. Commun. 15:5564, 2024).

An, Y. et al. In silico fragment-based discovery of CIB1-directed anti-tumor
agents by FRASE-bot. https://doi.org/10.1038/s41467-024-49892-9

Vendored from the companion ``fragenv`` tree (``src/fragment.py``) so this
repository is standalone; upstream remains the reference implementation.

The paper builds its FRASE library with Pipeline Pilot's "Enumerate Fragments":
every connected fragment obtainable by breaking single, non-cyclic bonds, where
alpha-atoms (the first acyclic atom attached to a ring) stay with their ring.
Only fragments of 50-300 Da are retained.

This module re-implements that rule in RDKit. By default it ENUMERATES all such
fragments (not a single disjoint partition): a ligand's rigid pieces (blocks,
joined by non-cuttable bonds) are the nodes of a graph whose edges are the
cuttable bonds, and every connected subset of blocks yields one fragment. Each
fragment is H-capped at its boundary, and its attachment count (number of cut
bonds at the boundary) tags it substituent/linker/hub.

Interpretation notes (each is a one-line flag in :func:`cuttable_bonds`):
* ``cut_biaryl`` (default True)  -- a single ring-ring bond (e.g. biphenyl) is
  non-cyclic and has no alpha-atom, so the literal rule breaks it. Set False to
  keep biaryls as a single fragment.
* ``cut_amide``  (default True)  -- an amide C-N bond is single and non-cyclic,
  so the literal rule breaks it. Set False to keep amides intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops


@dataclass
class PaperFragment:
    mol: Chem.Mol          # plain fragment, dummies stripped, H-capped
    smiles: str
    centroid: np.ndarray   # heavy-atom centroid in the parent ligand frame
    n_heavy: int
    n_attach: int          # number of cut bonds at this fragment's boundary


def _is_amide_bond(bond: Chem.Bond) -> bool:
    """True if ``bond`` is the C-N single bond of a C(=O)-N amide group."""
    u, v = bond.GetBeginAtom(), bond.GetEndAtom()
    c = u if u.GetAtomicNum() == 6 else (v if v.GetAtomicNum() == 6 else None)
    n = u if u.GetAtomicNum() == 7 else (v if v.GetAtomicNum() == 7 else None)
    if c is None or n is None:
        return False
    for nb in c.GetBonds():
        o = nb.GetOtherAtom(c)
        if o.GetAtomicNum() == 8 and nb.GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def cuttable_bonds(
    mol: Chem.Mol,
    *,
    cut_biaryl: bool = True,
    cut_amide: bool = True,
) -> list[int]:
    """Bond indices to break per the paper's rule.

    A bond is broken iff it is single, not in a ring, and not a ring<->alpha
    bond (one endpoint in a ring, the other acyclic).
    """
    cuts: list[int] = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        if b.IsInRing():
            continue
        u_ring = b.GetBeginAtom().IsInRing()
        v_ring = b.GetEndAtom().IsInRing()
        if u_ring != v_ring:
            continue  # alpha-bond: ring <-> acyclic substituent, keep
        if not cut_biaryl and u_ring and v_ring:
            continue  # ring <-> ring single bond (biaryl)
        if not cut_amide and _is_amide_bond(b):
            continue
        cuts.append(b.GetIdx())
    return cuts


def _strip_dummies(mol: Chem.Mol) -> Chem.Mol | None:
    """Cap [*] dummy atoms with H and sanitize; returns a fresh, sanitized mol.

    Dummies are replaced with explicit H (rather than removed) so that chiral
    centres do not end up as radicals after sanitization; the explicit H is then
    folded back into implicit H.
    """
    dummy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    rw = Chem.RWMol(mol)
    for d in dummy:
        rw.ReplaceAtom(d, Chem.Atom(1))
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return Chem.RemoveHs(out)
    except Exception:
        return None


def _blocks_and_edges(
    mol: Chem.Mol, cuts: list[int]
) -> tuple[list[list[int]], list[tuple[int, int, int, int, int]]]:
    """Contract non-cuttable bonds into rigid blocks; return block atoms + edges.

    Returns ``(blocks, cut_edges)`` where ``blocks[b]`` is the list of atom
    indices in block ``b``, and ``cut_edges`` is a list of
    ``(block_u, block_v, atom_u, atom_v, bond_idx)`` for each cuttable bond.
    """
    n = mol.GetNumAtoms()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cut_set = set(cuts)
    for b in mol.GetBonds():
        if b.GetIdx() not in cut_set:
            ra, rb = find(b.GetBeginAtomIdx()), find(b.GetEndAtomIdx())
            if ra != rb:
                parent[rb] = ra

    block_id: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        if r not in block_id:
            block_id[r] = len(block_id)

    nb = len(block_id)
    blocks: list[list[int]] = [[] for _ in range(nb)]
    for i in range(n):
        blocks[block_id[find(i)]].append(i)

    cut_edges: list[tuple[int, int, int, int, int]] = []
    for bidx in cuts:
        b = mol.GetBondWithIdx(bidx)
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bu, bv = block_id[find(u)], block_id[find(v)]
        if bu != bv:
            cut_edges.append((bu, bv, u, v, bidx))
    return blocks, cut_edges


def _extract_component(
    mol: Chem.Mol, atom_idxs: list[int], boundary_bonds: list[int]
) -> Chem.Mol | None:
    """Return the H-capped component containing ``atom_idxs``, cutting only the
    boundary bonds. Uses FragmentOnBonds so aromaticity/hybridization are
    inherited from the (sanitized) parent molecule.
    """
    if not boundary_bonds:
        return _strip_dummies(mol)
    split = rdmolops.FragmentOnBonds(mol, boundary_bonds, addDummies=True)
    frag_lists = Chem.GetMolFrags(split)
    frag_mols = Chem.GetMolFrags(split, asMols=True)
    for fl, fm in zip(frag_lists, frag_mols):
        if atom_idxs[0] in fl:
            return _strip_dummies(fm)
    return None


def _enumerate_fragments(
    mol: Chem.Mol,
    cuts: list[int],
    min_da: float,
    max_da: float,
    max_blocks: int | None = None,
    max_frags: int | None = None,
) -> list[PaperFragment]:
    """Enumerate every connected fragment over cuttable bonds (50-300 Da).

    ``max_frags`` caps how many fragments are emitted (the paper averages
    ~18 fragments per ligand; flexible "fragment factory" molecules can
    otherwise enumerate thousands of near-duplicate pieces).
    """
    blocks, cut_edges = _blocks_and_edges(mol, cuts)
    nb = len(blocks)
    if max_blocks is not None and nb > max_blocks:
        return []

    heavy_counts = [sum(1 for i in b if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
                    for b in blocks]
    max_heavy = int(max_da / 12.0) + 1  # any fragment with more heavy atoms is >max_da

    adj_masks = [0] * nb
    for bu, bv, _u, _v, _bidx in cut_edges:
        adj_masks[bu] |= 1 << bv
        adj_masks[bv] |= 1 << bu

    results: list[PaperFragment] = []
    seen: set[int] = set()

    def atom_union(sel: int) -> list[int]:
        out: list[int] = []
        for b in range(nb):
            if sel & (1 << b):
                out.extend(blocks[b])
        return out

    def boundary_bonds(sel: int) -> list[int]:
        bnd: list[int] = []
        for bu, bv, _u, _v, bidx in cut_edges:
            in_u = bool(sel & (1 << bu))
            in_v = bool(sel & (1 << bv))
            if in_u != in_v:
                bnd.append(bidx)
        return bnd

    def emit(sel: int) -> None:
        if max_frags is not None and len(results) >= max_frags:
            return
        if sel in seen:
            return
        seen.add(sel)
        atom_idxs = atom_union(sel)
        n_heavy = sum(heavy_counts[b] for b in range(nb) if sel & (1 << b))
        if n_heavy < 1 or n_heavy > max_heavy:
            return
        bnd = boundary_bonds(sel)
        n_attach = len(bnd)
        try:
            plain = _extract_component(mol, atom_idxs, bnd)
            if plain is None:
                return
        except Exception:
            return
        mw = Descriptors.MolWt(plain)
        if mw < min_da or mw > max_da:
            return
        conf = plain.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(a.GetIdx())) for a in plain.GetAtoms()],
            dtype=np.float32,
        )
        results.append(PaperFragment(
            mol=plain,
            smiles=Chem.MolToSmiles(plain),
            centroid=coords.mean(axis=0),
            n_heavy=plain.GetNumHeavyAtoms(),
            n_attach=n_attach,
        ))

    for root in range(nb):
        if heavy_counts[root] > max_heavy:
            continue
        root_heavy = heavy_counts[root]

        def rec(sel: int, cand: int, hc: int) -> None:
            if hc <= max_heavy:
                emit(sel)
            c = cand
            while c:
                v = (c & -c).bit_length() - 1
                c &= c - 1
                if v <= root:
                    continue
                if hc + heavy_counts[v] > max_heavy:
                    continue
                new_sel = sel | (1 << v)
                new_cand = (cand | adj_masks[v]) & ~new_sel
                rec(new_sel, new_cand, hc + heavy_counts[v])

        rec(1 << root, adj_masks[root], root_heavy)

    return results


def canonical_coords(mol: Chem.Mol, smiles: str) -> np.ndarray | None:
    """Heavy-atom coordinates of ``mol`` reordered to canonical-SMILES order.

    ``mol`` must carry a 3D conformer; ``smiles`` is ``Chem.MolToSmiles(mol)``.
    Returns an (N_atoms, 3) float32 array whose rows follow the atom order of
    ``Chem.MolFromSmiles(smiles)``, so the coordinates can be re-attached to a
    re-parsed SMILES molecule by index. Returns None on any mismatch.
    """
    canon = Chem.MolFromSmiles(smiles)
    if canon is None:
        return None
    if mol.GetNumAtoms() != canon.GetNumAtoms() or mol.GetNumConformers() == 0:
        return None
    match = mol.GetSubstructMatch(canon)
    if len(match) != canon.GetNumAtoms():
        return None
    conf = mol.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    return coords[list(match)]


def fragment_ligand(
    mol: Chem.Mol,
    *,
    min_da: float = 50.0,
    max_da: float = 300.0,
    cut_biaryl: bool = True,
    cut_amide: bool = True,
    enumerate: bool = True,
    max_blocks: int | None = 16,
    max_frags: int | None = None,
) -> list[PaperFragment]:
    """Fragment a ligand into 50-300 Da pieces per the paper's rule.

    ``mol`` must carry a 3D conformer (coordinates in the protein frame).
    ``enumerate=True`` (default) enumerates every connected fragment over the
    cuttable bonds (the paper's "Enumerate Fragments"); ``enumerate=False``
    cuts all cuttable bonds at once into a single disjoint partition.

    ``max_blocks`` guards against combinatorial explosion: ligands whose rigid
    blocks exceed this count (e.g. long polyol/PEG chains) are skipped.
    ``max_frags`` caps the number of emitted fragments per ligand.
    """
    cuts = cuttable_bonds(mol, cut_biaryl=cut_biaryl, cut_amide=cut_amide)
    if enumerate:
        return _enumerate_fragments(mol, cuts, min_da, max_da, max_blocks=max_blocks,
                                    max_frags=max_frags)

    if cuts:
        split = rdmolops.FragmentOnBonds(mol, cuts, addDummies=True)
        frag_mols = Chem.GetMolFrags(split, asMols=True)
    else:
        frag_mols = [mol]

    out: list[PaperFragment] = []
    for fm in frag_mols:
        n_attach = sum(1 for a in fm.GetAtoms() if a.GetAtomicNum() == 0)
        plain = _strip_dummies(fm)
        if plain is None:
            continue
        mw = Descriptors.MolWt(plain)
        if mw < min_da or mw > max_da:
            continue
        conf = plain.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(a.GetIdx())) for a in plain.GetAtoms()],
            dtype=np.float32,
        )
        centroid = coords.mean(axis=0)
        out.append(PaperFragment(
            mol=plain,
            smiles=Chem.MolToSmiles(plain),
            centroid=centroid,
            n_heavy=plain.GetNumHeavyAtoms(),
            n_attach=n_attach,
        ))
    return out
