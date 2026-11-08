"""Paper fragmentation of 2D SMILES, with explicit exit vectors.

WHY this exists: the library fragmenter (``frase_rich/fragmentation.py``,
vendored from ``fragenv/src/fragment.py``) needs a 3D conformer only to record
fragment centroids. Patent compounds are 2D, so this module re-runs the
identical cut/block/enumerate logic without coordinates, reusing the same cut
rule and block contraction.

The 3D route recovers attachment atoms geometrically
(``features.attachment_atoms``: nearest-atom match to the parent). Here every
cut bond has a known atom on the fragment side, so exit vectors are read
straight off the cut edge. That is the difference that matters downstream:
a fragment row without exit vectors cannot be placed onto a query pose.

Note on atom indices: ``FragmentOnBonds`` keeps original indices and appends
dummies; ``GetMolFrags(asMols=True)`` renumbers the per-fragment mols, so the
index list returned by ``GetMolFrags`` (same order as the mols) is what maps a
fragment atom back to its parent atom.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops

from .build import _attach_class, _ring_class
from .fragmentation import _blocks_and_edges, _strip_dummies, cuttable_bonds


def _smiles_order(mol: Chem.Mol) -> list[int]:
    """Original atom index at each canonical-SMILES position.

    Call right after ``Chem.MolToSmiles(mol)``: that call is what writes the
    ``_smilesAtomOutputOrder`` property.
    """
    return [int(i) for i in
            mol.GetPropsAsDict(includePrivate=True,
                               includeComputed=True)["_smilesAtomOutputOrder"]]


# Mass pre-filter margin, Da. A fragment's mass is additive over its parent
# atoms (parent H count + one cap H per cut), so it can be bounded without
# building the fragment. The bound is approximate for charged/tautomeric atoms;
# the exact MolWt check still runs on every fragment that survives.
_MASS_MARGIN = 6.0


def _component_with_map(mol: Chem.Mol, atom_idxs: list[int],
                        boundary: list[int]) -> tuple[Chem.Mol, list[int],
                                                      list[int], list[int]] | None:
    """H-capped component + maps from it back to the parent molecule.

    Returns ``(plain, rank_to_parent, attach_plain, attach_parent)`` where
    ``rank_to_parent[i]`` is the parent atom index of ``plain`` atom ``i``,
    ``attach_plain`` lists the fragment atoms that sat on a cut bond (in
    ``plain`` numbering) and ``attach_parent`` the same in parent numbering.
    """
    if not boundary:
        return Chem.Mol(mol), list(range(mol.GetNumAtoms())), [], []
    try:
        split = rdmolops.FragmentOnBonds(mol, boundary, addDummies=True)
        frag_lists = Chem.GetMolFrags(split)
        frag_mols = Chem.GetMolFrags(split, asMols=True)
    except Exception:
        return None
    for fl, fm in zip(frag_lists, frag_mols):
        if atom_idxs[0] not in fl:
            continue
        # dummy -> its neighbor is the fragment atom on the cut bond
        attach_fm = [d.GetNeighbors()[0].GetIdx()
                     for d in fm.GetAtoms() if d.GetAtomicNum() == 0]
        plain = _strip_dummies(fm)
        if plain is None:
            return None
        # _strip_dummies replaces dummies with H then RemoveHs; surviving heavy
        # atoms keep their relative order.
        heavy = [i for i in range(fm.GetNumAtoms())
                 if fm.GetAtomWithIdx(i).GetAtomicNum() > 1]
        rank = {fm_i: k for k, fm_i in enumerate(heavy)}
        rank_to_parent = [int(fl[i]) for i in heavy]
        attach_plain = [rank[i] for i in attach_fm]
        attach_parent = [int(fl[i]) for i in attach_fm]
        return plain, rank_to_parent, attach_plain, attach_parent
    return None


def decompose_parent(
    smiles: str,
    *,
    min_da: float = 50.0,
    max_da: float = 300.0,
    cut_biaryl: bool = True,
    cut_amide: bool = True,
    max_blocks: int = 16,
    max_frags: int = 200,
) -> list[dict] | None:
    """Enumerate library-rule fragments of one parent SMILES.

    Defaults match ``build.py``. Returns ``None`` if the parent does not parse,
    otherwise one dict per emitted fragment with the H-capped canonical SMILES,
    heavy count, attachment count, attachment atoms in canonical numbering and
    in parent numbering, the parent partner atom of each cut bond, and the role
    / ring-class tags used by the index.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    cuts = cuttable_bonds(mol, cut_biaryl=cut_biaryl, cut_amide=cut_amide)
    blocks, cut_edges = _blocks_and_edges(mol, cuts)
    nb = len(blocks)
    if max_blocks is not None and nb > max_blocks:
        return []

    heavy_counts = [sum(1 for i in b if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
                    for b in blocks]
    max_heavy = int(max_da / 12.0) + 1
    block_mass = [sum(mol.GetAtomWithIdx(i).GetMass()
                      + mol.GetAtomWithIdx(i).GetTotalNumHs() * 1.008
                      for i in b) for b in blocks]

    adj_masks = [0] * nb
    for bu, bv, _u, _v, _bidx in cut_edges:
        adj_masks[bu] |= 1 << bv
        adj_masks[bv] |= 1 << bu

    out: list[dict] = []
    seen: set[int] = set()
    # The frontier (cand) at a state depends only on the selected blocks, so a
    # block set reached twice explores an identical subtree. Upstream's rule
    # only stops *emitting* duplicates; without this visited set a branched
    # block graph re-traverses the same states along every path and the run
    # explodes (observed: single ligands exceeding a day of CPU).
    visited: set[int] = set()

    def atom_union(sel: int) -> list[int]:
        atoms: list[int] = []
        for b in range(nb):
            if sel & (1 << b):
                atoms.extend(blocks[b])
        return atoms

    def boundary_and_attach(sel: int) -> tuple[list[int], list[int], list[int]]:
        bnd: list[int] = []
        inside: list[int] = []
        outside: list[int] = []
        for bu, bv, u, v, bidx in cut_edges:
            in_u = bool(sel & (1 << bu))
            if in_u != bool(sel & (1 << bv)):
                bnd.append(bidx)
                inside.append(u if in_u else v)
                outside.append(v if in_u else u)
        return bnd, inside, outside

    def emit(sel: int, hc: int, mass: float) -> None:
        if hc < 1 or hc > max_heavy:
            return
        # mass bound: parent atom + H masses, plus one cap H per cut bond
        if mass - _MASS_MARGIN > max_da:
            return
        if len(out) >= max_frags or sel in seen:
            return
        seen.add(sel)
        bnd, inside, outside = boundary_and_attach(sel)
        pred_mw = mass + 1.008 * len(bnd)
        if pred_mw + _MASS_MARGIN < min_da or pred_mw - _MASS_MARGIN > max_da:
            return
        atom_idxs = atom_union(sel)
        got = _component_with_map(mol, atom_idxs, bnd)
        if got is None:
            return
        plain, rank_to_parent, attach_plain, _attach_parent = got
        mw = Descriptors.MolWt(plain)
        if mw < min_da or mw > max_da:
            return
        smi = Chem.MolToSmiles(plain)
        order = _smiles_order(plain)
        canon_pos = [0] * plain.GetNumAtoms()
        for pos, old in enumerate(order):
            canon_pos[old] = pos
        out.append({
            "frag_smiles": smi,
            "n_heavy": plain.GetNumHeavyAtoms(),
            "n_attach": len(bnd),
            "attach_atoms": sorted(canon_pos[i] for i in attach_plain),
            "attach_parent": [rank_to_parent[i] for i in attach_plain],
            "attach_partner": [int(a) for a in outside],
            "role": _attach_class(len(bnd)),
            "ring_class": _ring_class(plain),
        })

    for root in range(nb):
        if heavy_counts[root] > max_heavy:
            continue

        def rec(sel: int, cand: int, hc: int, mass: float) -> None:
            if sel in visited:
                return
            visited.add(sel)
            if mass + _MASS_MARGIN >= min_da:
                emit(sel, hc, mass)
            c = cand
            while c:
                v = (c & -c).bit_length() - 1
                c &= c - 1
                if v <= root:
                    continue
                if hc + heavy_counts[v] > max_heavy:
                    continue
                # mass only grows as blocks are added, so this prunes the branch
                if mass + block_mass[v] - _MASS_MARGIN > max_da:
                    continue
                new_sel = sel | (1 << v)
                new_cand = (cand | adj_masks[v]) & ~new_sel
                rec(new_sel, new_cand, hc + heavy_counts[v], mass + block_mass[v])

        rec(1 << root, adj_masks[root], heavy_counts[root], block_mass[root])

    return out


__all__ = ["decompose_parent"]
