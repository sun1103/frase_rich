"""Build a rich (weighted) FRASE index from cached RCSB structures.

Input layout matches the settled BindingDB path:
``bindingdb_selected.tsv`` maps PDB id -> ligand SMILES; ``rcsb_cache/`` holds
the PDB files. No downloads are performed -- uncached ids are skipped.

For each fragment the *environment membership* is unchanged from the original
builder (12 A around the fragment centroid), so the A/B isolates one change:
the continuous relevance weights. Enriching membership (contacts instead of a
CA sphere) is a separate ablation, kept out to avoid confounding.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .chem import charges_for, eem_charges_batch, with_explicit_hydrogens
from .features import attachment_atoms, relevance_weights
from .fragmentation import canonical_coords, fragment_ligand
from .geometry import residue_code
from .index import EnvResidue, WeightedTripletIndex
from .pdb_io import parse_pdb
from .protein import protein_residues
from .weighting import (
    environment,
    fragment_fp_energy,
    fragment_fp_weights,
    pocket_environment,
    triplet_table,
)

META_COLS = ["frag_smiles", "pdb_id", "role", "tags", "n_heavy",
             "parent_smiles", "n_attach"]


@dataclass
class _Residue:
    res_name: str
    chain: str
    res_seq: str
    coords: np.ndarray
    code: int
    atoms: list


def _attach_class(n_attach: int) -> str:
    if n_attach == 0:
        return "whole"
    if n_attach == 1:
        return "substituent"
    if n_attach == 2:
        return "linker"
    return "hub"


def _ring_class(mol: Chem.Mol) -> str:
    n = mol.GetRingInfo().NumRings()
    if n == 0:
        return "acyclic"
    if n == 1:
        return "monocyclic"
    return "polycyclic"


def _group_residues(atoms: list) -> list[_Residue]:
    """Group protein (non-hetero) atoms, keeping all atoms per residue."""
    groups: dict[tuple[str, str, str], list] = {}
    for atom in atoms:
        if atom.is_hetero:
            continue
        key = (atom.chain, atom.res_seq, atom.res_name)
        groups.setdefault(key, []).append(atom)

    residues: list[_Residue] = []
    for (chain, res_seq, res_name), res_atoms in groups.items():
        ca = next(
            (a for a in res_atoms if a.name.strip().upper() == "CA"), None)
        if ca is not None:
            pt = np.array([ca.x, ca.y, ca.z], dtype=np.float32)
        else:
            pt = np.array(
                [[a.x, a.y, a.z] for a in res_atoms], dtype=np.float32).mean(axis=0)
        residues.append(_Residue(
            res_name=res_name,
            chain=chain,
            res_seq=res_seq,
            coords=pt,
            code=residue_code(res_name),
            atoms=res_atoms,
        ))
    return residues


# ── ligand extraction (same logic as fetch_bindingdb_complexes.py) ─────────

def _submol_from_idxs(raw: Chem.Mol, keep: list[int]) -> Chem.Mol:
    sub = Chem.RWMol(Chem.Mol())
    conf = Chem.Conformer(len(keep))
    newmap: dict[int, int] = {}
    for new_i, old_i in enumerate(sorted(keep)):
        atom = raw.GetAtomWithIdx(old_i)
        newmap[old_i] = sub.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        conf.SetAtomPosition(new_i, raw.GetConformer().GetAtomPosition(old_i))
    sub.AddConformer(conf)
    for bond in raw.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in newmap and j in newmap:
            sub.AddBond(newmap[i], newmap[j], bond.GetBondType())
    return sub.GetMol()


def _ligand_in_block(raw, template, n_smi: int) -> Chem.Mol | None:
    """First HETATM residue whose heavy-atom count matches and whose bonds
    the template accepts."""
    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for atom in raw.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or not info.GetIsHeteroAtom():
            continue
        name = info.GetResidueName().strip().upper()
        if name == "HOH":
            continue
        key = (name, info.GetChainId(), info.GetResidueNumber(),
               info.GetInsertionCode())
        groups.setdefault(key, []).append(atom.GetIdx())

    for idxs in groups.values():
        n_heavy = sum(
            1 for i in idxs if raw.GetAtomWithIdx(i).GetAtomicNum() > 1)
        if n_heavy != n_smi:
            continue
        lig = _submol_from_idxs(raw, idxs)
        if any(a.GetAtomicNum() == 1 for a in lig.GetAtoms()):
            # ``sanitize=False`` parsing keeps explicit H; the template has
            # none, so the assigned double bond would exceed valence.
            try:
                lig = Chem.RemoveHs(lig)
            except Exception:
                continue
        try:
            out = AllChem.AssignBondOrdersFromTemplate(template, lig)
            if out is None:
                continue
            Chem.SanitizeMol(out)
        except Exception:
            continue
        return Chem.RemoveHs(out)
    return None


def _find_ligand(block: str, smi: str) -> Chem.Mol | None:
    """Find the HETATM residue that AssignBondOrdersFromTemplate accepts.

    Proximity bonding first (the PDBbind files are read that way); a crowded
    docked pose can get extra short contacts, so CONECT connectivity is
    tried when proximity over-bonds and the template no longer matches.
    """
    template = Chem.MolFromSmiles(smi)
    if template is None:
        return None
    n_smi = template.GetNumHeavyAtoms()
    has_conect = any(line.startswith("CONECT") for line in block.splitlines())
    for proximity in ((True, False) if has_conect else (True,)):
        raw = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True,
                                   proximityBonding=proximity)
        if raw is None:
            continue
        out = _ligand_in_block(raw, template, n_smi)
        if out is not None:
            return out
    return None


# ── main build loop ─────────────────────────────────────────────────────────

def build_index(
    *,
    selected: str,
    cache: str,
    out: str,
    max_complexes: int | None = None,
    env_radius: float = 12.0,
    d_lo: float = 8.0,
    d_hi: float = 12.0,
    min_da: float = 50.0,
    max_da: float = 300.0,
    max_frags: int | None = 200,
    max_blocks: int | None = 16,
    max_env_residues: int = 500,
    env_center: str = "fragment",
    temperature: float = 1.0,
    use_energy: bool = True,
    use_content: bool = True,
    use_pair_bias: bool = True,
    fp_aggregate: str = "mean",
    explicit_h: bool = True,
    backbone: bool = True,
    energy_pooling: str = "mean",
    contact_cutoff: float = 6.0,
    charge_method: str = "eem",
    strict_charges: bool = False,
    angles: bool = True,
    energy_scale: float = 0.2,
    content_scale: float = 0.3,
    legacy_hbond: bool = False,
    cut_biaryl: bool = True,
    cut_amide: bool = True,
) -> tuple[WeightedTripletIndex, list[dict]]:
    if not explicit_h:
        angles = False
        if charge_method != "formal":
            print("note: --no-explicit-h forces --charge-method formal",
                  flush=True)
            charge_method = "formal"
    features = {
        "explicit_h": explicit_h,
        "backbone": backbone,
        "angles": angles,
        "legacy_hbond": legacy_hbond,
        "energy_pooling": energy_pooling,
        "contact_cutoff": contact_cutoff,
        "charge_method": charge_method,
        "energy_scale": energy_scale,
        "content_scale": content_scale,
        "temperature": temperature,
        "fp_aggregate": fp_aggregate,
        "use_energy": use_energy,
        "use_content": use_content,
        "use_pair_bias": use_pair_bias,
        "marks": "mean,max,min_leg,cmean",
        "env_radius": env_radius,
        "d_lo": d_lo,
        "d_hi": d_hi,
        "env_center": env_center,
        "min_da": min_da,
        "max_da": max_da,
        "max_frags": max_frags,
        "max_blocks": max_blocks,
        "cut_biaryl": cut_biaryl,
        "cut_amide": cut_amide,
    }

    sel: dict[str, str] = {}
    with open(selected, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sel[row["pdb_id"].lower()] = row["ligand_smiles"]

    pids = [
        pid for pid in sel
        if os.path.isfile(os.path.join(cache, f"{pid}.pdb"))
        and os.path.getsize(os.path.join(cache, f"{pid}.pdb")) > 0
    ]
    pids.sort()
    print(f"{len(pids):,} cached complexes available for {len(sel):,} selected ids",
          flush=True)

    index = WeightedTripletIndex(features=features)
    meta_rows: list[dict] = []
    n_done = 0
    n_frags = 0
    n_skipped = {"parse": 0, "ligand": 0, "fragments": 0, "env": 0,
                 "protein": 0}
    n_triplets = 0
    charge_methods: dict[str, int] = {}

    for pid in pids:
        if max_complexes is not None and n_done >= max_complexes:
            break
        path = os.path.join(cache, f"{pid}.pdb")
        try:
            with open(path) as handle:
                block = handle.read()
            atoms = parse_pdb(path)
        except Exception:
            n_skipped["parse"] += 1
            continue

        if explicit_h:
            residues = protein_residues(block)
            if residues is None:
                # Documented fallback: heavy atoms only, no angles.
                n_skipped["protein"] += 1
                residues = _group_residues(atoms)
        else:
            residues = _group_residues(atoms)
        if not residues:
            n_skipped["parse"] += 1
            continue

        lig_mol = _find_ligand(block, sel[pid])
        if lig_mol is None:
            n_skipped["ligand"] += 1
            continue

        frags = fragment_ligand(
            lig_mol, min_da=min_da, max_da=max_da, max_frags=max_frags,
            max_blocks=max_blocks, cut_biaryl=cut_biaryl,
            cut_amide=cut_amide)
        if not frags:
            n_skipped["fragments"] += 1
            continue

        parent_smi = Chem.MolToSmiles(lig_mol)

        pocket_env, pocket_table = [], None
        if env_center == "pocket":
            conf = lig_mol.GetConformer()
            lig_xyz = np.array(
                [list(conf.GetAtomPosition(a.GetIdx()))
                 for a in lig_mol.GetAtoms() if a.GetAtomicNum() > 1],
                dtype=np.float64)
            pocket_env = pocket_environment(residues, lig_xyz, env_radius)
            pocket_table = triplet_table(
                pocket_env, d_lo=d_lo, d_hi=d_hi,
                use_pair_bias=use_pair_bias)

        mols, charge_sets, methods = [], [], []
        if explicit_h:
            mols = [with_explicit_hydrogens(f.mol) for f in frags]
            if charge_method == "eem":
                try:
                    charge_sets = eem_charges_batch(mols)
                    methods = ["eem"] * len(mols)
                except Exception:
                    charge_sets, methods = [], []
                    for mol in mols:
                        q, method = charges_for(
                            mol, method="eem", strict=strict_charges)
                        charge_sets.append(q)
                        methods.append(method)
            else:
                charge_sets = [None] * len(mols)
                methods = ["formal"] * len(mols)
        else:
            mols = [f.mol for f in frags]
            charge_sets = [None] * len(mols)
            methods = ["formal"] * len(mols)

        for k, frag in enumerate(frags):
            if env_center == "pocket":
                env, table = pocket_env, pocket_table
            else:
                env = environment(residues, frag.centroid, env_radius)
                table = None
            if len(env) < 3:
                n_skipped["env"] += 1
                continue
            if len(env) > max_env_residues:
                n_skipped["env"] += 1
                continue

            att = attachment_atoms(frag.mol, lig_mol) if explicit_h else set()
            weights, res_energies, res_contacts = relevance_weights(
                mols[k], env, temperature=temperature,
                use_energy=use_energy, use_content=use_content,
                frag_charges=charge_sets[k], exclude_donor_atoms=att,
                pooling=energy_pooling, contact_cutoff=contact_cutoff,
                use_angles=angles, backbone=backbone,
                energy_scale=energy_scale, content_scale=content_scale,
                legacy_hbond=legacy_hbond, return_energies=True,
                return_contact=True)
            env_res = [
                EnvResidue(
                    res_name=r.res_name, chain=r.chain, res_seq=r.res_seq,
                    coords=r.coords, code=r.code, atoms=r.atoms,
                    weight=float(w))
                for r, w in zip(env, weights)
            ]

            raw_fp = fragment_fp_weights(
                env_res, d_lo=d_lo, d_hi=d_hi,
                fp_aggregate=fp_aggregate, use_pair_bias=use_pair_bias,
                table=table)
            if not raw_fp:
                n_skipped["env"] += 1
                continue

            e_mean, e_max, e_min, e_cmean = fragment_fp_energy(
                env_res, res_energies, frag.n_heavy,
                d_lo=d_lo, d_hi=d_hi, table=table, return_min_leg=True,
                return_contact_mean=True, contact_energies=res_contacts)

            charge_methods[methods[k]] = charge_methods.get(methods[k], 0) + 1
            meta = {
                "frag_smiles": frag.smiles,
                "pdb_id": pid,
                "role": "",
                "tags": f"{_attach_class(frag.n_attach)},{_ring_class(frag.mol)}",
                "n_heavy": frag.n_heavy,
                "parent_smiles": parent_smi,
                "n_attach": frag.n_attach,
                "charge_method": methods[k],
                "_fp_e_atom_mean": e_mean,
                "_fp_e_atom_max": e_max,
                "_fp_e_atom_min": e_min,
                "_fp_e_atom_cmean": e_cmean,
            }
            fid = f"{pid}_{k}"
            n = index.add_fragment(
                fid, env_res, raw_fp, meta=meta,
                frag_coords=canonical_coords(frag.mol, frag.smiles))
            meta_rows.append({"frag_id": fid, "n_triplets": n,
                              "n_env": len(env), **meta})
            n_frags += 1
            n_triplets += n

        n_done += 1
        if n_done % 50 == 0:
            print(f"  [{n_done:,} complexes] {n_frags:,} fragments, "
                  f"{n_triplets:,} triplets, "
                  f"{len(index._postings):,} unique fps", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    index.save(out)
    print(f"\ncomplexes processed : {n_done:,}")
    print(f"fragments indexed   : {n_frags:,}")
    print(f"triplets            : {n_triplets:,}")
    print(f"unique fingerprints : {len(index._postings):,}")
    print("skipped:", ", ".join(f"{k}={v:,}" for k, v in n_skipped.items()))
    print("charges:", ", ".join(f"{k}={v:,}" for k, v in charge_methods.items()))
    print(f"saved               : {out}")
    return index, meta_rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selected", default="/Users/mai/Software/fragenv/data/bindingdb_selected.tsv")
    ap.add_argument("--cache", default="/Users/mai/Software/fragenv/data/rcsb_cache")
    ap.add_argument("--out", default="work/frase_rich_index.pkl")
    ap.add_argument("--max-complexes", type=int, default=None)
    ap.add_argument("--env-radius", type=float, default=12.0)
    ap.add_argument("--d-lo", type=float, default=8.0)
    ap.add_argument("--d-hi", type=float, default=12.0)
    ap.add_argument("--min-da", type=float, default=50.0)
    ap.add_argument("--max-da", type=float, default=300.0)
    ap.add_argument("--max-frags", type=int, default=200)
    ap.add_argument("--max-blocks", type=int, default=16)
    ap.add_argument("--max-env-residues", type=int, default=500)
    ap.add_argument("--env-center", choices=("fragment", "pocket"),
                    default="fragment",
                    help="fragment = 12 A sphere per fragment centroid; "
                         "pocket = union of 12 A spheres around ligand atoms, "
                         "shared by every fragment of the complex")
    ap.add_argument("--no-cut-biaryl", action="store_true")
    ap.add_argument("--no-cut-amide", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-energy", action="store_true",
                    help="ablation: drop the interaction-energy term")
    ap.add_argument("--no-content", action="store_true",
                    help="ablation: drop the pharmacophore-content term")
    ap.add_argument("--no-pair-bias", action="store_true",
                    help="ablation: drop the residue-residue pair bias")
    ap.add_argument("--fp-aggregate", choices=("mean", "max"), default="mean",
                    help="collapse multiple residue triplets to one code key")
    ap.add_argument("--no-explicit-h", action="store_true",
                    help="ablation: heavy atoms only (forces formal charges)")
    ap.add_argument("--no-backbone", action="store_true",
                    help="ablation: sidechain atoms only")
    ap.add_argument("--energy-pooling", choices=("mean", "sum"),
                    default="mean",
                    help="mean = per-contact average, sum = old extensive A/B")
    ap.add_argument("--contact-cutoff", type=float, default=6.0)
    ap.add_argument("--charge-method", choices=("eem", "formal"),
                    default="eem")
    ap.add_argument("--strict-charges", action="store_true",
                    help="fail instead of falling back to Gasteiger")
    ap.add_argument("--no-angles", action="store_true",
                    help="ablation: distance-only H-bonds")
    ap.add_argument("--energy-scale", type=float, default=0.2,
                    help="tanh scale for the interaction energy")
    ap.add_argument("--content-scale", type=float, default=0.3)
    ap.add_argument("--legacy-hbond", action="store_true",
                    help="v1 donor/acceptor sets (exact A/B control only)")
    args = ap.parse_args(argv)

    build_index(
        selected=args.selected, cache=args.cache, out=args.out,
        max_complexes=args.max_complexes, env_radius=args.env_radius,
        d_lo=args.d_lo, d_hi=args.d_hi, min_da=args.min_da, max_da=args.max_da,
        max_frags=args.max_frags, max_blocks=args.max_blocks,
        max_env_residues=args.max_env_residues,
        env_center=args.env_center,
        temperature=args.temperature,
        use_energy=not args.no_energy, use_content=not args.no_content,
        use_pair_bias=not args.no_pair_bias,
        fp_aggregate=args.fp_aggregate,
        explicit_h=not args.no_explicit_h,
        backbone=not args.no_backbone,
        energy_pooling=args.energy_pooling,
        contact_cutoff=args.contact_cutoff,
        charge_method=args.charge_method,
        strict_charges=args.strict_charges,
        angles=not args.no_angles,
        energy_scale=args.energy_scale,
        content_scale=args.content_scale,
        legacy_hbond=args.legacy_hbond,
        cut_biaryl=not args.no_cut_biaryl,
        cut_amide=not args.no_cut_amide,
    )


if __name__ == "__main__":
    main()
