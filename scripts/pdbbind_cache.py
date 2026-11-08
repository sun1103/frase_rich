#!/usr/bin/env python
"""PDBbind P-L tree -> one PDB per complex + a selected TSV.

The rich builder and the ref-frag query provider read ``{cache}/{pdb_id}.pdb``
and need the parent ligand as a HETATM residue in that same block. PDBbind
ships ``{pid}_protein.pdb`` and ``{pid}_ligand.sdf`` separately, so this script
merges them:

    P-L/{bucket}/{pid}/{pid}_protein.pdb  --.
                                            >-- {out_cache}/{pid}.pdb
    P-L/{bucket}/{pid}/{pid}_ligand.sdf   --'        + {out_tsv}

The ligand is written from the SDF (deposited bond orders), as a single
``HETATM`` group named ``LIG`` on chain ``L``. Protein HETATM records are
dropped so the builder's ``_find_ligand`` only ever sees that one group.

PDBbind is not small-molecule-curated: it contains peptides, cofactors,
phosphates and metallo-organic ligands. By default only ligands passing the
same filter as ``fragenv/scripts/select_bindingdb_complexes.py`` are kept
(Lipinski logP<=5/HBD<=5/HBA<=10, has carbon, no phosphorus, not peptide-like:
>=2 peptide amide motifs or >=2 acyclic amides), plus an allowed-element guard
(H C N O F P S Cl Br I) that drops metals, boron and selenium. The molecular
weight cutoff is 700 by default (the upstream selection used 500; 700 admits
966 more PDBbind complexes). ``--keep-all`` disables the filter; rejection
counts are printed.

Usage:
    conda run -n rxn-env python scripts/pdbbind_cache.py \
        --root /Users/mai/Software/fragenv/data/pdbbind_pl/P-L \
        --out-cache /Users/mai/Software/fragenv/data/pdbbind_cache \
        --out-tsv /Users/mai/Software/fragenv/data/pdbbind_selected.tsv

Smoke test:
    ... --max-complexes 20 --out-cache /tmp/pdbbind_smoke
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

LIGAND_RESIDUE = "LIG"
LIGAND_CHAIN = "L"
LIGAND_SEQ = 900
DEFAULT_MW_CUTOFF = 700.0
DEFAULT_LOGP_CUTOFF = 5.5
DEFAULT_HBD_CUTOFF = 5
DEFAULT_HBA_CUTOFF = 10

PEPTIDE_SMARTS = Chem.MolFromSmarts("[NX3,NX4+][CX4][CX3](=[OX1])")
AMIDE_SMARTS = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
ALLOWED_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H C N O F P S Cl Br I


def _n_acyclic_amides(mol: Chem.Mol) -> int:
    n = 0
    for n_idx, c_idx, _o_idx in mol.GetSubstructMatches(AMIDE_SMARTS):
        bond = mol.GetBondBetweenAtoms(n_idx, c_idx)
        if bond is not None and not bond.IsInRing():
            n += 1
    return n


def reject_reason(mol: Chem.Mol, *, mw_cutoff: float,
                  logp_cutoff: float, hbd_cutoff: int,
                  hba_cutoff: int) -> str | None:
    """First upstream-style reason to drop this ligand, or None to keep it."""
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    if 6 not in atoms:
        return "no_carbon"
    if 15 in atoms:
        return "phosphorus"
    if any(z not in ALLOWED_ELEMENTS for z in atoms):
        return "unsupported_element"
    if Descriptors.MolWt(mol) > mw_cutoff:
        return "mw>cutoff"
    if Descriptors.MolLogP(mol) > logp_cutoff:
        return "logp>cutoff"
    if rdMolDescriptors.CalcNumHBD(mol) > hbd_cutoff:
        return "hbd>cutoff"
    if rdMolDescriptors.CalcNumHBA(mol) > hba_cutoff:
        return "hba>cutoff"
    if (len(mol.GetSubstructMatches(PEPTIDE_SMARTS)) >= 2
            or _n_acyclic_amides(mol) >= 2):
        return "peptide"
    return None


def _ligand_block(mol: Chem.Mol) -> str:
    mol = Chem.RemoveHs(mol)
    for idx, atom in enumerate(mol.GetAtoms()):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{atom.GetSymbol()}{idx + 1}")
        info.SetResidueName(LIGAND_RESIDUE)
        info.SetResidueNumber(LIGAND_SEQ)
        info.SetChainId(LIGAND_CHAIN)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
    return Chem.MolToPDBBlock(mol)


def _merge(protein: str, ligand: str) -> str:
    prot_lines = [
        line for line in protein.splitlines()
        if line.startswith(("ATOM", "TER"))
    ]
    lig_lines = [
        line for line in ligand.splitlines()
        if line.startswith(("HETATM", "CONECT"))
    ]
    return "\n".join(prot_lines + lig_lines) + "\nEND\n"


def convert(root: str, out_cache: str, out_tsv: str,
            *, max_complexes: int | None = None,
            overwrite: bool = False, keep_all: bool = False,
            mw_cutoff: float = DEFAULT_MW_CUTOFF,
            logp_cutoff: float = DEFAULT_LOGP_CUTOFF,
            hbd_cutoff: int = DEFAULT_HBD_CUTOFF,
            hba_cutoff: int = DEFAULT_HBA_CUTOFF) -> None:
    os.makedirs(out_cache, exist_ok=True)
    prot_paths = sorted(glob.glob(os.path.join(root, "*", "*", "*_protein.pdb")))
    rows = []
    n_done = n_skip = 0
    skipped: list[tuple[str, str]] = []
    rejected: Counter[str] = Counter()
    for prot_path in prot_paths:
        if max_complexes is not None and n_done >= max_complexes:
            break
        pid = os.path.basename(prot_path)[: -len("_protein.pdb")]
        sdf_path = os.path.join(os.path.dirname(prot_path), f"{pid}_ligand.sdf")
        out_path = os.path.join(out_cache, f"{pid}.pdb")
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False, sanitize=True)
        if mol is None or mol.GetNumConformers() == 0:
            n_skip += 1
            skipped.append((pid, "ligand_parse"))
            continue
        if not keep_all:
            reason = reject_reason(
                mol, mw_cutoff=mw_cutoff, logp_cutoff=logp_cutoff,
                hbd_cutoff=hbd_cutoff, hba_cutoff=hba_cutoff)
            if reason is not None:
                n_skip += 1
                rejected[reason] += 1
                skipped.append((pid, reason))
                continue
        # The TSV row is needed even when the PDB already exists (resume).
        rows.append({"pdb_id": pid,
                     "ligand_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol))})
        if os.path.isfile(out_path) and not overwrite:
            n_done += 1
            continue
        try:
            protein = open(prot_path).read()
            block = _merge(protein, _ligand_block(mol))
        except Exception:  # noqa: BLE001 - one bad complex must not stop a run
            n_skip += 1
            rows.pop()
            skipped.append((pid, "merge"))
            continue
        tmp = out_path + ".tmp"
        with open(tmp, "w") as handle:
            handle.write(block)
        os.replace(tmp, out_path)
        n_done += 1
        if n_done % 500 == 0:
            print(f"  [{n_done:,} complexes]", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_tsv)) or ".", exist_ok=True)
    with open(out_tsv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pdb_id", "ligand_smiles"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"converted: {n_done:,}  skipped: {n_skip:,}  "
          f"tsv rows: {len(rows):,}")
    if rejected:
        print("filter rejections (first reason per ligand):")
        for why, count in rejected.most_common():
            print(f"  {why:20s} {count:,}")
    for pid, why in skipped[:20]:
        print(f"  skip {pid}: {why}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/Users/mai/Software/fragenv/data/pdbbind_pl/P-L")
    ap.add_argument("--out-cache",
                    default="/Users/mai/Software/fragenv/data/pdbbind_cache")
    ap.add_argument("--out-tsv",
                    default="/Users/mai/Software/fragenv/data/pdbbind_selected.tsv")
    ap.add_argument("--max-complexes", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-all", action="store_true",
                    help="skip the drug-like/peptide/element filter")
    ap.add_argument("--mw-cutoff", type=float, default=DEFAULT_MW_CUTOFF)
    ap.add_argument("--logp-cutoff", type=float, default=DEFAULT_LOGP_CUTOFF)
    ap.add_argument("--hbd-cutoff", type=int, default=DEFAULT_HBD_CUTOFF)
    ap.add_argument("--hba-cutoff", type=int, default=DEFAULT_HBA_CUTOFF)
    args = ap.parse_args(argv)
    convert(args.root, args.out_cache, args.out_tsv,
            max_complexes=args.max_complexes, overwrite=args.overwrite,
            keep_all=args.keep_all, mw_cutoff=args.mw_cutoff,
            logp_cutoff=args.logp_cutoff, hbd_cutoff=args.hbd_cutoff,
            hba_cutoff=args.hba_cutoff)


if __name__ == "__main__":
    main()
