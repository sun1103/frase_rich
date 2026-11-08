"""Merge pocket indexes built from disjoint PDB sets into one index.

Used to combine the PDBbind index with the RCSB/BindingDB shards. Feature
dicts must match exactly. Fragment ids are ``{pdb_id}_{k}``, so complexes
already present in the base index are dropped from the shards (their
fragments would collide) and the remaining ids are disjoint.

Usage:
    python scripts/merge_indexes.py --base work/rich_pocket_pdbbind.pkl \
        --shards work/rich_pocket_rcsbq0.pkl work/rich_pocket_rcsbq1.pkl \
        --out work/rich_pocket_union.pkl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frase_rich.index import WeightedTripletIndex  # noqa: E402


def merge(base_path: str, shard_paths: list[str], out_path: str) -> None:
    base = WeightedTripletIndex.load(base_path)
    existing = {str(i.get("pdb_id", "")).lower()
                for i in base._frags.values()}
    print(f"base: {len(base):,} fragments over {len(existing):,} complexes")

    for path in shard_paths:
        idx = WeightedTripletIndex.load(path)
        assert idx.features == base.features, f"feature mismatch: {path}"
        dup_fids: set[str] = set()
        added_pdbs: set[str] = set()
        for fid, info in idx._frags.items():
            pdb = str(info.get("pdb_id", "")).lower()
            if pdb in existing:
                dup_fids.add(fid)
                continue
            if fid in base._frags:
                raise SystemExit(
                    f"fragment id collision without pdb match: {fid}")
            base._frags[fid] = info
            added_pdbs.add(pdb)
        for key, postings in idx._postings.items():
            base._postings.setdefault(key, []).extend(
                p for p in postings if p[0] not in dup_fids)
        existing |= added_pdbs
        print(f"  merged {os.path.basename(path)}: "
              f"+{len(idx) - len(dup_fids):,} fragments over "
              f"{len(added_pdbs):,} complexes "
              f"(dropped {len(dup_fids):,} duplicate-pdb fragments)")
        del idx

    missing = {fid for postings in base._postings.values()
               for fid, _w in postings if fid not in base._frags}
    assert not missing, f"{len(missing)} postings point at unknown fids"

    n_pdbs = {str(i.get("pdb_id", "")).lower() for i in base._frags.values()}
    n_triplets = sum(int(i.get("_n_triplets", 0))
                     for i in base._frags.values())
    print(f"merged: {len(base):,} fragments over {len(n_pdbs):,} complexes, "
          f"{len(base._postings):,} keys, {n_triplets:,} triplets")
    base.save(out_path)
    print("saved:", out_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base index pickle")
    parser.add_argument("--shards", nargs="+", required=True,
                        help="shard index pickles (disjoint PDB ids)")
    parser.add_argument("--out", required=True, help="output pickle")
    args = parser.parse_args(argv)
    merge(args.base, args.shards, args.out)


if __name__ == "__main__":
    main()
