#!/usr/bin/env python
"""Emit a pid -> protein-group map and an optional same-protein subset TSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.proteins import group_by_sequence  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selected", default="/Users/mai/Software/fragenv/data/bindingdb_selected.tsv")
    ap.add_argument("--cache", default="/Users/mai/Software/fragenv/data/rcsb_cache")
    ap.add_argument("--groups-out", default="work/protein_groups.json")
    ap.add_argument("--selected-out", default=None,
                    help="write a pdb_id/ligand_smiles TSV for same-protein subsets")
    ap.add_argument("--min-group-size", type=int, default=2)
    ap.add_argument("--max-pids", type=int, default=None,
                    help="cap the selected subset, largest groups first")
    args = ap.parse_args(argv)

    rows: dict[str, str] = {}
    with open(args.selected, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pid = row["pdb_id"].lower()
            if os.path.isfile(os.path.join(args.cache, f"{pid}.pdb")):
                rows[pid] = row["ligand_smiles"]

    groups, group_of = group_by_sequence(list(rows), args.cache)
    multi = {gid: m for gid, m in groups.items() if len(m) >= args.min_group_size}

    os.makedirs(os.path.dirname(os.path.abspath(args.groups_out)) or ".", exist_ok=True)
    with open(args.groups_out, "w") as handle:
        json.dump(group_of, handle, indent=0, sort_keys=True)

    print(f"cached selected: {len(rows):,}  grouped: {len(group_of):,}")
    print(f"groups: {len(groups):,}  with >={args.min_group_size}: {len(multi):,}")
    print(f"complexes in same-protein groups: "
          f"{sum(len(m) for m in multi.values()):,}")

    if args.selected_out:
        chosen: list[str] = []
        for gid, members in multi.items():
            if args.max_pids is not None and len(chosen) + len(members) > args.max_pids:
                continue
            chosen.extend(members)
        os.makedirs(os.path.dirname(os.path.abspath(args.selected_out)) or ".", exist_ok=True)
        with open(args.selected_out, "w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["pdb_id", "ligand_smiles"])
            for pid in sorted(chosen):
                writer.writerow([pid, rows[pid]])
        print(f"subset: {len(chosen):,} complexes over "
              f"{sum(1 for m in multi.values() if set(m) <= set(chosen)):,} groups "
              f"-> {args.selected_out}")


if __name__ == "__main__":
    main()
