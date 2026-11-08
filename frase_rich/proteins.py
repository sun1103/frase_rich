"""Group PDB entries by exact protein sequence.

The leave-one-complex-out benchmark needs to distinguish two very different
tasks: retrieving the same fragment from another structure of the *same*
protein (environment transfer) and retrieving it from an unrelated protein
(chemical-only match). This module supplies the grouping used by
``evaluate.py --positive-mode same-protein``.
"""

from __future__ import annotations

import collections
import os

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def longest_chain_sequence(path: str) -> str | None:
    """One-letter sequence of the longest protein chain in a PDB file."""
    chains: dict[str, list[str]] = collections.defaultdict(list)
    seen: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    with open(path, errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip().upper() != "CA":
                continue
            res = line[17:20].strip().upper()
            if res not in AA3:
                continue
            chain = line[21]
            key = (line[22:26], line[26])
            if key in seen[chain]:
                continue
            seen[chain].add(key)
            chains[chain].append(AA3[res])
    if not chains:
        return None
    return "".join(max(chains.values(), key=len))


def group_by_sequence(
    pids: list[str], cache: str
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return ``(groups, group_of)`` for exact longest-chain sequence matches.

    ``groups`` maps group id -> pids; ``group_of`` maps pid -> group id.
    Entries without a parseable protein chain are omitted.
    """
    by_seq: dict[str, list[str]] = collections.defaultdict(list)
    for pid in sorted(set(pids)):
        path = os.path.join(cache, f"{pid.lower()}.pdb")
        if not os.path.isfile(path):
            continue
        seq = longest_chain_sequence(path)
        if seq:
            by_seq[seq].append(pid.lower())

    groups: dict[str, list[str]] = {}
    group_of: dict[str, str] = {}
    for i, (_seq, members) in enumerate(
            sorted(by_seq.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))):
        gid = f"g{i:04d}"
        groups[gid] = members
        for pid in members:
            group_of[pid] = gid
    return groups, group_of
