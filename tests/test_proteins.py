"""Tests for protein-sequence grouping (no cache needed)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.proteins import group_by_sequence, longest_chain_sequence  # noqa: E402


def _atom_line(serial: int, res: str, chain: str, resseq: int) -> str:
    return ("ATOM  %5d  CA  %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
            % (serial, res, chain, resseq, 0.0, 0.0, 0.0))


def _write_pdb(path: str, chains: dict[str, list[str]]) -> None:
    with open(path, "w") as handle:
        serial = 1
        for chain, residues in chains.items():
            for resseq, res in enumerate(residues, start=1):
                handle.write(_atom_line(serial, res, chain, resseq))
                serial += 1


class TestProteins(unittest.TestCase):
    def test_longest_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "1aaa.pdb")
            _write_pdb(path, {
                "A": ["ALA", "PHE", "ASP"],
                "B": ["GLY", "SER"],
            })
            self.assertEqual(longest_chain_sequence(path), "AFD")

    def test_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_pdb(os.path.join(tmp, "1aaa.pdb"), {"A": ["ALA", "PHE"]})
            _write_pdb(os.path.join(tmp, "2bbb.pdb"), {"A": ["ALA", "PHE"]})
            _write_pdb(os.path.join(tmp, "3ccc.pdb"), {"A": ["ALA", "GLY"]})
            groups, group_of = group_by_sequence(
                ["1aaa", "2bbb", "3ccc"], tmp)
            self.assertEqual(group_of["1aaa"], group_of["2bbb"])
            self.assertNotEqual(group_of["1aaa"], group_of["3ccc"])
            self.assertEqual(len(groups), 2)

    def test_unparseable_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "9zzz.pdb")
            with open(path, "w") as handle:
                handle.write("HETATM    1  C1  LIG A   1       0.000   0.000   0.000\n")
            self.assertIsNone(longest_chain_sequence(path))
            _groups, group_of = group_by_sequence(["9zzz"], tmp)
            self.assertEqual(group_of, {})


if __name__ == "__main__":
    unittest.main()
