"""Duplicate-site handling in the profile store build."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.evaluate import _FragmentSet  # noqa: E402
from frase_rich.fragmentation import canonical_coords  # noqa: E402
from scripts.build_profile_store import _group_copies  # noqa: E402


def _copy(smiles: str, offset: float):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(i) + offset, 0.0, 0.0))
    mol.AddConformer(conf)
    return SimpleNamespace(smiles=Chem.MolToSmiles(mol), mol=mol)


class GroupCopiesTest(unittest.TestCase):
    def setUp(self):
        self.copies = [_copy("CCO", 0.0), _copy("CCO", 10.0)]
        self.coords = [canonical_coords(f.mol, f.smiles)
                       for f in self.copies]
        self.by_smiles = _FragmentSet()
        for frag in self.copies:
            self.by_smiles.add(frag)

    def test_stored_coords_identify_the_copy(self):
        frags = {
            "f1": {"frag_smiles": "CCO", "_frag_coords": self.coords[0]},
            "f2": {"frag_smiles": "CCO", "_frag_coords": self.coords[1]},
            "f3": {"frag_smiles": "CCN"},
        }
        chosen, copy_fids = _group_copies(["f1", "f2", "f3"], frags,
                                          self.by_smiles)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(copy_fids, [["f1"], ["f2"]])
        self.assertIs(chosen[0][1], self.copies[0])
        self.assertIs(chosen[1][1], self.copies[1])

    def test_fids_on_the_same_copy_share_one_profile(self):
        frags = {
            "f1": {"frag_smiles": "CCO", "_frag_coords": self.coords[1]},
            "f2": {"frag_smiles": "CCO", "_frag_coords": self.coords[1]},
        }
        chosen, copy_fids = _group_copies(["f1", "f2"], frags,
                                          self.by_smiles)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(copy_fids, [["f1", "f2"]])

    def test_without_coords_the_first_copy_is_used(self):
        frags = {"f1": {"frag_smiles": "CCO"},
                 "f2": {"frag_smiles": "CCO"}}
        chosen, copy_fids = _group_copies(["f1", "f2"], frags,
                                          self.by_smiles)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(copy_fids, [["f1", "f2"]])


if __name__ == "__main__":
    unittest.main()
