"""Ref-fragment query: picking the right copy when a SMILES repeats.

A symmetric parent yields the same fragment constitution at two cut sites
(19% of indexed rows share their ``(pdb, smiles)`` key). Matching by SMILES
alone recomputes the wrong site's environment, so the query path must
disambiguate by the deposited coordinates.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.evaluate import _FragmentSet, _pick_fragment  # noqa: E402
from frase_rich.fragmentation import canonical_coords  # noqa: E402


def _fragment(smiles: str, shift: float) -> SimpleNamespace:
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol, randomSeed=7)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + shift, p.y, p.z))
    return SimpleNamespace(mol=mol, smiles=Chem.MolToSmiles(mol))


class FragmentSetTest(unittest.TestCase):
    def test_first_for_dict_callers_all_for_disambiguation(self):
        a = _fragment("CCC", 0.0)
        b = _fragment("CCC", 10.0)
        fs = _FragmentSet()
        fs.add(a)
        fs.add(b)
        self.assertIs(fs.get(a.smiles), a)
        self.assertEqual(fs.all(a.smiles), [a, b])
        self.assertEqual(list(fs.values()), [a])
        self.assertEqual(len(fs), 1)

    def test_registration_matches_setdefault_semantics(self):
        a = _fragment("CCC", 0.0)
        fs = _FragmentSet()
        fs.setdefault(a.smiles, a)
        self.assertIs(fs.get(a.smiles), a)
        self.assertEqual(fs.all(a.smiles), [])


class PickFragmentTest(unittest.TestCase):
    def setUp(self):
        self.a = _fragment("CCC", 0.0)
        self.b = _fragment("CCC", 10.0)
        self.ca = canonical_coords(self.a.mol, self.a.smiles)
        self.cb = canonical_coords(self.b.mol, self.b.smiles)

    def test_returns_the_coordinate_match(self):
        self.assertIs(_pick_fragment([self.a, self.b], self.cb), self.b)
        self.assertIs(_pick_fragment([self.a, self.b], self.ca), self.a)

    def test_single_candidate_ignores_coords(self):
        self.assertIs(_pick_fragment([self.b], self.ca), self.b)

    def test_no_coords_keeps_first(self):
        self.assertIs(_pick_fragment([self.a, self.b], None), self.a)

    def test_shape_mismatch_keeps_first(self):
        self.assertIs(_pick_fragment([self.a, self.b], np.zeros((2, 3))),
                      self.a)


if __name__ == "__main__":
    unittest.main()
