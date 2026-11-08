"""Unit tests for EEM charges, coordinate mapping and protein H placement."""

import os
import unittest

import numpy as np

from frase_rich.chem import (
    DEFAULT_OBABEL, _map_by_coordinate, charges_for, eem_charges_batch,
    formal_net_charge, with_explicit_hydrogens,
)
from frase_rich.protein import protein_residues

from rdkit import Chem
from rdkit.Chem import AllChem

HAS_OBABEL = os.path.isfile(DEFAULT_OBABEL)


def _embed(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    return mol


@unittest.skipUnless(HAS_OBABEL, "obabel not installed")
class TestEEM(unittest.TestCase):
    def test_quaternary_ammonium_charge_lands_on_hydrogens(self):
        mol = _embed("C[N+](C)(C)C")
        charges, method = charges_for(mol, method="eem", strict=True)
        self.assertEqual(method, "eem")
        self.assertAlmostEqual(float(charges.sum()), 1.0, places=6)
        n_idx = next(a.GetIdx() for a in mol.GetAtoms()
                     if a.GetAtomicNum() == 7)
        h_on_n = [a.GetIdx() for a in mol.GetAtoms()
                  if a.GetAtomicNum() == 1
                  and a.GetNeighbors()[0].GetIdx() == n_idx]
        # EEM keeps the ammonium N negative and puts the positive charge on
        # its hydrogens; the exact split depends on substitution.
        self.assertLess(charges[n_idx], -0.1)
        self.assertTrue(all(charges[i] > 0.1 for i in h_on_n))

    def test_batch_matches_single(self):
        mols = [_embed("CCO"), _embed("c1ccccc1O")]
        batch = eem_charges_batch(mols)
        single = eem_charges_batch([mols[1]])[0]
        self.assertTrue(np.allclose(batch[1], single, atol=1e-8))

    def test_fallback_records_method(self):
        mol = _embed("CCO")
        charges, method = charges_for(mol, method="eem",
                                      obabel="/nonexistent/obabel")
        self.assertEqual(method, "gasteiger")
        self.assertAlmostEqual(float(charges.sum()), 0.0, places=6)

    def test_strict_raises(self):
        mol = _embed("CCO")
        with self.assertRaises(Exception):
            charges_for(mol, method="eem", obabel="/nonexistent/obabel",
                        strict=True)


class TestCoordinateMapping(unittest.TestCase):
    def test_permutation_is_inverted(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(8, 3))
        perm = np.array([3, 7, 0, 5, 1, 6, 2, 4])
        mapping = _map_by_coordinate(src, src[perm])
        # dst row i is src[perm[i]], so src row perm[i] must map to i.
        for i, p in enumerate(perm):
            self.assertEqual(mapping[p], i)

    def test_collision_is_rejected(self):
        src = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            _map_by_coordinate(src, src.copy())

    def test_displacement_is_rejected(self):
        src = np.zeros((3, 3))
        src[1, 0] = 1.0
        src[2, 1] = 2.0
        dst = src.copy()
        dst[0, 2] = 0.5
        with self.assertRaises(ValueError):
            _map_by_coordinate(src, dst)


GLY_PDB = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.000   1.400   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.300   2.300   0.000  1.00  0.00           O
END
"""


class TestProteinHydrogens(unittest.TestCase):
    def test_gly_backbone_and_added_h_identity(self):
        residues = protein_residues(GLY_PDB)
        self.assertIsNotNone(residues)
        self.assertEqual(len(residues), 1)
        res = residues[0]
        heavy = [a for a in res.atoms if not a.is_hydrogen]
        self.assertEqual({a.name for a in heavy}, {"N", "CA", "C", "O"})
        hydrogens = [a for a in res.atoms if a.is_hydrogen]
        self.assertTrue(hydrogens)
        for h in hydrogens:
            self.assertGreaterEqual(h.parent, 0)
            self.assertLess(h.parent, len(res.atoms))
            self.assertFalse(res.atoms[h.parent].is_hydrogen)

    def test_unparseable_block_returns_none(self):
        self.assertIsNone(protein_residues("not a pdb"))

    def test_formal_net_charge_of_ammonium(self):
        mol = Chem.MolFromSmiles("C[N+](C)(C)C")
        self.assertEqual(formal_net_charge(mol), 1)


if __name__ == "__main__":
    unittest.main()
