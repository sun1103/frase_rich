import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from frase_rich.patent_decomp import decompose_parent
from frase_rich.fragmentation import fragment_ligand

SMILES = [
    "C[C@H](CS)C(=O)N1CCC[C@H]1C(=O)O",
    "COc1ccc(-c2ccccc2)cc1C(=O)Nc1ccc(Cl)cc1",
    "O=C(O)c1ccc(NC(=O)c2ccc(N3CCOCC3)cc2)cc1",
    "Cc1ccc2[nH]cc(CCN3CCC(C(=O)c4ccccc4)CC3)c2c1",
    "CN1CCN(CC1)c1ccc(C(=O)Nc2ccc(OC(F)(F)F)cc2)cc1",
]


class PatentDecompTest(unittest.TestCase):
    def test_matches_3d_rule(self):
        """Same fragment set as the conformer-carrying library fragmenter."""
        checked = 0
        for smi in SMILES:
            mol = Chem.MolFromSmiles(smi)
            molh = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(molh, randomSeed=11) < 0:
                continue
            mol3 = Chem.RemoveHs(molh)
            ref = sorted(f.smiles for f in fragment_ligand(
                mol3, min_da=50.0, max_da=300.0, max_blocks=16, max_frags=200))
            got = sorted(r["frag_smiles"] for r in decompose_parent(smi))
            self.assertEqual(ref, got, smi)
            checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_biphenyl_biaryl_cut(self):
        frags = decompose_parent("c1ccccc1-c1ccccc1")
        self.assertTrue(frags)
        for f in frags:
            if f["frag_smiles"] == "c1ccccc1":
                self.assertEqual(f["n_attach"], 1)
                self.assertEqual(len(f["attach_atoms"]), 1)
                self.assertEqual(len(f["attach_partner"]), 1)
                # benzene symmetry: any ring position is a valid canonical index
                plain = Chem.MolFromSmiles(f["frag_smiles"])
                atom = plain.GetAtomWithIdx(f["attach_atoms"][0])
                self.assertEqual(atom.GetSymbol(), "C")
                self.assertTrue(atom.GetIsAromatic())

    def test_attachment_parent_is_cut_bond(self):
        """attach_parent / attach_partner must be the two ends of a cut bond."""
        smi = "O=C(O)c1ccc(NC(=O)c2ccc(N3CCOCC3)cc2)cc1"
        mol = Chem.MolFromSmiles(smi)
        for f in decompose_parent(smi):
            self.assertEqual(len(f["attach_atoms"]), f["n_attach"])
            self.assertEqual(len(f["attach_parent"]), f["n_attach"])
            self.assertEqual(len(f["attach_partner"]), f["n_attach"])
            for a, b in zip(f["attach_parent"], f["attach_partner"]):
                bond = mol.GetBondBetweenAtoms(a, b)
                self.assertIsNotNone(bond)
                self.assertEqual(bond.GetBondType(), Chem.BondType.SINGLE)
                self.assertFalse(bond.IsInRing())

    def test_canonical_attachment_atoms(self):
        smi = "Cc1ccc2[nH]cc(CCN3CCC(C(=O)c4ccccc4)CC3)c2c1"
        for f in decompose_parent(smi):
            plain = Chem.MolFromSmiles(f["frag_smiles"])
            self.assertIsNotNone(plain)
            for a in f["attach_atoms"]:
                self.assertGreaterEqual(a, 0)
                self.assertLess(a, plain.GetNumAtoms())

    def test_whole_molecule_and_size_gate(self):
        frags = decompose_parent("c1ccccc1")
        self.assertEqual([f["role"] for f in frags], ["whole"])
        self.assertEqual(decompose_parent("CC"), [])  # below 50 Da, no cuts

    def test_role_tags(self):
        frags = decompose_parent("CCOC(=O)c1ccccc1C(=O)NCCOc1ccccc1")
        roles = {f["role"] for f in frags}
        self.assertIn("linker", roles)
        self.assertIn("substituent", roles)

    def test_unparseable_parent(self):
        self.assertIsNone(decompose_parent("not a smiles"))


if __name__ == "__main__":
    unittest.main()
