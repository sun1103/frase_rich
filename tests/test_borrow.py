"""Borrow channel: environment ranking, attach recovery, ligand-only context."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.evaluate import _RefFragProvider  # noqa: E402
from frase_rich.features import attachment_atoms  # noqa: E402
from frase_rich.index import WeightedTripletIndex  # noqa: E402
from frase_rich.weighting import normalize_marks  # noqa: E402
from scripts.patent_place import (  # noqa: E402
    _borrow_rows,
    _canonical_attach,
    _joint_pick,
    _prepare_borrow,
    _summarize_combined,
)


def _index(rows) -> WeightedTripletIndex:
    idx = WeightedTripletIndex()
    for fid, meta, fp in rows:
        idx.add_fragment(fid, [], fp, meta=meta)
    return idx


class CanonicalAttachTest(unittest.TestCase):
    def test_maps_into_canonical_smiles_order(self):
        # "OCC" canonicalizes to "CCO": the O moves from 0 to 2.
        smiles, attach = _canonical_attach(Chem.MolFromSmiles("OCC"), {0})
        self.assertEqual(smiles, "CCO")
        self.assertEqual(attach, [2])

    def test_multi_atom_fragment(self):
        mol = Chem.MolFromSmiles("CC(=O)OC")
        smiles, attach = _canonical_attach(mol, {0, 2})
        self.assertEqual(smiles, Chem.MolToSmiles(mol))
        self.assertEqual(len(attach), 2)
        self.assertEqual(len(set(attach)), 2)
        self.assertTrue(all(0 <= a < mol.GetNumAtoms() for a in attach))


class BorrowRowsTest(unittest.TestCase):
    ROWS = [
        ("a_1", {"pdb_id": "a", "n_attach": 1, "n_heavy": 10,
                 "_fp_e_atom_mean": {5: 4.0}}, {1: 2.0, 2: 1.0}),
        ("b_1", {"pdb_id": "b", "n_attach": 1, "n_heavy": 10,
                 "_fp_e_atom_mean": {5: 1.0, 7: 1.0}}, {1: 1.0, 3: 1.0}),
        ("b_2", {"pdb_id": "b", "n_attach": 2, "n_heavy": 10,
                 "_fp_e_atom_mean": {5: 3.0}}, {1: 3.0}),
        ("c_1", {"pdb_id": "c", "n_attach": 1, "n_heavy": 25,
                 "_fp_e_atom_mean": {5: 1.0}}, {1: 1.0}),
    ]

    def setUp(self):
        self.index = _index(self.ROWS)
        self.q_marks = normalize_marks({5: 1.0})
        self.q_weights = normalize_marks({1: 1.0})

    def _rows(self, borrow, *, band=8, exclude="a", include_self=False):
        rows, score = _borrow_rows(
            borrow, self.q_weights, self.q_marks, n_attach=1, n_heavy=10,
            heavy_band=band, exclude_pdb=exclude,
            include_self=include_self, combine="min")
        return [borrow["fids"][int(r)] for r in rows], score

    def test_mark_ranking_applies_identity_gates(self):
        borrow = _prepare_borrow(self.index, rank="mark", mark="mean")
        names, _ = self._rows(borrow)
        self.assertEqual(names, ["b_1"])

    def test_heavy_band_and_ranking(self):
        borrow = _prepare_borrow(self.index, rank="mark", mark="mean")
        names, _ = self._rows(borrow, band=20)
        # c_1 shares one key with a stronger mark overlap than b_1.
        self.assertEqual(names, ["c_1", "b_1"])

    def test_include_self_keeps_the_query_pdb(self):
        borrow = _prepare_borrow(self.index, rank="mark", mark="mean")
        names, _ = self._rows(borrow, include_self=True)
        self.assertEqual(names, ["a_1", "b_1"])

    def test_weight_ranking_uses_exact_key_overlap(self):
        borrow = _prepare_borrow(self.index, rank="weight", mark="mean")
        names, _ = self._rows(borrow, band=20)
        self.assertEqual(names, ["c_1", "b_1"])


def _hit(source, *, tanimoto=0.0, env_score=None, persist=0.5,
         energy=0.4, name=None):
    return {
        "source": source,
        "rank": 1,
        "cand_smiles": name or f"{source}_cand",
        "tanimoto": tanimoto,
        "env_score": env_score,
        "pool_row": 1 if source == "pool" else None,
        "borrow_fid": None if source == "pool" else "f1",
        "borrow_pdb": None if source == "pool" else "p1",
        "realistic": {"weighted": 0.9, "clashes": 0,
                      "shell_energy_r_mean": energy,
                      "shell_persist_energy_r": persist},
    }


class JointSelectionTest(unittest.TestCase):
    def test_picks_the_channel_with_better_contact_fit(self):
        record = {"top_hits": [_hit("pool", tanimoto=0.9, persist=0.5)],
                  "borrow_hits": [_hit("borrow", env_score=0.7,
                                       persist=0.7)]}
        pick = _joint_pick(record)
        self.assertEqual(pick["source"], "borrow")
        self.assertAlmostEqual(pick["retrieval_score"], 0.7)
        self.assertIsNone(pick["pool_row"])
        self.assertEqual(pick["borrow_fid"], "f1")

        record = {"top_hits": [_hit("pool", tanimoto=0.9, persist=0.8)],
                  "borrow_hits": [_hit("borrow", env_score=0.9,
                                       persist=0.7)]}
        pick = _joint_pick(record)
        self.assertEqual(pick["source"], "pool")
        self.assertAlmostEqual(pick["retrieval_score"], 0.9)

    def test_channel_gates_are_independent(self):
        record = {"top_hits": [_hit("pool", tanimoto=0.5, persist=0.9)],
                  "borrow_hits": [_hit("borrow", env_score=0.5,
                                       persist=0.9)]}
        self.assertIsNone(_joint_pick(record, shell_gate=0.6))
        record["borrow_hits"][0]["env_score"] = 0.6
        self.assertEqual(_joint_pick(record, shell_gate=0.6)["source"],
                         "borrow")

    def test_summary_counts_paired_wins_and_source_mix(self):
        records = [
            {"top_hits": [_hit("pool", tanimoto=0.8, persist=0.5)],
             "borrow_hits": [_hit("borrow", env_score=0.7, persist=0.6)]},
            {"top_hits": [_hit("pool", tanimoto=0.9, persist=0.7)],
             "borrow_hits": [_hit("borrow", env_score=0.4, persist=0.9)]},
            {"top_hits": [], "borrow_hits": []},
        ]
        for record in records:
            record["selected"] = _joint_pick(record)
        summary = _summarize_combined(records)
        self.assertEqual(summary["n_solved"],
                         {"pool": 2, "borrow": 1, "union": 2})
        self.assertEqual(summary["paired"]["n"], 1)
        self.assertEqual(summary["paired"]["wins"]["borrow"], 1)
        self.assertAlmostEqual(summary["paired"]["mean_delta"], 0.1)
        self.assertEqual(summary["selected_source"],
                         {"pool": 1, "borrow": 1, "none": 1})
        self.assertAlmostEqual(summary["mean_persist"]["union"], 0.65)


class LigandContextTest(unittest.TestCase):
    def test_recovers_fragment_cuts_without_the_pocket(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1-c1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        for atom in mol.GetAtoms():
            info = Chem.AtomPDBResidueInfo()
            info.SetIsHeteroAtom(True)
            info.SetResidueName("LIG")
            info.SetResidueNumber(1)
            info.SetChainId("A")
            info.SetName(f"{atom.GetSymbol()}{atom.GetIdx() + 1}")
            atom.SetMonomerInfo(info)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "1aaa.pdb")
            with open(path, "w") as handle:
                handle.write(Chem.MolToPDBBlock(mol))
            provider = _RefFragProvider(
                tmp, {"explicit_h": True}, paths={"1aaa": path})
            ctx = provider.ligand_context("1aaa", "c1ccccc1-c1ccccc1")
            self.assertIsNotNone(ctx)
            lig, by_smiles = ctx
            frags = [f for smiles in by_smiles
                     for f in by_smiles.all(smiles)]
            self.assertTrue(frags)
            self.assertTrue(any(attachment_atoms(f.mol, lig) for f in frags))


if __name__ == "__main__":
    unittest.main()
