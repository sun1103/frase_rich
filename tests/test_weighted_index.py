"""Unit tests for the key/weight mechanics (no data files needed)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.geometry import (  # noqa: E402
    canonical_triplet_fp,
    enumerate_triplet_idxs,
    residue_code,
)
from frase_rich.index import EnvResidue, WeightedTripletIndex  # noqa: E402


def _triangle(code_offsets=(0, 0, 0)):
    """Three residues at 10 A sides; codes default to three different residues."""
    names = ["ALA", "PHE", "ASP"]
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 8.660254, 0.0]])
    return [
        EnvResidue(res_name=names[i], chain="A", res_seq=str(i + 1),
                   coords=pts[i], code=residue_code(names[i]) + code_offsets[i])
        for i in range(3)
    ]


class TestGeometry(unittest.TestCase):
    def test_fp_is_order_invariant(self):
        a, b, c = residue_code("ALA"), residue_code("PHE"), residue_code("ASP")
        self.assertEqual(canonical_triplet_fp((a, b, c)),
                         canonical_triplet_fp((b, c, a)))
        self.assertEqual(canonical_triplet_fp((a, b, c)),
                         canonical_triplet_fp((c, b, a)))

    def test_triangle_in_band(self):
        pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 8.66, 0.0]])
        self.assertEqual(enumerate_triplet_idxs(pts), [(0, 1, 2)])

    def test_triangle_out_of_band(self):
        pts = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [10.0, 17.32, 0.0]])
        self.assertEqual(enumerate_triplet_idxs(pts), [])


class TestWeightedIndex(unittest.TestCase):
    def _index(self):
        idx = WeightedTripletIndex()
        env_a = _triangle()
        env_b = _triangle()  # same environment, different complex
        env_c = [
            EnvResidue(res_name="GLY", chain="A", res_seq=str(i + 1),
                       coords=r.coords, code=residue_code("GLY"))
            for i, r in enumerate(_triangle())
        ]
        fp_a = {canonical_triplet_fp(tuple(r.code for r in env_a)): 1.0}
        fp_b = {canonical_triplet_fp(tuple(r.code for r in env_b)): 1.0}
        fp_c = {canonical_triplet_fp(tuple(r.code for r in env_c)): 1.0}
        idx.add_fragment("a", env_a, fp_a, meta={"pdb_id": "1aaa",
                                                "frag_smiles": "CC"})
        idx.add_fragment("b", env_b, fp_b, meta={"pdb_id": "2bbb",
                                                "frag_smiles": "CCC"})
        idx.add_fragment("c", env_c, fp_c, meta={"pdb_id": "3ccc",
                                                "frag_smiles": "CCCC"})
        return idx

    def test_uniform_weights_reduce_to_jaccard(self):
        idx = self._index()
        hits = idx.query(_triangle(), top_k=None)
        by_id = {h["frag_id"]: h for h in hits}
        self.assertAlmostEqual(by_id["a"]["score_jaccard"],
                               by_id["a"]["score_weighted"], places=12)
        self.assertEqual(by_id["a"]["n_match"], 1)

    def test_exclude_pdb(self):
        idx = self._index()
        hits = idx.query(_triangle(), top_k=None, exclude_pdb="1aaa")
        self.assertEqual([h["frag_id"] for h in hits], ["b"])

    def test_weights_have_mean_one(self):
        idx = WeightedTripletIndex()
        env = _triangle()
        # two fingerprints with raw weights 1 and 3
        fp1 = canonical_triplet_fp(tuple(r.code for r in env))
        fp2 = fp1 ^ 1  # arbitrary distinct key
        idx.add_fragment("x", env, {fp1: 1.0, fp2: 3.0},
                         meta={"pdb_id": "9zzz"})
        weights = [w for _, w in idx._postings[fp1]] + \
                  [w for _, w in idx._postings[fp2]]
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=12)
        self.assertAlmostEqual(float(np.sum(weights)), 2.0, places=12)


if __name__ == "__main__":
    unittest.main()
