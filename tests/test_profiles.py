"""Tests for cross-structure residue profiles (no cache needed)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.profiles import (  # noqa: E402
    attach_sequence_keys,
    profile_similarity,
    rank_score,
    sequence_residue_map,
    store_features,
)


def _atom_line(serial: int, res: str, chain: str, resseq: int,
               icode: str = "", atom: str = "CA") -> str:
    return ("ATOM  %5d %4s %3s %1s%4d%1s    %8.3f%8.3f%8.3f  1.00  0.00\n"
            % (serial, atom, res, chain, resseq, icode, 0.0, 0.0, 0.0))


def _write_pdb(path: str, chains: dict[str, tuple[int, list[str]]],
               extra_lines: list[str] | None = None) -> None:
    """``chains`` maps a label to (first resseq, residue names)."""
    with open(path, "w") as handle:
        serial = 1
        for chain, (start, residues) in chains.items():
            for offset, res in enumerate(residues):
                handle.write(_atom_line(serial, res, chain, start + offset))
                serial += 1
        for line in extra_lines or ():
            handle.write(line)


def _res(chain: str, res_seq: str, res_name: str = "ALA"):
    return SimpleNamespace(chain=chain, res_seq=res_seq, res_name=res_name)


def _profile(keys, weights, contacts, mask, names=None):
    return {
        "seq_id": np.array([k[0] for k in keys], dtype=np.int64),
        "seq_pos": np.array([k[1] for k in keys], dtype=np.int64),
        "weights": np.asarray(weights, dtype=np.float64),
        "contacts": np.asarray(contacts, dtype=np.float64),
        "mask": np.asarray(mask, dtype=bool),
        "res_names": list(names or ["ALA"] * len(keys)),
    }


class TestSequenceMap(unittest.TestCase):
    def test_chain_labels_and_numbering_do_not_matter(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.pdb")
            b = os.path.join(tmp, "b.pdb")
            _write_pdb(a, {"A": (1, ["ALA", "GLY", "SER"])})
            _write_pdb(b, {"H": (101, ["ALA", "GLY", "SER"])})
            map_a = sequence_residue_map(a)
            map_b = sequence_residue_map(b)
            self.assertEqual(map_a[("A", "1")], map_b[("H", "101")])
            self.assertEqual(map_a[("A", "2")][1], 1)
            self.assertEqual(map_a[("A", "3")][1], 2)
            # CRC32 is a property of the sequence, not the chain label.
            self.assertEqual(map_a[("A", "1")][0], map_b[("H", "101")][0])

    def test_homodimer_shares_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dimer.pdb")
            _write_pdb(path, {
                "A": (1, ["ALA", "GLY"]),
                "B": (11, ["ALA", "GLY"]),
            })
            mapping = sequence_residue_map(path)
            self.assertEqual(mapping[("A", "1")], mapping[("B", "11")])
            self.assertEqual(mapping[("A", "2")], mapping[("B", "12")])

    def test_different_sequences_get_different_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hetero.pdb")
            _write_pdb(path, {
                "A": (1, ["ALA", "GLY", "SER"]),
                "B": (1, ["GLY", "SER", "ALA"]),
            })
            mapping = sequence_residue_map(path)
            self.assertNotEqual(mapping[("A", "1")][0], mapping[("B", "1")][0])

    def test_insertion_codes_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "icode.pdb")
            with open(path, "w") as handle:
                handle.write(_atom_line(1, "ALA", "A", 52))
                handle.write(_atom_line(2, "GLY", "A", 52, icode="A"))
            mapping = sequence_residue_map(path)
            self.assertIn(("A", "52"), mapping)
            self.assertIn(("A", "52A"), mapping)
            self.assertEqual(mapping[("A", "52A")][1], 1)

    def test_non_ca_and_unknown_residues_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "side.pdb")
            _write_pdb(
                path, {"A": (1, ["ALA"])},
                extra_lines=[
                    # CB of the first residue, then an unknown residue name.
                    _atom_line(2, "ALA", "A", 1, atom="CB"),
                    _atom_line(3, "UNK", "A", 2),
                ])
            mapping = sequence_residue_map(path)
            self.assertEqual(sorted(mapping), [("A", "1")])

    def test_duplicate_ca_records_collapsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dupe.pdb")
            with open(path, "w") as handle:
                handle.write(_atom_line(1, "ALA", "A", 1))
                handle.write(_atom_line(2, "ALA", "A", 1))  # alternate location
                handle.write(_atom_line(3, "GLY", "A", 2))
            mapping = sequence_residue_map(path)
            self.assertEqual(mapping[("A", "2")][1], 1)


class TestAttachKeys(unittest.TestCase):
    def _seq_map(self):
        return {("A", "1"): (7, 0), ("A", "2"): (7, 1),
                ("B", "11"): (7, 0), ("B", "12"): (7, 1)}

    def test_homodimer_aggregation(self):
        env = [_res("A", "1", "ALA"), _res("B", "11", "ALA"),
               _res("A", "2", "GLY"), _res("B", "12", "GLY")]
        raw = {
            "weights": np.array([0.25, 0.75, 0.10, 0.00]),
            "contacts": np.array([1.0, 2.0, -1.0, 3.0]),
            "mask": np.array([False, True, True, False]),
            "extra": "kept",
        }
        out = attach_sequence_keys(raw, env, self._seq_map())
        self.assertEqual(out["seq_id"].tolist(), [7, 7])
        self.assertEqual(out["seq_pos"].tolist(), [0, 1])
        np.testing.assert_allclose(out["weights"], [1.0, 0.10])
        np.testing.assert_allclose(out["contacts"], [3.0, 2.0])
        self.assertEqual(out["mask"].tolist(), [True, True])
        self.assertEqual(out["res_names"], ["ALA", "GLY"])
        self.assertEqual(out["extra"], "kept")
        # The input profile must not be mutated.
        np.testing.assert_allclose(raw["weights"], [0.25, 0.75, 0.10, 0.0])

    def test_unmapped_residues_dropped(self):
        env = [_res("A", "1"), _res("Z", "9")]
        raw = {"weights": np.array([0.5, 0.5]),
               "contacts": np.array([1.0, 1.0]),
               "mask": np.array([True, True])}
        out = attach_sequence_keys(raw, env, self._seq_map())
        self.assertEqual(len(out["seq_id"]), 1)


class TestProfileSimilarity(unittest.TestCase):
    def _identical(self):
        keys = [(7, i) for i in range(4)]
        return _profile(keys, [0.1, 0.2, 0.3, 0.4],
                        [1.0, 2.0, 3.0, 4.0], [True] * 4)

    def test_empty_shell(self):
        keys = [(7, i) for i in range(4)]
        q = _profile(keys, [0.25] * 4, [1.0] * 4, [False] * 4)
        m = profile_similarity(q, self._identical())
        self.assertEqual(m["n_shell"], 0)
        self.assertEqual(m["n_shared"], 0)
        self.assertEqual(m["weight_cos"], 0.0)
        self.assertIsNone(m["energy_r"])

    def test_no_shared_residues(self):
        q = _profile([(7, i) for i in range(3)], [0.3] * 3,
                     [1.0] * 3, [True] * 3)
        c = _profile([(9, i) for i in range(3)], [0.3] * 3,
                     [1.0] * 3, [True] * 3)
        m = profile_similarity(q, c)
        self.assertEqual(m["n_shell"], 3)
        self.assertEqual(m["n_shared"], 0)
        self.assertIsNone(m["energy_r"])
        self.assertEqual(rank_score(m, "energy"), -2.0)

    def test_identical_profiles(self):
        q = self._identical()
        m = profile_similarity(q, q)
        self.assertEqual(m["n_shared"], 4)
        self.assertAlmostEqual(m["weight_cos"], 1.0, places=12)
        self.assertAlmostEqual(m["all_weight_cos"], 1.0, places=12)
        for key in ("energy_r", "weighted_energy_r", "energy_r_fav",
                    "energy_r_engaged", "weighted_energy_r_strength",
                    "all_energy_r", "all_weighted_energy_r"):
            self.assertAlmostEqual(m[key], 1.0, places=12, msg=key)
        for key in ("shell_recall", "shell_weight_recall",
                    "contact_retention", "contact_jaccard",
                    "overlap_energy"):
            self.assertAlmostEqual(m[key], 1.0, places=12, msg=key)
        self.assertAlmostEqual(m["delta_energy"], 0.0, places=12)

    def test_anti_correlated_energies(self):
        keys = [(7, i) for i in range(4)]
        q = _profile(keys, [0.25] * 4, [1.0, 2.0, 3.0, 4.0], [True] * 4)
        c = _profile(keys, [0.25] * 4, [4.0, 3.0, 2.0, 1.0], [True] * 4)
        m = profile_similarity(q, c)
        self.assertAlmostEqual(m["energy_r"], -1.0, places=12)
        self.assertAlmostEqual(m["overlap_energy"], 0.0, places=12)

    def test_too_few_shared_suppresses_correlation(self):
        keys = [(7, 0), (7, 1)]
        q = _profile(keys, [0.5, 0.5], [1.0, 2.0], [True, True])
        c = _profile(keys, [0.5, 0.5], [2.0, 1.0], [True, True])
        m = profile_similarity(q, c)
        self.assertEqual(m["n_shared"], 2)
        self.assertIsNone(m["energy_r"])
        self.assertIsNone(m["weighted_energy_r"])
        self.assertAlmostEqual(m["weight_cos"], 1.0, places=12)

    def test_shell_restriction(self):
        keys = [(7, i) for i in range(4)]
        q = _profile(keys, [0.25] * 4, [1.0, 1.0, 0.0, 0.0],
                     [True, True, False, False])
        c = _profile(keys, [0.25] * 4, [1.0, -2.0, 5.0, 5.0],
                     [True] * 4)
        m = profile_similarity(q, c)
        self.assertEqual(m["n_shell"], 2)
        self.assertEqual(m["n_shared"], 2)
        # One of two shell residues is touched favourably.
        self.assertAlmostEqual(m["shell_recall"], 0.5)
        self.assertAlmostEqual(m["shell_weight_recall"], 0.5)
        self.assertAlmostEqual(m["contact_retention"], 0.5)
        # 0.5*(1-1) + 0.5*(-2-1)
        self.assertAlmostEqual(m["delta_energy"], -1.5)
        # Whole-pocket metrics still see all four residues.
        self.assertEqual(m["all_energy_r"] is not None, True)

    def test_unfavourable_candidate_contact_not_retained(self):
        keys = [(7, i) for i in range(3)]
        q = _profile(keys, [1 / 3] * 3, [1.0, 1.0, 1.0], [True] * 3)
        c = _profile(keys, [1 / 3] * 3, [1.0, -1.0, 1.0], [True] * 3)
        m = profile_similarity(q, c)
        self.assertAlmostEqual(m["contact_retention"], 2 / 3)
        self.assertAlmostEqual(m["contact_jaccard"], 2 / 3)
        self.assertAlmostEqual(m["shell_recall"], 2 / 3)
        self.assertAlmostEqual(m["delta_energy"], -2 / 3)

    def test_row_order_does_not_matter(self):
        q = self._identical()
        c = self._identical()
        order = [2, 0, 3, 1]
        shuffled = {
            "seq_id": c["seq_id"][order],
            "seq_pos": c["seq_pos"][order],
            "weights": c["weights"][order],
            "contacts": c["contacts"][order],
            "mask": c["mask"][order],
            "res_names": [c["res_names"][i] for i in order],
        }
        a = profile_similarity(q, c)
        b = profile_similarity(q, shuffled)
        for key, value in a.items():
            if isinstance(value, float):
                self.assertAlmostEqual(value, b[key], places=12, msg=key)
            else:
                self.assertEqual(value, b[key], msg=key)

    def test_zero_query_weights_fall_back_to_uniform(self):
        keys = [(7, i) for i in range(4)]
        q = _profile(keys, [0.0] * 4, [1.0, 2.0, 3.0, 4.0], [True] * 4)
        m = profile_similarity(q, q)
        self.assertAlmostEqual(m["weighted_energy_r"], 1.0, places=12)
        self.assertAlmostEqual(m["delta_energy"], 0.0, places=12)


class TestRankScore(unittest.TestCase):
    def _metrics(self):
        return {
            "weight_cos": 0.4, "energy_r": 0.2, "weighted_energy_r": 0.3,
            "energy_r_fav": 0.25, "energy_r_engaged": 0.15,
            "weighted_energy_r_strength": 0.35,
            "contact_jaccard": 0.5, "overlap_energy": 0.45,
            "all_weight_cos": 0.6, "all_energy_r": 0.1,
            "all_weighted_energy_r": 0.12, "shell_weight_recall": 0.8,
            "contact_retention": 0.7, "delta_energy": -0.2,
        }

    def test_values(self):
        m = self._metrics()
        expected = {
            "weight_cos": 0.4, "energy": 0.2, "weighted_energy": 0.3,
            "energy_fav": 0.25, "energy_engaged": 0.15,
            "weighted_energy_strength": 0.35, "contact_jaccard": 0.5,
            "overlap_energy": 0.45, "all_weight_cos": 0.6,
            "all_energy": 0.1, "all_weighted_energy": 0.12,
            "retention": 0.7, "delta_energy": 0.2,
        }
        for kind, value in expected.items():
            self.assertAlmostEqual(rank_score(m, kind), value, places=12)
        self.assertAlmostEqual(
            rank_score(m, "combined"),
            0.5 * (0.2 + 0.3) + 0.5 * 0.4, places=12)
        self.assertAlmostEqual(rank_score(m, "recall_energy"), 0.8 * 1.3,
                               places=12)

    def test_missing_energy_below_zero(self):
        m = self._metrics()
        m["energy_r"] = None
        m["weighted_energy_r"] = None
        m["all_energy_r"] = None
        self.assertEqual(rank_score(m, "energy"), -2.0)
        self.assertEqual(rank_score(m, "weighted_energy"), -2.0)
        self.assertEqual(rank_score(m, "all_energy"), -2.0)
        self.assertAlmostEqual(
            rank_score(m, "combined"), 0.5 * (-1 - 1) + 0.5 * 0.4, places=12)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            rank_score(self._metrics(), "nope")


class TestStoreFeatures(unittest.TestCase):
    def test_subset_only(self):
        features = {
            "explicit_h": True, "energy_pooling": "mean", "temperature": 1.0,
            "fp_aggregate": "mean", "use_pair_bias": True, "unrelated": 42,
        }
        self.assertEqual(store_features(features),
                         {"explicit_h": True, "energy_pooling": "mean",
                          "temperature": 1.0, "fp_aggregate": "mean",
                          "use_pair_bias": True})


if __name__ == "__main__":
    unittest.main()
