"""Unit tests for explicit-H features, two-sided H-bonds and mean pooling."""

import os
import tempfile
import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from frase_rich.features import (
    AtomProps, _hbond_contributions, fragment_atom_props, interaction_energy,
    residue_atom_props,
)
from frase_rich.geometry import canonical_triplet_fp
from frase_rich.index import EnvResidue, WeightedTripletIndex


def _props(xyz, *, radius=None, charge=None, formal=None, donor=None,
           acceptor=None, hydrophobic=None, aromatic=None, is_h=False,
           parent=None, donor_h=None, donor_rotatable=None, acc_ideal=None,
           neighbors=None):
    n = len(xyz)
    xyz = np.asarray(xyz, dtype=np.float64)
    radius = (np.ones(n) * 1.7 if radius is None else np.asarray(radius, float))
    charge = (np.zeros(n) if charge is None else np.asarray(charge, float))
    formal = charge.copy() if formal is None else np.asarray(formal, float)
    donor = (np.zeros(n, bool) if donor is None else np.asarray(donor, bool))
    acceptor = (np.zeros(n, bool) if acceptor is None
                else np.asarray(acceptor, bool))
    hydrophobic = (np.zeros(n, bool) if hydrophobic is None
                   else np.asarray(hydrophobic, bool))
    aromatic = (np.zeros(n, bool) if aromatic is None
                else np.asarray(aromatic, bool))
    is_h = np.asarray(is_h, bool) if not np.isscalar(is_h) \
        else np.full(n, bool(is_h), dtype=bool)
    parent = (np.full(n, -1, int) if parent is None
              else np.asarray(parent, int))
    donor_h = (np.zeros(n, bool) if donor_h is None
               else np.asarray(donor_h, bool))
    donor_rotatable = (np.zeros(n, bool) if donor_rotatable is None
                       else np.asarray(donor_rotatable, bool))
    acc_ideal = (np.full(n, np.nan) if acc_ideal is None
                 else np.asarray(acc_ideal, float))
    if neighbors is None:
        neighbors = tuple(() for _ in range(n))
    return AtomProps(
        xyz=xyz, radius=radius, charge=charge, formal=formal, donor=donor,
        acceptor=acceptor, hydrophobic=hydrophobic, aromatic=aromatic,
        is_hydrogen=is_h, parent=parent, donor_h=donor_h,
        donor_rotatable=donor_rotatable, acc_ideal=acc_ideal,
        neighbors=neighbors)


def _hbond_pair(h_pos=(1.0, 0.0, 0.0), y_pos=(3.3, 0.866, 0.0)):
    """Donor D at origin, acceptor A at (2.8, 0, 0), Y at ``y_pos``."""
    frag = _props(
        [(0.0, 0.0, 0.0), h_pos], is_h=[False, True], parent=[-1, 0],
        donor=[True, False], donor_h=[False, True], acc_ideal=[np.nan, np.nan])
    res = _props(
        [(2.8, 0.0, 0.0), y_pos], acceptor=[True, False],
        acc_ideal=[120.0, np.nan], neighbors=[(1,), ()])
    return frag, res


class TestTwoSidedHBond(unittest.TestCase):
    def test_linear_contact_is_accepted(self):
        frag, res = _hbond_pair()
        vals = _hbond_contributions(frag, res)
        self.assertEqual(len(vals), 1)
        self.assertGreater(vals[0], 0.9)

    def test_donor_bent_is_rejected(self):
        # H pulled sideways: D-H...A is ~106 deg.
        frag, res = _hbond_pair(h_pos=(0.0, 1.0, 0.0))
        self.assertEqual(_hbond_contributions(frag, res), [])

    def test_acceptor_bent_is_rejected(self):
        # Y along A->H: C-A...H is 0 deg, far below the 90 deg gate.
        frag, res = _hbond_pair(y_pos=(1.8, 0.0, 0.0))
        self.assertEqual(_hbond_contributions(frag, res), [])

    def test_acceptor_off_ideal_is_penalised(self):
        # 90 deg from the 120 deg ideal: cos^2(30) = 0.75, still accepted.
        frag, res = _hbond_pair(y_pos=(2.8, 1.0, 0.0))
        vals = _hbond_contributions(frag, res)
        self.assertEqual(len(vals), 1)
        self.assertGreater(vals[0], 0.5)
        self.assertLess(vals[0], 0.9)

    def test_rotatable_hydroxyl_uses_best_direction(self):
        # Arbitrary H placement must not kill a good O...O contact.
        frag = _props(
            [(0.0, 0.0, 0.0), (0.6, 0.8, 0.0)], is_h=[False, True],
            parent=[-1, 0], donor=[True, False], donor_h=[False, True],
            donor_rotatable=[False, True])
        res = _props(
            [(2.8, 0.0, 0.0), (3.3, 0.866, 0.0)], acceptor=[True, False],
            acc_ideal=[120.0, np.nan], neighbors=[(1,), ()])
        vals = _hbond_contributions(frag, res)
        self.assertEqual(len(vals), 1)
        self.assertGreater(vals[0], 0.5)


class TestPooling(unittest.TestCase):
    def _system(self, n_frag: int):
        # One hydrophobic contact at 3.9 A, duplicated n_frag times.
        frag = _props([(0.0, 0.0, 0.0)] * n_frag,
                      hydrophobic=[True] * n_frag)
        res = _props([(3.9, 0.0, 0.0)], hydrophobic=[True])
        return frag, res

    def test_mean_is_invariant_to_duplicate_contacts(self):
        e1, _ = interaction_energy(*self._system(1), pooling="mean")
        e2, _ = interaction_energy(*self._system(4), pooling="mean")
        self.assertAlmostEqual(e1, e2, places=12)

    def test_sum_scales_with_contacts(self):
        s1, _ = interaction_energy(*self._system(1), pooling="sum")
        s4, _ = interaction_energy(*self._system(4), pooling="sum")
        prox = 0.1 * np.exp(-0.5 * ((3.9 - 5.0) / 2.0) ** 2)
        self.assertAlmostEqual(s4 - prox, 4.0 * (s1 - prox), places=12)

    def test_mean_is_smaller_than_sum(self):
        frag, res = self._system(3)
        mean, _ = interaction_energy(frag, res, pooling="mean")
        total, _ = interaction_energy(frag, res, pooling="sum")
        self.assertLess(mean, total)


class TestTripletEnergy(unittest.TestCase):
    def _env(self):
        # CA triangle with sides 10, 10, 10 (all in the 8-12 A band).
        return [
            EnvResidue("ALA", "A", "1", np.array([0.0, 0.0, 0.0]), 1),
            EnvResidue("GLY", "A", "10", np.array([10.0, 0.0, 0.0]), 2),
            EnvResidue("SER", "A", "20", np.array([5.0, 8.660254, 0.0]), 3),
        ]

    def test_energy_is_mean_triplet_energy_per_atom(self):
        from frase_rich.weighting import fragment_fp_energy

        energies = np.array([0.2, 0.1, 0.3])
        mean, emax = fragment_fp_energy(self._env(), energies, n_atoms=10)
        self.assertEqual(len(mean), 1)
        self.assertAlmostEqual(next(iter(mean.values())),
                               float(energies.mean()) / 10.0, places=12)
        self.assertAlmostEqual(next(iter(emax.values())),
                               float(energies.mean()) / 10.0, places=12)

    def test_energy_scales_inversely_with_atom_count(self):
        from frase_rich.weighting import fragment_fp_energy

        energies = np.array([0.2, 0.1, 0.3])
        mean1, max1 = fragment_fp_energy(self._env(), energies, n_atoms=10)
        mean2, max2 = fragment_fp_energy(self._env(), energies, n_atoms=20)
        self.assertEqual(set(mean1), set(mean2))
        for key in mean1:
            self.assertAlmostEqual(mean2[key], mean1[key] / 2.0, places=12)
            self.assertAlmostEqual(max2[key], max1[key] / 2.0, places=12)


class TestTripletTable(unittest.TestCase):
    def _env(self):
        return [
            EnvResidue("ALA", "A", "1", np.array([0.0, 0.0, 0.0]), 1,
                       weight=0.5),
            EnvResidue("GLY", "A", "10", np.array([10.0, 0.0, 0.0]), 2,
                       weight=0.3),
            EnvResidue("SER", "A", "20", np.array([5.0, 8.660254, 0.0]), 3,
                       weight=0.2),
        ]

    def test_precomputed_table_matches_legacy(self):
        from frase_rich.weighting import fragment_fp_weights, triplet_table

        env = self._env()
        legacy = fragment_fp_weights(env)
        tabled = fragment_fp_weights(env, table=triplet_table(env))
        self.assertEqual(set(legacy), set(tabled))
        for key in legacy:
            self.assertAlmostEqual(legacy[key], tabled[key], places=12)

    def test_min_leg_requires_all_three_contacts(self):
        from frase_rich.weighting import fragment_fp_energy, triplet_table

        env = self._env()
        table = triplet_table(env)
        # Totals are all positive (the ungated proximity term does that), but
        # the third residue's contact energy is zero: the triangle is out.
        mean, _emax, emin = fragment_fp_energy(
            env, np.array([0.31, 0.21, 0.01]), 10, table=table,
            return_min_leg=True,
            contact_energies=np.array([0.3, 0.2, 0.0]))
        key = next(iter(mean))
        self.assertAlmostEqual(emin[key], 0.0, places=12)
        _, _, emin_all = fragment_fp_energy(
            env, np.array([0.31, 0.21, 0.11]), 10, table=table,
            return_min_leg=True,
            contact_energies=np.array([0.3, 0.2, 0.1]))
        self.assertAlmostEqual(emin_all[key], 0.1 / 10.0, places=12)

    def test_contact_mean_uses_contact_energies(self):
        from frase_rich.weighting import fragment_fp_energy, triplet_table

        env = self._env()
        out = fragment_fp_energy(
            env, np.array([0.31, 0.21, 0.01]), 10, table=triplet_table(env),
            return_min_leg=True, return_contact_mean=True,
            contact_energies=np.array([0.3, 0.2, 0.0]))
        self.assertEqual(len(out), 4)
        _mean, _emax, emin, ecmean = out
        key = next(iter(ecmean))
        # Mean of the three contact legs, including the non-contacting one.
        self.assertAlmostEqual(ecmean[key], (0.3 + 0.2 + 0.0) / 3 / 10,
                               places=12)
        self.assertAlmostEqual(emin[key], 0.0, places=12)

    def test_contact_mean_is_zero_without_contacts(self):
        from frase_rich.weighting import fragment_fp_energy, triplet_table

        env = self._env()
        _mean, _emax, _emin, ecmean = fragment_fp_energy(
            env, np.array([0.31, 0.21, 0.11]), 10, table=triplet_table(env),
            return_min_leg=True, return_contact_mean=True,
            contact_energies=np.zeros(3))
        self.assertEqual(set(ecmean.values()), {0.0})

    def test_pocket_environment_is_ligand_union(self):
        from frase_rich.weighting import pocket_environment

        residues = [
            EnvResidue("ALA", "A", "1", np.array([1.0, 0.0, 0.0]), 1),
            EnvResidue("GLY", "A", "2", np.array([30.0, 0.0, 0.0]), 2),
            EnvResidue("SER", "A", "3", np.array([15.0, 20.0, 0.0]), 3),
        ]
        lig = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
        got = pocket_environment(residues, lig, radius=12.0)
        # Both ends are included by the union; a centroid sphere at (15,0)
        # would drop residues 1 and 2.
        self.assertEqual([r.res_seq for r in got], ["1", "2"])


class TestMarkScoring(unittest.TestCase):
    def test_uniform_marks_reduce_to_jaccard(self):
        from frase_rich.weighting import mark_overlap, normalize_marks

        q = normalize_marks({1: 2.0, 2: 2.0, 3: 0.0})
        lib = normalize_marks({2: 5.0, 3: 5.0, 4: 5.0})
        # Supports {1, 2} and {2, 3, 4}: exact-key Jaccard is 1/4.
        self.assertAlmostEqual(mark_overlap(q, lib), 0.25, places=12)

    def test_normalize_drops_non_positive_support(self):
        from frase_rich.weighting import normalize_marks

        norm = normalize_marks({1: 2.0, 2: 2.0, 3: 0.0, 4: -1.0})
        self.assertEqual(set(norm), {1, 2})
        self.assertAlmostEqual(sum(norm.values()), 2.0, places=12)

    def test_min_overlap_is_bounded_by_both_sides(self):
        from frase_rich.weighting import mark_overlap, normalize_marks

        q = normalize_marks({1: 4.0, 2: 1.0})
        lib = normalize_marks({1: 1.0, 2: 4.0})
        # Mean-1 scaling gives (1.6, 0.4) and (0.4, 1.6); W = 0.8.
        self.assertAlmostEqual(mark_overlap(q, lib), 0.8 / (2 + 2 - 0.8),
                               places=12)

    def test_dense_marks_match_pairwise_overlap(self):
        from types import SimpleNamespace

        from frase_rich.evaluate import _build_dense_marks, _dense_mark_scores
        from frase_rich.weighting import mark_overlap, normalize_marks

        key = "_fp_e_atom_mean"
        idx = SimpleNamespace(_frags={
            "a": {key: {1: 1.0, 2: 3.3, 3: 0.0}},
            "b": {key: {2: 2.0, 4: 0.5}},
            "c": {},
            "d": {key: {4: 1.0, 5: 2.0, 6: -1.0}},
        })
        q = normalize_marks({1: 2.5, 2: 0.7, 9: 1.0})  # key 9 is query-only
        dense = _build_dense_marks(idx, key)
        scores = _dense_mark_scores(q, dense)
        row_of = dense[3]
        for fid, info in idx._frags.items():
            want = mark_overlap(
                q, normalize_marks(info.get(key) or {}))
            self.assertAlmostEqual(scores[row_of[fid]], want, places=12)


class TestResidueProps(unittest.TestCase):
    def test_backbone_flag_controls_atoms(self):
        from frase_rich.protein import AtomH

        atoms = [
            AtomH("N", "N", 0.0, 0.0, 0.0),
            AtomH("CA", "C", 1.4, 0.0, 0.0),
            AtomH("C", "C", 2.0, 1.4, 0.0),
            AtomH("O", "O", 1.3, 2.3, 0.0),
            AtomH("CB", "C", 2.0, -1.0, 0.0),
        ]
        full = residue_atom_props("ALA", atoms, backbone=True)
        side = residue_atom_props("ALA", atoms, backbone=False)
        self.assertEqual(len(full), 5)
        self.assertEqual(len(side), 1)

    def test_crystallographic_hydrogen_is_dropped(self):
        # parse_pdb atoms have no is_hydrogen flag; element H must still be
        # treated as hydrogen or legacy builds keep deposited H atoms.
        from frase_rich.features import residue_atom_props

        class LegacyAtom:
            def __init__(self, name, element, x, y, z):
                self.name = name
                self.element = element
                self.x, self.y, self.z = x, y, z

        atoms = [LegacyAtom("CB", "C", 0.0, 0.0, 0.0),
                 LegacyAtom("H1", "H", 1.0, 0.0, 0.0)]
        props = residue_atom_props("ALA", atoms, backbone=False)
        self.assertEqual(len(props), 1)
        self.assertFalse(props.is_hydrogen.any())

    def test_legacy_donor_acceptor_sets(self):
        from frase_rich.features import fragment_atom_props, residue_atom_props

        class LegacyAtom:
            def __init__(self, name, element, x, y, z):
                self.name = name
                self.element = element
                self.x, self.y, self.z = x, y, z

        gly = [LegacyAtom("N", "N", 0.0, 0.0, 0.0),
               LegacyAtom("CA", "C", 1.4, 0.0, 0.0)]
        new = residue_atom_props("GLY", gly, backbone=True)
        old = residue_atom_props("GLY", gly, backbone=True,
                                 legacy_hbond=True)
        self.assertFalse(new.acceptor[0])   # backbone amide N
        self.assertTrue(old.acceptor[0])    # v1 accepted every N

        mol = Chem.AddHs(Chem.MolFromSmiles("CS"))
        AllChem.EmbedMolecule(mol, randomSeed=7)
        self.assertTrue(fragment_atom_props(mol).donor[1])
        self.assertFalse(
            fragment_atom_props(mol, legacy_hbond=True).donor[1])

    def test_histidine_variants_differ_on_one_hydrogen(self):
        from frase_rich.protein import AtomH, _his_variants

        names = ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"]
        atoms = [AtomH(nm, nm[0], float(i), 0.0, 0.0)
                 for i, nm in enumerate(names)]
        atoms.append(AtomH("H", "H", 0.0, 0.0, 0.0,
                           is_hydrogen=True, parent=9))
        for i, a in enumerate(atoms[:-1]):
            a.neighbors = tuple(j for j in range(len(atoms) - 1) if j != i)[:2]
        variants = _his_variants("HIS", atoms)
        self.assertEqual(len(variants), 2)
        parents = [{a.parent for a in v if a.is_hydrogen} for v in variants]
        self.assertEqual(parents, [{6}, {9}])


class TestRefMolScoring(unittest.TestCase):
    def _index(self, weights: dict[int, float] | None = None):
        idx = WeightedTripletIndex()
        env = [
            EnvResidue("ALA", "A", "1", np.array([0.0, 0.0, 0.0]), 1,
                       weight=1 / 3),
            EnvResidue("GLY", "A", "10", np.array([10.0, 0.0, 0.0]), 2,
                       weight=1 / 3),
            EnvResidue("SER", "A", "20", np.array([5.0, 9.0, 0.0]), 3,
                       weight=1 / 3),
        ]
        fps = {canonical_triplet_fp((1, 2, 3)): 1.0}
        if weights:
            fps = weights
        idx.add_fragment("f1", env, fps)
        return idx, set(fps)

    def test_uniform_refmol_query_equals_jaccard(self):
        idx, fps = self._index()
        hits = idx.query([], query_fp_weights={fp: 1.0 for fp in fps},
                         top_k=None)
        self.assertTrue(hits)
        for row in hits:
            self.assertAlmostEqual(row["score_weighted"],
                                   row["score_jaccard"], places=12)

    def test_min_is_not_above_geomean(self):
        idx, fps = self._index()
        q = {fp: float(i + 2) for i, fp in enumerate(sorted(fps))}
        # Add a second shared key so the two combinations can differ.
        q[next(iter(fps)) + 1] = 1.0
        hits_min = idx.query([], query_fp_weights=q, combine="min", top_k=None)
        hits_geo = idx.query([], query_fp_weights=q, combine="geomean",
                             top_k=None)
        for a, b in zip(hits_min, hits_geo):
            self.assertLessEqual(a["score_weighted"], b["score_weighted"] + 1e-12)

    def test_v1_index_still_loads(self):
        import pickle

        idx, fps = self._index()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            pickle.dump({
                "schema": "frase_rich.weighted_index.v1",
                "weight_mode": idx.weight_mode,
                "postings": idx._postings,
                "frags": idx._frags,
            }, fh)
            path = fh.name
        loaded = WeightedTripletIndex.load(path)
        self.assertEqual(loaded.features, {})
        self.assertEqual(len(loaded), 1)
        os.unlink(path)

    def test_v2_features_roundtrip(self):
        idx, _ = self._index()
        idx.features = {"explicit_h": True, "energy_pooling": "mean"}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            path = fh.name
        idx.save(path)
        loaded = WeightedTripletIndex.load(path)
        self.assertEqual(loaded.features["explicit_h"], True)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
