import math
import unittest
from types import SimpleNamespace

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from frase_rich.patent_decomp import _smiles_order
from frase_rich.patent_place import (
    _pinned_fragment_from_ligand,
    _profile_pose_indices,
    attachment_fingerprint,
    attachment_identity,
    axis_rotation,
    build_fingerprints,  # noqa: F401  (import check)
    clash_stats,
    contact_shell,
    dummy_fragment_smiles,
    exit_directions,
    fuse_rankings,
    kabsch,
    persistence_overlap,
    profile_overlap,
    normalize_weights,
    placements,
    query_exit_vectors,
    query_fingerprint,
    tanimoto_topk,  # noqa: F401
    weight_overlap,
)
from frase_rich.fragmentation import fragment_ligand
from frase_rich.weighting import fragment_fp_weights

SMILE = "COc1ccc(-c2ccccc2)cc1C(=O)Nc1ccc(Cl)cc1"


def _embedded(smiles=SMILE, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    return Chem.RemoveHs(mol)


def _canonical_positions(fragment):
    """Fragment atom index -> canonical SMILES position."""
    Chem.MolToSmiles(fragment)  # writes _smilesAtomOutputOrder
    order = _smiles_order(fragment)
    pos = [0] * fragment.GetNumAtoms()
    for canon, orig in enumerate(order):
        pos[orig] = canon
    return pos


class KabschTest(unittest.TestCase):
    def test_recovers_rigid_transform(self):
        rng = np.random.default_rng(3)
        src = rng.normal(size=(6, 3))
        r = axis_rotation([0.3, -0.5, 0.8], 1.1)
        t = np.array([2.0, -1.0, 0.5])
        dst = src @ r.T + t
        r2, t2, rmsd = kabsch(src, dst)
        self.assertLess(rmsd, 1e-9)
        np.testing.assert_allclose(r2, r, atol=1e-9)
        np.testing.assert_allclose(t2, t, atol=1e-9)

    def test_axis_rotation_is_rigid(self):
        r = axis_rotation([1.0, 2.0, 3.0], 0.7)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1.0, places=12)
        v = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
        np.testing.assert_allclose(r @ v, v, atol=1e-12)


class ExitVectorTest(unittest.TestCase):
    def test_exit_vectors_are_cut_bonds(self):
        lig = _embedded()
        frags = [f for f in fragment_ligand(lig, max_frags=200)
                 if f.n_attach >= 1]
        self.assertTrue(frags)
        checked = 0
        for frag in frags[:20]:
            att, att_xyz, exit_xyz = query_exit_vectors(frag.mol, lig)
            self.assertEqual(len(att), frag.n_attach)
            self.assertEqual(att_xyz.shape, (frag.n_attach, 3))
            self.assertEqual(exit_xyz.shape, (frag.n_attach, 3))
            d = np.linalg.norm(att_xyz - exit_xyz, axis=1)
            self.assertTrue(np.all(d > 0.9) and np.all(d < 2.0), d)
            checked += 1
        self.assertGreater(checked, 3)

    def test_candidate_exit_directions_are_unit_h_bonds(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        mh = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mh, randomSeed=5)
        dirs = exit_directions(mh, [0])
        self.assertEqual(len(dirs), 1)
        self.assertEqual(len(dirs[0]), 1)  # one aromatic H at that carbon
        self.assertAlmostEqual(float(np.linalg.norm(dirs[0][0])), 1.0,
                               places=9)


class PlacementTest(unittest.TestCase):
    def test_placement_reproduces_pose_from_same_conformer(self):
        """Geometry-only check: the fragment's own 3D conformer, placed as a
        candidate, must land back on the query pose.

        Using the query conformer (rather than a fresh ETKDG one) isolates
        the placement math from conformer generation: exit directions, Kabsch
        and the roll enumeration must reproduce the pose exactly.
        """
        lig = _embedded()
        frag = next(f for f in fragment_ligand(lig, max_frags=200)
                    if f.n_attach == 1 and f.n_heavy >= 10)
        att, att_xyz, exit_xyz = query_exit_vectors(frag.mol, lig)
        self.assertEqual(len(att), 1)

        cand = Chem.AddHs(frag.mol, addCoords=True)
        poses = placements(cand, 0, att, att_xyz, exit_xyz)
        self.assertTrue(poses)

        heavy = [a.GetIdx() for a in cand.GetAtoms() if a.GetAtomicNum() > 1]
        qpos = frag.mol.GetConformer().GetPositions()
        best = None
        for r, t, rmsd in poses:
            xyz = cand.GetConformer().GetPositions() @ r.T + t
            dev = float(np.mean(np.linalg.norm(xyz[heavy] - qpos, axis=1)))
            if best is None or dev < best[0]:
                best = (dev, float(rmsd))
        # The cap H is ~4 deg off the true exit vector, so the irreducible
        # error of the self pose is ~0.3 A; the 24-step roll adds up to 7.5 deg.
        self.assertLess(best[0], 0.6)
        self.assertLess(best[1], 1e-6)

    def test_canonical_attachment_mapping(self):
        """The pool stores attachment atoms in canonical SMILES numbering, so
        a re-parsed candidate must put those indices on the same atom types
        and the same local environment as the posed fragment."""
        lig = _embedded()
        frag = next(f for f in fragment_ligand(lig, max_frags=200)
                    if f.n_attach >= 1)
        att, _att_xyz, _exit_xyz = query_exit_vectors(frag.mol, lig)
        pos_map = _canonical_positions(frag.mol)
        cand = Chem.MolFromSmiles(str(frag.smiles))
        self.assertIsNotNone(cand)
        for a in att:
            p = pos_map[a]
            self.assertEqual(
                cand.GetAtomWithIdx(p).GetSymbol(),
                frag.mol.GetAtomWithIdx(a).GetSymbol())
            self.assertEqual(
                cand.GetAtomWithIdx(p).GetDegree(),
                frag.mol.GetAtomWithIdx(a).GetDegree())

    def test_etkdg_candidate_matches_local_attachment_frame(self):
        """An ETKDG conformer is a different conformer, so global RMSD is not
        meaningful; the attachment atom and its exit direction must still be
        reproduced."""
        lig = _embedded()
        frag = next(f for f in fragment_ligand(lig, max_frags=200)
                    if f.n_attach == 1 and f.n_heavy >= 10)
        att, att_xyz, exit_xyz = query_exit_vectors(frag.mol, lig)
        pos_map = _canonical_positions(frag.mol)
        cand_attach = [pos_map[i] for i in att]
        cand = Chem.AddHs(Chem.MolFromSmiles(str(frag.smiles)))
        self.assertEqual(AllChem.EmbedMolecule(cand, randomSeed=11), 0)

        best = None
        for r, t, rmsd in placements(cand, 0, cand_attach,
                                     att_xyz, exit_xyz):
            xyz = cand.GetConformer().GetPositions() @ r.T + t
            dev = np.linalg.norm(xyz[cand_attach[0]] - att_xyz[0])
            if best is None or dev < best[0]:
                best = (float(dev), float(rmsd))
        self.assertLess(best[0], 0.05)
        self.assertLess(best[1], 0.05)


class ClashTest(unittest.TestCase):
    def test_legacy_atoms_are_filtered_by_element(self):
        """The fallback residue parser has no is_hydrogen flag; a protein H
        must still be ignored, or every pose counts a spurious clash."""
        mol = _embedded("c1ccccc1", seed=3)
        xyz = mol.GetConformer().GetPositions()
        p = xyz[0] + np.array([1.0, 0.0, 0.0])
        hp = xyz[0] + np.array([0.5, 0.2, 0.0])  # closer than the heavy atom
        heavy = SimpleNamespace(x=p[0], y=p[1], z=p[2], element="C")
        hydrogen = SimpleNamespace(x=hp[0], y=hp[1], z=hp[2], element="H")
        env = [SimpleNamespace(atoms=[heavy, hydrogen])]

        n, dmin = clash_stats(mol, env)
        d_heavy = np.linalg.norm(xyz - p, axis=1)
        self.assertAlmostEqual(dmin, float(d_heavy.min()), places=9)
        self.assertEqual(n, int(np.count_nonzero(d_heavy < 2.2)))


class ContactShellTest(unittest.TestCase):
    def _atom(self, xyz, element="C"):
        return SimpleNamespace(x=float(xyz[0]), y=float(xyz[1]),
                               z=float(xyz[2]), element=element)

    def test_heavy_atom_distance_defines_the_shell(self):
        mol = _embedded("c1ccccc1", seed=3)
        p = mol.GetConformer().GetPositions()[0]
        env = [
            SimpleNamespace(atoms=[self._atom(p + [3.0, 0.0, 0.0])]),
            SimpleNamespace(atoms=[self._atom(p + [9.0, 0.0, 0.0])]),
            # hydrogen-only contact must not count
            SimpleNamespace(atoms=[self._atom(p + [1.0, 0.0, 0.0], "H")]),
        ]
        np.testing.assert_array_equal(
            contact_shell(mol, env, cutoff=5.0),
            np.array([True, False, False]))

    def test_identical_profiles_score_one(self):
        mask = np.array([True, True, True, False])
        prof = {
            "weights": np.array([0.2, 0.3, 0.5, 0.0]),
            "contacts": np.array([1.0, 2.0, -1.0, 0.0]),
            "energies": np.array([1.0, 2.0, -1.0, 0.0]),
            "mask": mask,
        }
        out = profile_overlap(prof, prof)
        self.assertAlmostEqual(out["shell_recall"], 1.0)
        self.assertAlmostEqual(out["shell_jaccard"], 1.0)
        self.assertAlmostEqual(out["shell_weight_cos"], 1.0)
        self.assertAlmostEqual(out["shell_energy_r"], 1.0)
        self.assertEqual(out["n_shell"], 3)

    def test_disjoint_shells_have_no_energy_correlation(self):
        q = {
            "weights": np.array([0.5, 0.5, 0.0, 0.0]),
            "contacts": np.array([1.0, 2.0, 0.0, 0.0]),
            "energies": np.array([1.0, 2.0, 0.0, 0.0]),
            "mask": np.array([True, True, False, False]),
        }
        c = {
            "weights": np.array([0.0, 0.0, 0.5, 0.5]),
            "contacts": np.array([0.0, 0.0, 3.0, 4.0]),
            "energies": np.array([0.0, 0.0, 3.0, 4.0]),
            "mask": np.array([False, False, True, True]),
        }
        out = profile_overlap(q, c)
        self.assertAlmostEqual(out["shell_recall"], 0.0)
        self.assertAlmostEqual(out["shell_jaccard"], 0.0)
        self.assertIsNone(out["shell_energy_r"])


class ShellWeightsTest(unittest.TestCase):
    @staticmethod
    def _res(xyz, code):
        return SimpleNamespace(coords=np.array(xyz, dtype=np.float64),
                               code=code, chain="A", res_seq=str(code),
                               weight=0.25, atoms=[])

    def _env(self):
        # One in-band triangle (side 10 A) plus a residue too far for any
        # triplet, so the full and "far residue excluded" key sets agree.
        return [self._res([0, 0, 0], 1),
                self._res([10, 0, 0], 2),
                self._res([5, 8.660, 0], 3),
                self._res([30, 0, 0], 4)]

    def test_mask_keeps_only_triplets_fully_inside_the_shell(self):
        env = self._env()
        full = fragment_fp_weights(env, use_pair_bias=False)
        self.assertEqual(len(full), 1)

        far_excluded = np.array([True, True, True, False])
        self.assertEqual(
            fragment_fp_weights(env, use_pair_bias=False,
                                residue_mask=far_excluded),
            full)

        # Excluding one vertex leaves no complete triangle.
        vertex_excluded = np.array([True, True, False, False])
        self.assertEqual(
            fragment_fp_weights(env, use_pair_bias=False,
                                residue_mask=vertex_excluded), {})

    def test_mask_size_is_checked(self):
        with self.assertRaises(ValueError):
            fragment_fp_weights(self._env(), residue_mask=np.ones(2, bool))


class PersistenceTest(unittest.TestCase):
    @staticmethod
    def _prof(mask, contacts):
        return {"mask": np.array(mask, dtype=bool),
                "contacts": np.array(contacts, dtype=np.float64),
                "weights": np.ones(len(mask))}

    def test_contacts_weighted_by_pose_frequency(self):
        q = {
            "mask": np.array([True, True, True, False]),
            "weights": np.array([1.0, 1.0, 1.0, 0.0]),
            "contacts": np.array([1.0, 2.0, 3.0, 0.0]),
        }
        profiles = {
            0: self._prof([True, True, False, False], [1.0, 2.0, 0.0, 0.0]),
            1: self._prof([True, False, True, False], [1.0, 0.0, 3.0, 0.0]),
            2: self._prof([True, True, True, False], [1.0, 2.0, 3.0, 0.0]),
            3: self._prof([False, True, True, False], [0.0, 2.0, 3.0, 0.0]),
        }
        out = persistence_overlap(q, profiles)
        self.assertAlmostEqual(out["shell_persist_recall"], 0.75)
        self.assertAlmostEqual(out["shell_persist_all"], 0.0)
        self.assertAlmostEqual(out["shell_persist_var"], 0.1875)
        # Pose-averaged contacts (0.75, 1.5, 2.25) are collinear with the
        # query energies (1, 2, 3), so persistence preserves the correlation.
        self.assertAlmostEqual(out["shell_persist_energy_r"], 1.0, places=12)
        # Favourable contacts survive at r0 3/4, r1 3/4, r2 3/4.
        self.assertAlmostEqual(out["shell_contact_retention"], 0.75)
        self.assertAlmostEqual(out["shell_delta_energy"], -0.5)

    def test_query_weights_prioritise_stable_contacts(self):
        q = {
            "mask": np.array([True, True, True]),
            "weights": np.array([2.0, 1.0, 1.0]),
            "contacts": np.array([1.0, 0.0, 0.0]),
        }
        profiles = {
            0: self._prof([True, True, False], [1.0, 1.0, 0.0]),
            1: self._prof([True, False, False], [1.0, 0.0, 0.0]),
            2: self._prof([True, False, False], [1.0, 0.0, 0.0]),
            3: self._prof([True, False, False], [1.0, 0.0, 0.0]),
        }
        out = persistence_overlap(q, profiles)
        # r0: weight 2, p=1; r1: weight 1, p=0.25; r2: weight 1, p=0
        self.assertAlmostEqual(out["shell_persist_recall"], 2.25 / 4.0)
        self.assertAlmostEqual(out["shell_persist_all"], 2.0 / 4.0)
        self.assertAlmostEqual(out["shell_persist_var"], 0.1875 / 4.0)
        # Only r0 is a favourable query contact, and it is always kept.
        self.assertAlmostEqual(out["shell_contact_retention"], 1.0)
        # Pose-averaged candidate energies: (1.0, 0.25, 0.0).
        self.assertAlmostEqual(out["shell_delta_energy"], 0.25 / 4.0)

    def test_empty_inputs_are_zero(self):
        q = {"mask": np.array([True]), "weights": np.array([1.0]),
             "contacts": np.array([1.0])}
        out = persistence_overlap(q, {})
        self.assertEqual(out["shell_persist_recall"], 0.0)
        self.assertIsNone(out["shell_persist_energy_r"])
        empty_q = {"mask": np.array([False]), "weights": np.array([1.0]),
                   "contacts": np.array([0.0])}
        out = persistence_overlap(empty_q, {0: self._prof([True], [1.0])})
        self.assertEqual(out["shell_persist_all"], 0.0)
        self.assertIsNone(out["shell_persist_energy_r"])

    def test_no_favourable_query_contacts_gives_zero_retention(self):
        q = {"mask": np.array([True]), "weights": np.array([1.0]),
             "contacts": np.array([0.0])}
        out = persistence_overlap(q, {0: self._prof([True], [1.0])})
        self.assertEqual(out["shell_contact_retention"], 0.0)
        self.assertAlmostEqual(out["shell_delta_energy"], 1.0)


class ProfilePoseTest(unittest.TestCase):
    def test_keeps_both_arms_and_spreads_over_conformers(self):
        poses = [{"clashes": c, "rmsd": 0.1 * i}
                 for i, c in enumerate([3, 0, 2, 1, 5])]
        geom = [(None, 0), (None, 0), (None, 1), (None, 1), (None, 2)]
        idx = _profile_pose_indices(poses, geom, ri=1, oi=4, n=3)
        self.assertEqual(len(idx), 3)
        self.assertEqual(len(set(idx)), 3)
        self.assertIn(1, idx)  # realistic arm
        self.assertIn(4, idx)  # best-overlap arm
        self.assertIn(3, idx)  # best pose of the other conformer


class PinnedCutTest(unittest.TestCase):
    """A pinned fragment may need a cut the standard cutter never emits."""

    def test_ring_alpha_amide_cut(self):
        # Benzoyl pyrrolidine: the amide C-N bond is ring<->alpha, so the
        # paper cutter keeps it. Pinning the pyrrolidine must still work.
        mol = _embedded("O=C(N1CCCC1)c1ccccc1")
        frag = _pinned_fragment_from_ligand(mol, "C1CCNC1")
        self.assertEqual(frag.n_attach, 1)
        self.assertEqual(frag.n_heavy, 5)
        self.assertEqual(frag.smiles, Chem.MolToSmiles(Chem.MolFromSmiles(
            "C1CCNC1")))
        # Coordinates are inherited from the parent ligand frame.
        lpos = np.asarray(mol.GetConformer().GetPositions())
        fpos = np.asarray(
            [list(frag.mol.GetConformer().GetAtomPosition(a.GetIdx()))
             for a in frag.mol.GetAtoms() if a.GetAtomicNum() > 1])
        for p in fpos:
            self.assertLess(float(np.min(np.linalg.norm(lpos - p, axis=1))),
                            1e-6)

    def test_whole_ligand_is_rejected(self):
        mol = _embedded("C1CCNC1")
        with self.assertRaises(ValueError):
            _pinned_fragment_from_ligand(mol, "C1CCNC1")

    def test_non_substructure_is_rejected(self):
        mol = _embedded("O=C(N1CCCC1)c1ccccc1")
        with self.assertRaises(ValueError):
            _pinned_fragment_from_ligand(mol, "c1ccncc1")


class DummyFormatTest(unittest.TestCase):
    def test_each_attachment_becomes_one_dummy(self):
        out = Chem.MolFromSmiles(dummy_fragment_smiles("OCCO", [0, 3]))
        self.assertIsNotNone(out)
        self.assertEqual(
            sum(1 for a in out.GetAtoms() if a.GetAtomicNum() == 0), 2)
        self.assertEqual(
            Chem.MolToSmiles(out, isomericSmiles=False), "*OCCO*")

    def test_skeleton_is_unchanged(self):
        src = "NC1=NCCC(C2CNC2)N1"
        out = Chem.MolFromSmiles(dummy_fragment_smiles(src, [3]))
        self.assertIsNotNone(out)
        self.assertEqual(out.GetNumHeavyAtoms(),
                         Chem.MolFromSmiles(src).GetNumHeavyAtoms())
        self.assertEqual(Chem.MolToSmiles(out, isomericSmiles=False),
                         "*C1CC(C2CNC2)NC(N)=N1")


class RetrievalFingerprintTest(unittest.TestCase):
    """Pool fragments are stored neutral; the query is uncharged to match."""

    def test_charge_is_dropped_by_default(self):
        charged = query_fingerprint("NC1=[NH+]CCCN1")
        neutral = query_fingerprint("NC1=NCCCN1")
        np.testing.assert_array_equal(charged, neutral)

    def test_raw_form_keeps_the_charge_bits(self):
        charged = query_fingerprint("NC1=[NH+]CCCN1", neutralize=False)
        neutral = query_fingerprint("NC1=NCCCN1", neutralize=False)
        self.assertFalse(np.array_equal(charged, neutral))


class AttachmentFingerprintTest(unittest.TestCase):
    def test_marks_change_the_fingerprint(self):
        plain = query_fingerprint("OCCO")
        marked = attachment_fingerprint("OCCO", [0, 3])
        self.assertFalse(np.array_equal(plain, marked))

    def test_symmetric_sites_are_interchangeable(self):
        # Same graph either way, so a symmetric linker must not depend on
        # which end the pool happened to number first.
        left = attachment_fingerprint("OCCO", [0])
        right = attachment_fingerprint("OCCO", [3])
        np.testing.assert_array_equal(left, right)

    def test_attachment_order_does_not_matter(self):
        a = attachment_fingerprint("OCCO", [0, 3])
        b = attachment_fingerprint("OCCO", [3, 0])
        np.testing.assert_array_equal(a, b)

    def test_neutralized_like_the_pool(self):
        charged = attachment_fingerprint("NC1=[NH+]CCN1", [0])
        neutral = attachment_fingerprint("NC1=NCCN1", [0])
        np.testing.assert_array_equal(charged, neutral)


class AttachmentIdentityTest(unittest.TestCase):
    def test_element_and_hybridization(self):
        mol = Chem.MolFromSmiles("OCCO")  # SMILES order: O, C, C, O
        self.assertEqual(attachment_identity(mol, [0]),
                         (("O", "SP3"),))
        self.assertEqual(attachment_identity(mol, [1]),
                         (("C", "SP3"),))

    def test_multiset_is_sorted(self):
        mol = Chem.MolFromSmiles("OCCO")
        self.assertEqual(attachment_identity(mol, [0, 1]),
                         (("C", "SP3"), ("O", "SP3")))

    def test_aromatic_site_differs_from_sp3(self):
        aromatic = attachment_identity(Chem.MolFromSmiles("c1ccccc1"), [0])
        aliphatic = attachment_identity(Chem.MolFromSmiles("C1CCCCC1"), [0])
        self.assertNotEqual(aromatic, aliphatic)
        self.assertEqual(aromatic[0][1], "SP2")


class FuseRankingTest(unittest.TestCase):
    def test_reciprocal_rank_sum(self):
        fused = fuse_rankings([[1, 2, 3], [2, 1, 4]])
        self.assertEqual([rid for rid, _ in fused], [1, 2, 3, 4])
        self.assertAlmostEqual(fused[0][1], 1 / 11 + 1 / 12, places=12)
        self.assertAlmostEqual(fused[2][1], 1 / 13, places=12)

    def test_single_channel_keeps_its_order(self):
        fused = fuse_rankings([[7, 9, 11]])
        self.assertEqual([rid for rid, _ in fused], [7, 9, 11])


class GeminalAttachmentTest(unittest.TestCase):
    def test_two_marks_on_one_atom(self):
        # 17% of pool rows are cut twice at the same atom (a tertiary amine
        # or a quaternary carbon split off its substituents).
        out = Chem.MolFromSmiles(dummy_fragment_smiles("CC", [1, 1]))
        self.assertIsNotNone(out)
        marked = [a for a in out.GetAtoms() if a.GetAtomicNum() == 6
                  and sum(1 for n in a.GetNeighbors()
                          if n.GetAtomicNum() == 0) == 2]
        self.assertEqual(len(marked), 1)
        self.assertEqual(marked[0].GetTotalNumHs(), 1)


class OverlapTest(unittest.TestCase):
    def test_uniform_weights_reduce_to_jaccard(self):
        q = normalize_weights({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
        l = normalize_weights({3: 2.0, 4: 2.0, 5: 2.0, 6: 2.0})
        out = weight_overlap(q, l)
        # shared {3,4}; union {1,2,3,4,5,6}
        self.assertAlmostEqual(out["weighted"], 2.0 / 6.0, places=12)
        self.assertAlmostEqual(out["jaccard"], 2.0 / 6.0, places=12)
        self.assertEqual(out["shared"], 2)

    def test_min_weighting_penalises_weak_match(self):
        q = {1: 3.0, 2: 1.0}
        l = {1: 1.0, 2: 3.0}
        out = weight_overlap(q, l)
        # W = min(3,1)+min(1,3) = 2; denominator 2+2-2 = 2
        self.assertAlmostEqual(out["weighted"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
