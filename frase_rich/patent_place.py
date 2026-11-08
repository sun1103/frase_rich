"""Place patent-bank fragments on a query fragment's pose and score in time.

WHY this exists: a pool row knows a fragment's SMILES and the atom positions
where it was cut from its patent parent, but not where it sits in a protein.
The library side scores fragments in their experimental pose, so a candidate
cannot enter the same ``fp_weights_in_env`` path until it has coordinates in
the protein frame. This module builds that pose from the query fragment's exit
vectors, then runs the identical scoring call.

Placement rule
--------------
The query fragment has ``q_att`` attachment atoms (sitting on the cut bonds)
and ``q_exit`` partner positions in the parent ligand. A candidate has
``c_att`` attachment atoms in canonical SMILES numbering. A rigid transform is
fitted on the attachment atoms plus one point along each exit direction
(``atom + unit_vector * cut_bond_length``). The extra point is what resolves
the orientation the atom positions alone leave free:

* 1 attachment: Kabsch on two points leaves the roll about the exit axis
  undetermined, so it is sampled in ``rolls`` steps.
* 2+ attachments: the point set is over-determined; every attachment-atom
  permutation is tried and the best attachment fit is kept.
* a candidate's exit direction is ambiguous (the H-capped fragment has one
  alternative H at the cut atom, sometimes more), so every H direction on the
  attachment atom is tried.

Pose selection deliberately ignores the query's key set: it takes the pose
with the fewest protein heavy-atom clashes, ties broken by attachment RMSD.
The oracle arm (best key overlap over every pose and candidate) is reported
separately as an upper bound, never as a deployable score.

Cost: EEM is ~240 ms through obabel, so charges are computed once per
conformer (rigid poses share them) and batched across a candidate's
conformers; one scoring call is then ~15 ms.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import defaultdict
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

from .chem import (
    charges_for,
    eem_charges_batch,
    gasteiger_charges,
    with_explicit_hydrogens,
)
from .features import relevance_weights
from .patent_decomp import _smiles_order
from .weighting import fp_weights_in_env, normalize_marks

# Morgan settings for the retrieval index. Bit matrices are packed to bytes so
# 31.4M rows fit in 4 GB on disk; np.bitwise_count works directly on bytes, so
# no unpacking is needed at query time.
FP_RADIUS = 2
FP_BITS = 1024

_META_COLUMNS = ("frag_smiles", "n_heavy", "n_attach", "attach_atoms")


@dataclass
class _PinnedFragment:
    """Same fields the fragmenter emits, for an explicitly requested cut."""
    mol: Chem.Mol
    smiles: str
    centroid: np.ndarray
    n_heavy: int
    n_attach: int


# ---------------------------------------------------------------------------
# fingerprint index
# ---------------------------------------------------------------------------

def _stable_seed(text: str) -> int:
    """Process-independent seed (Python's hash() is salted per process).

    Clamped to int32: RDKit's ``randomSeed`` is a signed 32-bit property.
    """
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def fp_paths(prefix: str) -> dict[str, str]:
    return {
        "fp": prefix + "_fp.npy",
        "pop": prefix + "_pop.npy",
        "n_attach": prefix + "_n_attach.npy",
        "n_heavy": prefix + "_n_heavy.npy",
    }


def _open_meta(path: str, mode: str, dtype=None, shape=None):
    """numpy memmap; ``r+`` reads dtype/shape from the file header."""
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


def _fp_worker(job) -> tuple[int, int]:
    """Pack one parquet row group's Morgan fingerprints into the memmap."""
    (pool_path, paths, rg, row_start, n_rows, n_bits, radius,
     attach_aware) = job
    # Expected atom-valence misses are reported per row; without this the
    # attach-aware build writes megabytes of RDKit warnings to stderr.
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits)
    fp = _open_meta(paths["fp"], "r+")
    pop = _open_meta(paths["pop"], "r+")
    na = _open_meta(paths["n_attach"], "r+")
    nh = _open_meta(paths["n_heavy"], "r+")

    columns = ["frag_smiles", "n_attach", "n_heavy"]
    if attach_aware:
        columns.append("attach_atoms")
    pf = pq.ParquetFile(pool_path)
    done = 0
    remaining = n_rows
    batch_size = max(1, min(50_000, n_rows))
    for b in pf.iter_batches(batch_size=batch_size, row_groups=[rg],
                             columns=columns):
        # A truncated build may take only part of the last row group.
        m = min(b.num_rows, remaining)
        if m <= 0:
            break
        b = b.slice(0, m)
        smis = b.column("frag_smiles").to_pylist()
        atts = (b.column("attach_atoms").to_pylist()
                if attach_aware else None)
        packed = np.zeros((m, n_bits // 8), dtype=np.uint8)
        for j, smi in enumerate(smis):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if attach_aware:
                try:
                    # Mark, then uncharge: uncharging first would strip the
                    # cap H of a cut quaternary N and leave no valence for
                    # the dummy.
                    mol = _neutral(_with_dummy_marks(mol, atts[j] or []))
                except Exception:
                    continue
            bits = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.uint8)
            packed[j] = np.packbits(bits)
        sl = slice(row_start + done, row_start + done + m)
        fp[sl] = packed
        pop[sl] = np.bitwise_count(packed).sum(axis=1)
        na[sl] = b.column("n_attach").to_numpy()
        nh[sl] = b.column("n_heavy").to_numpy()
        done += m
        remaining -= m
    fp.flush()
    pop.flush()
    na.flush()
    nh.flush()
    return rg, row_start


def build_fingerprints(
    pool_path: str,
    prefix: str,
    *,
    n_bits: int = FP_BITS,
    radius: int = FP_RADIUS,
    workers: int = 6,
    limit_rows: int = 0,
    attach_aware: bool = False,
    verbose: bool = True,
) -> dict[str, str]:
    """Build (or reuse) the packed fingerprint matrix aligned to ``pool_path``.

    Row order is the parquet row order, so a matrix row index is a pool row
    index. One pass, sharded by parquet row group; workers write disjoint row
    slices of the same memmap. Unparseable SMILES (none expected) get an
    all-zero row instead of shifting the alignment.

    ``attach_aware`` marks the stored attachment atoms with unlabeled dummies,
    then uncharges (matching ``attachment_fingerprint``). Use a
    different ``prefix`` for the two modes; the cache check is shape-only, so
    the mode is not recorded in the file.
    """
    import multiprocessing as mp

    pf = pq.ParquetFile(pool_path)
    n_total = pf.metadata.num_rows
    n_rows = min(n_total, limit_rows) if limit_rows else n_total
    paths = fp_paths(prefix)

    try:
        have = np.load(paths["fp"], mmap_mode="r")
        if have.shape == (n_rows, n_bits // 8):
            if verbose:
                print(f"fingerprints cached: {paths['fp']} "
                      f"({n_rows:,} x {n_bits})", flush=True)
            return paths
    except (OSError, ValueError):
        pass

    fp = _open_meta(paths["fp"], "w+", np.uint8, (n_rows, n_bits // 8))
    del fp
    pop = _open_meta(paths["pop"], "w+", np.int16, (n_rows,))
    del pop
    na = _open_meta(paths["n_attach"], "w+", np.int8, (n_rows,))
    del na
    nh = _open_meta(paths["n_heavy"], "w+", np.int16, (n_rows,))
    del nh

    jobs = []
    start = 0
    for rg in range(pf.metadata.num_row_groups):
        rg_rows = pf.metadata.row_group(rg).num_rows
        if start >= n_rows:
            break
        take = min(rg_rows, n_rows - start)
        jobs.append((pool_path, paths, rg, start, take, n_bits, radius,
                     attach_aware))
        start += rg_rows

    if verbose:
        print(f"building fingerprints: {n_rows:,} rows in {len(jobs)} row "
              f"groups on {workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for k, (rg, _start) in enumerate(pool.imap_unordered(_fp_worker, jobs),
                                         1):
            if verbose and (k % 5 == 0 or k == len(jobs)):
                print(f"  {k}/{len(jobs)} row groups", flush=True)
    if verbose:
        print(f"saved: {paths['fp']}", flush=True)
    return paths


def tanimoto_topk(
    paths: dict[str, str],
    n_rows: int,
    query_bits: np.ndarray,
    *,
    n_attach: int,
    n_heavy: int,
    k: int = 64,
    heavy_band: int = 8,
    chunk_rows: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` pool rows by Morgan Tanimoto, gated to a placement-compatible
    attachment count and a heavy-atom band around the query fragment.

    A candidate with a different attachment count is not a drop-in replacement
    (an unused exit vector would dangle), so the count is an equality gate.
    The heavy band keeps replacements near the query's size; it is a ranking
    convenience, not a chemistry claim.
    """
    fp = np.load(paths["fp"], mmap_mode="r")
    pop = np.load(paths["pop"], mmap_mode="r")
    na = np.load(paths["n_attach"], mmap_mode="r")
    nh = np.load(paths["n_heavy"], mmap_mode="r")

    q = np.packbits(np.asarray(query_bits, dtype=np.uint8))
    qpop = int(np.bitwise_count(q).sum())
    best_idx = np.empty(0, dtype=np.int64)
    best_t = np.empty(0, dtype=np.float64)

    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        keep = ((na[start:stop] == n_attach)
                & (np.abs(nh[start:stop].astype(np.int16) - n_heavy)
                   <= heavy_band))
        sel = np.nonzero(keep)[0]
        if sel.size == 0:
            continue
        block = np.asarray(fp[start + sel])
        inter = np.bitwise_count(block & q).sum(axis=1, dtype=np.int64)
        den = qpop + pop[start + sel].astype(np.int64) - inter
        t = np.where(den > 0, inter / np.maximum(den, 1), 0.0)
        idx = start + sel
        if best_idx.size:
            t = np.concatenate([best_t, t])
            idx = np.concatenate([best_idx, idx])
        if t.size > k:
            part = np.argpartition(t, t.size - k)[t.size - k:]
            t, idx = t[part], idx[part]
        order = np.argsort(-t, kind="stable")
        best_t, best_idx = t[order], idx[order]

    return best_idx[:k], best_t[:k]


def query_fingerprint(smiles: str, *, n_bits: int = FP_BITS,
                      radius: int = FP_RADIUS,
                      neutralize: bool = True) -> np.ndarray | None:
    """Morgan fingerprint for pool retrieval.

    ``neutralize`` uncharges the query first: the pool stores patent fragments
    in their (neutral) writing form and ECFP encodes formal charge, so a
    protonated query under-matches the neutral pool. Verified on a +1 amidine
    fragment: max Tanimoto 0.387 charged vs 0.467 neutral over 31.4M rows.
    The charged form is still used for everything after retrieval.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if neutralize:
        try:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        except Exception:
            pass
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits)
    return np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.uint8)


def fuse_rankings(rankings, *, k0: float = 10.0) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion of per-channel retrieval lists.

    Channels score against different fingerprints, so their Tanimotos are not
    comparable; only each channel's order is fused. ``k0`` damps the top ranks
    so agreement between channels dominates a single strong hit.
    """
    scores: dict[int, float] = {}
    for ids in rankings:
        for rank, rid in enumerate(ids, start=1):
            rid = int(rid)
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k0 + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def random_row(
    paths: dict[str, str],
    n_rows: int,
    *,
    n_attach: int,
    n_heavy: int,
    heavy_band: int = 8,
    rng: np.random.Generator | None = None,
    max_tries: int = 20_000,
) -> int | None:
    """A random placement-compatible pool row: the no-retrieval control."""
    na = np.load(paths["n_attach"], mmap_mode="r")
    nh = np.load(paths["n_heavy"], mmap_mode="r")
    rng = rng or np.random.default_rng()
    for _ in range(max_tries):
        i = int(rng.integers(0, n_rows))
        if int(na[i]) == n_attach and abs(int(nh[i]) - n_heavy) <= heavy_band:
            return i
    return None


def read_pool_rows(
    pool_path: str, row_ids,
    columns: tuple[str, ...] = _META_COLUMNS,
) -> dict[int, dict]:
    """Fetch metadata for scattered row ids with one read per touched group."""
    pf = pq.ParquetFile(pool_path)
    starts = []
    total = 0
    for rg in range(pf.metadata.num_row_groups):
        starts.append(total)
        total += pf.metadata.row_group(rg).num_rows

    by_rg: dict[int, list[int]] = defaultdict(list)
    for rid in set(int(r) for r in row_ids):
        by_rg[bisect_right(starts, rid) - 1].append(rid)

    out: dict[int, dict] = {}
    for rg, ids in by_rg.items():
        table = pf.read_row_group(rg, columns=list(columns))
        cols = {c: table.column(c).to_pylist() for c in columns}
        base = starts[rg]
        for rid in ids:
            j = rid - base
            out[rid] = {c: cols[c][j] for c in columns}
    return out


# ---------------------------------------------------------------------------
# exit vectors and rigid placement
# ---------------------------------------------------------------------------

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                     float]:
    """Rigid ``(R, t)`` minimising ``|src @ R.T + t - dst|``; returns RMSD too."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    sc = src.mean(axis=0)
    dc = dst.mean(axis=0)
    h = (src - sc).T @ (dst - dc)
    u, _s, vt = np.linalg.svd(h)
    d = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = dc - sc @ r.T
    rmsd = float(np.sqrt(np.mean(np.sum((src @ r.T + t - dst) ** 2, axis=1))))
    return r, t, rmsd


def axis_rotation(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about ``axis``."""
    x, y, z = _unit(axis)
    c, s = math.cos(angle), math.sin(angle)
    cc = 1.0 - c
    return np.array([
        [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
        [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
        [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
    ])


def exit_directions(mol, attach_idx, conf_id: int = 0) -> list[list[np.ndarray]]:
    """Unit directions available at each candidate attachment atom.

    The fragmenter replaced the cut bond with an H, so the H positions are the
    free valences the new bond can occupy. For a CH2/CH3 attachment there are
    several; all are returned. Atoms with no H (rare) fall back to the
    direction opposite the heavy-neighbour centroid.
    """
    pos = np.asarray(mol.GetConformer(conf_id).GetPositions())
    out = []
    for a in attach_idx:
        atom = mol.GetAtomWithIdx(int(a))
        hs = [n.GetIdx() for n in atom.GetNeighbors()
              if n.GetAtomicNum() == 1]
        if hs:
            out.append([_unit(pos[h] - pos[a]) for h in hs])
        else:
            heavy = [n.GetIdx() for n in atom.GetNeighbors()
                     if n.GetAtomicNum() > 1]
            acc = np.zeros(3)
            for n in heavy:
                acc += _unit(pos[n] - pos[a])
            out.append([_unit(-acc)])
    return out


def query_exit_vectors(fragment, ligand, tol: float = 1e-3):
    """Attachment atoms and partner positions of an experimentally posed
    fragment.

    Fragment heavy atoms keep the ligand's coordinates, so a nearest-atom map
    is exact. Any ligand neighbour of a mapped atom that is not itself in the
    fragment is the partner on the other side of a cut bond. Returns
    ``(frag_atom_idxs, attachment_xyz, partner_xyz)``.
    """
    fpos = np.asarray(fragment.GetConformer().GetPositions())
    lpos = np.asarray(ligand.GetConformer().GetPositions())
    heavy = [a.GetIdx() for a in fragment.GetAtoms()
             if a.GetAtomicNum() > 1]
    fxyz = fpos[heavy]
    d = np.linalg.norm(fxyz[:, None, :] - lpos[None, :, :], axis=-1)
    nearest = np.argmin(d, axis=1)
    picked = d[np.arange(len(heavy)), nearest]
    if float(picked.max()) > tol:
        raise ValueError("fragment atoms do not map onto the ligand coordinates")
    mapped = set(int(j) for j in nearest)

    att_idx, att_xyz, exit_xyz = [], [], []
    for fi, lj in zip(heavy, nearest):
        for nb in ligand.GetAtomWithIdx(int(lj)).GetNeighbors():
            if nb.GetIdx() not in mapped:
                att_idx.append(int(fi))
                att_xyz.append(fpos[fi])
                exit_xyz.append(lpos[nb.GetIdx()])
    return att_idx, np.asarray(att_xyz), np.asarray(exit_xyz)


def placements(
    mol,
    conf_id: int,
    cand_attach,
    q_att_xyz: np.ndarray,
    q_exit_xyz: np.ndarray,
    *,
    rolls: int = 24,
    max_poses: int = 48,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Candidate rigid transforms ``(R, t, attachment_rmsd)``.

    Point set = attachment atoms + one point per exit direction at the query's
    cut-bond length, fitted by Kabsch to the query attachment atoms + partner
    positions. Single-attachment fits are completed by rolling about the exit
    axis. Combinations are capped at ``max_poses`` (evenly thinned, so the
    roll sampling stays spread).
    """
    pos = np.asarray(mol.GetConformer(conf_id).GetPositions())
    dirs = exit_directions(mol, cand_attach, conf_id)
    q_att = np.asarray(q_att_xyz, dtype=np.float64)
    q_exit = np.asarray(q_exit_xyz, dtype=np.float64)
    dist = np.linalg.norm(q_exit - q_att, axis=1)
    dst_all = np.vstack([q_att, q_exit])

    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    for perm in itertools.permutations(range(len(cand_attach))):
        ca = [int(cand_attach[p]) for p in perm]
        for combo in itertools.product(*[dirs[p] for p in perm]):
            src = np.array(
                [pos[a] for a in ca]
                + [pos[a] + combo[i] * dist[i] for i, a in enumerate(ca)])
            r, t, rmsd = kabsch(src, dst_all)
            if len(ca) == 1:
                axis = q_exit[0] - q_att[0]
                for k in range(rolls):
                    rr = axis_rotation(axis, 2.0 * math.pi * k / rolls)
                    out.append((rr @ r, q_att[0] + rr @ (t - q_att[0]),
                                rmsd))
            else:
                out.append((r, t, rmsd))

    if len(out) > max_poses:
        sel = np.linspace(0, len(out) - 1, max_poses).round().astype(int)
        out = [out[i] for i in dict.fromkeys(sel.tolist())]
    return out


def pose_mol(mol, conf_id: int, r: np.ndarray, t: np.ndarray):
    """Copy of ``mol`` with conformer ``conf_id`` rigidly transformed."""
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetConformer(conf_id))
    conf.SetPositions(np.asarray(conf.GetPositions()) @ r.T + t)
    out.AddConformer(conf, assignId=True)
    return out


def embed_candidate(smiles: str, n_confs: int = 3, *, seed: int | None = None):
    """H-added 3D conformers of a candidate fragment; ``None`` on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed if seed is not None else _stable_seed(smiles))
    params.useSmallRingTorsions = True
    cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_confs,
                                           params=params))
    if not cids:
        return None
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=200)
    except Exception:
        pass
    return mh, cids


def _single_conformer(mol, conf_id: int):
    out = Chem.Mol(mol)
    conf = Chem.Conformer(mol.GetConformer(conf_id))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def conformer_charges(mol, conf_ids) -> tuple[list[np.ndarray], str]:
    """EEM charges per conformer, one obabel call; Gasteiger on failure."""
    mols = [_single_conformer(mol, cid) for cid in conf_ids]
    try:
        return list(eem_charges_batch(mols)), "eem"
    except Exception:
        return [gasteiger_charges(m) for m in mols], "gasteiger"


def clash_stats(mol, env, cutoff: float = 2.2,
                extra_xyz=None) -> tuple[int, float]:
    """Heavy-atom contacts closer than ``cutoff`` and the minimum distance.

    Protein hydrogens are ignored (placement is heavy-atom only).
    ``extra_xyz`` adds fixed heavy atoms to the obstacle set -- used to keep
    the retained part of the parent ligand while one fragment is replaced.
    """
    conf = mol.GetConformer()
    pos = np.asarray(conf.GetPositions())
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return 0, float("inf")
    xyz = pos[heavy]
    # The legacy parse path (protein_residues returning None) yields atoms
    # without is_hydrogen; they identify hydrogens by element instead.
    prot = np.array(
        [[a.x, a.y, a.z] for res in env for a in res.atoms
         if not (getattr(a, "is_hydrogen", False)
                 or str(getattr(a, "element", "")).strip().upper() == "H")],
        dtype=np.float64)
    if extra_xyz is not None and len(extra_xyz):
        extra = np.asarray(extra_xyz, dtype=np.float64).reshape(-1, 3)
        prot = extra if prot.size == 0 else np.vstack([prot, extra])
    if prot.size == 0:
        return 0, float("inf")
    d = np.linalg.norm(xyz[:, None, :] - prot[None, :, :], axis=-1)
    return int(np.count_nonzero(d < cutoff)), float(d.min())


def retained_ligand_xyz(lig, frag_mol, q_att_xyz, *,
                        tol: float = 1e-3,
                        bond_cutoff: float = 1.8) -> np.ndarray | None:
    """Heavy atoms of the parent ligand that survive a one-fragment swap.

    Excludes the atoms of the replaced fragment (matched by coordinates: the
    fragment was cut from this very conformer) and the retained atoms bonded
    to the fragment's attachment atoms, since the candidate's attachment atom
    takes exactly that position and is bonded to them.
    """
    fpos = np.asarray(
        [list(frag_mol.GetConformer().GetAtomPosition(a.GetIdx()))
         for a in frag_mol.GetAtoms() if a.GetAtomicNum() > 1],
        dtype=np.float64)
    quat = np.asarray(q_att_xyz, dtype=np.float64)
    out = []
    conf = lig.GetConformer()
    for atom in lig.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        p = np.asarray(list(conf.GetAtomPosition(atom.GetIdx())),
                       dtype=np.float64)
        if fpos.size and np.min(np.linalg.norm(fpos - p, axis=1)) <= tol:
            continue
        if quat.size and np.min(np.linalg.norm(quat - p, axis=1)) <= bond_cutoff:
            continue
        out.append(p)
    return np.asarray(out, dtype=np.float64) if out else None


# ---------------------------------------------------------------------------
# weight comparison
# ---------------------------------------------------------------------------

def normalize_weights(raw: dict[int, float]) -> dict[int, float]:
    """Mean-1 scaling over positive support, as the index does."""
    pos = {int(k): float(v) for k, v in raw.items() if v > 0}
    if not pos:
        return {}
    scale = len(pos) / float(sum(pos.values()))
    return {k: v * scale for k, v in pos.items()}


def weight_overlap(q: dict[int, float], l: dict[int, float]) -> dict:
    """Shared keys, weighted overlap and query coverage.

    ``weighted`` is ``sum(min(q, l)) / (|q| + |l| - W)``; with uniform
    (all-ones) vectors it equals Jaccard exactly.
    """
    if not q or not l:
        return {"shared": 0, "weighted": 0.0, "coverage": 0.0,
                "jaccard": 0.0}
    shared = q.keys() & l.keys()
    w = sum(min(q[k], l[k]) for k in shared)
    union = max(len(q) + len(l) - w, 1e-9)
    return {
        "shared": len(shared),
        "weighted": float(w / union),
        "coverage": float(len(shared) / len(q)),
        "jaccard": float(len(shared) / max(len(q) + len(l) - len(shared),
                                           1e-9)),
    }


def score_posed(
    posed,
    env,
    table,
    attach_set: set[int],
    charges: np.ndarray,
    *,
    return_marks: bool = False,
    **kwargs,
):
    """``fp_weights_in_env`` on a posed candidate with precomputed charges.

    ``return_marks`` also returns the four per-key energy marks, normalized to
    mean 1 over their positive support -- the same vectors the library index
    stores, so a placed analog can be queried against them.
    """
    fp, _env, _method, energies = fp_weights_in_env(
        posed, env, table=table, exclude_donor_atoms=attach_set,
        frag_charges=charges, return_energies=True, **kwargs)
    if not return_marks:
        return fp
    marks = {name: normalize_marks(values)
             for name, values in energies.items()}
    return fp, marks


def query_marks(frag_mol, env, table, attach_idx, *, mark: str = "mean",
                **kwargs) -> dict[int, float]:
    """The query fragment's own per-key mark in its experimental pose."""
    molh = with_explicit_hydrogens(frag_mol)
    try:
        charges, _method = charges_for(molh, method="eem")
    except Exception:
        charges = gasteiger_charges(molh)
    _fp, marks = score_posed(
        molh, env, table, set(int(i) for i in attach_idx), charges,
        return_marks=True, **kwargs)
    return marks.get(mark, {})


# ``relevance_weights`` takes only the interaction parameters; the fingerprint
# aggregation flags in the same kwargs dict would be rejected.
_RELEVANCE_KWARGS = frozenset({
    "temperature", "use_energy", "use_content", "pooling", "contact_cutoff",
    "use_angles", "backbone", "energy_scale", "content_scale",
    "legacy_hbond"})


def contact_shell(mol, env, cutoff: float = 5.0) -> np.ndarray:
    """Residues with a heavy atom within ``cutoff`` of the molecule.

    This is fragment-specific, unlike the fixed 12 A pocket membership: it is
    the set of residues the query actually touches.
    """
    pos = np.asarray(mol.GetConformer().GetPositions())
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    mask = np.zeros(len(env), dtype=bool)
    if not heavy:
        return mask
    fxyz = pos[heavy]
    for i, res in enumerate(env):
        xyz = np.array(
            [[a.x, a.y, a.z] for a in res.atoms
             if not (getattr(a, "is_hydrogen", False)
                     or str(getattr(a, "element", "")).strip().upper() == "H")],
            dtype=np.float64)
        if xyz.size == 0:
            continue
        d2 = ((xyz[:, None, :] - fxyz[None, :, :]) ** 2).sum(axis=-1)
        mask[i] = d2.min() <= cutoff * cutoff
    return mask


def residue_profile(
    mol,
    env,
    charges,
    *,
    attach_set: set[int] | None = None,
    shell_cutoff: float = 5.0,
    **kwargs,
) -> dict:
    """Per-residue weights, contact-gated energies and the contact shell.

    Same ``relevance_weights`` call the fingerprint weights are built from, but
    the per-residue values are kept instead of collapsed onto triplet keys.
    """
    molh = with_explicit_hydrogens(mol)
    rk = {k: v for k, v in kwargs.items() if k in _RELEVANCE_KWARGS}
    weights, energies, contacts = relevance_weights(
        molh, env, frag_charges=charges, exclude_donor_atoms=attach_set,
        return_energies=True, return_contact=True, **rk)
    return {
        "weights": np.asarray(weights, dtype=np.float64),
        "energies": np.asarray(energies, dtype=np.float64),
        "contacts": np.asarray(contacts, dtype=np.float64),
        "mask": contact_shell(molh, env, shell_cutoff),
    }


def profile_overlap(q: dict, c: dict) -> dict:
    """Contact-local agreement between query and placed candidate.

    ``shell_recall`` is the fraction of the query's contact residues the
    candidate also touches; ``shell_weight_cos`` compares the softmax weights
    on that shell; ``shell_energy_r`` correlates the per-residue contact-gated
    energies (``None`` when fewer than three residues are shared or one side is
    constant).
    """
    q_mask, c_mask = q["mask"], c["mask"]
    n_q = int(q_mask.sum())
    shared = q_mask & c_mask
    idx = np.flatnonzero(shared)
    qw = q["weights"][q_mask]
    cw = c["weights"][q_mask]
    denom = float(np.linalg.norm(qw) * np.linalg.norm(cw))
    weight_cos = float(qw @ cw / denom) if denom > 0 else 0.0
    energy_r = None
    if idx.size >= 3:
        qe = q["contacts"][idx]
        ce = c["contacts"][idx]
        if qe.std() > 1e-9 and ce.std() > 1e-9:
            energy_r = float(np.corrcoef(qe, ce)[0, 1])
    union = int((q_mask | c_mask).sum())
    return {
        "shell_recall": float(shared.sum() / n_q) if n_q else 0.0,
        "shell_jaccard": float(shared.sum() / union) if union else 0.0,
        "shell_weight_cos": weight_cos,
        "shell_energy_r": energy_r,
        "n_shell": n_q,
    }


def persistence_overlap(q: dict, profiles: dict) -> dict:
    """Pose-ensemble agreement on the query fragment's contact shell.

    ``profile_overlap`` scores one candidate pose. Averaging its energy
    correlation over poses reduces scatter but still lets a contact seen in a
    single pose count fully. Here each query-shell residue gets
    ``p_r = P(contact | profiled poses)``, and the scores are weighted by the
    query's own residue weights:

    - ``shell_persist_recall``: expected coverage of the query shell;
    - ``shell_persist_all``: weight fraction contacted in every pose;
    - ``shell_persist_var``: weight-averaged contact variance (instability);
    - ``shell_persist_energy_r``: correlation of query energies with
      pose-averaged candidate energies, so a transient contact contributes in
      proportion to its persistence instead of its best pose.

    It also reports the interaction *displacement* relative to the query
    fragment in its experimental pose (candidate minus query, favourable
    energies positive):

    - ``shell_contact_retention``: weight-averaged probability that a residue
      the query actually engages favourably is engaged favourably again;
    - ``shell_delta_energy``: pose-averaged signed energy change over the
      query shell, negative when the replacement loses interactions.
    """
    empty = {"shell_persist_recall": 0.0, "shell_persist_all": 0.0,
             "shell_persist_var": 0.0, "shell_persist_energy_r": None,
             "shell_contact_retention": 0.0, "shell_delta_energy": 0.0}
    if not profiles:
        return empty
    masks = np.stack([p["mask"] for p in profiles.values()])
    q_mask = q["mask"]
    n_q = int(q_mask.sum())
    if n_q == 0:
        return empty

    p = masks.mean(axis=0)
    w = np.clip(np.asarray(q["weights"])[q_mask], 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        w = np.ones(n_q, dtype=np.float64)
        total = float(n_q)

    p_shell = p[q_mask]
    energy_r = None
    qe = np.asarray(q["contacts"])[q_mask]
    contact_energies = np.stack([
        np.asarray(prof["contacts"])[q_mask] for prof in profiles.values()])
    ce = contact_energies.mean(axis=0)
    if n_q >= 3 and qe.std() > 1e-9 and ce.std() > 1e-9:
        energy_r = float(np.corrcoef(qe, ce)[0, 1])

    engaged = contact_energies > 0.0
    p_engaged = engaged.mean(axis=0)
    q_fav = qe > 0.0
    retention = 0.0
    if q_fav.any():
        w_fav = w[q_fav]
        retention = float((w_fav * p_engaged[q_fav]).sum() / w_fav.sum())

    return {
        "shell_persist_recall": float((w * p_shell).sum() / total),
        "shell_persist_all": float((w * (p_shell >= 1.0 - 1e-12)).sum()
                                   / total),
        "shell_persist_var": float((w * p_shell * (1.0 - p_shell)).sum()
                                   / total),
        "shell_persist_energy_r": energy_r,
        "shell_contact_retention": retention,
        "shell_delta_energy": float((w * (ce - qe)).sum() / total),
    }


def query_profile(frag_mol, env, attach_idx, *, shell_cutoff: float = 5.0,
                  **kwargs) -> dict:
    """Per-residue profile of the query fragment in its experimental pose."""
    molh = with_explicit_hydrogens(frag_mol)
    try:
        charges, _method = charges_for(molh, method="eem")
    except Exception:
        charges = gasteiger_charges(molh)
    return residue_profile(
        molh, env, charges, attach_set=set(int(i) for i in attach_idx),
        shell_cutoff=shell_cutoff, **kwargs)


# ---------------------------------------------------------------------------
# query preparation
# ---------------------------------------------------------------------------

def _constitution(smiles: str) -> str:
    """Canonical constitution-only SMILES (stereo ignored)."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else smiles


def _with_dummy_marks(mol, attach_idx):
    """Copy of ``mol`` with each attachment H cap replaced by a dummy bond.

    A cap can be an implicit H, an explicit H count, or a separate ``[H]``
    atom; each attachment claims one, so a geminally cut atom gets one dummy
    per cut. Isotope-labelled H (``[2H]``/``[3H]``) are real atoms, not caps,
    and are left alone. The stored positions index this mol directly, so no
    renumbering happens apart from dropping claimed ``[H]`` atoms.
    """
    attach_idx = [int(i) for i in attach_idx]
    rw = Chem.RWMol(mol)
    claimed: list[int | None] = []
    for i in attach_idx:
        free = [n.GetIdx() for n in rw.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() == 1 and n.GetIsotope() == 0
                and n.GetIdx() not in claimed]
        claimed.append(free[0] if free else None)
    h_atoms = sorted({h for h in claimed if h is not None}, reverse=True)
    if h_atoms:
        gone = set(h_atoms)
        remap, kept = {}, 0
        for old in range(rw.GetNumAtoms()):
            if old not in gone:
                remap[old] = kept
                kept += 1
        for h in h_atoms:
            rw.RemoveAtom(h)
        attach_idx = [remap[i] for i in attach_idx]
    for i in attach_idx:
        atom = rw.GetAtomWithIdx(i)
        if atom.GetNumExplicitHs() > 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
        d = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(i, d, Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _dummy_marks(mol, attach_idx) -> str:
    return Chem.MolToSmiles(_with_dummy_marks(mol, attach_idx))


_UNCHARGER = rdMolStandardize.Uncharger()


def _neutral(mol):
    """Uncharged form; ``Uncharger`` keeps heavy-atom indices for implicit-H
    mols, which is what the stored attachment positions rely on."""
    try:
        out = _UNCHARGER.uncharge(mol)
    except Exception:
        return mol
    return out if out.GetNumAtoms() == mol.GetNumAtoms() else mol


def attachment_fingerprint(smiles: str, attach_atoms, *,
                           n_bits: int = FP_BITS, radius: int = FP_RADIUS,
                           neutralize: bool = True) -> np.ndarray | None:
    """Morgan fingerprint of a fragment with its attachment points marked.

    The marks are *unlabeled* dummies: the pool's attachment positions are in
    canonical-SMILES order, not role order, so labeled ``[*:1]/[*:2]`` would
    make the score depend on how a symmetric linker happened to be numbered.
    Unlabeled dummies keep the channel invariant to that, and still encode the
    atom's attachment environment at radii 1-2.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # Mark first: the cap H is what makes room for the dummy, and a cut at
        # a saturated atom (cut quaternary N) only balances once marked.
        marked = _with_dummy_marks(mol, attach_atoms)
    except Exception:
        return None
    if neutralize:
        marked = _neutral(marked)
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits)
    return np.asarray(gen.GetFingerprintAsNumPy(marked), dtype=np.uint8)


def attachment_identity(mol, attach_atoms) -> tuple | None:
    """Sorted (element, hybridization) per attachment atom.

    A replacement must be able to make the same kind of bond as the site it
    replaces; a carbon-attached candidate at a nitrogen site is a different
    edit even when the geometry fits.
    """
    out = []
    try:
        for i in attach_atoms:
            atom = mol.GetAtomWithIdx(int(i))
            out.append((atom.GetSymbol(), str(atom.GetHybridization())))
    except Exception:
        return None
    return tuple(sorted(out))


def dummy_fragment_smiles(smiles: str, attach_atoms) -> str | None:
    """Attachment-marked form of an H-capped fragment.

    Pool ``attach_atoms`` are canonical-SMILES positions, and re-parsing the
    stored SMILES reproduces that numbering, so the positions index this mol
    directly. Each capped H becomes a dummy bond, i.e. the fragment as an
    R-group ready to be joined at the same site. Returns None on any failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return _dummy_marks(mol, attach_atoms)
    except Exception:
        return None


def _pinned_fragment_from_ligand(lig, query_smiles: str):
    """H-capped fragment of ``lig`` covering ``query_smiles``.

    The standard cutter never breaks ring<->alpha bonds (an amide C-N at a
    ring nitrogen, for instance), so an explicitly requested fragment can lie
    outside the generated set. This finds the query as a substructure, cuts
    every ligand bond with exactly one endpoint inside the match, and returns
    the H-capped component in the parent ligand frame. Raises ValueError when
    the query does not match, covers the whole ligand, or matches more than
    one site equally well.
    """
    q = Chem.MolFromSmiles(query_smiles)
    if q is None:
        raise ValueError(f"cannot parse query SMILES {query_smiles!r}")
    matches = lig.GetSubstructMatches(q, uniquify=True)
    if not matches:
        raise ValueError("query is not a substructure of the ligand")

    options = []
    for match in matches:
        inside = set(int(i) for i in match)
        boundary = [b.GetIdx() for b in lig.GetBonds()
                    if (b.GetBeginAtomIdx() in inside)
                    != (b.GetEndAtomIdx() in inside)]
        if not boundary:
            raise ValueError("query covers the whole ligand; nothing to "
                             "replace")
        options.append((len(boundary), inside, boundary))
    options.sort(key=lambda o: o[0])
    n_bnd, inside, boundary = options[0]
    rivals = [o for o in options[1:] if o[0] == n_bnd
              and o[1] != inside]
    if rivals:
        raise ValueError(
            f"query matches {1 + len(rivals)} sites with {n_bnd} boundary "
            f"bonds; make the query more specific")

    split = rdmolops.FragmentOnBonds(lig, boundary, addDummies=True)
    frag_lists = Chem.GetMolFrags(split)
    frag_mols = Chem.GetMolFrags(split, asMols=True)
    comp = next((fm for fl, fm in zip(frag_lists, frag_mols)
                 if int(next(iter(inside))) in fl), None)
    if comp is None:
        raise ValueError("could not isolate the matched component")

    rw = Chem.RWMol(comp)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            rw.ReplaceAtom(atom.GetIdx(), Chem.Atom(1))
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        out = Chem.RemoveHs(out)
    except Exception as exc:
        raise ValueError(f"cannot sanitize the pinned fragment: {exc}") from exc
    if out.GetNumConformers() == 0:
        raise ValueError("pinned fragment lost its coordinates")

    conf = out.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(a.GetIdx()))
                       for a in out.GetAtoms() if a.GetAtomicNum() > 1],
                      dtype=np.float64)
    return _PinnedFragment(
        mol=out,
        smiles=Chem.MolToSmiles(out),
        centroid=coords.mean(axis=0),
        n_heavy=out.GetNumHeavyAtoms(),
        n_attach=n_bnd,
    )


def select_queries(
    provider,
    selected_rows,
    *,
    max_queries: int = 50,
    max_per_complex: int = 2,
    min_heavy: int = 6,
    max_heavy: int = 25,
    query_smiles: str | None = None,
    all_fragments: bool = False,
) -> list[dict]:
    """Pick easily placeable query fragments from each complex.

    By default only ``n_attach`` 1 and 2 are kept (the two cases placement
    handles differently), and within each class the largest fragment is chosen
    so the test is not dominated by tiny rigid groups. ``all_fragments`` keeps
    every eligible fragment instead and is not capped by ``max_per_complex``;
    the downstream placement scores then decide which fragment is useful.

    ``query_smiles`` pins one exact fragment instead (matched on
    constitution, stereo ignored). A pin bypasses the attachment/heavy
    filters because it is an explicit request, and any failure to derive its
    exit vectors is raised rather than skipped. When the requested cut is not
    in the standard fragment set (the cutter never breaks ring<->alpha bonds,
    e.g. an amide at a ring nitrogen), the fragment is built directly from the
    parent ligand and registered on the shared ``by_smiles`` table. Returns at
    most one query in that mode.
    """
    want = _constitution(query_smiles) if query_smiles else None
    out: list[dict] = []
    for row in selected_rows:
        if len(out) >= max_queries:
            break
        pid = row["pdb_id"].lower()
        ctx = provider._context(pid, row["ligand_smiles"])
        if ctx is None:
            if want is not None:
                raise ValueError(
                    f"{pid}: no usable context; the ligand must be a HETATM "
                    f"with experimental coordinates in {pid}.pdb, and the "
                    f"pocket, triplet table and fragment set must be non-empty")
            continue
        lig, env, table, by_smiles = ctx
        if want is not None:
            frag = next((f for f in by_smiles.values()
                         if _constitution(str(f.smiles)) == want), None)
            if frag is None:
                # The requested cut is outside the standard fragment set;
                # build it from the parent ligand and register it so the
                # downstream lookups (by_smiles) still resolve it.
                try:
                    frag = _pinned_fragment_from_ligand(lig, query_smiles)
                except ValueError as exc:
                    have = sorted(str(f.smiles) for f in by_smiles.values()
                                  if f.n_attach in (1, 2))
                    raise ValueError(
                        f"{pid}: --query-smiles not among this complex's "
                        f"fragments ({len(have)} with 1-2 attachments) and "
                        f"not derivable from the ligand: {exc}; e.g. "
                        f"{have[:10]}") from exc
                if hasattr(by_smiles, "add"):
                    by_smiles.add(frag)
                else:
                    by_smiles.setdefault(str(frag.smiles), frag)
            picked = [frag]
        else:
            frags = [f for f in by_smiles.values()
                     if f.n_attach in (1, 2)
                     and min_heavy <= f.n_heavy <= max_heavy]
            if all_fragments:
                picked = sorted(frags, key=lambda f: (
                    int(f.n_attach), -int(f.n_heavy), str(f.smiles)))
            else:
                picked = []
                for na in (1, 2):
                    group = sorted(
                        (f for f in frags if f.n_attach == na),
                        key=lambda f: (-f.n_heavy, str(f.smiles)))
                    if group:
                        picked.append(group[0])
        limit = len(picked) if all_fragments else max_per_complex
        for frag in picked[:limit]:
            try:
                att, att_xyz, exit_xyz = query_exit_vectors(frag.mol, lig)
            except ValueError as exc:
                if want is not None:
                    raise ValueError(f"{pid}: pinned fragment "
                                     f"{frag.smiles}: {exc}") from exc
                continue
            if len(att) != frag.n_attach:
                if want is not None:
                    raise ValueError(
                        f"{pid}: pinned fragment {frag.smiles} has "
                        f"{frag.n_attach} attachment atoms but "
                        f"{len(att)} exit vectors")
                continue
            # Canonical-SMILES positions of the attachment atoms, matching
            # the pool's attach_atoms convention. MolToSmiles rewrites the
            # atom-order property that _smiles_order reads.
            smi = Chem.MolToSmiles(frag.mol)
            order = _smiles_order(frag.mol)
            old_to_canon = {int(old): pos for pos, old in enumerate(order)}
            attach_canon = sorted(old_to_canon.get(int(i), -1) for i in att)
            if len(attach_canon) != len(att) or -1 in attach_canon:
                if want is not None:
                    raise ValueError(
                        f"{pid}: pinned fragment {frag.smiles}: cannot map "
                        f"attachment atoms into canonical order")
                continue
            out.append({
                "pdb_id": pid,
                "parent_smiles": row["ligand_smiles"],
                "smiles": str(frag.smiles),
                "n_attach": int(frag.n_attach),
                "n_heavy": int(frag.n_heavy),
                "_att_idx": att,
                "_att_canon": attach_canon,
                "_att_identity": attachment_identity(frag.mol, att),
                "_att_xyz": att_xyz,
                "_exit_xyz": exit_xyz,
            })
            if want is not None:
                return out
    return out[:max_queries]


def query_weights(frag_mol, env, table, attach_idx,
                  **kwargs) -> dict[int, float]:
    """The query fragment scored in its experimental pose (mean-1 weights)."""
    molh = with_explicit_hydrogens(frag_mol)
    try:
        charges, _method = charges_for(molh, method="eem")
    except Exception:
        charges = gasteiger_charges(molh)
    raw = score_posed(molh, env, table, set(int(i) for i in attach_idx),
                      charges, **kwargs)
    return normalize_weights(raw)


def _profile_pose_indices(poses, geom, ri: int, oi: int,
                          n: int) -> list[int]:
    """Diverse poses to profile: the realistic and oracle picks, then the
    fewest-clash pose of each other conformer, then an even spread."""
    chosen = [ri]
    if oi != ri:
        chosen.append(oi)
    reps = []
    for conf in sorted({g[1] for g in geom}):
        idxs = [i for i, g in enumerate(geom) if g[1] == conf]
        reps.append(min(idxs,
                        key=lambda i: (poses[i]["clashes"], poses[i]["rmsd"])))
    for i in sorted(reps,
                    key=lambda i: (poses[i]["clashes"], poses[i]["rmsd"])):
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)
    if len(chosen) < n:
        step = max(1, len(poses) // n)
        for i in range(0, len(poses), step):
            if len(chosen) >= n:
                break
            if i not in chosen:
                chosen.append(i)
    return chosen[:n]


def score_candidate(
    smiles: str,
    cand_attach,
    q_att_xyz,
    q_exit_xyz,
    env,
    table,
    q_weights: dict[int, float],
    *,
    n_confs: int = 3,
    rolls: int = 24,
    max_poses: int = 48,
    n_profile_poses: int = 4,
    return_weights: bool = False,
    q_profile: dict | None = None,
    obstacles=None,
    **kwargs,
) -> dict | None:
    """Embed, place and score one candidate; returns realistic + oracle arms.

    ``realistic`` picks the pose with the fewest clashes (ties: attachment
    RMSD), then scores its key overlap with the query. ``oracle`` keeps the
    best overlap over every pose. The gap between them is the placement cost.
    ``obstacles`` adds fixed heavy-atom coordinates to the clash set -- the
    retained parent-ligand atoms when a single fragment is replaced. It
    changes pose selection and the clash count, not the key overlap.

    ``return_weights`` attaches the mean-1 normalized per-key weights of the
    selected poses, so the placed analog can itself be used as an index query;
    the realistic pose also gets ``weights_shell``, the same weights restricted
    to triplets inside its own contact shell, and ``marks``, the four per-key
    energy marks the library index stores.
    ``q_profile`` adds contact-local scores (``profile_overlap``); they are
    computed against the query fragment's own contact shell instead of the
    whole 12 A pocket, which the fingerprint overlap is dominated by.
    ``shell_energy_r_mean`` averages over ``n_profile_poses`` diverse poses,
    because the per-pose correlation is noisy. The same pose ensemble also
    yields the ``shell_persist_*`` scores and the interaction displacement
    (``persistence_overlap``): contacts weighted by how often they survive
    across poses, plus retention and signed energy change relative to the
    query fragment.
    """
    embedded = embed_candidate(smiles, n_confs)
    if embedded is None:
        return None
    mh, cids = embedded
    charges, charge_method = conformer_charges(mh, cids)
    attach_set = set(int(a) for a in cand_attach)

    poses = []
    geom = []  # (posed mol, conformer index) parallel to ``poses``
    for conf_i, cid in enumerate(cids):
        for r, t, rmsd in placements(
                mh, cid, cand_attach, q_att_xyz, q_exit_xyz,
                rolls=rolls, max_poses=max_poses):
            posed = pose_mol(mh, cid, r, t)
            n_clash, min_d = clash_stats(posed, env, extra_xyz=obstacles)
            raw = score_posed(posed, env, table, attach_set,
                              charges[conf_i], **kwargs)
            norm = normalize_weights(raw)
            compared = weight_overlap(q_weights, norm)
            entry = {
                "conf": conf_i,
                "rmsd": rmsd,
                "clashes": n_clash,
                "min_dist": min_d,
                **compared,
            }
            if return_weights:
                entry["weights"] = norm
            poses.append(entry)
            geom.append((posed, conf_i))
    if not poses:
        return None

    ri = min(range(len(poses)),
             key=lambda i: (poses[i]["clashes"], poses[i]["rmsd"]))
    oi = max(range(len(poses)),
             key=lambda i: (poses[i]["weighted"], poses[i]["coverage"],
                            -poses[i]["clashes"]))
    if q_profile is not None:
        # A handful of poses, not all ~144: the per-residue energy correlation
        # is pose-noisy (r ~ 0.4 between two poses of the same candidate), so
        # the candidate score is the mean over a diverse subset. Profiling
        # stays ~2% of the scoring cost.
        profiles = {}
        for i in _profile_pose_indices(poses, geom, ri, oi, n_profile_poses):
            posed, conf_i = geom[i]
            prof = residue_profile(posed, env, charges[conf_i],
                                   attach_set=attach_set, **kwargs)
            profiles[i] = prof
            poses[i].update(profile_overlap(q_profile, prof))
        vals = [poses[i]["shell_energy_r"] for i in profiles
                if poses[i].get("shell_energy_r") is not None]
        mean_r = float(np.mean(vals)) if vals else None
        ensemble = persistence_overlap(q_profile, profiles)
        for i in (ri, oi):
            poses[i]["shell_energy_r_mean"] = mean_r
            poses[i].update(ensemble)
        if return_weights and ri in profiles:
            posed, conf_i = geom[ri]
            raw = score_posed(posed, env, table, attach_set, charges[conf_i],
                              residue_mask=profiles[ri]["mask"], **kwargs)
            poses[ri]["weights_shell"] = normalize_weights(raw)
            _fp, poses[ri]["marks"] = score_posed(
                posed, env, table, attach_set, charges[conf_i],
                return_marks=True, **kwargs)
    realistic = poses[ri]
    oracle = poses[oi]
    return {
        "charge_method": charge_method,
        "n_poses": len(poses),
        "n_conformers": len(cids),
        "realistic": realistic,
        "oracle": oracle,
    }


__all__ = [
    "FP_BITS",
    "FP_RADIUS",
    "attachment_fingerprint",
    "attachment_identity",
    "build_fingerprints",
    "clash_stats",
    "conformer_charges",
    "contact_shell",
    "dummy_fragment_smiles",
    "embed_candidate",
    "exit_directions",
    "fp_paths",
    "fuse_rankings",
    "kabsch",
    "normalize_weights",
    "persistence_overlap",
    "placements",
    "pose_mol",
    "profile_overlap",
    "query_exit_vectors",
    "query_fingerprint",
    "query_marks",
    "query_profile",
    "query_weights",
    "random_row",
    "read_pool_rows",
    "residue_profile",
    "score_candidate",
    "score_posed",
    "select_queries",
    "tanimoto_topk",
    "weight_overlap",
]
