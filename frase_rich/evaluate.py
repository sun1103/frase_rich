"""Leave-one-complex-out retrieval benchmark for the weighted index.

Protocol
--------
For every fragment whose canonical SMILES occurs in at least two complexes:

1. Query with that fragment's environment, in one of three modes:
   ``stored`` uses the rough query path (CA coordinates + 11-bit codes only);
   ``refmol`` re-derives the environment from the query complex's parent
   ligand in its experimental pose and computes per-fingerprint weights with
   the same library-side code; ``reffrag`` uses the fragment itself in that
   pose against the per-complex unified pocket (requires an index built with
   ``--env-center pocket``).
2. Exclude all fragments from the query's own PDB id.
3. Rank candidates by weighted and plain scores and find the first hit with
   the same fragment SMILES.

The ``stored`` mode tests the asymmetry we care about most: a rich library
template matched by a coarse query. The ``refmol`` mode is a stronger task and
carries its own nested baseline: its per-key Jaccard score is exactly the
all-uniform version of the same query weights. If weighted scoring only wins on
the complexes it was built from, the signal is memorization; excluding the
source PDB makes that failure visible.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from collections import OrderedDict, defaultdict

import numpy as np

from .build import _find_ligand, _group_residues
from .features import attachment_atoms
from .fragmentation import canonical_coords, fragment_ligand
from .geometry import canonical_triplet_fp, enumerate_triplet_idxs
from .index import WeightedTripletIndex
from .pdb_io import parse_pdb
from .protein import protein_residues
from .weighting import (
    MARKS as MARK_META,
    fp_weights_in_env,
    normalize_marks,
    pocket_environment,
    query_fp_weights,
    triplet_table,
)


def _trim_block(block: str, lig_xyz: np.ndarray,
                radius: float = 15.0) -> str:
    """Drop residue records farther than ``radius`` from the ligand.

    ``protein_residues`` runs RDKit over the whole deposited structure, which
    costs minutes on large assemblies (8ana: 86k atoms, ~105 s). The pocket
    environment only ever contains residues within 12 A of the ligand, so
    anything past 15 A is dead weight. Residues are kept whole so their atom
    sets -- and therefore the per-residue weights -- are unchanged.
    """
    lines = block.splitlines()
    groups: dict[tuple, list[int]] = {}
    xyz: dict[int, tuple[float, float, float]] = {}
    for i, line in enumerate(lines):
        if line[:6].strip() in ("ATOM", "HETATM"):
            groups.setdefault((line[21], line[22:27]), []).append(i)
            xyz[i] = (float(line[30:38]), float(line[38:46]),
                      float(line[46:54]))
    keep: set[int] = set()
    r2 = radius * radius
    for idxs in groups.values():
        pts = np.array([xyz[i] for i in idxs], dtype=np.float64)
        d2 = ((pts[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(axis=-1)
        if float(d2.min()) <= r2:
            keep.update(idxs)
    return "\n".join(
        line for i, line in enumerate(lines)
        if i in keep or line[:6].strip() not in ("ATOM", "HETATM")) + "\n"


def _het_block(block: str) -> str:
    """Ligand extraction only inspects HETATM groups; parsing the protein
    atoms too is pure overhead on large structures. CONECT records are kept
    so the ligand lookup can use explicit connectivity when proximity
    bonding over-bonds a crowded pose."""
    return "".join(line + "\n" for line in block.splitlines()
                   if line.startswith(("HETATM", "CONECT"))) + "END\n"


class _RefMolProvider:
    """Per-PDB ref-mol query contexts, shared by all fragments of a complex."""

    def __init__(self, cache: str, features: dict, max_contexts: int = 32,
                 paths: dict[str, str] | None = None):
        self.cache = cache
        self.features = dict(features or {})
        self.max_contexts = max_contexts
        self.paths = {str(k).lower(): v for k, v in (paths or {}).items()}
        self._by_pdb: OrderedDict[str, tuple | None] = OrderedDict()
        self.n_failed = 0

    def _store(self, key: str, ctx) -> None:
        self._by_pdb[key] = ctx
        self._by_pdb.move_to_end(key)
        while len(self._by_pdb) > self.max_contexts:
            self._by_pdb.popitem(last=False)

    def _kwargs(self) -> dict:
        f = self.features
        return {
            "env_radius": f.get("env_radius", 12.0),
            "d_lo": f.get("d_lo", 8.0),
            "d_hi": f.get("d_hi", 12.0),
            "temperature": f.get("temperature", 1.0),
            "use_energy": f.get("use_energy", True),
            "use_content": f.get("use_content", True),
            "pooling": f.get("energy_pooling", "mean"),
            "contact_cutoff": f.get("contact_cutoff", 6.0),
            "use_angles": f.get("angles", True),
            "backbone": f.get("backbone", True),
            "energy_scale": f.get("energy_scale", 0.2),
            "content_scale": f.get("content_scale", 0.3),
            "charge_method": f.get("charge_method", "eem"),
            "fp_aggregate": f.get("fp_aggregate", "mean"),
            "use_pair_bias": f.get("use_pair_bias", True),
            "legacy_hbond": f.get("legacy_hbond", False),
        }

    def get(self, pdb_id: str, parent_smiles: str | None):
        key = pdb_id.lower()
        if key in self._by_pdb:
            self._by_pdb.move_to_end(key)
            return self._by_pdb[key]
        ctx = None
        path = self.paths.get(key) or os.path.join(self.cache, f"{key}.pdb")
        try:
            block = open(path).read()
            lig = (_find_ligand(_het_block(block), parent_smiles)
                   if parent_smiles else None)
            if lig is not None:
                conf = lig.GetConformer()
                lig_xyz = np.array(
                    [list(conf.GetAtomPosition(a.GetIdx()))
                     for a in lig.GetAtoms() if a.GetAtomicNum() > 1],
                    dtype=np.float64)
                block = _trim_block(block, lig_xyz)
            residues = protein_residues(block)
            if residues is None:
                residues = _group_residues(parse_pdb(path))
            if lig is not None and residues:
                fp, env, method = query_fp_weights(
                    lig, residues, **self._kwargs())
                if fp:
                    ctx = (fp, env, method)
        except Exception:
            ctx = None
        if ctx is None:
            self.n_failed += 1
        self._store(key, ctx)
        return ctx


class _FragmentSet(dict):
    """``smiles -> first fragment`` plus every fragment sharing that SMILES.

    A symmetric parent yields the same constitution at two sites (19% of
    indexed rows share their ``(pdb, smiles)`` key with another row), so
    matching a stored fragment by SMILES alone can pick the wrong copy and
    recompute its energies in the wrong local environment. ``dict`` behaviour
    is unchanged for the callers that only need one fragment per SMILES;
    ``all()`` exposes the duplicates for coordinate disambiguation.
    """

    def __init__(self):
        super().__init__()
        self._all: dict[str, list] = defaultdict(list)

    def add(self, frag) -> None:
        smiles = str(frag.smiles)
        self._all[smiles].append(frag)
        self.setdefault(smiles, frag)

    def all(self, smiles: str) -> list:
        return self._all.get(str(smiles), [])


def _pick_fragment(cands: list, frag_coords) -> object:
    """The candidate whose deposited coordinates match ``frag_coords``.

    Both sides are in canonical-SMILES atom order (``canonical_coords`` at
    build time and here), so the match is a direct comparison, not a fit.
    """
    if len(cands) == 1 or frag_coords is None:
        return cands[0]
    want = np.asarray(frag_coords, dtype=np.float64)
    best, best_d = cands[0], None
    for frag in cands:
        try:
            got = np.asarray(canonical_coords(frag.mol, str(frag.smiles)),
                             dtype=np.float64)
        except Exception:
            continue
        if got.shape != want.shape:
            continue
        d = float(np.max(np.abs(got - want)))
        if best_d is None or d < best_d:
            best, best_d = frag, d
    return best


class _RefFragProvider:
    """Per-PDB ref-fragment contexts on the unified pocket.

    The environment is the pocket the library fragments were scored on (one
    per complex); the query molecule is the fragment itself, rebuilt from the
    deposited parent so its pose is in the protein frame. All fragments of a
    complex share the pocket and the triplet table.

    ``paths`` maps a ``pdb_id`` to an explicit structure file, so a target
    outside ``cache`` can be used without copying it in. ``extra_caches``
    are searched in order after ``cache``: borrowed fragments may come from
    the union cache while the target complex lives in the PDBbind cache.
    """

    def __init__(self, cache: str, features: dict, max_contexts: int = 32,
                 paths: dict[str, str] | None = None,
                 extra_caches: list[str] | None = None):
        self.cache = cache
        self.features = dict(features or {})
        self.max_contexts = max_contexts
        self.paths = {str(k).lower(): v for k, v in (paths or {}).items()}
        self.extra_caches = [str(p) for p in (extra_caches or []) if p]
        self._by_pdb: OrderedDict[str, tuple | None] = OrderedDict()
        self._by_ligand: OrderedDict[str, tuple | None] = OrderedDict()
        self.n_failed = 0

    def _resolve_path(self, key: str) -> str:
        path = self.paths.get(key)
        if path:
            return path
        for base in (self.cache, *self.extra_caches):
            candidate = os.path.join(base, f"{key}.pdb")
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.cache, f"{key}.pdb")

    def _store(self, key: str, ctx) -> None:
        self._by_pdb[key] = ctx
        self._by_pdb.move_to_end(key)
        while len(self._by_pdb) > self.max_contexts:
            self._by_pdb.popitem(last=False)

    def _kwargs(self) -> dict:
        f = self.features
        return {
            "d_lo": f.get("d_lo", 8.0),
            "d_hi": f.get("d_hi", 12.0),
            "temperature": f.get("temperature", 1.0),
            "use_energy": f.get("use_energy", True),
            "use_content": f.get("use_content", True),
            "pooling": f.get("energy_pooling", "mean"),
            "contact_cutoff": f.get("contact_cutoff", 6.0),
            "use_angles": f.get("angles", True),
            "backbone": f.get("backbone", True),
            "energy_scale": f.get("energy_scale", 0.2),
            "content_scale": f.get("content_scale", 0.3),
            "charge_method": f.get("charge_method", "eem"),
            "fp_aggregate": f.get("fp_aggregate", "mean"),
            "use_pair_bias": f.get("use_pair_bias", True),
            "legacy_hbond": f.get("legacy_hbond", False),
        }

    def _context(self, pdb_id: str, parent_smiles: str | None):
        key = pdb_id.lower()
        if key in self._by_pdb:
            self._by_pdb.move_to_end(key)
            return self._by_pdb[key]
        ctx = None
        path = self._resolve_path(key)
        try:
            block = open(path).read()
            lig = (_find_ligand(_het_block(block), parent_smiles)
                   if parent_smiles else None)
            if lig is not None:
                conf = lig.GetConformer()
                lig_xyz = np.array(
                    [list(conf.GetAtomPosition(a.GetIdx()))
                     for a in lig.GetAtoms() if a.GetAtomicNum() > 1],
                    dtype=np.float64)
                block = _trim_block(block, lig_xyz)
            residues = protein_residues(block)
            if residues is None:
                residues = _group_residues(parse_pdb(path))
            if lig is not None and residues:
                f = self.features
                env = pocket_environment(
                    residues, lig_xyz, f.get("env_radius", 12.0))
                table = triplet_table(
                    env, d_lo=f.get("d_lo", 8.0),
                    d_hi=f.get("d_hi", 12.0),
                    use_pair_bias=f.get("use_pair_bias", True))
                frags = fragment_ligand(
                    lig, min_da=f.get("min_da", 50.0),
                    max_da=f.get("max_da", 300.0),
                    max_frags=f.get("max_frags", 200),
                    max_blocks=f.get("max_blocks", 16),
                    cut_biaryl=f.get("cut_biaryl", True),
                    cut_amide=f.get("cut_amide", True))
                by_smiles = _FragmentSet()
                for frag in frags:
                    by_smiles.add(frag)
                if env and not table.empty and by_smiles:
                    ctx = (lig, env, table, by_smiles)
        except Exception:
            ctx = None
        if ctx is None:
            self.n_failed += 1
        self._store(key, ctx)
        return ctx

    def ligand_context(self, pdb_id: str, parent_smiles: str | None):
        """Parent ligand and its fragment cuts, without the pocket machinery.

        Borrowing a fragment only needs to recover which indexed atoms sat on
        the cut bond, so the deposited ligand pose and the same cut
        enumeration the build used are enough. Building the 12 A environment
        and triplet table here would cost seconds per source complex for data
        the query site already supplies.
        """
        key = pdb_id.lower()
        if key in self._by_ligand:
            self._by_ligand.move_to_end(key)
            return self._by_ligand[key]
        ctx = None
        try:
            with open(self._resolve_path(key)) as handle:
                block = handle.read()
            lig = _find_ligand(_het_block(block), parent_smiles)
            if lig is not None:
                f = self.features
                frags = fragment_ligand(
                    lig, min_da=f.get("min_da", 50.0),
                    max_da=f.get("max_da", 300.0),
                    max_frags=f.get("max_frags", 200),
                    max_blocks=f.get("max_blocks", 16),
                    cut_biaryl=f.get("cut_biaryl", True),
                    cut_amide=f.get("cut_amide", True))
                by_smiles = _FragmentSet()
                for frag in frags:
                    by_smiles.add(frag)
                if by_smiles:
                    ctx = (lig, by_smiles)
        except Exception:
            ctx = None
        if ctx is None:
            self.n_failed += 1
        self._by_ligand[key] = ctx
        self._by_ligand.move_to_end(key)
        while len(self._by_ligand) > self.max_contexts * 8:
            self._by_ligand.popitem(last=False)
        return ctx

    def get(self, pdb_id: str, parent_smiles: str | None,
            frag_smiles: str, frag_coords=None):
        ctx = self._context(pdb_id, parent_smiles)
        if ctx is None:
            return None
        lig, env, table, by_smiles = ctx
        cands = (by_smiles.all(frag_smiles)
                 if hasattr(by_smiles, "all") else [])
        if not cands:
            first = by_smiles.get(str(frag_smiles))
            if first is None:
                self.n_failed += 1
                return None
            cands = [first]
        frag = _pick_fragment(cands, frag_coords)
        att = (attachment_atoms(frag.mol, lig)
               if self.features.get("explicit_h", True) else set())
        try:
            fp, _env, method, energies = fp_weights_in_env(
                frag.mol, env, table=table, exclude_donor_atoms=att,
                return_energies=True, **self._kwargs())
        except Exception:
            self.n_failed += 1
            return None
        if not fp:
            return None
        return fp, energies, method


def _build_dense_marks(index: WeightedTripletIndex, meta_key: str):
    """Stack every fragment's normalized mark vector into one dense matrix.

    The per-pair ``mark_overlap`` loop is the evaluation bottleneck: one query
    costs ``n_candidates * n_keys`` Python-level ``min`` calls. Stacking the
    library marks once turns that into one numpy pass per query. ``key_ids``
    holds the union of mark keys in sorted order; rows with no positive
    support stay all-zero (their score is 0 for every query).
    """
    key_set: set[int] = set()
    for info in index._frags.values():
        raw = info.get(meta_key)
        if raw:
            key_set.update(int(k) for k, v in raw.items() if v > 0)
    if not key_set:
        return None
    key_ids = np.array(sorted(key_set), dtype=np.int64)
    fids = list(index._frags)
    row_of = {fid: i for i, fid in enumerate(fids)}
    lib = np.zeros((len(fids), len(key_ids)), dtype=np.float64)
    n_pos = np.zeros(len(fids), dtype=np.int64)
    for i, fid in enumerate(fids):
        raw = index._frags[fid].get(meta_key)
        if not raw:
            continue
        items = [(int(k), float(v)) for k, v in raw.items() if v > 0]
        if not items:
            continue
        keys = np.fromiter((k for k, _ in items), dtype=np.int64,
                           count=len(items))
        vals = np.fromiter((v for _, v in items), dtype=np.float64,
                           count=len(items))
        # Same mean-1 scaling as normalize_marks: a uniform vector is all ones.
        lib[i, np.searchsorted(key_ids, keys)] = vals * (
            len(items) / float(vals.sum()))
        n_pos[i] = len(items)
    return lib, n_pos, key_ids, row_of


def _trim_index(index: WeightedTripletIndex, *, keep_env: bool = False,
                keep_frag_coords: bool = False) -> None:
    """Free the parts of a loaded index the dense evaluation path no longer
    needs: the posting lists (53M tuples on the union index) and, unless the
    stored-query mode needs them, the per-fragment environment arrays.
    ``keep_frag_coords`` is for ``reffrag``, which uses the deposited
    coordinates to pick the right copy when a SMILES occurs twice."""
    for info in index._frags.values():
        if not keep_env:
            for key in ("_env_coords", "_env_codes", "_env_weights"):
                info.pop(key, None)
        if not (keep_env or keep_frag_coords):
            info.pop("_frag_coords", None)
    index._postings.clear()
    gc.collect()


def _dense_mark_scores(q_marks: dict[int, float], dense) -> np.ndarray:
    """``mark_overlap`` for every library row at once (see _build_dense_marks)."""
    lib, n_pos, key_ids, _row_of = dense
    keys = np.fromiter(q_marks.keys(), dtype=np.int64, count=len(q_marks))
    vals = np.fromiter(q_marks.values(), dtype=np.float64,
                       count=len(q_marks))
    cols = np.searchsorted(key_ids, keys)
    inside = cols < len(key_ids)
    inside[inside] = key_ids[cols[inside]] == keys[inside]
    order = np.argsort(cols[inside], kind="stable")
    cols = cols[inside][order]
    vals = vals[inside][order]
    n_query = len(q_marks)
    out = np.zeros(lib.shape[0], dtype=np.float64)
    step = 4096
    for start in range(0, lib.shape[0], step):
        stop = min(start + step, lib.shape[0])
        w = np.minimum(lib[start:stop][:, cols], vals).sum(axis=1)
        out[start:stop] = w / np.maximum(n_query + n_pos[start:stop] - w,
                                         1e-9)
    return out


class _DenseLibrary:
    """Row-aligned numpy view of the index for array-based ranking.

    The benchmark only needs the *rank* of the first same-SMILES fragment, but
    ``WeightedTripletIndex.query`` materialises and sorts a dict row for every
    candidate (nearly the whole library on a dense fingerprint set). Stacking
    the library weights once lets each query score all fragments in one numpy
    pass and count how many rows beat the best positive, avoiding ~10^5 dict
    rows and three sorts per query.
    """

    def __init__(self, index: WeightedTripletIndex):
        self.fids = list(index._frags)
        self.row_of = {fid: i for i, fid in enumerate(self.fids)}
        self.key_ids = np.array(sorted(index._postings), dtype=np.int64)
        self.n_lib = np.fromiter(
            (int(index._frags[f].get("_n_distinct", 0)) for f in self.fids),
            dtype=np.int64, count=len(self.fids))
        self.weights = np.zeros((len(self.fids), len(self.key_ids)),
                                dtype=np.float64)
        for col, fp in enumerate(self.key_ids):
            postings = index._postings[fp]
            rows = np.fromiter((self.row_of[fid] for fid, _w in postings),
                               dtype=np.int64, count=len(postings))
            vals = np.fromiter((w for _fid, w in postings),
                               dtype=np.float64, count=len(postings))
            self.weights[rows, col] = vals
        order = {fid: i for i, fid in enumerate(sorted(self.fids))}
        self.fid_rank = np.fromiter((order[f] for f in self.fids),
                                    dtype=np.int64, count=len(self.fids))
        self.pdb = np.array(
            [str(index._frags[f].get("pdb_id", "")).lower()
             for f in self.fids], dtype=object)

    def columns(self, keys) -> tuple[np.ndarray, np.ndarray]:
        """Map fingerprint keys to matrix columns; second value marks hits."""
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size == 0:
            return keys, np.zeros(0, dtype=bool)
        cols = np.searchsorted(self.key_ids, keys)
        inside = cols < len(self.key_ids)
        if inside.any():
            inside[inside] = self.key_ids[cols[inside]] == keys[inside]
        return cols, inside

    def _accumulate(self, cols, vals, combine):
        n_match = np.zeros(len(self.fids), dtype=np.int64)
        w_match = np.zeros(len(self.fids), dtype=np.float64)
        if cols.size == 0:
            return n_match, w_match
        step = 4096
        for start in range(0, len(self.fids), step):
            stop = min(start + step, len(self.fids))
            block = self.weights[start:stop][:, cols]
            if combine == "min":
                match = np.minimum(block, vals)
            else:
                match = np.sqrt(block * vals)
            n_match[start:stop] = np.count_nonzero(match > 0, axis=1)
            w_match[start:stop] = match.sum(axis=1)
        return n_match, w_match

    def score_fp_weights(self, q_weights, *, combine="min"):
        """Per-fragment (n_match, w_match) for a weighted exact-key query."""
        raw = {int(k): float(v) for k, v in q_weights.items() if v > 0}
        n_query = len(raw)
        if not raw:
            zeros = np.zeros(len(self.fids), dtype=np.int64)
            return n_query, zeros, zeros.astype(np.float64)
        scale = n_query / float(sum(raw.values()))
        keys = np.fromiter(raw.keys(), dtype=np.int64, count=n_query)
        vals = np.fromiter(raw.values(), dtype=np.float64,
                           count=n_query) * scale
        cols, inside = self.columns(keys)
        n_match, w_match = self._accumulate(cols[inside], vals[inside],
                                            combine)
        return n_query, n_match, w_match

    def score_stored(self, query_fps: set[int]):
        """Per-fragment (n_match, w_match) for the coarse stored-query path.

        There are no query weights: ``w_match`` is the sum of library weights
        over shared keys, exactly as ``WeightedTripletIndex.query`` does.
        """
        keys = sorted(query_fps)
        cols, inside = self.columns(keys)
        n_match = np.zeros(len(self.fids), dtype=np.int64)
        w_match = np.zeros(len(self.fids), dtype=np.float64)
        # Stored mode has no query-side weights: every shared key adds the
        # library weight itself rather than a pairwise minimum.
        if inside.any():
            sub = cols[inside]
            step = 4096
            for start in range(0, len(self.fids), step):
                stop = min(start + step, len(self.fids))
                block = self.weights[start:stop][:, sub]
                n_match[start:stop] = np.count_nonzero(block > 0, axis=1)
                w_match[start:stop] = block.sum(axis=1)
        return len(keys), n_match, w_match

    def scores(self, n_query: int, n_match: np.ndarray,
               w_match: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        union = np.maximum(n_query + self.n_lib - n_match, 1e-9)
        w_union = np.maximum(n_query + self.n_lib - w_match, 1e-9)
        return n_match / union, w_match / w_union

    def candidate_mask(self, n_match: np.ndarray, exclude_pdb: str) -> np.ndarray:
        mask = n_match >= 1
        if exclude_pdb:
            mask = mask & (self.pdb != exclude_pdb.lower())
        return mask

    def rank_of_first(self, score: np.ndarray, n_match: np.ndarray,
                      mask: np.ndarray,
                      positive_ids: set[str]) -> int | None:
        """Rank of the best positive under index.query's ordering.

        The ordering key is ``(-round(score, 12), -n_match, frag_id)``: rounded
        keys suppress float noise (e.g. the all-off arm differs from Jaccard
        at ~1e-16), and frag_id rather than incoming order keeps the weighted
        sort from leaking into the Jaccard baseline. Counting rows strictly
        better than the best positive gives the same position as sorting.
        """
        rounded = np.round(score, 12)
        rows = [self.row_of[p] for p in positive_ids if p in self.row_of]
        if not rows:
            return None
        for row in rows:
            if not mask[row]:
                return None
        best = max(rows, key=lambda i: (rounded[i], n_match[i],
                                        -self.fid_rank[i]))
        bs, bn, bf = float(rounded[best]), int(n_match[best]), int(
            self.fid_rank[best])
        tie = mask & (rounded == bs)
        better = (mask & (rounded > bs)) | (tie & (n_match > bn))
        better |= tie & (n_match == bn) & (self.fid_rank < bf)
        return int(np.count_nonzero(better)) + 1


def evaluate(
    index: WeightedTripletIndex,
    *,
    max_queries: int | None = 2000,
    seed: int = 0,
    output: str | None = None,
    protein_groups: str | None = None,
    positive_mode: str = "any",
    query_mode: str = "stored",
    cache: str | None = None,
    combine: str = "min",
    query_noise: float = 0.0,
    score_mode: str = "fp",
    mark: str = "mean",
    allow_feature_mismatch: bool = False,
) -> dict:
    if score_mode not in ("fp", "emark"):
        raise ValueError(f"unknown score_mode: {score_mode}")
    if mark not in MARK_META:
        raise ValueError(f"unknown mark: {mark}")
    provider = None
    if query_mode in ("refmol", "reffrag"):
        if not cache:
            raise ValueError(f"{query_mode} mode requires the structure cache")
        if not index.features and not allow_feature_mismatch:
            raise ValueError(
                f"{query_mode} mode needs an index with feature metadata "
                "(v2); pass allow_feature_mismatch=True to use defaults")
        if query_mode == "reffrag" and not allow_feature_mismatch:
            center = index.features.get("env_center", "fragment")
            if center != "pocket":
                raise ValueError(
                    "reffrag requires an index built with "
                    "--env-center pocket (found "
                    f"'{center}'); rebuild or pass allow_feature_mismatch")
        cls = _RefMolProvider if query_mode == "refmol" else _RefFragProvider
        provider = cls(cache, index.features)
    elif query_mode != "stored":
        raise ValueError(f"unknown query mode: {query_mode}")

    if score_mode == "emark":
        if query_mode != "reffrag":
            raise ValueError(
                "score_mode=emark needs query_mode=reffrag (only the ref-frag "
                "provider returns per-key interaction marks)")
        if not allow_feature_mismatch:
            stored = str(index.features.get("marks", ""))
            if mark not in [m.strip() for m in stored.split(",")]:
                raise ValueError(
                    f"index does not store the '{mark}' mark "
                    f"(marks={stored!r}); rebuild or pass "
                    "allow_feature_mismatch")

    group_of: dict[str, str] = {}
    if protein_groups:
        with open(protein_groups) as handle:
            group_of = {str(k).lower(): str(v)
                        for k, v in json.load(handle).items()}

    def _pdb(fid: str) -> str:
        return str(index._frags[fid].get("pdb_id", "")).lower()

    by_smiles: dict[str, list[str]] = defaultdict(list)
    for fid, info in index._frags.items():
        smiles = str(info.get("frag_smiles", ""))
        if smiles:
            by_smiles[smiles].append(fid)

    query_ids: list[str] = []
    positives: dict[str, set[str]] = {}
    n_unknown_group = 0
    for smiles, fids in by_smiles.items():
        for fid in fids:
            q_pdb = _pdb(fid)
            q_gid = group_of.get(q_pdb)
            pos: set[str] = set()
            for other in fids:
                if other == fid or _pdb(other) == q_pdb:
                    continue
                if positive_mode != "any":
                    o_gid = group_of.get(_pdb(other))
                    if q_gid is None or o_gid is None:
                        n_unknown_group += 1
                        continue
                    same = q_gid == o_gid
                    if positive_mode == "same-protein" and not same:
                        continue
                    if positive_mode == "cross-protein" and same:
                        continue
                pos.add(other)
            if pos:
                query_ids.append(fid)
                positives[fid] = pos
    query_ids.sort()
    if n_unknown_group:
        print(f"skipped {n_unknown_group:,} comparisons with no protein group",
              flush=True)

    if max_queries is not None and len(query_ids) > max_queries:
        rng = random.Random(seed)
        # Sort after sampling so fragments of one complex are queried
        # consecutively and the per-PDB context cache is reused.
        query_ids = sorted(rng.sample(query_ids, max_queries))
    mean_pos = (float(np.mean([len(positives[f]) for f in query_ids]))
                if query_ids else 0.0)
    print(f"eligible queries: {len(query_ids):,}  mode={positive_mode}  "
          f"mean positives/query={mean_pos:.1f}", flush=True)

    dense = _DenseLibrary(index)
    print(f"prepared dense library: {len(dense.fids):,} fragments x "
          f"{len(dense.key_ids):,} keys", flush=True)
    dense_marks = None
    if score_mode == "emark":
        dense_marks = _build_dense_marks(index, MARK_META[mark])
        if dense_marks is None:
            raise ValueError(f"index stores no '{mark}' marks; rebuild it")
        # _build_dense_marks iterates the same _frags order as _DenseLibrary,
        # so its rows index into dense's scores directly.
        print(f"prepared '{mark}' marks for "
              f"{int(np.count_nonzero(dense_marks[1])):,} fragments",
              flush=True)
    _trim_index(index, keep_env=(query_mode == "stored"),
                keep_frag_coords=(query_mode == "reffrag"))

    wrapped = 0
    misses = 0
    ranks = {"weighted": [], "jaccard": []}
    if score_mode == "emark":
        ranks["mark"] = []
    wins = losses = ties = 0
    mark_wins = mark_losses = mark_ties = 0
    n_candidates: list[int] = []

    for q_idx, fid in enumerate(query_ids, start=1):
        info = index._frags[fid]
        pdb_id = str(info.get("pdb_id", ""))
        q_marks = None
        if query_mode in ("refmol", "reffrag"):
            if query_mode == "reffrag":
                ctx = provider.get(
                    pdb_id, info.get("parent_smiles"),
                    info.get("frag_smiles", ""),
                    frag_coords=info.get("_frag_coords"))
            else:
                ctx = provider.get(pdb_id, info.get("parent_smiles"))
            if ctx is None:
                misses += 1
                continue
            fp = ctx[0]
            if score_mode == "emark":
                energies = ctx[1] if len(ctx) > 2 else None
                raw_marks = (energies or {}).get(mark) or {}
                if not raw_marks:
                    misses += 1
                    continue
                if query_noise > 0:
                    rng = random.Random(f"{seed}:{fid}")
                    raw_marks = {
                        k: v * float(np.exp(rng.gauss(0.0, query_noise)))
                        for k, v in raw_marks.items()}
                q_marks = normalize_marks(raw_marks)
                if not q_marks:
                    misses += 1
                    continue
            elif query_noise > 0:
                rng = random.Random(f"{seed}:{fid}")
                fp = {k: v * np.exp(rng.gauss(0.0, query_noise))
                      for k, v in fp.items()}
            n_query, n_match, w_match = dense.score_fp_weights(
                fp, combine=combine)
        else:
            coords = info.get("_env_coords")
            codes = info.get("_env_codes")
            if coords is None or codes is None or len(coords) < 3:
                misses += 1
                continue
            pts = np.asarray(coords, dtype=np.float64)
            int_codes = np.asarray(codes, dtype=np.int64)
            query_fps = {
                canonical_triplet_fp((int(int_codes[i]), int(int_codes[j]),
                                      int(int_codes[k])))
                for i, j, k in enumerate_triplet_idxs(pts, d_lo=8.0,
                                                      d_hi=12.0)
            }
            n_query, n_match, w_match = dense.score_stored(query_fps)
        mask = dense.candidate_mask(n_match, pdb_id)
        n_cand = int(np.count_nonzero(mask))
        if n_cand == 0:
            misses += 1
            continue
        n_candidates.append(n_cand)
        score_j, score_w = dense.scores(n_query, n_match, w_match)
        pos = positives[fid]
        r_w = dense.rank_of_first(score_w, n_match, mask, pos)
        r_j = dense.rank_of_first(score_j, n_match, mask, pos)
        if r_w is None or r_j is None:
            misses += 1
            continue
        ranks["weighted"].append(r_w)
        ranks["jaccard"].append(r_j)
        if r_w < r_j:
            wins += 1
        elif r_w > r_j:
            losses += 1
        else:
            ties += 1
        if score_mode == "emark":
            mark_scores = _dense_mark_scores(q_marks, dense_marks)
            r_m = dense.rank_of_first(mark_scores, n_match, mask, pos)
            if r_m is None:
                misses += 1
                continue
            ranks["mark"].append(r_m)
            if r_m < r_j:
                mark_wins += 1
            elif r_m > r_j:
                mark_losses += 1
            else:
                mark_ties += 1
        wrapped += 1

        if q_idx % 50 == 0:
            print(f"  {q_idx:,}/{len(query_ids):,} queries", flush=True)

    result = {}
    for method, values in ranks.items():
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == 0:
            result[method] = {"n": 0}
            continue
        result[method] = {
            "n": int(len(arr)),
            "mrr": float(np.mean(1.0 / arr)),
            "recall@10": float(np.mean(arr <= 10)),
            "recall@50": float(np.mean(arr <= 50)),
            "recall@100": float(np.mean(arr <= 100)),
            "median_rank": float(np.median(arr)),
            "mean_rank": float(np.mean(arr)),
        }

    summary = {
        "positive_mode": positive_mode,
        "query_mode": query_mode,
        "combine": combine,
        "score_mode": score_mode,
        "mark": mark if score_mode == "emark" else None,
        "query_noise": query_noise,
        "n_queries_scored": wrapped,
        "n_misses": misses,
        "query_context_failures": provider.n_failed if provider else 0,
        "refmol_context_failures": provider.n_failed if provider else 0,
        "mean_positives": mean_pos,
        "mean_candidates": float(np.mean(n_candidates)) if n_candidates else 0.0,
        "paired": {"weighted_better": wins, "jaccard_better": losses, "ties": ties},
        "metrics": result,
    }
    if score_mode == "emark":
        summary["paired_mark"] = {
            "mark_better": mark_wins, "jaccard_better": mark_losses,
            "ties": mark_ties}

    print(f"\n=== leave-one-complex-out retrieval ({query_mode}) ===")
    print(f"scored={wrapped:,}  misses={misses:,}  "
          f"mean candidates={summary['mean_candidates']:.1f}")
    print(f"{'method':>10} {'MRR':>8} {'R@10':>8} {'R@50':>8} {'R@100':>8} {'median':>9}")
    for method in ("jaccard", "weighted", "mark"):
        m = result.get(method)
        if m and m.get("n"):
            print(f"{method:>10} {m['mrr']:8.4f} {m['recall@10']:8.4f} "
                  f"{m['recall@50']:8.4f} {m['recall@100']:8.4f} "
                  f"{m['median_rank']:9.1f}")
    print(f"paired: weighted better={wins:,}  jaccard better={losses:,}  "
          f"ties={ties:,}")
    if score_mode == "emark":
        print(f"paired: {mark} better={mark_wins:,}  "
              f"jaccard better={mark_losses:,}  ties={mark_ties:,}")

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
        with open(output, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"saved: {output}")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="work/frase_rich_index.pkl")
    ap.add_argument("--max-queries", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="work/eval_summary.json")
    ap.add_argument("--protein-groups", default=None,
                    help="pid -> protein-group JSON (scripts/protein_groups.py)")
    ap.add_argument("--positive-mode",
                    choices=("any", "same-protein", "cross-protein"),
                    default="any")
    ap.add_argument("--query-mode",
                    choices=("stored", "refmol", "reffrag"),
                    default="stored")
    ap.add_argument("--cache",
                    default="/Users/mai/Software/fragenv/data/rcsb_cache")
    ap.add_argument("--combine", choices=("min", "geomean"), default="min")
    ap.add_argument("--score-mode", choices=("fp", "emark"), default="fp",
                    help="fp = weighted exact-key overlap of softmax weights; "
                         "emark = per-key interaction-energy profile "
                         "(requires --query-mode reffrag)")
    ap.add_argument("--mark", choices=("mean", "max", "min_leg", "cmean"),
                    default="mean",
                    help="per-key interaction mark used by --score-mode emark")
    ap.add_argument("--query-noise", type=float, default=0.0,
                    help="lognormal sigma on query weights/marks (0 = off)")
    ap.add_argument("--allow-feature-mismatch", action="store_true")
    args = ap.parse_args(argv)

    index = WeightedTripletIndex.load(args.index)
    print(f"loaded index: {len(index):,} fragments", flush=True)
    evaluate(index, max_queries=args.max_queries,
             seed=args.seed, output=args.output,
             protein_groups=args.protein_groups,
             positive_mode=args.positive_mode,
             query_mode=args.query_mode, cache=args.cache,
             combine=args.combine, query_noise=args.query_noise,
             score_mode=args.score_mode, mark=args.mark,
             allow_feature_mismatch=args.allow_feature_mismatch)


if __name__ == "__main__":
    main()
