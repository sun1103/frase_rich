"""Placement prototype: patent analogs -> query fragment pose -> in-time score.

For ~50 experimentally posed query fragments (PDBbind):

1. retrieve placement-compatible analogs from the patent pool by Morgan
   Tanimoto (same attachment count, nearby heavy-atom count);
2. embed the analogs, place them on the query fragment's exit vectors and
   score every pose with ``fp_weights_in_env`` against the same pocket;
3. compare the candidate's per-fingerprint key set with the query fragment's
   scored keys, under a realistic pose choice (fewest clashes) and under an
   oracle pose choice (best key overlap, an upper bound only).

The realistic arm answers "can a patent analog reproduce the environment
fingerprint once placed"; the oracle arm separates retrieval failure from
placement failure.

With ``--index``, the placed analog is also used as a query against the
weighted library index: the rank of the first fragment from another complex of
the same protein is recorded for the original fragment (``ref``), the best
placed analog, and the random control. That arm is the end-to-end test — the
library fragments live in their own pockets, so the shared-key ceiling of the
same-pocket overlap does not apply.

With ``--borrow-index``, the same weighted index is also a *source* of
candidates: the query fragment's environment retrieves fragments that occupied
similar pockets elsewhere (per-key energy marks by default, weighted key
overlap optionally), and those fragments are cut out of their source complexes
and placed into the query site. The query side is scored with the index's own
feature flags so extraction matches the build. ``--borrow-only`` skips patent
retrieval and runs the borrow channel alone.

Usage:
    # batch over PDBbind
    python scripts/patent_place.py --n-queries 50 \
        --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
        --out work/patent_place.json

    # one cached complex, default fragment choice (largest 1- and 2-attach)
    python scripts/patent_place.py --pdb-id 13gs \
        --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
        --out work/patent_place_13gs.json

    # reference in, scored candidates out: no manual fragment pick -- select
    # every eligible fragment, place every retrieved candidate, rank by score
    python scripts/patent_place.py --pdb-id 13gs --all-fragments --place-all ...

    # same, for any target structure instead of a cached PDBbind complex
    python scripts/patent_place.py --structure target.pdb \
        --ligand-smiles '<co-crystal SMILES>' --all-fragments --place-all ...

    # an explicit override, not the normal path: pin one fragment
    python scripts/patent_place.py --pdb-id 13gs --query-smiles 'CC(=O)Nc1ccc(O)cc1' ...

``--pdb-id`` reads ``<cache>/<id>.pdb`` and needs the parent ligand SMILES:
``--ligand-smiles`` if given, else looked up in ``--selected`` (so an ID absent
from that TSV needs the flag). The ligand must be a HETATM record with
experimental coordinates.

``--structure`` accepts any PDB with the reference ligand present, so the
cache is only a convenience source, not a requirement. The ligand must be a
HETATM residue with experimental coordinates matching ``--ligand-smiles``
(heavy-atom count + bond-order template), because the query fragment poses
and exit vectors are cut from it; an apo structure has nothing to cut.

``--all-fragments`` and ``--place-all`` remove the two arbitrary truncations
(largest-fragment-per-class selection, stop after N placeable candidates), so
the score ranking is what selects. The flat ``candidates`` list in the output
is sorted by ``--rank-by``.

For a single replacement at an assigned site, pin the fragment that occupies
that site with ``--query-smiles``; the rest of the parent ligand is then kept
in its experimental pose and, under ``--ligand-context clash`` (the default
for a pinned site), counted as a clash obstacle (except the atoms bonded
across the cut, which the candidate's attachment atom replaces).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.evaluate import (
    _DenseLibrary,
    _RefFragProvider,
    _build_dense_marks,
    _dense_mark_scores,
    _pick_fragment,
)
from frase_rich.features import attachment_atoms
from frase_rich.patent_decomp import _smiles_order
from frase_rich.patent_place import (
    attachment_fingerprint,
    attachment_identity,
    build_fingerprints,
    dummy_fragment_smiles,
    fuse_rankings,
    normalize_weights,
    query_fingerprint,
    query_marks,
    query_profile,
    query_weights,
    random_row,
    read_pool_rows,
    retained_ligand_xyz,
    score_candidate,
    select_queries,
    tanimoto_topk,
)
from frase_rich.weighting import (
    MARKS as MARK_META,
    fp_weights_in_env,
    normalize_marks,
)

# Mirrors the union index build (--env-center pocket, mean pooling, EEM,
# backbone, angles). The index itself is not loaded: only the scoring kwargs
# and the pocket membership matter here.
DEFAULT_FEATURES = {
    "env_radius": 12.0,
    "d_lo": 8.0,
    "d_hi": 12.0,
    "temperature": 1.0,
    "use_energy": True,
    "use_content": True,
    "energy_pooling": "mean",
    "contact_cutoff": 6.0,
    "angles": True,
    "backbone": True,
    "energy_scale": 0.2,
    "content_scale": 0.3,
    "charge_method": "eem",
    "fp_aggregate": "mean",
    "use_pair_bias": True,
    "legacy_hbond": False,
    "env_center": "pocket",
}

DEFAULT_SELECTED = "/Users/mai/Software/fragenv/data/pdbbind_selected.tsv"
DEFAULT_CACHE = "/Users/mai/Software/fragenv/data/pdbbind_cache"
DEFAULT_UNION_CACHE = "/Users/mai/Software/fragenv/data/union_cache"


def _read_selected(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _strip_private(query: dict) -> dict:
    return {k: v for k, v in query.items() if not k.startswith("_")}


def _nonisomeric(smiles: str) -> str:
    """Canonical constitution-only SMILES.

    The fragment SMILES cut from a pose carries no stereo, while the pool row
    it came from may; string equality would call the same molecule an analog.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else smiles


def _index_summary(records: list[dict]) -> dict:
    """MRR / recall over the placed-query arms, per arm."""
    arms: dict[str, list[int]] = {}
    for r in records:
        for name, rank in (r.get("index_eval") or {}).items():
            if rank is not None:
                arms.setdefault(name, []).append(int(rank))
    out = {}
    for name, vals in arms.items():
        out[name] = {
            "n": len(vals),
            "mrr": float(np.mean([1.0 / v if v else 0.0 for v in vals])),
            "r@10": float(np.mean([1.0 if 0 < v <= 10 else 0.0
                                   for v in vals])),
            "r@50": float(np.mean([1.0 if 0 < v <= 50 else 0.0
                                   for v in vals])),
            "no_positive": int(sum(1 for v in vals if v == 0)),
            "median_rank": float(np.median(vals)),
        }
    return out


_SHELL_KEYS = ("shell_recall", "shell_jaccard", "shell_weight_cos",
               "shell_energy_r", "shell_energy_r_mean",
               "shell_persist_recall", "shell_persist_all",
               "shell_persist_var", "shell_persist_energy_r",
               "shell_contact_retention", "shell_delta_energy")


def _mean_field(poses: list[dict], key: str):
    vals = [float(p[key]) for p in poses if p.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _shell_value(pose: dict, key: str) -> float:
    value = pose.get(key)
    return 0.0 if value is None else float(value)


def _best_real(hits):
    return max(hits, key=lambda h: (h["realistic"]["weighted"],
                                    -h["realistic"]["clashes"]))["realistic"]


def _best_shell_hit(hits, gate: float, key: str = "shell_energy_r_mean",
                    score_key: str = "tanimoto"):
    """Candidate chosen by a multi-pose contact-local score.

    Only analogs above the retrieval gate compete: the shell score is
    independent of retrieval, so the gate keeps the replacement plausible
    without discarding the contact signal. ``key`` selects the mean per-pose
    energy correlation or its persistence-weighted counterpart;
    ``score_key`` is the retrieval score the gate applies (2D Tanimoto for
    pool candidates, environment overlap for borrowed ones).
    """
    passing = [h for h in hits
               if h.get(score_key, 0.0) >= gate
               and h["realistic"].get(key) is not None]
    if not passing:
        return None
    return max(passing, key=lambda h: h["realistic"][key])


def _best_shell(hits, gate: float, key: str = "shell_energy_r_mean",
                score_key: str = "tanimoto"):
    hit = _best_shell_hit(hits, gate, key=key, score_key=score_key)
    return None if hit is None else hit["realistic"]


def _shell_summary(scored: list[dict], gate: float = 0.6,
                   hits_key: str = "top_hits",
                   score_key: str = "tanimoto") -> dict:
    groups = {
        "top1_realistic": [r[hits_key][0]["realistic"] for r in scored],
        "best_realistic": [_best_real(r[hits_key]) for r in scored],
        "best_oracle": [max(r[hits_key],
                            key=lambda h: h["oracle"]["weighted"])["oracle"]
                        for r in scored],
        "control": [r["controls"][0]["realistic"] for r in scored
                    if r.get("controls")],
    }
    gated = [(r, _best_shell(r[hits_key], gate, score_key=score_key))
             for r in scored]
    groups["best_shell"] = [pose for _r, pose in gated
                            if pose is not None]
    persist_gated = [(r, _best_shell(r[hits_key], gate,
                                     key="shell_persist_energy_r",
                                     score_key=score_key))
                     for r in scored]
    groups["best_persist"] = [pose for _r, pose in persist_gated
                              if pose is not None]
    out = {}
    for label, poses in groups.items():
        out[label] = {k: _mean_field(poses, k) for k in _SHELL_KEYS}
        out[label]["n"] = len(poses)
    out["best_shell_gate"] = gate
    for key in ("shell_energy_r", "shell_energy_r_mean",
                "shell_persist_energy_r", "shell_persist_recall",
                "shell_contact_retention", "shell_delta_energy"):
        gaps = []
        for r, hit in gated:
            if hit is None or not r.get("controls"):
                continue
            gaps.append(_shell_value(hit, key)
                        - _shell_value(r["controls"][0]["realistic"], key))
        out[f"{key}_gap_best_shell_minus_control"] = (
            float(np.mean(gaps)) if gaps else None)
        gaps = []
        for r, hit in persist_gated:
            if hit is None or not r.get("controls"):
                continue
            gaps.append(_shell_value(hit, key)
                        - _shell_value(r["controls"][0]["realistic"], key))
        out[f"{key}_gap_best_persist_minus_control"] = (
            float(np.mean(gaps)) if gaps else None)
    return out


def _summarize(records: list[dict], shell_gate: float = 0.6) -> dict:
    scored = [r for r in records if r.get("top_hits")]
    if not scored:
        return {"n_queries": len(records), "n_scored": 0,
                "index_arms": _index_summary(records)}

    gaps = []
    for r in scored:
        h = r["top_hits"][0]
        gaps.append(h["oracle"]["weighted"] - h["realistic"]["weighted"])

    best_real = np.array([max(h["realistic"]["weighted"] for h in r["top_hits"])
                          for r in scored])
    best_oracle = np.array([max(h["oracle"]["weighted"] for h in r["top_hits"])
                            for r in scored])
    controls = [r["controls"][0] for r in scored if r.get("controls")]
    control_real = np.array([c["realistic"]["weighted"] for c in controls]) \
        if controls else np.zeros(0)
    control_oracle = np.array([c["oracle"]["weighted"] for c in controls]) \
        if controls else np.zeros(0)
    return {
        "n_queries": len(records),
        "n_scored": len(scored),
        "exact_available_mean": float(np.mean(
            [r.get("n_exact_available", 0) for r in scored])),
        "mean_top1_tanimoto": float(np.mean(
            [r["top_hits"][0]["tanimoto"] for r in scored])),
        "control_realistic_mean": (float(control_real.mean())
                                   if control_real.size else None),
        "control_oracle_mean": (float(control_oracle.mean())
                                if control_oracle.size else None),
        "best_realistic": {
            "mean": float(best_real.mean()),
            "median": float(np.median(best_real)),
            "frac_ge_0.5": float(np.mean(best_real >= 0.5)),
            "frac_ge_0.3": float(np.mean(best_real >= 0.3)),
        },
        "best_oracle": {
            "mean": float(best_oracle.mean()),
            "median": float(np.median(best_oracle)),
            "frac_ge_0.5": float(np.mean(best_oracle >= 0.5)),
            "frac_ge_0.3": float(np.mean(best_oracle >= 0.3)),
        },
        "top1_realistic_weighted_mean": float(np.mean(
            [r["top_hits"][0]["realistic"]["weighted"] for r in scored])),
        "top1_oracle_weighted_mean": float(np.mean(
            [r["top_hits"][0]["oracle"]["weighted"] for r in scored])),
        "pose_gap_mean": float(np.mean(gaps)),
        "realistic_clashes_mean": float(np.mean(
            [r["top_hits"][0]["realistic"]["clashes"] for r in scored])),
        "realistic_min_dist_mean": float(np.mean(
            [r["top_hits"][0]["realistic"]["min_dist"] for r in scored])),
        "n_query_keys_mean": float(np.mean(
            [r["n_query_keys"] for r in scored])),
        "n_shell_mean": float(np.mean(
            [r.get("n_shell", 0) for r in scored])),
        "shell": _shell_summary(scored, gate=shell_gate),
        "index_arms": _index_summary(records),
    }


def _prepare_borrow(index, *, rank: str, mark: str, marks=None) -> dict:
    """Row-aligned arrays and the retrieval score matrix for one index.

    The index is queried in its own feature space, so the query fragment must
    be scored with ``fp_weights_in_env`` (the build-parity path). Rows follow
    ``index._frags`` order, the same order ``_DenseLibrary`` and
    ``_build_dense_marks`` use. ``marks`` reuses a matrix already built for
    the ``--index`` arms when both roles load the same file.
    """
    fids = list(index._frags)
    pdb = np.array([str(index._frags[f].get("pdb_id", "")).lower()
                    for f in fids], dtype=object)
    n_attach = np.fromiter(
        (int(index._frags[f].get("n_attach", -1)) for f in fids),
        dtype=np.int64, count=len(fids))
    n_heavy = np.fromiter(
        (int(index._frags[f].get("n_heavy", 0)) for f in fids),
        dtype=np.int64, count=len(fids))
    out = {"fids": fids, "pdb": pdb, "n_attach": n_attach,
           "n_heavy": n_heavy, "rank": rank}
    if rank == "weight":
        out["lib"] = _DenseLibrary(index)
        print(f"borrow dense weights: {len(fids):,} fragments x "
              f"{len(out['lib'].key_ids):,} keys", flush=True)
    else:
        dense = marks if marks is not None else _build_dense_marks(
            index, MARK_META[mark])
        if dense is None:
            raise SystemExit(f"borrow index stores no '{mark}' marks")
        out["marks"] = dense
        print(f"borrow dense '{mark}' marks: {len(fids):,} fragments x "
              f"{dense[0].shape[1]:,} keys"
              f"{' (shared)' if marks is not None else ''}", flush=True)
    return out


def _borrow_rows(borrow: dict, q_weights, q_marks, *, n_attach: int,
                 n_heavy: int, heavy_band: int, exclude_pdb: str,
                 include_self: bool, combine: str):
    """Index rows that pass the cheap gates, best retrieval score first."""
    mask = borrow["n_attach"] == int(n_attach)
    mask &= np.abs(borrow["n_heavy"] - int(n_heavy)) <= int(heavy_band)
    if not include_self and exclude_pdb:
        mask &= borrow["pdb"] != exclude_pdb.lower()
    if borrow["rank"] == "weight":
        lib = borrow["lib"]
        n_query, n_match, w_match = lib.score_fp_weights(
            q_weights, combine=combine)
        _jaccard, score = lib.scores(n_query, n_match, w_match)
        mask &= n_match >= 1
        second = n_match
    else:
        score = _dense_mark_scores(q_marks, borrow["marks"])
        mask &= score > 0
        second = borrow["marks"][1]
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return rows, score
    order = np.lexsort((rows, -second[rows], -score[rows]))
    return rows[order], score


def _canonical_attach(mol, attach_idx):
    """Attachment atoms as positions in the canonical SMILES.

    ``score_candidate`` re-parses the SMILES, so the cut atoms must be given
    in that atom order. ``MolToSmiles`` writes the ``_smilesAtomOutputOrder``
    property ``_smiles_order`` reads, hence the call order.
    """
    smiles = Chem.MolToSmiles(mol)
    order = _smiles_order(mol)
    old_to_canon = {int(old): pos for pos, old in enumerate(order)}
    out = sorted(old_to_canon.get(int(i), -1) for i in attach_idx)
    if len(out) != len(attach_idx) or -1 in out:
        return None, None
    return smiles, out


def _summarize_borrow(records: list[dict], shell_gate: float = 0.6) -> dict:
    scored = [r for r in records if r.get("borrow_hits")]
    if not scored:
        return {"n_queries": len(records), "n_scored": 0}
    top = [r["borrow_hits"][0] for r in scored]
    best_real = np.array([max(h["realistic"]["weighted"]
                              for h in r["borrow_hits"]) for r in scored])
    best_oracle = np.array([max(h["oracle"]["weighted"]
                                for h in r["borrow_hits"]) for r in scored])
    return {
        "n_queries": len(records),
        "n_scored": len(scored),
        "n_candidates_mean": float(np.mean(
            [len(r["borrow_hits"]) for r in scored])),
        "top1_env_score_mean": float(np.mean(
            [float(h.get("env_score") or 0.0) for h in top])),
        "best_realistic": {
            "mean": float(best_real.mean()),
            "median": float(np.median(best_real)),
            "frac_ge_0.5": float(np.mean(best_real >= 0.5)),
            "frac_ge_0.3": float(np.mean(best_real >= 0.3)),
        },
        "best_oracle": {
            "mean": float(best_oracle.mean()),
            "median": float(np.median(best_oracle)),
            "frac_ge_0.5": float(np.mean(best_oracle >= 0.5)),
            "frac_ge_0.3": float(np.mean(best_oracle >= 0.3)),
        },
        "shell": _shell_summary(scored, gate=shell_gate,
                                hits_key="borrow_hits",
                                score_key="env_score"),
    }


_CHANNEL_GATE = {"pool": ("top_hits", "tanimoto"),
                 "borrow": ("borrow_hits", "env_score")}


def _channel_best(record: dict, source: str, shell_gate: float = 0.6):
    """Best gate-passing candidate from one channel by contact persistence."""
    hits_key, gate_key = _CHANNEL_GATE[source]
    best = None
    for hit in record.get(hits_key) or []:
        if (hit.get(gate_key) or 0.0) < shell_gate:
            continue
        real = hit.get("realistic") or {}
        value = real.get("shell_persist_energy_r")
        if value is None:
            continue
        rank = (float(value), real.get("shell_energy_r_mean") or 0.0)
        if best is None or rank > best[0]:
            best = (rank, hit)
    return None if best is None else best[1]


def _joint_pick(record: dict, shell_gate: float = 0.6):
    """Recommended replacement across both channels.

    Each channel keeps its own retrieval gate (2D Tanimoto for patent-pool
    analogs, environment overlap for borrowed fragments); the channels are
    compared only on the placement contact score.
    """
    pick = None
    for source in ("pool", "borrow"):
        hit = _channel_best(record, source, shell_gate)
        if hit is None:
            continue
        real = hit["realistic"]
        rank = (float(real["shell_persist_energy_r"]),
                real.get("shell_energy_r_mean") or 0.0)
        if pick is None or rank > pick[0]:
            pick = (rank, source, hit)
    if pick is None:
        return None
    _rank, source, hit = pick
    real = hit["realistic"]
    gate_key = _CHANNEL_GATE[source][1]
    return {
        "source": source,
        "cand_smiles": hit["cand_smiles"],
        "retrieval_rank": int(hit["rank"]),
        "retrieval_score": float(hit.get(gate_key) or 0.0),
        "weighted": real.get("weighted"),
        "clashes": real.get("clashes"),
        "shell_energy_r_mean": real.get("shell_energy_r_mean"),
        "shell_persist_energy_r": real.get("shell_persist_energy_r"),
        "pool_row": hit.get("pool_row"),
        "borrow_fid": hit.get("borrow_fid"),
        "borrow_pdb": hit.get("borrow_pdb"),
    }


def _summarize_combined(records: list[dict], shell_gate: float = 0.6) -> dict:
    """Cross-channel recommendation: means, paired channel delta, source mix.

    ``union`` takes the better channel per query and is optimistic by
    construction (best of two); the unbiased complement is the paired
    ``borrow - pool`` delta on queries where both channels pass their gate.
    """
    pool: list[float] = []
    borrow: list[float] = []
    union: list[float] = []
    deltas: list[float] = []
    wins = {"pool": 0, "borrow": 0, "tie": 0}
    sources = {"pool": 0, "borrow": 0, "none": 0}
    for record in records:
        pick = record.get("selected")
        sources["none" if pick is None else pick["source"]] += 1
        values = {}
        for source in ("pool", "borrow"):
            hit = _channel_best(record, source, shell_gate)
            values[source] = (None if hit is None else float(
                hit["realistic"]["shell_persist_energy_r"]))
        if values["pool"] is not None:
            pool.append(values["pool"])
        if values["borrow"] is not None:
            borrow.append(values["borrow"])
        if values["pool"] is not None and values["borrow"] is not None:
            delta = values["borrow"] - values["pool"]
            deltas.append(delta)
            if delta > 1e-12:
                wins["borrow"] += 1
            elif delta < -1e-12:
                wins["pool"] += 1
            else:
                wins["tie"] += 1
        available = [v for v in values.values() if v is not None]
        if available:
            union.append(max(available))

    def _mean(vals):
        return float(np.mean(vals)) if vals else None

    return {
        "gate": float(shell_gate),
        "rule": ("max shell_persist_energy_r among gate-passing candidates; "
                 "pool gate = 2D Tanimoto, borrow gate = env_score"),
        "n_queries": len(records),
        "n_solved": {"pool": len(pool), "borrow": len(borrow),
                     "union": len(union)},
        "mean_persist": {"pool": _mean(pool), "borrow": _mean(borrow),
                         "union": _mean(union)},
        "paired": {"n": len(deltas), "wins": wins,
                   "mean_delta": _mean(deltas),
                   "median_delta": (float(np.median(deltas))
                                    if deltas else None)},
        "selected_source": sources,
        "union_is_best_of_two_channels": True,
    }


def write_candidates_csv(candidates: list[dict], path: str) -> None:
    """Write the flat per-candidate table (the JSON ``candidates`` list).

    Columns are the union of record keys in first-seen order, so runs with and
    without the attachment channel produce compatible-but-different headers.
    """
    columns: list[str] = []
    seen: set[str] = set()
    for row in candidates:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(candidates)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="work/patent_pool.parquet")
    ap.add_argument("--fp-prefix", default="work/patent_pool")
    ap.add_argument("--selected", default=DEFAULT_SELECTED)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default="work/patent_place.json")
    ap.add_argument("--out-csv", default=None,
                    help="also write the flat candidate table as CSV (default: "
                         "the --out path with a .csv extension; pass '' to "
                         "skip)")
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--max-per-complex", type=int, default=2)
    ap.add_argument("--min-heavy", type=int, default=6)
    ap.add_argument("--max-heavy", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=40,
                    help="pool rows retrieved per query before placement")
    ap.add_argument("--n-candidates", type=int, default=6,
                    help="retrieved candidates actually placed and scored")
    ap.add_argument("--n-conformers", type=int, default=3)
    ap.add_argument("--rolls", type=int, default=24)
    ap.add_argument("--max-poses", type=int, default=48)
    ap.add_argument("--n-profile-poses", type=int, default=4,
                    help="poses per candidate whose contact profile is scored "
                         "(averaged into shell_energy_r_mean)")
    ap.add_argument("--shell-gate", type=float, default=0.6,
                    help="minimum 2D Tanimoto for the contact-local ranking "
                         "arm")
    ap.add_argument("--heavy-band", type=int, default=8)
    ap.add_argument("--attach-fp-prefix", default="",
                    help="second retrieval channel over attachment-marked "
                         "Morgan fingerprints. Needs its own index prefix "
                         "(built on first use, ~4 GB); the union is fused by "
                         "reciprocal rank, not by comparing Tanimotos")
    ap.add_argument("--attach-gate", choices=("auto", "on", "off"),
                    default="auto",
                    help="drop retrieved candidates whose attachment-atom "
                         "element/hybridization multiset differs from the "
                         "query site. 'auto' = on when --query-smiles pins a "
                         "single site, off for the all-fragment benchmark "
                         "protocol (which must stay comparable)")
    ap.add_argument("--keep-exact", action="store_true",
                    help="allow candidates whose SMILES equals the query's "
                         "(excluded by default: an exact match is not an analog)")
    ap.add_argument("--n-controls", type=int, default=1,
                    help="random placement-compatible candidates scored as a "
                         "no-retrieval control")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-fp-rows", type=int, default=0,
                    help="debug: only fingerprint the first N pool rows")
    ap.add_argument("--self-check", action="store_true",
                    help="also place the query fragment as its own candidate")
    ap.add_argument("--max-complexes", type=int, default=0,
                    help="debug: scan only the first N selected complexes")
    ap.add_argument("--pdb-id", default="",
                    help="single-complex mode: read <cache>/<pdb_id>.pdb and "
                         "skip the --selected batch (it is still read to look "
                         "up the ligand SMILES unless --ligand-smiles is set). "
                         "With --structure it is only the label for the run")
    ap.add_argument("--structure", default="",
                    help="target structure file (PDB) used instead of "
                         "<cache>/<pdb_id>.pdb; needs --ligand-smiles. The "
                         "co-crystal ligand must be a HETATM record with "
                         "experimental coordinates, since the query fragment "
                         "poses are cut from it")
    ap.add_argument("--ligand-smiles", default="",
                    help="parent ligand SMILES for --pdb-id/--structure; "
                         "overrides the --selected lookup (needed for IDs not "
                         "in that TSV)")
    ap.add_argument("--query-smiles", default="",
                    help="pin the exact fragment (matched on constitution, "
                         "stereo ignored) instead of the largest fragment per "
                         "attachment class; requires --pdb-id and bypasses the "
                         "attachment/heavy filters")
    ap.add_argument("--all-fragments", action="store_true",
                    help="select every eligible query fragment (1-2 "
                         "attachments, --min/max-heavy) instead of the largest "
                         "per attachment class; ignores --max-per-complex")
    ap.add_argument("--place-all", action="store_true",
                    help="place every deduped retrieved candidate (up to "
                         "--top-k) instead of stopping at --n-candidates")
    ap.add_argument("--ligand-context", choices=("auto", "ignore", "clash"),
                    default="auto",
                    help="treat the retained parent-ligand atoms as clash "
                         "obstacles while one fragment is replaced. 'auto' = "
                         "clash when --query-smiles pins a single site, "
                         "ignore otherwise (the all-fragment benchmark "
                         "protocol); 'ignore' preserves the old numbers")
    ap.add_argument("--rank-by", choices=("weighted", "persist", "energy"),
                    default="energy",
                    help="score used to order the flat candidate list: "
                         "contact energy (shell_energy_r_mean, default), "
                         "contact persistence (shell_persist_energy_r), or "
                         "realistic weighted overlap ('weighted' is the "
                         "literal FRASE arm; on same-pocket scoring it is "
                         "saturated by generic keys and nearly ties)")
    ap.add_argument("--top-show", type=int, default=-1,
                    help="ranked candidate rows to print (default: 10 with "
                         "--pdb-id, 0 otherwise)")
    ap.add_argument("--index", default="",
                    help="weighted index for the end-to-end arm: the placed "
                         "analog's per-key weights are used as the query")
    ap.add_argument("--borrow-index", default="",
                    help="weighted index used as a fragment-borrow source: "
                         "the query fragment's environment retrieves indexed "
                         "fragments from similar pockets, which are then "
                         "placed into the query site. Feature flags are "
                         "taken from the index so extraction matches the "
                         "build")
    ap.add_argument("--borrow-cache", default=DEFAULT_UNION_CACHE,
                    help="structure cache for borrow sources (the union "
                         "cache holds both PDBbind and RCSB complexes)")
    ap.add_argument("--borrow-rank", choices=("mark", "weight"),
                    default="mark",
                    help="borrow retrieval score: per-key energy marks "
                         "('mark') or weighted exact-key overlap ('weight')")
    ap.add_argument("--borrow-combine", choices=("min", "geomean"),
                    default="min",
                    help="pairing for weighted borrow overlap")
    ap.add_argument("--borrow-n-candidates", type=int, default=0,
                    help="borrowed candidates placed per query (default: "
                         "--n-candidates); the pool arm keeps its own depth")
    ap.add_argument("--borrow-scan", type=int, default=400,
                    help="ranked index rows examined per query; exact and "
                         "attachment gates may reject many")
    ap.add_argument("--borrow-only", action="store_true",
                    help="skip patent-pool retrieval; borrow from the index "
                         "only")
    ap.add_argument("--borrow-include-self", action="store_true",
                    help="allow fragments from the query's own PDB "
                         "(excluded by default)")
    ap.add_argument("--protein-groups",
                    default="work/pdbbind_protein_groups.json",
                    help="pdb_id -> protein group map for same-protein "
                         "positives")
    ap.add_argument("--mark", choices=tuple(MARK_META), default="mean",
                    help="per-key interaction mark for the emark index arms")
    ap.add_argument("--positive-mode", choices=("smiles", "protein"),
                    default="smiles",
                    help="index-arm positives: 'smiles' = the query "
                         "fragment's chemotype in other same-protein "
                         "complexes; 'protein' = any fragment of those "
                         "complexes (pocket-only, not discriminative)")
    args = ap.parse_args(argv)

    t0 = time.time()
    paths = None
    n_rows = 0
    attach_paths = None
    if not args.borrow_only:
        paths = build_fingerprints(
            args.pool, args.fp_prefix, workers=args.workers,
            limit_rows=args.limit_fp_rows)
        n_rows = int(np.load(paths["fp"], mmap_mode="r").shape[0])
        print(f"pool fingerprints: {n_rows:,} rows", flush=True)

        if args.attach_fp_prefix:
            attach_paths = build_fingerprints(
                args.pool, args.attach_fp_prefix, workers=args.workers,
                limit_rows=args.limit_fp_rows, attach_aware=True)
            n_a = int(np.load(attach_paths["fp"], mmap_mode="r").shape[0])
            if n_a != n_rows:
                ap.error(f"--attach-fp-prefix has {n_a:,} rows but the "
                         f"global index has {n_rows:,}; rebuild both from "
                         f"the same pool")
            print(f"attachment fingerprints: {n_a:,} rows", flush=True)

    provider_paths: dict[str, str] = {}
    if args.structure:
        if args.max_complexes:
            ap.error("--max-complexes is batch mode only")
        if not os.path.exists(args.structure):
            ap.error(f"--structure: no such file: {args.structure}")
        if not args.ligand_smiles:
            ap.error("--structure needs --ligand-smiles (the co-crystal "
                     "ligand must carry experimental coordinates as HETATM)")
        pid = (args.pdb_id or
               os.path.splitext(os.path.basename(args.structure))[0]).lower()
        provider_paths[pid] = args.structure
        rows = [{"pdb_id": pid, "ligand_smiles": args.ligand_smiles}]
    elif args.pdb_id:
        if args.max_complexes:
            ap.error("--max-complexes is batch mode only")
        pid = args.pdb_id.lower()
        ligand = args.ligand_smiles
        if not ligand and os.path.exists(args.selected):
            ligand = next((r["ligand_smiles"]
                           for r in _read_selected(args.selected)
                           if r["pdb_id"].lower() == pid), "")
        if not ligand:
            ap.error(f"--pdb-id {pid}: no ligand SMILES; pass "
                     f"--ligand-smiles or add the ID to --selected")
        rows = [{"pdb_id": pid, "ligand_smiles": ligand}]
    else:
        if args.ligand_smiles or args.query_smiles:
            ap.error("--ligand-smiles/--query-smiles require --pdb-id "
                     "or --structure")
        rows = _read_selected(args.selected)
        if args.max_complexes:
            rows = rows[:args.max_complexes]
    borrow_index = None
    features = DEFAULT_FEATURES
    if args.borrow_index:
        from frase_rich.index import WeightedTripletIndex

        print(f"loading borrow index {args.borrow_index} ...", flush=True)
        borrow_index = WeightedTripletIndex.load(args.borrow_index)
        if borrow_index.features:
            features = dict(borrow_index.features)
        n_pdbs = len({str(i.get("pdb_id", "")).lower()
                      for i in borrow_index._frags.values()})
        print(f"borrow index: {len(borrow_index):,} fragments over "
              f"{n_pdbs:,} complexes, "
              f"{len(borrow_index._postings):,} keys", flush=True)
    provider = _RefFragProvider(args.cache, features, paths=provider_paths,
                                extra_caches=[args.borrow_cache])
    try:
        queries = select_queries(
            provider, rows, max_queries=args.n_queries,
            max_per_complex=args.max_per_complex,
            min_heavy=args.min_heavy, max_heavy=args.max_heavy,
            query_smiles=args.query_smiles or None,
            all_fragments=args.all_fragments)
    except ValueError as exc:
        raise SystemExit(f"query selection failed: {exc}")
    if not queries:
        raise SystemExit("no query fragments found")
    print(f"query fragments: {len(queries)} across "
          f"{len({q['pdb_id'] for q in queries})} complexes", flush=True)
    if args.pdb_id or args.structure:
        for i, q in enumerate(queries, start=1):
            print(f"  q{i} {q['pdb_id']} {q['smiles']} "
                  f"(n_attach={q['n_attach']}, n_heavy={q['n_heavy']})",
                  flush=True)

    # Pass 1: retrieve. Collect all wanted row ids so metadata is read once.
    rng = np.random.default_rng(args.seed)
    wanted: set[int] = set()
    if args.borrow_only:
        for q in queries:
            q["_hits"] = []
            q["_controls"] = []
    else:
        for q in queries:
            bits = query_fingerprint(q["smiles"])
            if bits is None:
                q["_hits"] = []
                q["_controls"] = []
                continue
            idx, tan = tanimoto_topk(
                paths, n_rows, bits, n_attach=q["n_attach"],
                n_heavy=q["n_heavy"], k=args.top_k,
                heavy_band=args.heavy_band)
            hits = [(int(i), float(t), None) for i, t in zip(idx, tan)]
            if attach_paths is not None:
                abits = attachment_fingerprint(q["smiles"], q["_att_canon"])
                if abits is None:
                    print(f"  q {q['smiles']}: cannot mark the attachment "
                          f"site; global channel only", flush=True)
                aidx, atan = (tanimoto_topk(
                    attach_paths, n_rows, abits, n_attach=q["n_attach"],
                    n_heavy=q["n_heavy"], k=args.top_k,
                    heavy_band=args.heavy_band)
                    if abits is not None
                    else (np.empty(0, dtype=np.int64),
                          np.empty(0, dtype=np.float64)))
                gmap = {int(i): float(t) for i, t in zip(idx, tan)}
                amap = {int(i): float(t) for i, t in zip(aidx, atan)}
                # Channels score different fingerprints, so their Tanimotos
                # are not comparable; fuse the orders and keep per-channel
                # values.
                fused = fuse_rankings([idx, aidx])
                hits = [(rid, gmap.get(rid), amap.get(rid))
                        for rid, _ in fused][:args.top_k]
            q["_hits"] = hits
            wanted.update(rid for rid, _g, _a in hits)
            controls = []
            for _ in range(args.n_controls):
                rid = random_row(paths, n_rows, n_attach=q["n_attach"],
                                 n_heavy=q["n_heavy"],
                                 heavy_band=args.heavy_band, rng=rng)
                if rid is not None:
                    controls.append(int(rid))
                    wanted.add(int(rid))
            q["_controls"] = controls
    meta = read_pool_rows(args.pool, wanted) if wanted else {}
    if not args.borrow_only:
        print(f"retrieved {len(wanted):,} unique pool rows in "
              f"{time.time() - t0:.0f}s", flush=True)

    kwargs = provider._kwargs()
    index = None
    dense_marks = None
    row_fids: list[str] = []
    row_pdbs: list[str] = []
    group_of: dict[str, str] = {}
    if args.index:
        from frase_rich.index import WeightedTripletIndex

        if (borrow_index is not None
                and os.path.abspath(args.index)
                == os.path.abspath(args.borrow_index)):
            index = borrow_index
            print(f"index loaded: {len(index):,} fragments (shared with the "
                  f"borrow source)", flush=True)
        else:
            print(f"loading index {args.index} ...", flush=True)
            index = WeightedTripletIndex.load(args.index)
        with open(args.protein_groups) as handle:
            group_of = {str(k).lower(): str(v)
                        for k, v in json.load(handle).items()}
        checks = [
            ("env_center", "pocket"),
            ("energy_pooling", kwargs["pooling"]),
            ("charge_method", kwargs["charge_method"]),
            ("angles", kwargs["use_angles"]),
            ("backbone", kwargs["backbone"]),
            ("fp_aggregate", kwargs["fp_aggregate"]),
        ]
        mismatch = [(k, index.features.get(k), want)
                    for k, want in checks
                    if k in index.features and index.features[k] != want]
        if mismatch:
            print(f"WARNING: index/query feature mismatch: {mismatch}",
                  flush=True)
        else:
            print(f"index loaded: {len(index):,} fragments; features match",
                  flush=True)
        dense_marks = _build_dense_marks(index, MARK_META[args.mark])
        if dense_marks is None:
            print(f"WARNING: index stores no '{args.mark}' marks; "
                  f"emark arms disabled", flush=True)
        else:
            _lib, n_pos, _keys, row_of = dense_marks
            row_fids = [""] * len(row_of)
            for fid, i in row_of.items():
                row_fids[i] = fid
            row_pdbs = [str(index._frags[fid].get("pdb_id", "")).lower()
                        for fid in row_fids]
            print(f"dense '{args.mark}' marks: "
                  f"{int(np.count_nonzero(n_pos)):,} fragments, "
                  f"{_lib.shape[1]} keys", flush=True)
    borrow = None
    if borrow_index is not None:
        shared_marks = (dense_marks
                        if index is borrow_index
                        and args.borrow_rank == "mark" else None)
        borrow = _prepare_borrow(borrow_index, rank=args.borrow_rank,
                                 mark=args.mark, marks=shared_marks)
    borrow_depth = args.borrow_n_candidates or args.n_candidates
    ligand_context = args.ligand_context
    if ligand_context == "auto":
        ligand_context = "clash" if args.query_smiles else "ignore"
    if ligand_context == "clash":
        print("ligand context: retained parent-ligand atoms count as clashes",
              flush=True)
    attach_gate = args.attach_gate
    if attach_gate == "auto":
        attach_gate = "on" if args.query_smiles else "off"
    gate_on = attach_gate == "on"
    if gate_on:
        print("attachment gate: candidate cut atom (element, hybridization) "
              "must match the query site", flush=True)
    records: list[dict] = []
    for qi, q in enumerate(queries, start=1):
        row = _strip_private(q)
        ctx = provider._context(q["pdb_id"], q["parent_smiles"])
        if ctx is None:
            row["error"] = "context"
            records.append(row)
            continue
        lig, env, table, by_smiles = ctx
        frag = by_smiles.get(q["smiles"])
        if frag is None:
            row["error"] = "fragment"
            records.append(row)
            continue
        obstacles = None
        if ligand_context == "clash":
            obstacles = retained_ligand_xyz(lig, frag.mol, q["_att_xyz"])
        row["ligand_context"] = ligand_context
        row["attach_gate"] = attach_gate
        row["n_obstacle_atoms"] = 0 if obstacles is None else int(len(obstacles))
        q_w = query_weights(frag.mol, env, table, q["_att_idx"], **kwargs)
        q_prof = query_profile(frag.mol, env, q["_att_idx"], **kwargs)
        q_m = (query_marks(frag.mol, env, table, q["_att_idx"],
                           mark=args.mark, **kwargs)
               if dense_marks is not None else {})
        b_weights = None
        b_marks = {}
        if borrow is not None:
            # Score the query fragment exactly as the build scored the
            # library: same environment, same weighting code, same
            # mean-1 normalization.
            raw_b, _env_b, _method_b, energies_b = fp_weights_in_env(
                frag.mol, env, table=table,
                exclude_donor_atoms=set(q["_att_idx"]),
                return_energies=True, **kwargs)
            b_weights = normalize_weights(raw_b)
            # ``energies_b`` is keyed by mark name ("mean", "max", ...);
            # ``MARK_META`` holds the index metadata key those marks are
            # stored under.
            b_marks = normalize_marks(energies_b.get(args.mark) or {})
        row["n_query_keys"] = len(q_w)
        row["n_shell"] = int(q_prof["mask"].sum())

        hit_list = q["_hits"]
        q_noniso = _nonisomeric(q["smiles"])
        exact_cache: dict[str, bool] = {}

        def _is_exact(smiles: str) -> bool:
            if smiles not in exact_cache:
                exact_cache[smiles] = _nonisomeric(smiles) == q_noniso
            return exact_cache[smiles]

        n_exact = sum(1 for rid, _g, _a in hit_list
                      if _is_exact((meta.get(int(rid)) or {}).get(
                          "frag_smiles", "")))
        row["n_exact_available"] = int(n_exact)

        q_ident = q.get("_att_identity")
        ident_cache: dict[str, tuple | None] = {}

        def _cand_identity(smiles: str, attach_atoms) -> tuple | None:
            if smiles not in ident_cache:
                mol = Chem.MolFromSmiles(smiles)
                ident_cache[smiles] = (attachment_identity(mol, attach_atoms)
                                       if mol is not None else None)
            return ident_cache[smiles]

        def _score(info, pool_row, rank, tan_global, tan_attach, *,
                   source="pool", env_score=None):
            res = score_candidate(
                info["frag_smiles"], info["attach_atoms"],
                q["_att_xyz"], q["_exit_xyz"], env, table, q_w,
                n_confs=args.n_conformers, rolls=args.rolls,
                max_poses=args.max_poses,
                n_profile_poses=args.n_profile_poses,
                return_weights=bool(args.index),
                q_profile=q_prof, obstacles=obstacles, **kwargs)
            if res is not None:
                best_t = max((t for t in (tan_global, tan_attach)
                              if t is not None), default=0.0)
                ident = _cand_identity(info["frag_smiles"],
                                       info["attach_atoms"])
                res.update({
                    "source": source,
                    "pool_row": pool_row,
                    "env_score": env_score,
                    "rank": rank,
                    "tanimoto": float(best_t),
                    "tanimoto_global": tan_global,
                    "tanimoto_attach": tan_attach,
                    "cand_attach_identity": ident,
                    "cand_smiles": info["frag_smiles"],
                    "cand_dummy_smiles": dummy_fragment_smiles(
                        info["frag_smiles"], info["attach_atoms"]),
                    "n_attach": int(info["n_attach"]),
                    "n_heavy": int(info["n_heavy"]),
                    "exact": _is_exact(info["frag_smiles"]),
                })
            return res

        # One candidate per (constitution, attachment position): pool rows
        # differing only in stereo write the same replacement, since placement
        # generates its own conformers anyway. A different attachment atom on
        # the same skeleton is a genuinely different replacement, so it stays.
        seen: set[tuple[str, tuple[int, ...]]] = set()
        hits = []
        n_gate_dropped = 0
        for rank, (rid, tan_global, tan_attach) in enumerate(hit_list,
                                                            start=1):
            info = meta.get(int(rid))
            if info is None:
                continue
            smiles = info["frag_smiles"]
            key = (_nonisomeric(smiles),
                   tuple(sorted(int(a) for a in info["attach_atoms"] or ())))
            if key in seen:
                continue
            if not args.keep_exact and _is_exact(smiles):
                continue
            if gate_on and q_ident is not None:
                if _cand_identity(smiles, info["attach_atoms"]) != q_ident:
                    n_gate_dropped += 1
                    continue
            seen.add(key)
            res = _score(info, pool_row=int(rid), rank=rank,
                         tan_global=tan_global, tan_attach=tan_attach)
            if res is None:
                continue
            hits.append(res)
            if not args.place_all and len(hits) >= args.n_candidates:
                break
        row["top_hits"] = hits
        row["n_attach_gate_dropped"] = int(n_gate_dropped)

        borrow_hits: list[dict] = []
        if borrow is not None:
            rows, b_score = _borrow_rows(
                borrow, b_weights, b_marks,
                n_attach=q["n_attach"], n_heavy=q["n_heavy"],
                heavy_band=args.heavy_band,
                exclude_pdb="" if args.borrow_include_self else q["pdb_id"],
                include_self=args.borrow_include_self,
                combine=args.borrow_combine)
            seen_b: set[tuple[str, tuple[int, ...]]] = set()
            n_dropped_b = 0
            n_miss_b = 0
            n_scanned = 0
            for irow in rows:
                if (not args.place_all
                        and len(borrow_hits) >= borrow_depth):
                    break
                n_scanned += 1
                if n_scanned > args.borrow_scan:
                    break
                fid = borrow["fids"][int(irow)]
                info_b = borrow_index._frags.get(fid)
                if info_b is None:
                    continue
                smiles_b = str(info_b.get("frag_smiles", ""))
                if not smiles_b:
                    continue
                if not args.keep_exact and _is_exact(smiles_b):
                    continue
                ctx_b = provider.ligand_context(
                    str(info_b.get("pdb_id", "")).lower(),
                    info_b.get("parent_smiles"))
                if ctx_b is None:
                    n_miss_b += 1
                    continue
                lig_b, by_b = ctx_b
                cands = by_b.all(smiles_b)
                if not cands:
                    n_miss_b += 1
                    continue
                bfrag = _pick_fragment(cands, info_b.get("_frag_coords"))
                att_b = attachment_atoms(bfrag.mol, lig_b)
                if not att_b or len(att_b) != int(
                        info_b.get("n_attach", -1)):
                    n_miss_b += 1
                    continue
                smiles_c, attach_canon = _canonical_attach(bfrag.mol, att_b)
                if smiles_c is None:
                    n_miss_b += 1
                    continue
                key = (_nonisomeric(smiles_c), tuple(attach_canon))
                if key in seen_b:
                    continue
                if gate_on and q_ident is not None:
                    if _cand_identity(smiles_c, attach_canon) != q_ident:
                        n_dropped_b += 1
                        continue
                seen_b.add(key)
                res = _score(
                    {"frag_smiles": smiles_c,
                     "attach_atoms": attach_canon,
                     "n_attach": len(attach_canon),
                     "n_heavy": int(info_b.get("n_heavy", 0))},
                    pool_row=None, rank=n_scanned, tan_global=None,
                    tan_attach=None, source="borrow",
                    env_score=float(b_score[int(irow)]))
                if res is None:
                    continue
                res["borrow_fid"] = fid
                res["borrow_pdb"] = str(
                    info_b.get("pdb_id", "")).lower()
                borrow_hits.append(res)
            row["n_borrow_scanned"] = int(n_scanned)
            row["n_borrow_gate_dropped"] = int(n_dropped_b)
            row["n_borrow_context_miss"] = int(n_miss_b)
        row["borrow_hits"] = borrow_hits
        row["selected"] = _joint_pick(row, args.shell_gate)

        controls = []
        for rid in q["_controls"]:
            info = meta.get(int(rid))
            if info is None:
                continue
            res = _score(info, pool_row=int(rid), rank=0,
                         tan_global=0.0, tan_attach=None)
            if res is not None:
                controls.append(res)
        row["controls"] = controls

        if args.self_check:
            res = score_candidate(
                q["smiles"], q["_att_idx"], q["_att_xyz"], q["_exit_xyz"],
                env, table, q_w,
                n_confs=args.n_conformers, rolls=args.rolls,
                max_poses=args.max_poses,
                n_profile_poses=args.n_profile_poses,
                q_profile=q_prof, obstacles=obstacles, **kwargs)
            if res is not None:
                row["self_check"] = res

        if index is not None:
            q_pdb = q["pdb_id"].lower()
            gid = group_of.get(q_pdb)
            pos_fids = None
            if args.positive_mode == "smiles" and gid is not None:
                # Same fragment in another complex of the same protein. Rows
                # from the query's own PDB are excluded from the search below,
                # so they cannot be positives either: counting them here would
                # turn "no eligible positive" into a rank-0 miss.
                pos_fids = {
                    fid for fid, info in index._frags.items()
                    if str(info.get("frag_smiles", "")) == q["smiles"]
                    and str(info.get("pdb_id", "")).lower() != q_pdb
                    and group_of.get(
                        str(info.get("pdb_id", "")).lower()) == gid}
            row["n_pos_fids"] = len(pos_fids) if pos_fids is not None else None

            def _arm_rank(weights):
                """Rank of the first positive; 0 if none is retrieved, None
                when the query has no positive in the index."""
                if not weights or gid is None:
                    return None
                if args.positive_mode == "smiles" and not pos_fids:
                    return None
                ranked = index.query(
                    None, query_fp_weights=weights, combine="min",
                    exclude_pdb=q["pdb_id"], top_k=None)
                for pos, item in enumerate(ranked, start=1):
                    if args.positive_mode == "smiles":
                        if item.get("frag_id") in pos_fids:
                            return pos
                        continue
                    pid = str(item.get("pdb_id", "")).lower()
                    if pid and group_of.get(pid) == gid:
                        return pos
                return 0

            def _arm_rank_marks(marks):
                """The same, against the stored per-key energy marks."""
                if not marks or dense_marks is None or gid is None:
                    return None
                if args.positive_mode == "smiles" and not pos_fids:
                    return None
                scores = _dense_mark_scores(marks, dense_marks)
                order = np.lexsort((-dense_marks[1], -scores))
                pos = 0
                for row in order:
                    if scores[row] <= 0:
                        break
                    if row_pdbs[row] == q_pdb:
                        continue
                    pos += 1
                    if args.positive_mode == "smiles":
                        if row_fids[row] in pos_fids:
                            return pos
                    elif group_of.get(row_pdbs[row]) == gid:
                        return pos
                return 0

            arms = {"ref": q_w}
            # Contact-local variant of the same query: only triplets inside
            # the query fragment's own contact shell survive.
            arms["ref_shell"] = query_weights(
                frag.mol, env, table, q["_att_idx"],
                residue_mask=q_prof["mask"], **kwargs)
            if hits:
                best_real = max(
                    hits, key=lambda h: (h["realistic"]["weighted"],
                                         -h["realistic"]["clashes"]))
                arms["placed_best_real"] = \
                    best_real["realistic"].get("weights")
                arms["placed_best_real_shell"] = \
                    best_real["realistic"].get("weights_shell")
                arms["placed_rank1"] = hits[0]["realistic"].get("weights")
                arms["placed_rank1_shell"] = \
                    hits[0]["realistic"].get("weights_shell")
                best_shell = _best_shell(hits, args.shell_gate)
                if best_shell is not None:
                    arms["placed_shell_best"] = best_shell.get("weights")
                    arms["placed_shell_best_shell"] = \
                        best_shell.get("weights_shell")
                best_persist = _best_shell(hits, args.shell_gate,
                                           key="shell_persist_energy_r")
                if best_persist is not None:
                    arms["placed_persist_best"] = best_persist.get("weights")
                    arms["placed_persist_best_shell"] = \
                        best_persist.get("weights_shell")
                best_oracle = max(hits,
                                  key=lambda h: h["oracle"]["weighted"])
                arms["placed_oracle"] = best_oracle["oracle"].get("weights")
            if controls:
                arms["control"] = controls[0]["realistic"].get("weights")
                arms["control_shell"] = \
                    controls[0]["realistic"].get("weights_shell")

            def _mark(hit):
                return hit["realistic"].get("marks", {}).get(args.mark)

            mark_arms = {"ref_mark": q_m}
            if hits:
                mark_arms["placed_best_real_mark"] = _mark(best_real)
                mark_arms["placed_rank1_mark"] = _mark(hits[0])
                best_shell_hit = _best_shell_hit(hits, args.shell_gate)
                if best_shell_hit is not None:
                    mark_arms["placed_shell_best_mark"] = _mark(best_shell_hit)
                best_persist_hit = _best_shell_hit(
                    hits, args.shell_gate, key="shell_persist_energy_r")
                if best_persist_hit is not None:
                    mark_arms["placed_persist_best_mark"] = \
                        _mark(best_persist_hit)
            if controls:
                mark_arms["control_mark"] = _mark(controls[0])

            row["index_eval"] = {name: _arm_rank(w)
                                 for name, w in arms.items()}
            row["index_eval"].update({name: _arm_rank_marks(m)
                                      for name, m in mark_arms.items()})
            for item in hits + controls:
                for arm in ("realistic", "oracle"):
                    if isinstance(item.get(arm), dict):
                        item[arm].pop("weights", None)
                        item[arm].pop("weights_shell", None)
                        item[arm].pop("marks", None)
        records.append(row)
        if qi % 5 == 0 or qi == len(queries):
            print(f"  {qi}/{len(queries)} queries, "
                  f"{time.time() - t0:.0f}s", flush=True)

    # Flat, score-ranked view across every query fragment and every placed
    # candidate: this is the answer to "reference in, scored candidates out".
    rank_key = {"weighted": "weighted",
                "persist": "shell_persist_energy_r",
                "energy": "shell_energy_r_mean"}[args.rank_by]
    candidates: list[dict] = []
    for qi, rec in enumerate(records, start=1):
        sel = rec.get("selected")
        sel_key = (None if sel is None
                   else (sel["source"], sel["cand_smiles"],
                         sel["retrieval_rank"]))
        for source, hits_key in (("pool", "top_hits"),
                                 ("borrow", "borrow_hits")):
            for hit in rec.get(hits_key, []):
                real = hit.get("realistic") or {}
                orac = hit.get("oracle") or {}
                pool_row = hit.get("pool_row")
                hit_key = (hit.get("source", source), hit["cand_smiles"],
                           int(hit["rank"]))
                candidates.append({
                    "source": hit.get("source", source),
                    "selected": hit_key == sel_key,
                    "query_index": qi,
                    "pdb_id": rec["pdb_id"],
                    "query_smiles": rec["smiles"],
                    "query_n_attach": rec.get("n_attach"),
                    "cand_smiles": hit["cand_smiles"],
                    "cand_dummy_smiles": hit.get("cand_dummy_smiles"),
                    "cand_n_attach": hit.get("n_attach"),
                    "pool_row": (None if pool_row is None else int(pool_row)),
                    "borrow_fid": hit.get("borrow_fid"),
                    "borrow_pdb": hit.get("borrow_pdb"),
                    "env_score": hit.get("env_score"),
                    "retrieval_rank": int(hit["rank"]),
                    "tanimoto": float(hit["tanimoto"]),
                    "tanimoto_global": hit.get("tanimoto_global"),
                    "tanimoto_attach": hit.get("tanimoto_attach"),
                    "cand_attach_identity": hit.get("cand_attach_identity"),
                    "weighted": real.get("weighted"),
                    "oracle_weighted": orac.get("weighted"),
                    "clashes": real.get("clashes"),
                    "rmsd": real.get("rmsd"),
                    "shell_energy_r_mean": real.get("shell_energy_r_mean"),
                    "shell_persist_energy_r":
                        real.get("shell_persist_energy_r"),
                })

    def _desc(v):
        return float("-inf") if v is None else float(v)

    candidates.sort(key=lambda c: (-_desc(c[rank_key]),
                                   -_desc(c.get("weighted")),
                                   -c["tanimoto"]))
    summary = _summarize(records, shell_gate=args.shell_gate)
    if borrow is not None:
        summary["borrow"] = _summarize_borrow(
            records, shell_gate=args.shell_gate)
        summary["combined"] = _summarize_combined(
            records, shell_gate=args.shell_gate)
    out = {
        "pool": args.pool,
        "fp_prefix": args.fp_prefix,
        "attach_fp_prefix": args.attach_fp_prefix,
        "borrow_index": args.borrow_index or None,
        "n_pool_rows": n_rows,
        "features": kwargs,
        "args": {k: v for k, v in vars(args).items()},
        "summary": summary,
        "candidates": candidates,
        "queries": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2)
    csv_path = (os.path.splitext(args.out)[0] + ".csv"
                if args.out_csv is None else args.out_csv)
    if csv_path:
        write_candidates_csv(candidates, csv_path)

    print("\n=== placement prototype ===")
    print(f"scored queries: {summary.get('n_scored', 0)}/{len(records)}  "
          f"mean top-1 Tanimoto: {summary.get('mean_top1_tanimoto', 0):.3f}  "
          f"exact matches in top-{args.top_k}: "
          f"{summary.get('exact_available_mean', 0):.1f}")
    for arm in ("best_realistic", "best_oracle"):
        s = summary.get(arm)
        if s:
            print(f"{arm:>15}: mean={s['mean']:.3f} median={s['median']:.3f} "
                  f"P(>0.5)={s['frac_ge_0.5']:.2f} "
                  f"P(>0.3)={s['frac_ge_0.3']:.2f}")
    if summary.get("control_realistic_mean") is not None:
        print(f"random control: realistic="
              f"{summary['control_realistic_mean']:.3f} "
              f"oracle={summary['control_oracle_mean']:.3f}")
    if "pose_gap_mean" in summary:
        print(f"pose gap (oracle - realistic, top-1): "
              f"{summary['pose_gap_mean']:+.3f}")
        print(f"top-1 realistic clashes mean: "
              f"{summary['realistic_clashes_mean']:.1f}  "
              f"min protein distance mean: "
              f"{summary['realistic_min_dist_mean']:.2f} A")
    if summary.get("shell"):
        print("\ncontact-shell scores (query's own 5 A shell):")

        def _fmt(v):
            return "  n/a" if v is None else f"{v:.3f}"

        for label in ("top1_realistic", "best_realistic", "best_oracle",
                      "best_shell", "best_persist", "control"):
            st = summary["shell"].get(label)
            if not st:
                continue
            print(f"  {label:>15}: recall={_fmt(st['shell_recall'])} "
                  f"cos={_fmt(st['shell_weight_cos'])} "
                  f"E_r={_fmt(st['shell_energy_r'])} "
                  f"E_r_mean={_fmt(st['shell_energy_r_mean'])} "
                  f"P_r={_fmt(st['shell_persist_recall'])} "
                  f"P_all={_fmt(st['shell_persist_all'])} "
                  f"P_E={_fmt(st['shell_persist_energy_r'])} "
                  f"ret={_fmt(st['shell_contact_retention'])} "
                  f"dE={_fmt(st['shell_delta_energy'])} n={st['n']}")
        gate = summary["shell"].get("best_shell_gate")
        for key in ("shell_energy_r_mean", "shell_persist_energy_r"):
            gap = summary["shell"].get(f"{key}_gap_best_shell_minus_control")
            if gap is not None:
                print(f"  {key} gap (gated best, T>={gate:.2f} - control): "
                      f"{gap:+.3f}")
            gap = summary["shell"].get(
                f"{key}_gap_best_persist_minus_control")
            if gap is not None:
                print(f"  {key} gap (persist-selected, T>={gate:.2f} - "
                      f"control): {gap:+.3f}")
    if summary.get("borrow"):
        b = summary["borrow"]
        print("\n=== borrow (environment-matched fragments from the index) ===")
        if not b.get("n_scored"):
            print(f"  no placed borrow candidates ({b['n_queries']} queries)")
        else:
            print(f"  scored queries: {b['n_scored']}/{len(records)}  "
                  f"candidates/query: {b['n_candidates_mean']:.1f}  "
                  f"mean env score: {b['top1_env_score_mean']:.3f}")
            for arm in ("best_realistic", "best_oracle"):
                s = b.get(arm)
                if s:
                    print(f"{arm:>15}: mean={s['mean']:.3f} "
                          f"median={s['median']:.3f} "
                          f"P(>0.5)={s['frac_ge_0.5']:.2f} "
                          f"P(>0.3)={s['frac_ge_0.3']:.2f}")
            bs = b.get("shell", {}).get("best_shell")
            if bs:
                print(f"  best_shell (env-gated): E_r_mean="
                      f"{'n/a' if bs['shell_energy_r_mean'] is None else format(bs['shell_energy_r_mean'], '.3f')} "
                      f"n={bs['n']}")
    if summary.get("combined"):
        cmb = summary["combined"]
        ms = cmb["mean_persist"]

        def _m(value):
            return "n/a" if value is None else f"{value:.3f}"

        print("\n=== combined selection (gated contact persistence) ===")
        print(f"  persist: pool={_m(ms['pool'])} "
              f"(n={cmb['n_solved']['pool']})  "
              f"borrow={_m(ms['borrow'])} (n={cmb['n_solved']['borrow']})  "
              f"union={_m(ms['union'])} (n={cmb['n_solved']['union']})")
        paired = cmb["paired"]
        if paired["n"]:
            print(f"  paired (both gated, n={paired['n']}): "
                  f"pool {paired['wins']['pool']} / "
                  f"borrow {paired['wins']['borrow']} / "
                  f"tie {paired['wins']['tie']}  "
                  f"mean delta={paired['mean_delta']:+.3f}  "
                  f"median={paired['median_delta']:+.3f}")
        src = cmb["selected_source"]
        print(f"  selected source: pool={src['pool']}  "
              f"borrow={src['borrow']}  none={src['none']}")
    if summary.get("index_arms"):
        print(f"\nindex retrieval (positives={args.positive_mode}, "
              f"same-protein / other complexes):")
        for name, st in summary["index_arms"].items():
            print(f"  {name:>16}: n={st['n']} MRR={st['mrr']:.3f} "
                  f"R@10={st['r@10']:.2f} R@50={st['r@50']:.2f} "
                  f"median_rank={st['median_rank']:.0f} "
                  f"no-positive={st['no_positive']}")

    show = (args.top_show if args.top_show >= 0
            else (10 if (args.pdb_id or args.structure) else 0))
    if candidates and show:
        def _v(x, spec="{:.3f}"):
            return "  n/a" if x is None else spec.format(x)

        print(f"\n=== ranked candidates (by {args.rank_by}) ===")
        for i, c in enumerate(candidates[:show], start=1):
            dummy = c.get("cand_dummy_smiles")
            tail = f"  ->  {dummy}" if dummy else ""
            print(f"  {i:>2}. q{c['query_index']} T={c['tanimoto']:.3f} "
                  f"w={_v(c['weighted'])} clash={_v(c['clashes'], '{}'):>2} "
                  f"Er={_v(c['shell_energy_r_mean'])} "
                  f"PE={_v(c['shell_persist_energy_r'])}  "
                  f"{c['cand_smiles']}{tail}")
    print(f"saved: {args.out}  ({time.time() - t0:.0f}s)")
    if csv_path:
        print(f"saved: {csv_path}  ({len(candidates)} rows)")


if __name__ == "__main__":
    main()
