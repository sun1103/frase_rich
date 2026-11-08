#!/usr/bin/env python
"""Probe: does steric shape distance predict replacement fidelity?

Exploratory check, not part of the pipeline. For every candidate already
placed by ``patent_place.py`` the persisted run stores
``realistic.weighted``: the weighted triplet overlap between the placed
candidate and the query fragment, both scored in the query pocket. That is
the per-candidate replacement-fidelity label we can analyse without
re-placing anything.

The probe asks whether morfeus Sterimol distance between the candidate and
the query fragment (B1, B5, L) predicts that label, and whether it adds
anything over the 2D Tanimoto that selected the candidates:

* query Sterimol uses the fragment's experimental pose and its true exit
  vector (attachment atom -> parent partner atom);
* candidate Sterimol uses ETKDG/MMFF conformers (the same protocol as
  placement) and every free-valence direction at each attachment atom, then
  keeps the best-matching axis. That is deliberately optimistic: it is the
  most favourable shape match a direction-agnostic prior could see, so a
  null result is not an artefact of the ambiguous exit direction.

Sterimol uses heavy atoms only (H-capped fragment H's are implicit), and
heavy-atom indices are unchanged by ``AddHs``.

Usage:
    conda run --no-capture-output -n rxn-env python scripts/probe_steric_fidelity.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morfeus import Sterimol  # noqa: E402
from morfeus.utils import get_radii  # noqa: E402

from frase_rich.evaluate import _RefFragProvider  # noqa: E402
from frase_rich.patent_place import (  # noqa: E402
    embed_candidate,
    exit_directions,
    read_pool_rows,
    select_queries,
)
from scripts.patent_place import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_FEATURES,
    DEFAULT_SELECTED,
    _read_selected,
)


def sterimol(elements, coords, attach_pos: int, direction) -> np.ndarray:
    """``[B1, B5, L]`` (A) along ``direction`` from ``coords[attach_pos]``."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    dummy = coords[attach_pos] + d
    elems = list(elements) + [0]
    xyz = np.vstack([coords, dummy])
    radii = get_radii(elements, radii_type="crc") + [0.0]
    s = Sterimol(elems, xyz, len(elems), attach_pos + 1, radii=radii)
    return np.array([s.B_1_value, s.B_5_value, s.L_value], dtype=np.float64)


def query_sterimols(mol, query: dict) -> np.ndarray:
    """One ``[B1, B5, L]`` per true exit vector of the posed query fragment."""
    n_h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    if n_h:
        raise SystemExit("query fragment unexpectedly carries explicit H")
    xyz = np.asarray(mol.GetConformer().GetPositions())
    elems = [a.GetAtomicNum() for a in mol.GetAtoms()]
    out = []
    for a, att, ex in zip(query["_att_idx"], query["_att_xyz"],
                          query["_exit_xyz"]):
        out.append(sterimol(elems, xyz, int(a), ex - att))
    return np.asarray(out)


def candidate_sterimols(smiles: str, attach, n_confs: int) -> np.ndarray:
    """``[B1, B5, L]`` for every conformer x attachment x free valence."""
    got = embed_candidate(smiles, n_confs)
    if got is None:
        return np.zeros((0, 3))
    mol, cids = got
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    pos_of = {idx: k for k, idx in enumerate(heavy)}
    elems = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in heavy]
    out = []
    for cid in cids:
        xyz = np.asarray(mol.GetConformer(cid).GetPositions())[heavy]
        for atom in sorted(set(int(a) for a in attach)):
            if atom not in pos_of:
                continue
            for direction in exit_directions(mol, [atom], cid)[0]:
                out.append(sterimol(elems, xyz, pos_of[atom], direction))
    return np.asarray(out)


def best_match(cands: np.ndarray, queries: np.ndarray) -> tuple[float, ...]:
    """Smallest (dB5 + dL) over candidate axis x query exit; returns dB1,dB5,dL."""
    if cands.size == 0 or queries.size == 0:
        return (np.nan, np.nan, np.nan)
    best = None
    for q in queries:
        for c in cands:
            db1, db5, dl = abs(c[0] - q[0]), abs(c[1] - q[1]), abs(c[2] - q[2])
            score = db5 + dl
            if best is None or score < best[0]:
                best = (score, db1, db5, dl)
    return tuple(float(x) for x in best[1:])


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x), dtype=np.float64)
    out[order] = np.arange(len(x), dtype=np.float64)
    return out


def _spearman(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    from scipy.stats import spearmanr

    r, p = spearmanr(x[ok], y[ok])
    return float(r), float(p)


def _ols(y, columns: dict[str, np.ndarray]) -> dict:
    """OLS with intercept; returns R2 and per-column coefficients + t stats."""
    names = ["intercept"] + list(columns)
    X = np.column_stack([np.ones(len(y))] + list(columns.values()))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = float(resid @ resid / (n - k))
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(n),
        "r2": r2,
        "coef": {name: float(b) for name, b in zip(names, beta)},
        "t": {name: float(b / s) if s > 0 else float("nan")
              for name, b, s in zip(names, beta, se)},
    }


def _partial_spearman(x, y, z) -> float:
    """Rank partial correlation of x and y controlling for z."""
    rx, ry, rz = _ranks(np.asarray(x)), _ranks(np.asarray(y)), _ranks(np.asarray(z))
    Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--placement", default="work/patent_place_persist.json")
    ap.add_argument("--pool", default="work/patent_pool.parquet")
    ap.add_argument("--selected", default=DEFAULT_SELECTED)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--n-conformers", type=int, default=3)
    ap.add_argument("--out", default="work/steric_probe.json")
    args = ap.parse_args(argv)

    with open(args.placement) as handle:
        run = json.load(handle)
    probes = run["queries"]
    print(f"persisted queries: {len(probes)}", flush=True)

    provider = _RefFragProvider(args.cache, DEFAULT_FEATURES)
    rows = _read_selected(args.selected)
    queries = select_queries(
        provider, rows,
        max_queries=int(run["args"]["n_queries"]),
        max_per_complex=int(run["args"]["max_per_complex"]),
        min_heavy=int(run["args"]["min_heavy"]),
        max_heavy=int(run["args"]["max_heavy"]))
    by_key = {(q["pdb_id"], q["smiles"]): q for q in queries}
    print(f"reconstructed query fragments: {len(queries)}", flush=True)

    records = []
    for n, probe in enumerate(probes, start=1):
        key = (probe["pdb_id"], probe["smiles"])
        q = by_key.get(key)
        if q is None:
            print(f"  missing query context: {key}", flush=True)
            continue
        ctx = provider._context(probe["pdb_id"], probe["parent_smiles"])
        frag = None if ctx is None else ctx[3].get(probe["smiles"])
        if frag is None:
            print(f"  missing fragment mol: {key}", flush=True)
            continue
        q_desc = query_sterimols(frag.mol, q)
        row_ids = [int(h["pool_row"]) for h in probe["top_hits"]]
        meta = read_pool_rows(args.pool, row_ids)
        for hit in probe["top_hits"]:
            info = meta[int(hit["pool_row"])]
            c_desc = candidate_sterimols(
                info["frag_smiles"], info["attach_atoms"], args.n_conformers)
            db1, db5, dl = best_match(c_desc, q_desc)
            records.append({
                "pdb_id": probe["pdb_id"],
                "query_smiles": probe["smiles"],
                "rank": int(hit["rank"]),
                "tanimoto": float(hit["tanimoto"]),
                "cand_smiles": info["frag_smiles"],
                "n_attach": int(info["n_attach"]),
                "weighted_real": float(hit["realistic"]["weighted"]),
                "weighted_oracle": float(hit["oracle"]["weighted"]),
                "db1": db1, "db5": db5, "dl": dl,
                "dist": db5 + dl,
                "n_cand_axes": int(len(c_desc)),
                "n_query_axes": int(len(q_desc)),
            })
        if n % 10 == 0:
            print(f"  {n}/{len(probes)} queries", flush=True)

    ok = [r for r in records if np.isfinite(r["db5"])]
    print(f"\nlabelled candidates: {len(records)} ({len(ok)} with descriptors)")
    if len(ok) < 20:
        raise SystemExit("too few candidates with descriptors")

    y = np.array([r["weighted_real"] for r in ok])
    y_or = np.array([r["weighted_oracle"] for r in ok])
    tan = np.array([r["tanimoto"] for r in ok])
    db1 = np.array([r["db1"] for r in ok])
    db5 = np.array([r["db5"] for r in ok])
    dl = np.array([r["dl"] for r in ok])
    dist = np.array([r["dist"] for r in ok])

    summary: dict = {"n_candidates": len(ok), "n_queries": len({r["pdb_id"] for r in ok})}
    for label, values in (("weighted_real", y), ("weighted_oracle", y_or)):
        summary[label] = {
            "vs_tanimoto": _spearman(values, tan),
            "vs_db1": _spearman(values, db1),
            "vs_db5": _spearman(values, db5),
            "vs_dl": _spearman(values, dl),
            "vs_dist": _spearman(values, dist),
            "partial_dist_given_tanimoto": _partial_spearman(values, dist, tan),
        }
    summary["ols_tanimoto_only"] = _ols(y, {"tanimoto": tan})
    summary["ols_tanimoto_plus_shape"] = _ols(
        y, {"tanimoto": tan, "db5": db5, "dl": dl})

    # Selection-relevant test: within one query's six candidates, does the
    # fidelity ordering follow the steric ordering?
    from scipy.stats import spearmanr

    within = []
    for pid in sorted({r["pdb_id"] for r in ok}):
        group = [r for r in ok if r["pdb_id"] == pid]
        if len(group) < 4:
            continue
        gd = np.array([r["dist"] for r in group])
        gy = np.array([r["weighted_real"] for r in group])
        if gd.std() < 1e-12 or gy.std() < 1e-12:
            continue
        rho = float(spearmanr(gy, gd).statistic)
        best = min(group, key=lambda r: -r["weighted_real"])
        worst = max(group, key=lambda r: -r["weighted_real"])
        within.append({"pdb_id": pid, "rho_fidelity_vs_dist": rho,
                       "best_dist": best["dist"], "worst_dist": worst["dist"],
                       "best_closer": bool(best["dist"] < worst["dist"])})
    closer = sum(1 for w in within if w["best_closer"])
    n_within = len(within)
    from scipy.stats import binomtest

    summary["within_query"] = {
        "n_queries": n_within,
        "mean_rho_fidelity_vs_dist": (float(np.mean(
            [w["rho_fidelity_vs_dist"] for w in within])) if within else None),
        "frac_negative_rho": (float(np.mean(
            [w["rho_fidelity_vs_dist"] < 0 for w in within])) if within else None),
        "best_fidelity_candidate_is_closer": {"n": closer, "of": n_within,
                                              "p": float(binomtest(
                                                  closer, n_within, 0.5).pvalue)
                                              if n_within else None},
    }

    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
