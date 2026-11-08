#!/usr/bin/env python
"""Recon: could a shape filter prune the Tanimoto top-120 before placement?

The placement run retrieves 120 pool rows per query by Morgan Tanimoto and
places the first six that embed (the persisted ``top_hits``). The proposed
second stage is to compute a Sterimol distance for every retrieved candidate
and drop some before the expensive pose scoring. This probe runs the same
retrieval, computes the same optimistic shape distance as
``probe_steric_fidelity.py`` for every candidate, and asks how a shape
filter would have changed the placed set:

* where the six placed candidates sit in the shape ordering of the 120
  (percentile; 0 = closest shape, 1 = farthest);
* whether the best placed candidate by the downstream selection score
  (``shell_persist_energy_r``, the best index arm in the placement study)
  survives a shape top-6 / top-20% / worst-quartile-drop filter;
* how Tanimoto rank and shape rank relate within one query.

No placement is run, so the 114 candidates that were never placed have no
fidelity label; the answer is about candidate-set coverage, not quality.

Usage:
    conda run --no-capture-output -n rxn-env python scripts/probe_shape_filter.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.evaluate import _RefFragProvider  # noqa: E402
from frase_rich.patent_place import (  # noqa: E402
    build_fingerprints,
    query_fingerprint,
    read_pool_rows,
    select_queries,
    tanimoto_topk,
)
from scripts.patent_place import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_FEATURES,
    DEFAULT_SELECTED,
    _nonisomeric,
    _read_selected,
)
from scripts.probe_steric_fidelity import (  # noqa: E402
    best_match,
    candidate_sterimols,
    query_sterimols,
)


def _pct_of(dists: np.ndarray, i: int) -> float:
    """Percentile rank of ``dists[i]`` among finite entries (0 best)."""
    ok = np.isfinite(dists)
    if ok.sum() < 2:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(dists[ok], method="average")
    sub = np.flatnonzero(ok)
    pos = int(np.flatnonzero(sub == i)[0])
    return float((ranks[pos] - 1) / (ok.sum() - 1))


def _recon_query(probe, q, args, paths, n_rows, provider, desc_cache):
    pid, smiles = probe["pdb_id"], probe["smiles"]
    ctx = provider._context(pid, probe["parent_smiles"])
    if ctx is None:
        return {"pdb_id": pid, "smiles": smiles, "error": "context"}
    frag = ctx[3].get(smiles)
    if frag is None:
        return {"pdb_id": pid, "smiles": smiles, "error": "fragment"}

    q_desc = query_sterimols(frag.mol, q)
    bits = query_fingerprint(smiles)
    if bits is None:
        return {"pdb_id": pid, "smiles": smiles, "error": "fingerprint"}
    idx, tan = tanimoto_topk(
        paths, n_rows, bits, n_attach=int(q["n_attach"]),
        n_heavy=int(q["n_heavy"]), k=args.top_k,
        heavy_band=args.heavy_band)

    placed_rows = {int(h["pool_row"]): h for h in probe["top_hits"]}
    wanted = [int(i) for i in idx]
    meta = read_pool_rows(args.pool, wanted)

    # Same stream as placement: Tanimoto order, exact matches dropped,
    # duplicate SMILES dropped. Placement additionally drops candidates that
    # fail to embed; those stay in this stream with a NaN distance.
    q_noniso = _nonisomeric(smiles)
    stream = []
    seen: set[str] = set()
    for rank, (rid, t) in enumerate(zip(idx, tan), start=1):
        info = meta.get(int(rid))
        if info is None:
            continue
        cand = info["frag_smiles"]
        if _nonisomeric(cand) == q_noniso:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        stream.append({"pool_row": int(rid), "tanimoto_rank": int(rank),
                       "tanimoto": float(t), "smiles": cand,
                       "attach": info["attach_atoms"]})
        if args.max_stream and len(stream) >= args.max_stream:
            break

    for entry in stream:
        key = (entry["smiles"],
               tuple(int(a) for a in (entry["attach"] or [])))
        if key not in desc_cache:
            try:
                desc_cache[key] = candidate_sterimols(
                    entry["smiles"], entry["attach"], args.n_conformers)
            except Exception as exc:  # pathological SMILES can break ETKDG
                desc_cache[key] = np.zeros((0, 3))
                entry["embed_error"] = type(exc).__name__
        db1, db5, dl = best_match(desc_cache[key], q_desc)
        entry["dist"] = None if not np.isfinite(db5) else float(db5 + dl)
        entry["db5"], entry["dl"] = (None if not np.isfinite(db5)
                                     else float(db5)), \
            (None if not np.isfinite(dl) else float(dl))

    dists = np.array([np.nan if e["dist"] is None else e["dist"]
                      for e in stream], dtype=np.float64)
    rows = [e["pool_row"] for e in stream]
    pos_of = {rid: i for i, rid in enumerate(rows)}

    placed = []
    for rid, hit in placed_rows.items():
        if rid not in pos_of:
            placed.append({"pool_row": rid, "missing_from_stream": True})
            continue
        i = pos_of[rid]
        e = stream[i]
        placed.append({
            "pool_row": rid,
            "tanimoto_rank": e["tanimoto_rank"],
            "tanimoto": e["tanimoto"],
            "dist": e["dist"],
            "shape_pct": _pct_of(dists, i),
            "weighted": float(hit["realistic"]["weighted"]),
            "shell_energy_r_mean":
                hit["realistic"].get("shell_energy_r_mean"),
            "shell_persist_energy_r":
                hit["realistic"].get("shell_persist_energy_r"),
        })

    finite = np.isfinite(dists)
    n_fin = int(finite.sum())
    # Shape-order candidate sets (only candidates with a distance).
    order = np.argsort(np.where(finite, dists, np.inf), kind="stable")
    top_idx = {int(i) for i in order[:args.top_n]}
    top20_idx = {int(i) for i in order[:max(1, int(round(0.2 * n_fin)))]}
    q75 = float(np.quantile(dists[finite], 0.75)) if n_fin else float("nan")
    # percentiles of the placed pool rows
    placed_rows_in_stream = {r["pool_row"] for r in placed
                             if not r.get("missing_from_stream")}

    def _survives(rid, keep_idx):
        return rid in placed_rows_in_stream and pos_of[rid] in keep_idx

    def _best_by(field):
        cands = [r for r in placed
                 if r.get("missing_from_stream") is not True
                 and r.get(field) is not None]
        if not cands:
            return None
        best = max(cands, key=lambda r: r[field])
        return {
            "pool_row": best["pool_row"],
            "field": field,
            "shape_pct": best["shape_pct"],
            "dist": best["dist"],
            "in_shape_top_n": _survives(best["pool_row"], top_idx),
            "in_shape_top_20pct": _survives(best["pool_row"], top20_idx),
            "survives_worst_quartile_drop":
                (pos_of[best["pool_row"]] in np.flatnonzero(
                    finite & (dists <= q75))),
        }

    overlap = len({r["pool_row"] for r in placed
                   if not r.get("missing_from_stream")} & placed_rows_in_stream
                  & {rows[i] for i in top_idx})
    sp = float("nan")
    ok = [e for e in stream if e["dist"] is not None]
    if len(ok) >= 3:
        from scipy.stats import spearmanr

        sp = float(spearmanr([e["tanimoto"] for e in ok],
                             [e["dist"] for e in ok]).statistic)

    return {
        "pdb_id": pid,
        "smiles": smiles,
        "n_attach": int(q["n_attach"]),
        "n_stream": len(stream),
        "n_finite": n_fin,
        "n_placed": len(placed),
        "placed": placed,
        "best_by_persist": _best_by("shell_persist_energy_r"),
        "best_by_weighted": _best_by("weighted"),
        "overlap_shape_top_n_with_placed": overlap,
        "top_n": args.top_n,
        "spearman_tanimoto_dist": sp,
        "dist_quantiles": (None if not n_fin else {
            "p10": float(np.quantile(dists[finite], 0.10)),
            "median": float(np.quantile(dists[finite], 0.50)),
            "p90": float(np.quantile(dists[finite], 0.90)),
        }),
        "stream": stream,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--placement", default="work/patent_place_persist.json")
    ap.add_argument("--pool", default="work/patent_pool.parquet")
    ap.add_argument("--fp-prefix", default="work/patent_pool")
    ap.add_argument("--selected", default=DEFAULT_SELECTED)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--n-conformers", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=120)
    ap.add_argument("--heavy-band", type=int, default=8)
    ap.add_argument("--max-stream", type=int, default=0,
                    help="cap stream length per query (0 = all retrieved)")
    ap.add_argument("--top-n", type=int, default=6,
                    help="size of a shape top-N filter")
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--out", default="work/shape_filter_recon.json")
    args = ap.parse_args(argv)

    with open(args.placement) as handle:
        run = json.load(handle)
    probes = run["queries"]
    if args.limit_queries:
        probes = probes[:args.limit_queries]
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

    paths = build_fingerprints(args.pool, args.fp_prefix,
                               workers=int(run["args"].get("workers", 6)))
    n_rows = int(np.load(paths["fp"], mmap_mode="r").shape[0])
    print(f"pool fingerprints: {n_rows:,} rows", flush=True)

    desc_cache: dict = {}
    out = []
    if os.path.exists(args.out):
        with open(args.out) as handle:
            prev = json.load(handle)
        out = [r for r in prev.get("queries", []) if "error" not in r]
        print(f"resuming: {len(out)} queries already done", flush=True)
    done = {(r["pdb_id"], r["smiles"]) for r in out}
    for n, probe in enumerate(probes, start=1):
        key = (probe["pdb_id"], probe["smiles"])
        if key in done:
            continue
        q = by_key.get(key)
        if q is None:
            print(f"  missing query context: {key}", flush=True)
            continue
        rec = _recon_query(probe, q, args, paths, n_rows, provider,
                           desc_cache)
        rec["descriptor_cache_size"] = len(desc_cache)
        out.append(rec)
        with open(args.out, "w") as handle:
            json.dump({"args": vars(args), "summary": None,
                       "queries": out}, handle)
        if n % 5 == 0 or n == len(probes):
            print(f"  {n}/{len(probes)} queries, "
                  f"{len(desc_cache):,} unique candidate descriptors",
                  flush=True)

    ok = [r for r in out if "error" not in r and r["n_finite"]]
    print(f"\nqueries with descriptors: {len(ok)}", flush=True)

    placed_pcts, within_half, sps = [], 0, []
    best_hits = {"persist": {"top_n": 0, "top20": 0, "q75": 0, "n": 0,
                            "pcts": []},
                 "weighted": {"top_n": 0, "top20": 0, "q75": 0, "n": 0,
                              "pcts": []}}
    overlaps, drops = [], []
    for r in ok:
        pcts = [p["shape_pct"] for p in r["placed"]
                if p.get("shape_pct") is not None]
        if pcts:
            placed_pcts.extend(pcts)
            if float(np.mean(pcts)) < 0.5:
                within_half += 1
        if np.isfinite(r["spearman_tanimoto_dist"]):
            sps.append(r["spearman_tanimoto_dist"])
        overlaps.append(r["overlap_shape_top_n_with_placed"])
        drops.append(sum(1 for p in r["placed"]
                         if p.get("shape_pct") is not None
                         and p["shape_pct"] > 0.75))
        for tag, field in (("persist", "best_by_persist"),
                           ("weighted", "best_by_weighted")):
            b = r.get(field)
            if b is None:
                continue
            best_hits[tag]["n"] += 1
            best_hits[tag]["pcts"].append(b["shape_pct"])
            best_hits[tag]["top_n"] += int(b["in_shape_top_n"])
            best_hits[tag]["top20"] += int(b["in_shape_top_20pct"])
            best_hits[tag]["q75"] += int(b["survives_worst_quartile_drop"])

    from scipy.stats import binomtest

    n_q = len(ok)
    summary = {
        "n_queries": n_q,
        "n_placed": len(placed_pcts),
        "placed_shape_pct": {
            "mean": float(np.mean(placed_pcts)) if placed_pcts else None,
            "median": float(np.median(placed_pcts)) if placed_pcts else None,
            "frac_queries_mean_below_0.5": within_half / n_q if n_q else None,
            "sign_test_p": (float(binomtest(within_half, n_q, 0.5).pvalue)
                            if n_q else None),
        },
        "shape_top_n_vs_placed": {
            "n": args.top_n,
            "mean_overlap": float(np.mean(overlaps)) if overlaps else None,
            "chance_overlap": (args.top_n ** 2
                               / float(np.mean([r["n_stream"] for r in ok])))
            if ok else None,
        },
        "placed_in_worst_quartile": {
            "mean_per_query": float(np.mean(drops)) if drops else None,
            "chance_per_query": 0.25 * (len(placed_pcts) / n_q) if n_q else None,
        },
        "spearman_tanimoto_dist": {
            "mean": float(np.mean(sps)) if sps else None,
            "frac_positive": (float(np.mean([s > 0 for s in sps]))
                              if sps else None),
        },
        "best_candidate_survival": {},
    }
    for tag, h in best_hits.items():
        summary["best_candidate_survival"][tag] = {
            "n": h["n"],
            "mean_shape_pct": (float(np.mean(h["pcts"])) if h["pcts"]
                               else None),
            "in_top_n": f"{h['top_n']}/{h['n']}",
            "in_top_20pct": f"{h['top20']}/{h['n']}",
            "survives_worst_quartile_drop": f"{h['q75']}/{h['n']}",
        }

    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as handle:
        json.dump({"args": vars(args), "summary": summary, "queries": out},
                  handle, indent=2)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
