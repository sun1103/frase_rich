#!/usr/bin/env python
"""Recon: can shape choose the attachment pattern among duplicate-SMILES rows?

The patent pool is keyed by ``(frag_smiles, attach_atoms)``, so one chemotype
can appear in the top-120 several times with different attachment patterns.
Tanimoto cannot separate those rows: the fingerprint is computed on the
H-capped fragment SMILES, which is identical for every variant. Placement
dedupes by SMILES and keeps the first row in retrieval order, so the pattern
actually scored is arbitrary.

For every retrieved group with more than one variant this probe computes the
same optimistic Sterimol distance used by the other probes for each variant
and asks whether the shape-best variant differs from the first one, and by
how much. It also restricts the comparison to the six chemotypes that were
actually placed, since those are the only ones whose geometry reached the
scoring step.

This measures a potential, not an outcome: the distance improvement is
geometry only. Whether a better-shaped exit vector places and scores better
needs the A/B placement run.

Usage:
    conda run --no-capture-output -n rxn-env python scripts/probe_attach_variants.py
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


def _dist(desc, q_desc) -> tuple[float, float, float]:
    db1, db5, dl = best_match(desc, q_desc)
    if not np.isfinite(db5):
        return (float("nan"), float("nan"), float("nan"))
    return float(db5 + dl), float(db5), float(dl)


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
    meta = read_pool_rows(args.pool, [int(i) for i in idx])

    # Same row filter as placement, but keep every attachment pattern: exact
    # matches are dropped; duplicate SMILES are not.
    q_noniso = _nonisomeric(smiles)
    groups: dict[str, list[dict]] = {}
    for rank, (rid, t) in enumerate(zip(idx, tan), start=1):
        info = meta.get(int(rid))
        if info is None:
            continue
        cand = info["frag_smiles"]
        if _nonisomeric(cand) == q_noniso:
            continue
        groups.setdefault(cand, []).append({
            "pool_row": int(rid), "rank": int(rank),
            "tanimoto": float(t),
            "attach": tuple(int(a) for a in (info["attach_atoms"] or [])),
        })

    placed_rows = {int(h["pool_row"]) for h in probe["top_hits"]}
    records = []
    for cand, variants in groups.items():
        dists = []
        for v in variants:
            key = (cand, v["attach"])
            if key not in desc_cache:
                try:
                    desc_cache[key] = candidate_sterimols(
                        cand, v["attach"], args.n_conformers)
                except Exception as exc:  # pathological SMILES can break ETKDG
                    desc_cache[key] = np.zeros((0, 3))
                    v["embed_error"] = type(exc).__name__
            d, db5, dl = _dist(desc_cache[key], q_desc)
            v.update({"dist": d, "db5": db5, "dl": dl})
            if np.isfinite(d):
                dists.append((d, v))
        first = variants[0]
        best = min(dists, key=lambda x: x[0]) if dists else None
        rec = {
            "smiles": cand,
            "n_variants": len(variants),
            "attachments": [list(v["attach"]) for v in variants],
            "rows": [v["pool_row"] for v in variants],
            "ranks": [v["rank"] for v in variants],
            "placed": any(v["pool_row"] in placed_rows for v in variants),
            "first_dist": first["dist"],
            "first_attach": list(first["attach"]),
            "first_is_best": (best is not None
                              and abs(best[1]["dist"] - first["dist"]) < 1e-9),
        }
        if best is not None:
            rec.update({
                "best_dist": best[0],
                "best_attach": list(best[1]["attach"]),
                "n_finite_variants": len(dists),
            })
            if np.isfinite(first["dist"]):
                rec["improvement"] = first["dist"] - best[0]
                rec["rel_improvement"] = ((first["dist"] - best[0])
                                          / first["dist"]
                                          if first["dist"] > 0 else 0.0)
            else:
                rec["improvement"] = None
                rec["rel_improvement"] = None
        records.append(rec)

    return {
        "pdb_id": pid,
        "smiles": smiles,
        "n_attach": int(q["n_attach"]),
        "n_rows": int(len(idx)),
        "n_groups": len(groups),
        "n_multi": sum(1 for r in records if r["n_variants"] > 1),
        "rows_in_multi": sum(r["n_variants"] for r in records
                             if r["n_variants"] > 1),
        "groups": records,
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
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--out", default="work/attach_variant_recon.json")
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
        out.append(rec)
        with open(args.out, "w") as handle:
            json.dump({"args": vars(args), "summary": None,
                       "queries": out}, handle)
        if n % 5 == 0 or n == len(probes):
            print(f"  {n}/{len(probes)} queries, "
                  f"{len(desc_cache):,} unique candidate descriptors",
                  flush=True)

    ok = [r for r in out if "error" not in r]
    print(f"\nqueries: {len(ok)}", flush=True)

    multi = [(r, g) for r in ok for g in r["groups"] if g["n_variants"] > 1
             and g.get("improvement") is not None]
    placed = [(r, g) for r, g in multi if g["placed"]]
    imp = np.array([g["improvement"] for _, g in multi])
    rel = np.array([g["rel_improvement"] for _, g in multi])
    pimp = np.array([g["improvement"] for _, g in placed])
    prel = np.array([g["rel_improvement"] for _, g in placed])

    def _q(x):
        return None if len(x) == 0 else {
            "p10": float(np.quantile(x, .10)),
            "median": float(np.quantile(x, .50)),
            "p90": float(np.quantile(x, .90)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
        }

    summary = {
        "n_queries": len(ok),
        "rows_per_query": float(np.mean([r["n_rows"] for r in ok])),
        "groups_per_query": float(np.mean([r["n_groups"] for r in ok])),
        "multi_variant_groups": {
            "n": len(multi),
            "frac_of_groups": float(len(multi)
                                    / sum(r["n_groups"] for r in ok)),
            "rows_in_multi": int(sum(r["rows_in_multi"] for r in ok)),
            "frac_of_rows": float(sum(r["rows_in_multi"] for r in ok)
                                  / sum(r["n_rows"] for r in ok)),
            "variant_count_dist": {
                str(k): sum(1 for _, g in multi if g["n_variants"] == k)
                for k in sorted({g["n_variants"] for _, g in multi})},
        },
        "first_is_best": {
            "n": int(sum(1 for _, g in multi if g["first_is_best"])),
            "of": len(multi),
            "frac": (float(np.mean([g["first_is_best"] for _, g in multi]))
                     if multi else None),
        },
        "improvement_all_groups": _q(imp),
        "relative_improvement_all_groups": _q(rel),
        "placed_chemotypes": {
            "n": len(placed),
            "first_is_best": int(sum(1 for _, g in placed if g["first_is_best"])),
            "improvement": _q(pimp),
            "relative_improvement": _q(prel),
            "frac_improved_gt_0.2A": (float(np.mean(pimp > 0.2))
                                      if len(pimp) else None),
            "frac_improved_gt_10pct": (float(np.mean(prel > 0.10))
                                       if len(prel) else None),
        },
    }

    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as handle:
        json.dump({"args": vars(args), "summary": summary, "queries": out},
                  handle, indent=2)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
