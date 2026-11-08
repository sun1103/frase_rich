#!/usr/bin/env python
"""A/B: does the shape-best attachment pattern score better in the pocket?

The attach-variant recon showed that 41% of retrieved rows are alternate
attachment patterns of an already-retrieved chemotype, that Tanimoto cannot
rank them (identical fingerprints), and that placement keeps the first one
arbitrarily. It also showed that picking the shape-closest pattern reduces
the Sterimol mismatch by a median ~0.3 A (17%). This probe tests whether
that geometry buys anything at the only place it matters: the pose score.

For every chemotype that was actually placed and has an alternative pattern,
``score_candidate`` is run twice with the identical query context, once with
the first (current) pattern and once with the shape-best pattern. The
realistic arm's metrics are paired. The first-pattern rescore is also checked
against the value persisted by the original placement run, which validates
that the A/B is reproducing the original protocol.

No retrieval and no new candidates: this is a controlled change of one
categorical variable (which atoms are the exit vectors).

Usage:
    conda run --no-capture-output -n rxn-env python scripts/probe_attach_ab.py
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
    query_profile,
    query_weights,
    score_candidate,
    select_queries,
)
from scripts.patent_place import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_FEATURES,
    DEFAULT_SELECTED,
    _read_selected,
)

METRICS = ("weighted", "shell_weight_cos", "shell_jaccard",
           "shell_energy_r_mean", "shell_persist_energy_r", "clashes")


def _scores(rec: dict | None) -> dict:
    if rec is None:
        return {"failed": True}
    real = rec["realistic"]
    out = {m: real.get(m) for m in METRICS}
    out.update({"rmsd": real.get("rmsd"), "n_poses": rec.get("n_poses")})
    return out


def _ab_query(probe, q, variant_rec, args, provider, kwargs):
    pid, smiles = probe["pdb_id"], probe["smiles"]
    ctx = provider._context(pid, probe["parent_smiles"])
    if ctx is None:
        return {"pdb_id": pid, "smiles": smiles, "error": "context"}
    _lig, env, table, by_smiles = ctx
    frag = by_smiles.get(smiles)
    if frag is None:
        return {"pdb_id": pid, "smiles": smiles, "error": "fragment"}
    q_w = query_weights(frag.mol, env, table, q["_att_idx"], **kwargs)
    q_prof = query_profile(frag.mol, env, q["_att_idx"], **kwargs)
    hits = {int(h["pool_row"]): h for h in probe["top_hits"]}

    pairs = []
    for group in variant_rec["groups"]:
        if not group.get("placed") or group.get("best_attach") is None:
            continue
        first = [int(a) for a in group["first_attach"]]
        best = [int(a) for a in group["best_attach"]]
        if set(first) == set(best):
            continue
        row = group["rows"][0]
        hit = hits.get(row)
        if hit is None:
            pairs.append({"smiles": group["smiles"],
                          "error": "placed row not the first variant"})
            continue
        common = dict(
            n_confs=args.n_conformers, rolls=args.rolls,
            max_poses=args.max_poses,
            n_profile_poses=args.n_profile_poses,
            q_profile=q_prof, **kwargs)
        a = _scores(score_candidate(
            group["smiles"], first, q["_att_xyz"], q["_exit_xyz"],
            env, table, q_w, **common))
        b = _scores(score_candidate(
            group["smiles"], best, q["_att_xyz"], q["_exit_xyz"],
            env, table, q_w, **common))
        stored = {m: hit["realistic"].get(m) for m in METRICS}
        sanity = {m: (a.get(m) is not None and stored.get(m) is not None
                      and abs(a[m] - stored[m]) < 1e-6)
                  for m in METRICS}
        pairs.append({
            "pdb_id": pid,
            "query_smiles": smiles,
            "cand_smiles": group["smiles"],
            "n_attach": len(group["attachments"][0]),
            "shape_improvement": group.get("improvement"),
            "rel_shape_improvement": group.get("rel_improvement"),
            "n_variants": group["n_variants"],
            "first_attach": first,
            "best_attach": best,
            "first": a,
            "best": b,
            "stored": stored,
            "sanity_match": sanity,
        })
    return {"pdb_id": pid, "smiles": smiles, "pairs": pairs}


def _summarise(pairs: list[dict]) -> dict:
    from scipy.stats import binomtest, spearmanr, wilcoxon

    ok = [p for p in pairs if "error" not in p
          and not p["first"].get("failed") and not p["best"].get("failed")]
    sanity = [p for p in ok
              if all(p["sanity_match"].get(m) for m in METRICS)]
    out = {
        "n_pairs": len(ok),
        "n_rescored_pairs": len(pairs),
        "sanity_first_rescore_matches_stored": len(sanity),
        "sanity_of": len(ok),
        "by_n_attach": {},
        "metrics": {},
    }
    for n_attach in sorted({p["n_attach"] for p in ok}):
        out["by_n_attach"][str(n_attach)] = sum(
            1 for p in ok if p["n_attach"] == n_attach)

    for m in METRICS:
        a = np.array([p["first"][m] for p in ok], dtype=np.float64)
        b = np.array([p["best"][m] for p in ok], dtype=np.float64)
        d = b - a
        finite = np.isfinite(d)
        if finite.sum() < 3:
            continue
        wins = int((d[finite] > 1e-12).sum())
        losses = int((d[finite] < -1e-12).sum())
        ties = int(finite.sum()) - wins - losses
        entry = {
            "first_mean": float(np.nanmean(a)),
            "best_mean": float(np.nanmean(b)),
            "median_delta": float(np.median(d[finite])),
            "mean_delta": float(np.mean(d[finite])),
            "wins": wins, "losses": losses, "ties": ties,
            "sign_test_p": (float(binomtest(wins, wins + losses, 0.5).pvalue)
                            if wins + losses else None),
        }
        try:
            entry["wilcoxon_p"] = float(
                wilcoxon(d[finite]).pvalue) if finite.sum() >= 6 else None
        except Exception:
            entry["wilcoxon_p"] = None
        imp = np.array([p["shape_improvement"] if p["shape_improvement"]
                        is not None else np.nan for p in ok])
        z = np.isfinite(imp) & finite
        entry["spearman_shape_improvement_vs_delta"] = (
            float(spearmanr(imp[z], d[z]).statistic) if z.sum() >= 3 else None)
        out["metrics"][m] = entry
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--placement", default="work/patent_place_persist.json")
    ap.add_argument("--variants", default="work/attach_variant_recon.json")
    ap.add_argument("--selected", default=DEFAULT_SELECTED)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--n-conformers", type=int, default=3)
    ap.add_argument("--rolls", type=int, default=24)
    ap.add_argument("--max-poses", type=int, default=48)
    ap.add_argument("--n-profile-poses", type=int, default=4)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--out", default="work/attach_variant_ab.json")
    args = ap.parse_args(argv)

    with open(args.placement) as handle:
        run = json.load(handle)
    with open(args.variants) as handle:
        variants = json.load(handle)
    probes = {(q["pdb_id"], q["smiles"]): q for q in run["queries"]}
    vqs = [q for q in variants["queries"] if "error" not in q]
    if args.limit_queries:
        vqs = vqs[:args.limit_queries]
    print(f"queries to A/B: {len(vqs)}", flush=True)

    provider = _RefFragProvider(args.cache, DEFAULT_FEATURES)
    rows = _read_selected(args.selected)
    queries = select_queries(
        provider, rows,
        max_queries=int(run["args"]["n_queries"]),
        max_per_complex=int(run["args"]["max_per_complex"]),
        min_heavy=int(run["args"]["min_heavy"]),
        max_heavy=int(run["args"]["max_heavy"]))
    by_key = {(q["pdb_id"], q["smiles"]): q for q in queries}
    kwargs = provider._kwargs()

    out = []
    if os.path.exists(args.out):
        with open(args.out) as handle:
            prev = json.load(handle)
        out = [r for r in prev.get("queries", []) if "error" not in r]
        print(f"resuming: {len(out)} queries already done", flush=True)
    done = {(r["pdb_id"], r["smiles"]) for r in out}

    for n, vrec in enumerate(vqs, start=1):
        key = (vrec["pdb_id"], vrec["smiles"])
        if key in done:
            continue
        q = by_key.get(key)
        probe = probes.get(key)
        if q is None or probe is None:
            print(f"  missing context: {key}", flush=True)
            continue
        rec = _ab_query(probe, q, vrec, args, provider, kwargs)
        out.append(rec)
        with open(args.out, "w") as handle:
            json.dump({"args": vars(args), "summary": None, "queries": out},
                      handle, indent=2)
        if n % 5 == 0 or n == len(vqs):
            print(f"  {n}/{len(vqs)} queries", flush=True)

    pairs = [p for r in out for p in r.get("pairs", [])]
    summary = _summarise(pairs)
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as handle:
        json.dump({"args": vars(args), "summary": summary, "queries": out},
                  handle, indent=2)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
