#!/usr/bin/env python
"""Rank same-group library fragments by residue-profile agreement.

Usage
-----
    conda run --no-capture-output -n rxn-env python \\
        scripts/eval_profile_index.py \\
        --store work/profile_store.pkl \\
        --queries work/patent_place_persist.json \\
        --out work/profile_eval.json

The task is the profile-index analogue of the triplet leave-one-complex-out
benchmark: for a query fragment, rank the fragments of the *other* complexes
of the same protein and find the same fragment SMILES. The candidate universe
is the profile store (all fragments of those complexes), so the triplet arms
computed here are restricted to exactly the same rows -- the only fair
comparison for a profile index, which cannot score fragments it has no
structure for.

Arms
----
profile   weight_cos, energy_r, weighted_energy_r, combined, recall_energy,
          contact_retention, delta_energy (see ``frase_rich.profiles``)
triplet   query fragment weights vs stored per-key weights (weighted
          Jaccard; ``ref``), the same restricted to the query contact shell
          (``ref_shell``), and plain Jaccard (``uniform``)
control   a seeded random candidate order; a per-candidate permutation of the
          profile vectors, which keeps every marginal but destroys residue
          correspondence.

Ranks are deterministic: score descending, then (frag_smiles, pdb_id, fid).
``tie_at_positive`` records how many candidates share the best positive's
score, so a collapse to ties is visible rather than hidden by the tie-break.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.evaluate import _RefFragProvider  # noqa: E402
from frase_rich.patent_place import (  # noqa: E402
    query_exit_vectors,
    query_profile,
    query_weights,
)
from frase_rich.profiles import (  # noqa: E402
    attach_sequence_keys,
    profile_similarity,
    rank_score,
    sequence_residue_map,
)

PROFILE_ARMS = (
    ("p_weight_cos", "weight_cos"),
    ("p_energy", "energy"),
    ("p_wenergy", "weighted_energy"),
    ("p_combined", "combined"),
    ("p_recall_energy", "recall_energy"),
    ("p_retention", "retention"),
    ("p_delta", "delta_energy"),
    ("p_energy_fav", "energy_fav"),
    ("p_energy_engaged", "energy_engaged"),
    ("p_wenergy_strength", "weighted_energy_strength"),
    ("p_contact_jaccard", "contact_jaccard"),
    ("p_overlap_energy", "overlap_energy"),
    ("p_all_weight_cos", "all_weight_cos"),
    ("p_all_energy", "all_energy"),
    ("p_all_wenergy", "all_weighted_energy"),
)


def _jaccard(qw: dict, lw: dict) -> float:
    """Weighted exact-key Jaccard over mean-1 weight vectors."""
    if not qw or not lw:
        return 0.0
    if len(lw) < len(qw):
        qw, lw = lw, qw
    w = 0.0
    for key, value in qw.items():
        other = lw.get(key)
        if other is not None:
            w += min(value, other)
    n_q, n_l = len(qw), len(lw)
    return float(w / max(n_q + n_l - w, 1e-9))


def _uniform_jaccard(qw: dict, lw: dict) -> float:
    if not qw or not lw:
        return 0.0
    inter = len(set(qw) & set(lw))
    return float(inter / max(len(qw) + len(lw) - inter, 1))


def _rank(scores: np.ndarray, cands: list[dict], pos: set[int]) -> tuple:
    order = sorted(
        range(len(cands)),
        key=lambda i: (-float(scores[i]), cands[i]["frag_smiles"],
                       cands[i]["pdb_id"], cands[i]["frag_id"]))
    for rank, i in enumerate(order, start=1):
        if i in pos:
            best = float(scores[i])
            tie = int(np.count_nonzero(np.asarray(scores) >= best - 1e-12))
            return rank, tie
    raise ValueError("no positive in candidate list")


def _permuted_similarity(q: dict, c: dict, rng: np.random.Generator) -> dict:
    """Profile agreement after destroying the residue correspondence.

    Only the per-residue *values* are permuted; the key arrays stay in place.
    Permuting keys and values together would preserve the key -> value
    pairing and leave every metric unchanged (which is exactly what a broken
    control looks like).
    """
    n = len(c["weights"])
    order = rng.permutation(n)
    shuffled = dict(c)
    shuffled["weights"] = np.asarray(c["weights"])[order]
    shuffled["contacts"] = np.asarray(c["contacts"])[order]
    shuffled["mask"] = np.asarray(c["mask"])[order]
    return profile_similarity(q, shuffled)


def _random_rank(cands: list[dict], pos: set[int],
                 seeds: list[int]) -> float:
    ranks = []
    for seed in seeds:
        order = list(range(len(cands)))
        random.Random(seed).shuffle(order)
        ranks.append(next(i for i, j in enumerate(order, start=1) if j in pos))
    return float(np.mean(ranks))


def _summary(ranks: list[int | None], ties: list[int]) -> dict:
    values = [r for r in ranks if r]
    n = len(values)
    if not n:
        return {"n": 0, "mrr": 0.0, "r@10": 0.0, "r@50": 0.0,
                "median_rank": None, "no_hit": len(ranks) - n,
                "mean_tie_at_positive": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": n,
        "mrr": float(np.mean(1.0 / arr)),
        "r@10": float(np.mean(arr <= 10)),
        "r@50": float(np.mean(arr <= 50)),
        "median_rank": float(np.median(arr)),
        "no_hit": len(ranks) - n,
        "mean_tie_at_positive": float(np.mean(ties)) if ties else None,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="work/profile_store.pkl")
    ap.add_argument("--queries", default="work/patent_place_persist.json")
    ap.add_argument("--groups", default="work/pdbbind_protein_groups.json")
    ap.add_argument("--cache",
                    default="/Users/mai/Software/fragenv/data/pdbbind_cache")
    ap.add_argument("--eligibility", default="work/placement_eligible.json")
    ap.add_argument("--out", default="work/profile_eval.json")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--shuffle-seeds", type=int, default=3)
    ap.add_argument("--save-scores", default="",
                    help="write per-candidate arm scores for offline fusion "
                         "analysis (large-ish, eligible queries only)")
    args = ap.parse_args(argv)

    t0 = time.time()
    with open(args.store, "rb") as handle:
        store = pickle.load(handle)
    profiles: dict[str, dict] = store["profiles"]
    for fid, info in profiles.items():
        info["frag_id"] = fid
    with open(args.groups) as handle:
        group_of = {str(k).lower(): str(v) for k, v in json.load(handle).items()}
    with open(args.queries) as handle:
        queries = json.load(handle)["queries"]
    if args.max_queries:
        queries = queries[:args.max_queries]
    eligible = {}
    if os.path.isfile(args.eligibility):
        with open(args.eligibility) as handle:
            for row in json.load(handle):
                eligible[(row["pdb_id"].lower(), row["smiles"])] = int(row["n_pos"])

    by_group: dict[str, list[dict]] = defaultdict(list)
    for info in profiles.values():
        gid = group_of.get(info["pdb_id"])
        if gid is not None:
            by_group[gid].append(info)

    provider = _RefFragProvider(args.cache, store["features"])
    kwargs = provider._kwargs()
    seeds = list(range(max(1, args.shuffle_seeds)))
    records: list[dict] = []
    arm_ranks: dict[str, list[int | None]] = defaultdict(list)
    arm_ties: dict[str, list[int]] = defaultdict(list)
    score_dump: list[dict] = []

    for qi, q in enumerate(queries, start=1):
        pid = q["pdb_id"].lower()
        gid = group_of.get(pid)
        row = {"pdb_id": pid, "smiles": q["smiles"], "group": gid,
               "eligible_prior": eligible.get((pid, q["smiles"]))}
        ctx = provider._context(pid, q["parent_smiles"])
        if ctx is None or gid is None:
            row["error"] = "context"
            records.append(row)
            continue
        lig, env, table, by_smiles = ctx
        frag = by_smiles.get(q["smiles"])
        if frag is None:
            row["error"] = "fragment"
            records.append(row)
            continue
        try:
            att, _att_xyz, _exit_xyz = query_exit_vectors(frag.mol, lig)
        except ValueError:
            row["error"] = "exit vectors"
            records.append(row)
            continue
        raw = query_profile(frag.mol, env, att, **kwargs)
        seq_map = sequence_residue_map(os.path.join(args.cache, f"{pid}.pdb"))
        q_prof = attach_sequence_keys(raw, env, seq_map)
        q_w = query_weights(frag.mol, env, table, att, **kwargs)
        q_w_shell = query_weights(
            frag.mol, env, table, att, residue_mask=raw["mask"], **kwargs)

        cands = [info for info in by_group.get(gid, ())
                 if info["pdb_id"] != pid]
        pos = {i for i, info in enumerate(cands)
               if info["frag_smiles"] == q["smiles"]}
        row["n_candidates"] = len(cands)
        row["n_pos"] = len(pos)
        if not pos:
            records.append(row)
            continue

        sims = [profile_similarity(q_prof, info) for info in cands]
        scores: dict[str, np.ndarray] = {
            name: np.array([rank_score(m, kind) for m in sims])
            for name, kind in PROFILE_ARMS
        }
        rng = np.random.default_rng(0)
        perm_scores = np.array([
            rank_score(_permuted_similarity(q_prof, info, rng), "combined")
            for info in cands])
        scores["p_combined_permuted"] = perm_scores
        scores["triplet_ref"] = np.array(
            [_jaccard(q_w, info["fp_norm"]) for info in cands])
        scores["triplet_ref_shell"] = np.array(
            [_jaccard(q_w_shell, info["fp_norm"]) for info in cands])
        scores["triplet_uniform"] = np.array(
            [_uniform_jaccard(q_w, info["fp_norm"]) for info in cands])
        scores["random"] = None  # handled separately

        row["arms"] = {}
        for name, values in scores.items():
            if values is None:
                continue
            rank, tie = _rank(values, cands, pos)
            row["arms"][name] = {"rank": rank, "tie": tie}
            arm_ranks[name].append(rank)
            arm_ties[name].append(tie)
        rand = _random_rank(cands, pos, seeds)
        row["arms"]["random"] = {"rank": rand, "tie": None}
        arm_ranks["random"].append(rand)
        if args.save_scores:
            score_dump.append({
                "pdb_id": pid,
                "smiles": q["smiles"],
                "fids": [info["frag_id"] for info in cands],
                "frag_smiles": [info["frag_smiles"] for info in cands],
                "pdb_ids": [info["pdb_id"] for info in cands],
                "pos": sorted(pos),
                "scores": {name: np.asarray(values).tolist()
                           for name, values in scores.items()
                           if values is not None},
            })
        records.append(row)
        if qi % 5 == 0:
            print(f"  {qi}/{len(queries)} queries, {time.time() - t0:.0f}s",
                  flush=True)

    summary = {name: _summary(arm_ranks[name], arm_ties[name])
               for name in arm_ranks}
    out = {
        "store": args.store,
        "features": store["features"],
        "shuffle_seeds": seeds,
        "summary": summary,
        "queries": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2)
    if args.save_scores:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_scores)) or ".",
                    exist_ok=True)
        with open(args.save_scores, "w") as handle:
            json.dump(score_dump, handle)

    print("\n=== profile index (same-group universe) ===")
    print(f"eligible queries: {summary.get('triplet_ref', {}).get('n', 0)}"
          f" / {len(records)}")
    print(f"{'arm':>22} {'n':>3} {'MRR':>6} {'R@10':>5} {'R@50':>5} "
          f"{'med':>7} {'tie':>6}")
    order = ([n for n, _ in PROFILE_ARMS]
             + ["p_combined_permuted", "triplet_ref", "triplet_ref_shell",
                "triplet_uniform", "random"])
    for name in order:
        st = summary.get(name)
        if not st:
            continue
        median = "-" if st["median_rank"] is None else f"{st['median_rank']:.0f}"
        tie = ("-" if st["mean_tie_at_positive"] is None
               else f"{st['mean_tie_at_positive']:.1f}")
        print(f"{name:>22} {st['n']:>3} {st['mrr']:>6.3f} "
              f"{st['r@10']:>5.2f} {st['r@50']:>5.2f} {median:>7} {tie:>6}")
    print(f"saved: {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
