#!/usr/bin/env python
"""Fuse per-candidate profile scores with the triplet baseline.

Usage
-----
    conda run --no-capture-output -n rxn-env python \\
        scripts/fuse_profile_scores.py \\
        --scores work/profile_scores.json \\
        --out work/profile_fusion.json

``scripts/eval_profile_index.py --save-scores`` writes one record per eligible
query with every arm's per-candidate scores. This script ranks each arm with
the evaluation's deterministic tie-break, then fuses the triplet arm with each
profile arm:

- ``rrf``: reciprocal-rank fusion, ``1/(k + rank)`` from each arm;
- ``zave``: sum of per-query z-scored arm scores;
- ``tiebreak``: triplet order, ties broken by the profile score.

It reports MRR / R@10 / R@50, paired wins/losses/ties against the triplet, the
mean ΔMRR, and a paired bootstrap interval for the best RRF arm. Oracle (the
better of the two ranks) is reported as a bound, never as a deployable arm.

The permutation control is not a separate arm here: its score is not meant to
be fused, only to show that the profile signal depends on residue
correspondence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASELINE = "triplet_ref"
NOT_FUSED = {BASELINE, "triplet_ref_shell", "triplet_uniform", "random"}


def _ranks(scores, row) -> np.ndarray:
    """Deterministic rank (1 = best) matching eval_profile_index._rank."""
    order = sorted(
        range(len(row["fids"])),
        key=lambda i: (-float(scores[i]), row["frag_smiles"][i],
                       row["pdb_ids"][i], row["fids"][i]))
    out = np.zeros(len(order), dtype=np.int64)
    for rank, i in enumerate(order, start=1):
        out[i] = rank
    return out


def _positive_rank(ranks: np.ndarray, pos: set[int]) -> int:
    return int(min(ranks[i] for i in pos))


def _z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sd = float(values.std())
    return (values - values.mean()) / sd if sd > 1e-12 else values * 0.0


def _summary(ranks: list[int]) -> dict:
    arr = np.asarray(ranks, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mrr": float(np.mean(1.0 / arr)),
        "r@10": float(np.mean(arr <= 10)),
        "r@50": float(np.mean(arr <= 50)),
        "median_rank": float(np.median(arr)),
    }


def _paired(base: list[int], other: list[int]) -> dict:
    d = np.array([1.0 / o - 1.0 / b for b, o in zip(base, other)])
    wins = int(np.sum(np.array(other) < np.array(base)))
    losses = int(np.sum(np.array(other) > np.array(base)))
    return {
        "wins": wins, "losses": losses,
        "ties": len(base) - wins - losses,
        "mean_dmrr": float(d.mean()),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="work/profile_scores.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--rerank-k", default="10,25",
                    help="comma-separated triplet top-K sizes to rerank")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rerank_ks = [int(k) for k in args.rerank_k.split(",") if k.strip()]

    with open(args.scores) as handle:
        rows = json.load(handle)
    if not rows:
        raise SystemExit("no scored queries")
    arms = [name for name in rows[0]["scores"] if name not in NOT_FUSED]

    base_ranks: list[int] = []
    arm_ranks: dict[str, list[int]] = {name: [] for name in arms}
    fused: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        pos = set(row["pos"])
        scores = row["scores"]
        tr = _ranks(scores[BASELINE], row)
        base_ranks.append(_positive_rank(tr, pos))
        pr = {name: _ranks(scores[name], row) for name in arms}
        for name in arms:
            arm_ranks[name].append(_positive_rank(pr[name], pos))
        for name in arms:
            rrf = 1.0 / (args.k + tr) + 1.0 / (args.k + pr[name])
            zave = _z(scores[BASELINE]) + _z(scores[name])
            rrf_rank = _ranks(rrf, row)
            zave_rank = _ranks(zave, row)
            fused[("rrf", name)] = fused.get(("rrf", name), [])
            fused[("rrf", name)].append(_positive_rank(rrf_rank, pos))
            fused[("zave", name)] = fused.get(("zave", name), [])
            fused[("zave", name)].append(_positive_rank(zave_rank, pos))
        # Triplet primary, profile secondary; deterministic keys still break
        # any remaining ties.
        for name in arms:
            order = sorted(
                range(len(tr)),
                key=lambda i: (-float(scores[BASELINE][i]),
                               -float(scores[name][i]),
                               row["frag_smiles"][i], row["pdb_ids"][i],
                               row["fids"][i]))
            tb_rank = np.zeros(len(order), dtype=np.int64)
            for rank, i in enumerate(order, start=1):
                tb_rank[i] = rank
            fused[("tiebreak", name)] = fused.get(("tiebreak", name), [])
            fused[("tiebreak", name)].append(_positive_rank(tb_rank, pos))
            # Rerank only the triplet's top K; everything else keeps the
            # triplet order, so this isolates the profile's top-of-list vote.
            for top_k in rerank_ks:
                inside = set(np.argsort(tr, kind="stable")[:top_k])
                rr = sorted(
                    range(len(tr)),
                    key=lambda i: (
                        0 if i in inside else 1,
                        -float(scores[name][i]) if i in inside
                        else int(tr[i]),
                        row["frag_smiles"][i], row["pdb_ids"][i],
                        row["fids"][i]))
                rr_rank = np.zeros(len(tr), dtype=np.int64)
                for rank, i in enumerate(rr, start=1):
                    rr_rank[i] = rank
                key = (f"rerank@{top_k}", name)
                fused[key] = fused.get(key, [])
                fused[key].append(_positive_rank(rr_rank, pos))

    table: dict[str, dict] = {BASELINE: _summary(base_ranks)}
    for name in arms:
        table[name] = _summary(arm_ranks[name])
    for (kind, name), ranks in fused.items():
        entry = _summary(ranks)
        entry.update(_paired(base_ranks, ranks))
        table[f"{kind}:{name}"] = entry
    oracle = {
        name: [_positive_rank(
            np.minimum(_ranks(rows[i]["scores"][BASELINE], rows[i]),
                       _ranks(rows[i]["scores"][name], rows[i])),
            set(rows[i]["pos"])) for i in range(len(rows))]
        for name in arms}
    for name in arms:
        entry = _summary(oracle[name])
        entry["oracle"] = True
        table[f"oracle:{name}"] = entry

    best = max((key for key in table if key.startswith("rrf:")),
               key=lambda key: table[key]["mrr"])
    diffs = np.array([
        1.0 / r - 1.0 / b
        for b, r in zip(base_ranks, fused[tuple(best.split(":", 1))])])
    rng = np.random.default_rng(args.seed)
    boot = [float(np.mean(rng.choice(diffs, size=diffs.size, replace=True)))
            for _ in range(max(1, args.bootstrap))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    table[best]["bootstrap"] = {
        "mean_dmrr": float(diffs.mean()),
        "ci95": [float(lo), float(hi)],
        "p_le_zero": float(np.mean(np.asarray(boot) <= 0.0)),
    }

    order = sorted(table, key=lambda key: -table[key]["mrr"])
    print(f"queries: {len(rows)}  arms fused: {len(arms)}  k={args.k}")
    print(f"{'arm':>28} {'MRR':>6} {'R@10':>5} {'R@50':>5} {'med':>5} "
          f"{'W/L/T':>8} {'dMRR':>7}")
    for key in order:
        st = table[key]
        wlt = (f"{st['wins']}/{st['losses']}/{st['ties']}"
               if "wins" in st else "-")
        dmrr = f"{st['mean_dmrr']:+.3f}" if "mean_dmrr" in st else "-"
        print(f"{key:>28} {st['mrr']:>6.3f} {st['r@10']:>5.2f} "
              f"{st['r@50']:>5.2f} {st['median_rank']:>5.0f} {wlt:>8} "
              f"{dmrr:>7}")
    b = table[best]["bootstrap"]
    print(f"\nbest RRF: {best}  mean dMRR {b['mean_dmrr']:+.3f}  "
          f"95% CI [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]  "
          f"P(d<=0)={b['p_le_zero']:.2f}")

    if args.out:
        out = {
            "scores": args.scores,
            "k": args.k,
            "bootstrap_seed": args.seed,
            "best_rrf": best,
            "table": table,
            "queries": [
                {"pdb_id": row["pdb_id"], "smiles": row["smiles"],
                 "n_candidates": len(row["fids"]),
                 "n_pos": len(row["pos"]),
                 "triplet_rank": base_ranks[i],
                 "ranks": {name: arm_ranks[name][i] for name in arms},
                 "rrf_rank": fused[tuple(best.split(":", 1))][i]}
                for i, row in enumerate(rows)
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(out, handle, indent=2)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
