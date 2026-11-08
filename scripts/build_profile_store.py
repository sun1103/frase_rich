#!/usr/bin/env python
"""Build the same-group residue-profile store used by the profile-index test.

Usage
-----
    conda run --no-capture-output -n rxn-env python \\
        scripts/build_profile_store.py \\
        --index work/rich_pocket_union.pkl \\
        --cache /Users/mai/Software/fragenv/data/pdbbind_cache \\
        --groups work/pdbbind_protein_groups.json \\
        --queries work/patent_place_persist.json \\
        --out work/profile_store.pkl

Only the complexes belonging to the groups of the placement queries are
profiled. Scoring the same-fragment retrieval task requires the positives'
structures, and the whole library would be 16,663 complexes; the same-group
universe is ~10^2 complexes and is the only universe in which a profile
comparison is defined anyway (residues in different proteins are not
comparable).

Each fragment is profiled with exactly the index's feature settings
(``EEM`` + explicit H, mean-pooled contact energies, 12 A pocket), and the
per-key triplet weights are recomputed from the same residue weights so the
evaluation has an in-store triplet baseline with *identical* candidates.

The output is a pickle::

    {
      "schema": "frase_rich.profile_store.v1",
      "features": {...},
      "groups": {gid: [pdb, ...]},
      "profiles": {fid: {
          frag_smiles, pdb_id, parent_smiles, n_heavy, n_attach,
          seq_id, seq_pos, weights, contacts, mask, res_names,
          fp_norm,            # mean-1 normalized triplet weights
          n_env, n_mapped, n_shell, n_shell_mapped, n_keys,
      }},
    }

``seq_id``/``seq_pos`` are cross-structure residue keys (see
``frase_rich.profiles``); homodimer copies are already merged there.
``fp_norm`` reproduces ``index.add_fragment``'s mean-1 scaling, so weighted
Jaccard over it is the triplet-index score.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.chem import (  # noqa: E402
    charges_for,
    eem_charges_batch,
    with_explicit_hydrogens,
)
from frase_rich.evaluate import _RefFragProvider, _pick_fragment  # noqa: E402
from frase_rich.features import attachment_atoms  # noqa: E402
from frase_rich.index import EnvResidue  # noqa: E402
from frase_rich.patent_place import residue_profile  # noqa: E402
from frase_rich.profiles import (  # noqa: E402
    attach_sequence_keys,
    sequence_residue_map,
    store_features,
)
from frase_rich.weighting import fragment_fp_weights  # noqa: E402


def _normalize_fp(raw: dict[int, float]) -> dict[int, float]:
    raw = {int(k): float(v) for k, v in raw.items() if v > 0}
    if not raw:
        return {}
    scale = len(raw) / float(sum(raw.values()))
    return {k: v * scale for k, v in raw.items()}


def _group_copies(fids: list[str], frags: dict, by_smiles) -> tuple[list, list]:
    """Resolve each index row to the deposited fragment copy it was built from.

    A symmetric parent can deposit the same SMILES at two sites, and keying
    by SMILES alone profiles one arbitrary copy and copies it onto the rest.
    ``_frag_coords`` (stored by the index build in canonical atom order)
    identifies the copy. Returns ``(chosen, copy_fids)``: one ``(smiles,
    fragment)`` entry per distinct copy, and the fids resolving to it. Fids
    whose SMILES the re-cut parent does not produce appear in neither list.
    """
    chosen: list[tuple[str, object]] = []
    copy_fids: list[list[str]] = []
    copy_of: dict[int, int] = {}
    for fid in fids:
        smiles = str(frags[fid].get("frag_smiles", ""))
        cands = by_smiles.all(smiles)
        if not cands:
            continue
        frag = _pick_fragment(cands, frags[fid].get("_frag_coords"))
        slot = copy_of.get(id(frag))
        if slot is None:
            slot = len(chosen)
            copy_of[id(frag)] = slot
            chosen.append((smiles, frag))
            copy_fids.append([])
        copy_fids[slot].append(fid)
    return chosen, copy_fids


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="work/rich_pocket_union.pkl")
    ap.add_argument("--cache", default="/Users/mai/Software/fragenv/data/pdbbind_cache")
    ap.add_argument("--groups", default="work/pdbbind_protein_groups.json")
    ap.add_argument("--queries", default="work/patent_place_persist.json")
    ap.add_argument("--out", default="work/profile_store.pkl")
    ap.add_argument("--shell-cutoff", type=float, default=5.0)
    args = ap.parse_args(argv)

    t0 = time.time()
    print(f"loading index {args.index} ...", flush=True)
    with open(args.index, "rb") as handle:
        index = pickle.load(handle)
    features = index["features"]
    frags = index["frags"]
    with open(args.groups) as handle:
        group_of = {str(k).lower(): str(v) for k, v in json.load(handle).items()}
    with open(args.queries) as handle:
        queries = json.load(handle)["queries"]
    q_groups = {group_of.get(q["pdb_id"].lower()) for q in queries}
    q_groups.discard(None)
    print(f"query groups: {len(q_groups)}", flush=True)

    by_pdb: dict[str, list[str]] = {}
    for fid, info in frags.items():
        pid = str(info.get("pdb_id", "")).lower()
        if group_of.get(pid) in q_groups:
            by_pdb.setdefault(pid, []).append(fid)
    pdbs = sorted(by_pdb)
    n_fids = sum(len(v) for v in by_pdb.values())
    print(f"same-group universe: {len(pdbs)} complexes, {n_fids:,} fragments",
          flush=True)

    provider = _RefFragProvider(args.cache, features)
    kwargs = provider._kwargs()
    profiles: dict[str, dict] = {}
    n_missing = 0
    n_failed = 0
    for i, pid in enumerate(pdbs, start=1):
        fids = by_pdb[pid]
        parent = str(frags[fids[0]].get("parent_smiles", ""))
        ctx = provider._context(pid, parent)
        if ctx is None:
            n_missing += len(fids)
            print(f"  {pid}: no context ({len(fids)} fragments skipped)",
                  flush=True)
            continue
        lig, env, table, by_smiles = ctx
        seq_map = sequence_residue_map(
            os.path.join(args.cache, f"{pid}.pdb"))

        chosen, copy_fids = _group_copies(fids, frags, by_smiles)
        n_missing += len(fids) - sum(len(v) for v in copy_fids)

        mols = [with_explicit_hydrogens(f.mol) for _s, f in chosen]
        try:
            charge_sets = eem_charges_batch(mols)
            methods = ["eem"] * len(mols)
        except Exception:
            charge_sets, methods = [], []
            for mol in mols:
                q, method = charges_for(mol, method="eem")
                charge_sets.append(q)
                methods.append(method)
        for k, (smiles, frag) in enumerate(chosen):
            molh = mols[k]
            try:
                att = attachment_atoms(frag.mol, lig)
                raw = residue_profile(
                    molh, env, charge_sets[k], attach_set=att,
                    shell_cutoff=args.shell_cutoff, **kwargs)
                env_res = [
                    EnvResidue(
                        res_name=r.res_name, chain=r.chain, res_seq=r.res_seq,
                        coords=r.coords, code=r.code, atoms=r.atoms,
                        weight=float(w))
                    for r, w in zip(env, raw["weights"])
                ]
                raw_fp = fragment_fp_weights(
                    env_res, d_lo=features.get("d_lo", 8.0),
                    d_hi=features.get("d_hi", 12.0),
                    fp_aggregate=features.get("fp_aggregate", "mean"),
                    use_pair_bias=features.get("use_pair_bias", True),
                    table=table)
                profile = attach_sequence_keys(raw, env, seq_map)
            except Exception as exc:
                n_failed += len(copy_fids[k])
                print(f"  {pid}/{smiles}: profile failed ({exc})", flush=True)
                continue
            shell = np.asarray(raw["mask"], dtype=bool)
            mapped = np.array([(r.chain, r.res_seq) in seq_map for r in env],
                              dtype=bool)
            base = {
                "frag_smiles": smiles,
                "pdb_id": pid,
                "parent_smiles": parent,
                "n_heavy": int(frag.n_heavy),
                "n_attach": int(frag.n_attach),
                "seq_id": profile["seq_id"],
                "seq_pos": profile["seq_pos"],
                "weights": profile["weights"],
                "contacts": profile["contacts"],
                "mask": profile["mask"],
                "res_names": profile["res_names"],
                "fp_norm": _normalize_fp(raw_fp),
                "charge_method": methods[k],
                "n_env": int(len(env)),
                "n_mapped": int(mapped.sum()),
                "n_shell": int(shell.sum()),
                "n_shell_mapped": int((shell & mapped).sum()),
                "n_keys": int(len(profile["seq_id"])),
            }
            for fid in copy_fids[k]:
                profiles[fid] = base
        if i % 10 == 0 or i == len(pdbs):
            print(f"  {i}/{len(pdbs)} complexes, {len(profiles):,} profiles, "
                  f"{time.time() - t0:.0f}s", flush=True)

    out = {
        "schema": "frase_rich.profile_store.v1",
        "features": store_features(features),
        "groups": {g: sorted(p for p in pdbs if group_of.get(p) == g)
                   for g in sorted(q_groups)},
        "profiles": profiles,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mapped = np.array([p["n_mapped"] / max(p["n_env"], 1)
                       for p in profiles.values()])
    shell_mapped = np.array([
        p["n_shell_mapped"] / max(p["n_shell"], 1) for p in profiles.values()])
    print(f"\nprofiles         : {len(profiles):,}")
    print(f"complexes        : {len({p['pdb_id'] for p in profiles.values()})}"
          f" / {len(pdbs)}")
    print(f"missing fragments: {n_missing:,}  failed: {n_failed:,}")
    print(f"mapped residues  : median {np.median(mapped):.2f}, "
          f"p10 {np.percentile(mapped, 10):.2f}")
    print(f"mapped shell     : median {np.median(shell_mapped):.2f}, "
          f"p10 {np.percentile(shell_mapped, 10):.2f}")
    print(f"saved            : {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
