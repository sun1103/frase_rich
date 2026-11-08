"""Cross-structure residue profiles for environment-based retrieval.

A residue-profile index stores, for every library fragment, the per-residue
interaction weight and contact energy that the triplet index throws away. The
query side has the same vectors, so the two can be compared residue by
residue. That comparison only means something when residue *i* in one
structure is the same residue as residue *i* in another.

PDBbind-like corpora cannot supply that identity from ``(chain, res_seq)``:
structures of one protein routinely use different chain labels (heavy chain
``H`` vs ``A``) or a different numbering frame, so the raw keys are disjoint
even though the sequences are identical (the groups in
``pdbbind_protein_groups.json`` were built by exact longest-chain sequence
match). ``sequence_residue_map`` therefore keys a residue by

    (sequence id, position along that sequence)

The sequence id is a CRC32 of the chain's one-letter sequence and the position
is the residue's index among that chain's CA records. This is stable across
numbering frames. Chains with identical sequences (homodimer copies) share
keys: a ligand that binds the dimer interface is described by its aggregate
(summed) contact with each sequence position rather than by which copy it
happens to touch, which also makes the descriptor robust to chain
relabelling. Chains
with different sequences (thrombin light chain, peptides, a different
construct) get their own sequence id, so they never collide with the main
chain.

Profiles are plain dicts so the builder does not need to know how they were
produced::

    profile = {
        "seq_id": int64 array,      # CRC32 of the source chain sequence
        "seq_pos": int64 array,     # position along that chain
        "weights": float64 array,   # softmax relevance weight per residue
        "contacts": float64 array,  # contact-gated energy per residue
        "mask": bool array,         # contact shell (heavy atom <= 5 A)
        "res_names": list[str],
    }

``profile_similarity`` compares two such dicts. The ``*_weight_*`` / energy
metrics are contact-local on the query's shell, matching the placement
prototype's finding that the whole-pocket profile is too pocket-dominated to
rank replacements. ``rank_score`` maps metrics to a scalar for ranking.
"""

from __future__ import annotations

import zlib
from typing import Iterable

import numpy as np

from .proteins import AA3

_EPS = 1e-12


def _chain_records(path: str) -> dict[str, list[tuple[str, str]]]:
    """Ordered ``(res_seq, res_name)`` records per chain, CA-only."""
    chains: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[tuple[int, str]]] = {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip().upper() != "CA":
                continue
            name = line[17:20].strip().upper()
            if name not in AA3:
                continue
            chain = line[21]
            num = int(line[22:26])
            icode = line[26].strip()
            bucket = seen.setdefault(chain, set())
            if (num, icode) in bucket:
                continue
            bucket.add((num, icode))
            chains.setdefault(chain, []).append((f"{num}{icode}", name))
    return chains


def sequence_residue_map(path: str) -> dict[tuple[str, str], tuple[int, int]]:
    """Map ``(chain, res_seq)`` to ``(sequence id, position)``.

    The sequence id is a CRC32 of the chain's one-letter sequence. Identical
    chains (homodimer copies) therefore produce the same ids, and their
    residues collapse onto the same keys; ``attach_sequence_keys`` aggregates
    those copies.
    """
    chains = _chain_records(path)
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for chain, records in chains.items():
        seq = "".join(AA3[name] for _res, name in records)
        seq_id = zlib.crc32(seq.encode("ascii"))
        for pos, (res_seq, _name) in enumerate(records):
            out[(chain, res_seq)] = (seq_id, pos)
    return out


def attach_sequence_keys(
    profile: dict,
    env: Iterable,
    seq_map: dict[tuple[str, str], tuple[int, int]],
) -> dict:
    """Project a per-environment profile onto cross-structure residue keys.

    Residues are aggregated by ``(sequence id, position)``. Homodimer copies
    contribute additively -- weights and contact energies sum and the shell
    mask is the union -- so the result describes the sequence position, not a
    particular chain. Residues on chains absent from ``seq_map`` (never true
    for a parsed PDB) are dropped. ``env`` must be the residue list the
    profile vectors were computed over.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    names: dict[tuple[int, int], str] = {}
    for i, res in enumerate(env):
        key = seq_map.get((res.chain, res.res_seq))
        if key is None:
            continue
        groups.setdefault(key, []).append(i)
        names.setdefault(key, str(getattr(res, "res_name", "")).strip().upper())

    weights = np.asarray(profile["weights"], dtype=np.float64)
    contacts = np.asarray(profile["contacts"], dtype=np.float64)
    mask = np.asarray(profile["mask"], dtype=bool)
    keys = sorted(groups)
    out = dict(profile)
    out["seq_id"] = np.array([k[0] for k in keys], dtype=np.int64)
    out["seq_pos"] = np.array([k[1] for k in keys], dtype=np.int64)
    out["weights"] = np.array(
        [float(weights[groups[k]].sum()) for k in keys], dtype=np.float64)
    out["contacts"] = np.array(
        [float(contacts[groups[k]].sum()) for k in keys], dtype=np.float64)
    out["mask"] = np.array([bool(mask[groups[k]].any()) for k in keys],
                           dtype=bool)
    out["res_names"] = [names[k] for k in keys]
    return out


def _key_index(profile: dict) -> dict[tuple[int, int], int]:
    ids = np.asarray(profile["seq_id"])
    pos = np.asarray(profile["seq_pos"])
    return {(int(i), int(p)): n for n, (i, p) in enumerate(zip(ids, pos))}


def _weighted_pearson(x: np.ndarray, y: np.ndarray,
                      w: np.ndarray) -> float | None:
    """Weighted Pearson r; ``None`` when either side is constant."""
    total = float(w.sum())
    if total <= _EPS or x.size < 3:
        return None
    w = w / total
    mx = float((w * x).sum())
    my = float((w * y).sum())
    dx = x - mx
    dy = y - my
    vx = float((w * dx * dx).sum())
    vy = float((w * dy * dy).sum())
    if vx <= _EPS or vy <= _EPS:
        return None
    return float((w * dx * dy).sum() / np.sqrt(vx * vy))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > _EPS else 0.0


_EMPTY = {
    "n_shell": 0, "n_shared": 0,
    "shell_recall": 0.0, "shell_weight_recall": 0.0,
    "weight_cos": 0.0, "energy_r": None, "weighted_energy_r": None,
    "energy_r_fav": None, "energy_r_engaged": None,
    "weighted_energy_r_strength": None,
    "contact_retention": 0.0, "delta_energy": 0.0,
    "contact_jaccard": 0.0, "overlap_energy": 0.0,
    "all_weight_cos": 0.0, "all_energy_r": None,
    "all_weighted_energy_r": None,
}


def profile_similarity(q: dict, c: dict, *,
                       min_shared: int = 3) -> dict:
    """Compare a query profile with one library-fragment profile.

    All metrics are contact-local on the query's shell, except
    ``shell_recall`` (unweighted coverage) and ``shell_weight_recall``
    (query-weight coverage). The energy metrics use the shared residues only:

    - ``energy_r``: Pearson r between per-residue contact energies;
    - ``weighted_energy_r``: the same correlation with query relevance
      weights, so a disagreement on a residue the query depends on costs
      more;
    - ``contact_retention``: query-weight fraction of the query's favourable
      contacts that the candidate also engages favourably;
    - ``delta_energy``: query-weight mean of ``candidate - query`` contact
      energy (negative = the replacement loses interactions).

    ``None`` energies mean the query shell has too few shared residues or one
    side is constant. Rankers should map that to the bottom, not to zero:
    zero is a legitimate correlation.
    """
    c_pos = _key_index(c)
    q_shell_idx = np.flatnonzero(np.asarray(q["mask"], dtype=bool))
    if q_shell_idx.size == 0 or not c_pos:
        return dict(_EMPTY)

    q_ids = np.asarray(q["seq_id"])
    q_pos = np.asarray(q["seq_pos"])
    shared_q: list[int] = []
    shared_c: list[int] = []
    for qi in q_shell_idx:
        ci = c_pos.get((int(q_ids[qi]), int(q_pos[qi])))
        if ci is not None:
            shared_q.append(int(qi))
            shared_c.append(int(ci))
    n_shell = int(q_shell_idx.size)
    if not shared_q:
        out = dict(_EMPTY)
        out["n_shell"] = n_shell
        return out

    sq = np.asarray(shared_q, dtype=np.int64)
    sc = np.asarray(shared_c, dtype=np.int64)
    q_weights = np.asarray(q["weights"], dtype=np.float64)
    c_weights = np.asarray(c["weights"], dtype=np.float64)
    q_contacts = np.asarray(q["contacts"], dtype=np.float64)
    c_contacts = np.asarray(c["contacts"], dtype=np.float64)
    c_mask = np.asarray(c["mask"], dtype=bool)

    w = np.clip(q_weights[sq], 0.0, None)
    total = float(w.sum())
    if total <= _EPS:
        w = np.ones(sq.size, dtype=np.float64)
        total = float(w.size)
    wn = w / total

    qe = q_contacts[sq]
    ce = c_contacts[sc]
    # Contact-shell coverage counts residues the candidate actually touches,
    # not merely residues present in its 12 A environment.
    c_touch = c_mask[sc] & (ce > 0.0)
    recall = float(c_touch.sum() / n_shell) if n_shell else 0.0
    weight_recall = float((wn * c_touch).sum())
    q_fav = qe > 0.0
    retention = (float((wn[q_fav] * (ce[q_fav] > 0.0)).sum())
                 if q_fav.any() else 0.0)

    def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
        if x.size < min_shared or x.std() <= 1e-9 or y.std() <= 1e-9:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    energy_r = _corr(qe, ce)
    q_contact_pos = qe > 0.0
    c_contact_pos = ce > 0.0
    engaged = q_contact_pos | c_contact_pos
    energy_r_fav = _corr(qe[q_contact_pos], ce[q_contact_pos])
    energy_r_engaged = _corr(qe[engaged], ce[engaged])
    # Weight by the query's own interaction strength rather than by the
    # softmax relevance, which is nearly flat over a 12 A pocket.
    strength = np.clip(qe, 0.0, None)
    wer = _weighted_pearson(qe, ce, w)
    wer_strength = _weighted_pearson(qe, ce, strength)
    delta = float((wn * (ce - qe)).sum())
    contact_union = int((q_contact_pos | c_contact_pos).sum())
    contact_jaccard = (float(
        (q_contact_pos & c_contact_pos).sum() / contact_union)
        if contact_union else 0.0)
    er_for_overlap = 0.0 if energy_r_engaged is None else energy_r_engaged
    overlap_energy = contact_jaccard * (er_for_overlap + 1.0) / 2.0

    # Whole-environment variant: every shared residue, not just the query's
    # 5 A shell. Kept as an explicit comparison, not as the default, because
    # the placement prototype found the whole-pocket profile to be dominated
    # by the common pocket rather than by the fragment.
    all_q: list[int] = []
    all_c: list[int] = []
    for qi in range(len(q_ids)):
        ci = c_pos.get((int(q_ids[qi]), int(q_pos[qi])))
        if ci is not None:
            all_q.append(qi)
            all_c.append(ci)
    aq = np.asarray(all_q, dtype=np.int64)
    ac = np.asarray(all_c, dtype=np.int64)
    all_weight_cos = (_cosine(q_weights[aq], c_weights[ac])
                      if aq.size else 0.0)
    all_energy_r = (_corr(q_contacts[aq], c_contacts[ac])
                    if aq.size else None)
    all_wenergy = (_weighted_pearson(
        q_contacts[aq], c_contacts[ac],
        np.clip(q_weights[aq], 0.0, None) + _EPS)
        if aq.size else None)

    return {
        "n_shell": n_shell,
        "n_shared": int(sq.size),
        "shell_recall": recall,
        "shell_weight_recall": weight_recall,
        "weight_cos": _cosine(q_weights[sq], c_weights[sc]),
        "energy_r": energy_r,
        "weighted_energy_r": wer,
        "energy_r_fav": energy_r_fav,
        "energy_r_engaged": energy_r_engaged,
        "weighted_energy_r_strength": wer_strength,
        "contact_retention": retention,
        "delta_energy": delta,
        "contact_jaccard": contact_jaccard,
        "overlap_energy": overlap_energy,
        "all_weight_cos": all_weight_cos,
        "all_energy_r": all_energy_r,
        "all_weighted_energy_r": all_wenergy,
    }


def rank_score(metrics: dict, kind: str) -> float:
    """Scalar ordering for one metric family.

    ``energy`` / ``weighted_energy`` / ``combined`` put missing correlations
    below every real value (``-2``) instead of at zero, because a true zero
    correlation is still a measurement.
    """
    er = metrics.get("energy_r")
    wer = metrics.get("weighted_energy_r")
    if kind == "weight_cos":
        return float(metrics.get("weight_cos", 0.0))
    if kind == "energy":
        return -2.0 if er is None else float(er)
    if kind == "weighted_energy":
        return -2.0 if wer is None else float(wer)
    if kind == "energy_fav":
        value = metrics.get("energy_r_fav")
        return -2.0 if value is None else float(value)
    if kind == "energy_engaged":
        value = metrics.get("energy_r_engaged")
        return -2.0 if value is None else float(value)
    if kind == "weighted_energy_strength":
        value = metrics.get("weighted_energy_r_strength")
        return -2.0 if value is None else float(value)
    if kind == "contact_jaccard":
        return float(metrics.get("contact_jaccard", 0.0))
    if kind == "overlap_energy":
        return float(metrics.get("overlap_energy", 0.0))
    if kind == "all_weight_cos":
        return float(metrics.get("all_weight_cos", 0.0))
    if kind == "all_energy":
        value = metrics.get("all_energy_r")
        return -2.0 if value is None else float(value)
    if kind == "all_weighted_energy":
        value = metrics.get("all_weighted_energy_r")
        return -2.0 if value is None else float(value)
    if kind == "combined":
        er = -1.0 if er is None else float(er)
        wer = -1.0 if wer is None else float(wer)
        return 0.5 * (er + wer) + 0.5 * float(
            metrics.get("weight_cos", 0.0))
    if kind == "recall_energy":
        wer = -1.0 if wer is None else float(wer)
        return float(metrics.get("shell_weight_recall", 0.0)) * (wer + 1.0)
    if kind == "retention":
        return float(metrics.get("contact_retention", 0.0))
    if kind == "delta_energy":
        # More negative = more lost interaction; rank the least negative first.
        return -float(metrics.get("delta_energy", 0.0))
    raise ValueError(f"unknown profile score: {kind}")


def store_features(index_features: dict) -> dict:
    """The subset of index features a profile store was computed under."""
    keys = (
        "explicit_h", "backbone", "angles", "legacy_hbond",
        "energy_pooling", "contact_cutoff", "charge_method",
        "energy_scale", "content_scale", "temperature",
        "use_energy", "use_content", "env_radius", "d_lo", "d_hi",
        "fp_aggregate", "use_pair_bias",
        "min_da", "max_da", "max_frags", "max_blocks",
        "cut_biaryl", "cut_amide",
    )
    return {k: index_features[k] for k in keys if k in index_features}
