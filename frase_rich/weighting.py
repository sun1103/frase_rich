"""Shared residue weighting and triplet-to-fingerprint aggregation.

The library builder and the ref-mol query must use the *same* path here,
otherwise the two sides drift and the weighted overlap stops being symmetric:

* ``environment``      -- the 12 A CA sphere (membership is deliberately fixed)
* ``fragment_fp_weights`` -- per-triplet weights and their collapse to one
  value per distinct 33-bit fingerprint.

Nothing in this module writes the index; ``WeightedTripletIndex.add_fragment``
still rescales every fragment's fingerprint weights to mean 1, which is what
makes the all-uniform case reduce to plain Jaccard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chem import charges_for, with_explicit_hydrogens
from .features import relevance_weights
from .geometry import (
    N_BITS,
    canonical_triplet_fp,
    enumerate_triplet_idxs,
    seq_number,
    triplet_bias,
)
from .index import EnvResidue


def environment(residues: list, centroid, radius: float = 12.0) -> list:
    """Residues whose representative point lies within ``radius``.

    This is the fragment-centred membership used by the original FRASE
    convention (and by every index built before ``env_center="pocket"``).
    """
    if not residues:
        return []
    pts = np.stack([r.coords for r in residues]).astype(np.float32)
    d = np.linalg.norm(pts - np.asarray(centroid, dtype=np.float32), axis=1)
    return [r for r, dist in zip(residues, d) if dist <= radius]


def pocket_environment(residues: list, ligand_xyz, radius: float = 12.0) -> list:
    """Residues within ``radius`` of *any* ligand heavy atom.

    One residue set per complex, shared by every fragment and by the whole
    molecule. A union of spheres (rather than a sphere at the ligand centroid)
    keeps the peripheral subpockets that a wide ligand reaches; a centroid
    sphere would silently drop the residues around its ends.
    """
    if not residues:
        return []
    lig = np.asarray(ligand_xyz, dtype=np.float64).reshape(-1, 3)
    if lig.size == 0:
        return []
    pts = np.stack([r.coords for r in residues]).astype(np.float64)
    diff = pts[:, None, :] - lig[None, :, :]
    d2 = (diff * diff).sum(axis=-1).min(axis=1)
    mask = d2 <= radius * radius
    return [r for r, keep in zip(residues, mask.tolist()) if keep]


@dataclass
class TripletTable:
    """Residue triplets for one environment, enumerated once.

    A pocket-wide environment is reused by every fragment of a complex, so the
    enumeration and the residual-residue bias are computed once instead of per
    fragment. Arrays are indexed in the environment's residue order.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    fp: np.ndarray
    bias: np.ndarray
    n_res: int

    @property
    def empty(self) -> bool:
        return self.fp.size == 0


def triplet_table(
    env: list,
    *,
    d_lo: float = 8.0,
    d_hi: float = 12.0,
    use_pair_bias: bool = True,
) -> TripletTable:
    """Enumerate and fingerprint the environment's triplets once."""
    empty = TripletTable(
        np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64),
        np.zeros(0, np.int64), np.zeros(0, np.float64), len(env))
    if len(env) < 3:
        return empty

    pts = np.stack([r.coords for r in env]).astype(np.float64)
    tri = enumerate_triplet_idxs(pts, d_lo=d_lo, d_hi=d_hi)
    if not tri:
        return empty
    arr = np.asarray(tri, dtype=np.int64)
    a, b, c = arr[:, 0], arr[:, 1], arr[:, 2]

    codes = np.array([r.code for r in env], dtype=np.int64)
    ordered = np.sort(
        np.stack([codes[a], codes[b], codes[c]], axis=1), axis=1)[:, ::-1]
    fp = ((ordered[:, 0] << (2 * N_BITS))
          | (ordered[:, 1] << N_BITS) | ordered[:, 2])

    if use_pair_bias:
        diff = pts[:, None, :] - pts[None, :, :]
        dmat = np.sqrt((diff * diff).sum(axis=-1))
        chains = np.array([r.chain for r in env])
        seqs = np.array([seq_number(r.res_seq) for r in env], dtype=np.int64)
        same = chains[:, None] == chains[None, :]
        sep = np.abs(seqs[:, None] - seqs[None, :])
        seq_w = np.where(
            same, 0.6 + 0.4 * (1.0 - np.exp(-sep / 3.0)), 1.0)
        pair = seq_w * np.exp(-0.5 * ((dmat - 10.0) / 2.0) ** 2)
        bias = (pair[a, b] + pair[a, c] + pair[b, c]) / 3.0
    else:
        bias = np.ones(len(a), dtype=np.float64)

    return TripletTable(a, b, c, fp, bias, len(env))


def fragment_fp_weights(
    env: list,
    *,
    d_lo: float = 8.0,
    d_hi: float = 12.0,
    fp_aggregate: str = "mean",
    use_pair_bias: bool = True,
    table: TripletTable | None = None,
    residue_mask: np.ndarray | None = None,
) -> dict[int, float]:
    """Collapse in-band residue triplets to one weight per fingerprint key.

    ``env`` items carry ``coords``, ``code``, ``chain``, ``res_seq`` and a
    ``weight`` that sums to 1 over the environment. Averaging (rather than
    summing) over the residue triplets that produce the same key is required:
    summing would let triplet multiplicity act as a weight, so the all-off arm
    would no longer reproduce Jaccard.

    ``table`` skips enumeration when a complex's pocket is reused by many
    fragments; the environment order must match the table's.
    ``residue_mask`` (over the environment) keeps only triplets whose three
    residues are all selected -- the contact-local query variant. Keys that
    arose only from shell-external residues disappear.
    """
    if table is None:
        if len(env) < 3:
            return {}
        table = triplet_table(
            env, d_lo=d_lo, d_hi=d_hi, use_pair_bias=use_pair_bias)
    if table.empty:
        return {}

    weights = np.array([float(r.weight) for r in env], dtype=np.float64)
    if weights.shape[0] != table.n_res:
        raise ValueError("environment does not match triplet table")
    wt = (weights[table.a] + weights[table.b] + weights[table.c]) \
        / 3.0 * table.bias

    keep = None
    if residue_mask is not None:
        mask = np.asarray(residue_mask, dtype=bool)
        if mask.shape[0] != table.n_res:
            raise ValueError("residue_mask does not match triplet table")
        keep = mask[table.a] & mask[table.b] & mask[table.c]
        if not keep.any():
            return {}
        wt = wt[keep]

    keys, inv = np.unique(table.fp[keep] if keep is not None else table.fp,
                          return_inverse=True)
    if fp_aggregate == "max":
        vals = np.full(len(keys), -np.inf, dtype=np.float64)
        np.maximum.at(vals, inv, wt)
    elif fp_aggregate == "mean":
        counts = np.bincount(inv)
        vals = np.bincount(inv, weights=wt) / counts
    else:
        raise ValueError(f"unknown fp_aggregate: {fp_aggregate}")
    return dict(zip(keys.tolist(), vals.tolist()))


def fragment_fp_energy(
    env: list,
    energies,
    n_atoms: int,
    *,
    d_lo: float = 8.0,
    d_hi: float = 12.0,
    table: TripletTable | None = None,
    return_min_leg: bool = False,
    return_contact_mean: bool = False,
    contact_energies=None,
):
    """Per-fingerprint triplet energy, divided by fragment heavy-atom count.

    Uses the same residue triplets and keys as ``fragment_fp_weights``, but
    stores the underlying interaction energy instead of the softmax weight.
    The per-atom normalization is what makes two fragments comparable: raw
    energy grows with fragment size.

    ``mean`` and ``max`` aggregate the mean energy of the three legs over the
    triplets that hash to a key. ``cmean`` is the same aggregation on
    contact-only energies: the ungated proximity term is removed, so a key is
    positive only when some residue in its triangles genuinely contacts the
    fragment, and non-contacting legs contribute 0 rather than a floor.
    ``min_leg`` keeps, per key, the best triangle in which **all three**
    residues are genuinely in contact (max over triplets of the weakest leg):
    one non-contacting leg disqualifies the triangle. ``contact_energies``
    supplies scores with the always-on proximity term removed; without it,
    every residue in the pocket would count as contacting.

    Returns ``(mean, max)``, plus ``min_leg`` when ``return_min_leg`` and
    ``cmean`` when ``return_contact_mean`` (appended in that order).
    """
    n_out = 2 + int(return_min_leg) + int(return_contact_mean)
    out = tuple({} for _ in range(n_out))
    if n_atoms <= 0:
        return out
    if table is None:
        if len(env) < 3:
            return out
        table = triplet_table(env, d_lo=d_lo, d_hi=d_hi)
    if table.empty:
        return out

    e = np.asarray(energies, dtype=np.float64)
    if e.shape[0] != table.n_res:
        raise ValueError("energies do not match triplet table")
    legs = np.stack([e[table.a], e[table.b], e[table.c]], axis=1) / n_atoms
    avg = legs.mean(axis=1)

    keys, inv = np.unique(table.fp, return_inverse=True)
    counts = np.bincount(inv)
    e_mean = np.bincount(inv, weights=avg) / counts
    e_max = np.full(len(keys), -np.inf, dtype=np.float64)
    np.maximum.at(e_max, inv, avg)

    result = (dict(zip(keys.tolist(), e_mean.tolist())),
              dict(zip(keys.tolist(), e_max.tolist())))

    if contact_energies is None:
        gate = legs
    else:
        ce = np.asarray(contact_energies, dtype=np.float64)
        if ce.shape[0] != table.n_res:
            raise ValueError("contact energies do not match triplet table")
        gate = np.stack(
            [ce[table.a], ce[table.b], ce[table.c]], axis=1) / n_atoms

    if return_min_leg:
        # Best fully-engaged triangle per key. The proximity term must be
        # excluded, otherwise every residue in the 12 A pocket is "contacting".
        e_min = np.full(len(keys), -np.inf, dtype=np.float64)
        np.maximum.at(e_min, inv, gate.min(axis=1))
        result += (dict(zip(keys.tolist(), e_min.tolist())),)

    if return_contact_mean:
        e_cmean = np.bincount(inv, weights=gate.mean(axis=1)) / counts
        result += (dict(zip(keys.tolist(), e_cmean.tolist())),)

    return result


MARKS = {
    "mean": "_fp_e_atom_mean",
    "max": "_fp_e_atom_max",
    "min_leg": "_fp_e_atom_min",
    "cmean": "_fp_e_atom_cmean",
}


def normalize_marks(values) -> dict[int, float]:
    """Mean-1 scaling over the positive support of a per-key mark vector.

    Zero/negative keys carry no interaction information under a contact-gated
    mark; leaving them in the denominator would make a fragment look sparse.
    After scaling, a uniform vector is all ones, so ``mark_overlap`` reduces
    to exact-key Jaccard.
    """
    pos = {int(k): float(v) for k, v in values.items() if v > 0}
    if not pos:
        return {}
    scale = len(pos) / float(sum(pos.values()))
    return {k: v * scale for k, v in pos.items()}


def mark_overlap(q_marks, l_marks) -> float:
    """Weighted exact-key overlap of two normalized mark vectors.

    ``W = sum(min(q, l))`` over shared keys; the denominator is the weighted
    union ``|A| + |B| - W``, so uniform vectors give Jaccard exactly.
    """
    if not q_marks or not l_marks:
        return 0.0
    w = sum(min(qv, l_marks[k]) for k, qv in q_marks.items() if k in l_marks)
    return w / max(len(q_marks) + len(l_marks) - w, 1e-9)


def query_fp_weights(
    ref_mol,
    residues: list,
    *,
    env_radius: float = 12.0,
    d_lo: float = 8.0,
    d_hi: float = 12.0,
    temperature: float = 1.0,
    use_energy: bool = True,
    use_content: bool = True,
    pooling: str = "mean",
    contact_cutoff: float = 6.0,
    use_angles: bool = True,
    backbone: bool = True,
    energy_scale: float = 0.2,
    content_scale: float = 0.3,
    charge_method: str = "eem",
    strict_charges: bool = False,
    fp_aggregate: str = "mean",
    use_pair_bias: bool = True,
    legacy_hbond: bool = False,
    return_energies: bool = False,
) -> tuple[dict[int, float], list[EnvResidue], str] | tuple[
        dict[int, float], list[EnvResidue], str, dict[str, dict[int, float]]]:
    """Per-fingerprint weights for a reference-molecule query.

    The ref-mol is the query complex's parent ligand in its experimental pose.
    Membership is the same 12 A CA sphere around its centroid and the weights
    go through exactly the same ``relevance_weights`` / ``fragment_fp_weights``
    path as the library, so a uniform build matched by this query reproduces
    Jaccard. Returns ``(fp_weights, environment, charge_method_used)``.
    """
    molh = with_explicit_hydrogens(ref_mol)
    if charge_method == "eem":
        charges, method = charges_for(
            molh, method="eem", strict=strict_charges)
    else:
        charges, method = None, "formal"

    conf = molh.GetConformer()
    heavy = [a.GetIdx() for a in molh.GetAtoms() if a.GetAtomicNum() > 1]
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in heavy],
                   dtype=np.float32)
    centroid = xyz.mean(axis=0)

    env = environment(residues, centroid, env_radius)
    if len(env) < 3:
        return {}, [], method
    weights, energies, contacts = relevance_weights(
        molh, env, temperature=temperature,
        use_energy=use_energy, use_content=use_content,
        frag_charges=charges, pooling=pooling,
        contact_cutoff=contact_cutoff, use_angles=use_angles,
        backbone=backbone, energy_scale=energy_scale,
        content_scale=content_scale, legacy_hbond=legacy_hbond,
        return_energies=True, return_contact=True)
    env_res = [
        EnvResidue(
            res_name=r.res_name, chain=r.chain, res_seq=r.res_seq,
            coords=r.coords, code=r.code, weight=float(w))
        for r, w in zip(env, weights)
    ]
    raw = fragment_fp_weights(
        env_res, d_lo=d_lo, d_hi=d_hi, fp_aggregate=fp_aggregate,
        use_pair_bias=use_pair_bias)
    if return_energies:
        e_mean, e_max, e_min, e_cmean = fragment_fp_energy(
            env_res, energies, len(heavy), d_lo=d_lo, d_hi=d_hi,
            return_min_leg=True, return_contact_mean=True,
            contact_energies=contacts)
        return raw, env_res, method, {
            "mean": e_mean, "max": e_max, "min_leg": e_min,
            "cmean": e_cmean}
    return raw, env_res, method


def fp_weights_in_env(
    ref_mol,
    env: list,
    *,
    table: TripletTable | None = None,
    d_lo: float = 8.0,
    d_hi: float = 12.0,
    temperature: float = 1.0,
    use_energy: bool = True,
    use_content: bool = True,
    pooling: str = "mean",
    contact_cutoff: float = 6.0,
    use_angles: bool = True,
    backbone: bool = True,
    energy_scale: float = 0.2,
    content_scale: float = 0.3,
    charge_method: str = "eem",
    strict_charges: bool = False,
    fp_aggregate: str = "mean",
    use_pair_bias: bool = True,
    legacy_hbond: bool = False,
    exclude_donor_atoms: set[int] | None = None,
    frag_charges: np.ndarray | None = None,
    return_energies: bool = False,
    residue_mask: np.ndarray | None = None,
):
    """Weights for ``ref_mol`` scored against a caller-supplied environment.

    Used by the ref-fragment query: the pocket is fixed per complex (the same
    residue set the library fragments were scored on), and only the fragment's
    energies change. Unlike ``query_fp_weights`` this never re-centers the
    environment on the molecule.

    ``frag_charges`` skips the charge computation when the caller already has
    them, aligned to ``with_explicit_hydrogens(ref_mol)`` atom order. EEM
    charges depend on internal geometry but not on rigid pose, so pose
    enumeration computes them once per conformer instead of once per pose
    (EEM is ~240 ms through obabel, the rest of the call ~15 ms).
    """
    molh = with_explicit_hydrogens(ref_mol)
    if frag_charges is not None:
        charges, method = np.asarray(frag_charges, dtype=np.float64), \
            "precomputed"
    elif charge_method == "eem":
        charges, method = charges_for(
            molh, method="eem", strict=strict_charges)
    else:
        charges, method = None, "formal"

    if len(env) < 3:
        return ({}, [], method, {}) if return_energies else ({}, [], method)
    weights, energies, contacts = relevance_weights(
        molh, env, temperature=temperature,
        use_energy=use_energy, use_content=use_content,
        frag_charges=charges, exclude_donor_atoms=exclude_donor_atoms,
        pooling=pooling, contact_cutoff=contact_cutoff,
        use_angles=use_angles, backbone=backbone,
        energy_scale=energy_scale, content_scale=content_scale,
        legacy_hbond=legacy_hbond, return_energies=True,
        return_contact=True)
    env_res = [
        EnvResidue(
            res_name=r.res_name, chain=r.chain, res_seq=r.res_seq,
            coords=r.coords, code=r.code, weight=float(w))
        for r, w in zip(env, weights)
    ]
    raw = fragment_fp_weights(
        env_res, d_lo=d_lo, d_hi=d_hi, fp_aggregate=fp_aggregate,
        use_pair_bias=use_pair_bias, table=table, residue_mask=residue_mask)
    if not return_energies:
        return raw, env_res, method

    n_heavy = sum(1 for a in molh.GetAtoms() if a.GetAtomicNum() > 1)
    e_mean, e_max, e_min, e_cmean = fragment_fp_energy(
        env_res, energies, n_heavy, d_lo=d_lo, d_hi=d_hi, table=table,
        return_min_leg=True, return_contact_mean=True,
        contact_energies=contacts)
    return raw, env_res, method, {
        "mean": e_mean, "max": e_max, "min_leg": e_min, "cmean": e_cmean}
