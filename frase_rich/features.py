"""Library-side pharmacophore features and fragment-residue relevance.

This is where the rich information that the query cannot see is computed. The
index key stays the discrete 33-bit triplet; these continuous scores enter only
as per-triplet weights.

Feature axes follow the pocketfield convention so the two prototypes can be
compared directly:

    [hydrophobic, donor, acceptor, positive, negative, aromatic]

The complementarity pairing crosses donor/acceptor and positive/negative; like
pairs hydrophobic/aromatic. That is the same ``_PHARM_PAIR`` idea used by
``MOLGEN/pocketfield/attention.py``, re-implemented here without dependencies.

WHY explicit hydrogens: EEM does not keep a formal charge on its atom. A
quaternary ammonium comes out as N -0.97 with each N-H at +0.37, so the
attractive partners of the cation are the *hydrogens*. A heavy-atom-only
electrostatic term scores the salt bridge backwards. Hydrogens also make the
donor angle D-H...A computable at all; they never enter the hydrophobic,
aromatic, clash or proximity terms.

WHY mean pooling: summing pair contributions makes residue size the score. An
aromatic stack (36 matching pairs) beats a carboxylate contact (2-4 pairs)
regardless of quality. Each term is averaged over *matching pairs that are
actually in contact*; multi-contact reinforcement is deliberately dropped, and
``pooling="sum"`` keeps the old behaviour for A/B.

H-bonds are two-sided: D-H...A with D-A <= 3.5 A, H-A <= 2.6 A and donor angle
>= 120 deg, plus the acceptor angle Y-A...H (Y = acceptor's heavy neighbour,
best over Y) within the acceptor's hybridization-specific ideal (sp2 ~ 120 deg,
sp3 ~ 109.5 deg) and >= 90 deg.

The tanh scale (0.2) was calibrated on a 50-complex sample: mean pooling shrank
the term magnitudes by roughly an order of magnitude against the old sums
(median per-pair energy 0.073, p90 0.18, p99 0.96). Residue partial charges
remain table-based while fragment charges are EEM -- a mixed scheme, and a
documented approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BACKBONE = {"N", "CA", "C", "O", "OXT"}
DONOR_O = {"OG", "OG1", "OH"}
DONOR_S = {"SG"}
SP3_ACCEPTOR_NAMES = {"OG", "OG1", "OH", "SG", "SD"}

_AROMATIC_NAMES = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIP": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HID": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIE": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

_VDW_BY_ELEMENT = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
}
_VDW_BY_Z = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80,
    16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}

# fragment dim -> residue dim it complements (donor<->acceptor, pos<->neg).
_PHARM_PAIR = np.array([0, 2, 1, 4, 3, 5])

_HBOND_DA_MAX = 3.5
_HBOND_HA_MAX = 2.6
_HBOND_DONOR_ANGLE = 120.0
_HBOND_ACCEPTOR_ANGLE = 90.0


@dataclass
class AtomProps:
    xyz: np.ndarray
    radius: np.ndarray
    charge: np.ndarray          # electrostatic charge (EEM for fragments)
    formal: np.ndarray          # formal charge (pharmacophore axes)
    donor: np.ndarray
    acceptor: np.ndarray
    hydrophobic: np.ndarray
    aromatic: np.ndarray
    is_hydrogen: np.ndarray
    parent: np.ndarray          # local parent index for H, -1 otherwise
    donor_h: np.ndarray         # H atoms that can donate
    donor_rotatable: np.ndarray  # O-H / S-H: orientation from AddHs is arbitrary
    acc_ideal: np.ndarray       # ideal Y-A...H angle in degrees, nan if not acceptor
    neighbors: tuple = field(default=())

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def heavy(self) -> np.ndarray:
        return ~self.is_hydrogen


def empty_atom_props() -> AtomProps:
    z = np.zeros((0, 3))
    b = np.zeros(0, dtype=bool)
    return AtomProps(
        xyz=z, radius=np.zeros(0), charge=np.zeros(0), formal=np.zeros(0),
        donor=b, acceptor=b.copy(), hydrophobic=b.copy(), aromatic=b.copy(),
        is_hydrogen=b.copy(), parent=np.zeros(0, dtype=int),
        donor_h=b.copy(), donor_rotatable=b.copy(),
        acc_ideal=np.zeros(0), neighbors=(),
    )


def sidechain_atoms(atoms: list) -> list:
    """Heavy sidechain atoms only (backbone N/CA/C/O and hydrogens removed)."""
    return [
        a for a in atoms
        if a.name.strip().upper() not in BACKBONE
        and getattr(a, "element", "C").strip().upper() != "H"
    ]


def _residue_charges(res_name: str, atoms: list) -> np.ndarray:
    q = np.zeros(len(atoms), dtype=np.float64)
    rn = res_name.strip().upper()
    for idx, atom in enumerate(atoms):
        nm = atom.name.strip().upper()
        if rn == "ARG" and nm in {"NE", "NH1", "NH2"}:
            q[idx] = 1.0 / 3.0
        elif rn == "LYS" and nm == "NZ":
            q[idx] = 1.0
        elif rn in {"HIS", "HIP", "HID", "HIE"} and nm in {"ND1", "NE2"}:
            q[idx] = 0.5
        elif rn == "ASP" and nm in {"OD1", "OD2"}:
            q[idx] = -0.5
        elif rn == "GLU" and nm in {"OE1", "OE2"}:
            q[idx] = -0.5
    return q


def _residue_acceptor(rn: str, name: str, element: str,
                      has_h: bool) -> bool:
    if element == "O":
        return True
    if element == "S":
        return name in {"SG", "SD"}
    if element != "N":
        return False
    if name in BACKBONE:
        return False                 # amide N, never an acceptor (Pro too)
    if rn in {"ARG", "LYS", "HIP"}:
        return False                 # protonated / delocalized
    if rn in {"ASN", "GLN"} and name in {"ND2", "NE2"}:
        return False                 # amide N
    # His: the tautomer without H on that ring N is the acceptor; indole NE1
    # always carries H and is therefore excluded by has_h.
    return not has_h


def residue_atom_props(
    res_name: str,
    atoms: list,
    *,
    backbone: bool = True,
    legacy_hbond: bool = False,
) -> AtomProps:
    """Atom-level properties for one residue, including hydrogens.

    Legacy atom lists (no ``is_hydrogen``/``parent`` attributes) are still
    accepted: they simply have no hydrogens and no angles. ``legacy_hbond``
    restores the pre-rework donor/acceptor set so the angle-free ablation can
    reproduce v1 weights exactly.
    """
    rn = res_name.strip().upper()

    def is_h_atom(a) -> bool:
        # Legacy parsed atoms carry no is_hydrogen flag but do carry element H.
        return (bool(getattr(a, "is_hydrogen", False))
                or str(getattr(a, "element", "")).strip().upper() == "H")

    full = list(atoms)
    keep = [i for i, a in enumerate(full)
            if backbone or a.name.strip().upper() not in BACKBONE]
    if not keep:
        return empty_atom_props()
    local = {g: n for n, g in enumerate(keep)}

    # Hydrogens whose parent did not survive the backbone filter are dropped.
    final = []
    for g in keep:
        atom = full[g]
        if is_h_atom(atom):
            pg = int(getattr(atom, "parent", -1))
            if pg < 0 or pg not in local:
                continue
            final.append(g)
        else:
            final.append(g)
    keep = final
    if not keep:
        return empty_atom_props()
    local = {g: n for n, g in enumerate(keep)}
    sel = [full[g] for g in keep]

    xyz = np.array([[a.x, a.y, a.z] for a in sel], dtype=np.float64)
    elements = [getattr(a, "element", "C").strip().upper() for a in sel]
    names = [a.name.strip().upper() for a in sel]
    is_h = np.array([is_h_atom(a) for a in sel], dtype=bool)
    parent = np.array(
        [int(local.get(int(getattr(a, "parent", -1)), -1)) if is_h[i] else -1
         for i, a in enumerate(sel)],
        dtype=int)
    radius = np.array([_VDW_BY_ELEMENT.get(el, 1.7) for el in elements])

    heavy_charge = _residue_charges(rn, sel)
    charge = heavy_charge.copy()
    formal = heavy_charge.copy()

    h_children: dict[int, int] = {}
    for i, a in enumerate(sel):
        if is_h[i] and parent[i] >= 0:
            h_children[parent[i]] = h_children.get(parent[i], 0) + 1

    donor, acceptor, hydrophobic, aromatic = [], [], [], []
    for i, (el, nm) in enumerate(zip(elements, names)):
        has_h = h_children.get(i, 0) > 0
        if is_h[i]:
            donor.append(False)
            acceptor.append(False)
            hydrophobic.append(False)
            aromatic.append(False)
            continue
        explicit = bool(h_children)
        if legacy_hbond:
            # v1 rules: every N donates, carboxylate/hydroxyl O always
            # accepts, and no amide/backbone/tautomer filtering.
            donor.append(el == "N" or (el == "O" and nm in DONOR_O)
                         or (el == "S" and nm in DONOR_S))
            acceptor.append(
                el == "O"
                or (el == "N" and not (rn == "ARG"
                                       or (rn == "LYS" and nm == "NZ")))
                or (el == "S" and nm in {"SG", "SD"}))
        else:
            donor.append(el in {"N", "O", "S"} and (
                has_h if explicit else
                (el == "N" or (el == "O" and nm in DONOR_O)
                 or (el == "S" and nm in DONOR_S))))
            acceptor.append(_residue_acceptor(rn, nm, el, has_h))
        non_hyd_carbon = (
            (rn == "ARG" and nm == "CZ")
            or (rn in {"ASN", "GLN", "ASP", "GLU"} and nm in {"CG", "CD"})
        )
        hydrophobic.append(el in {"C", "S", "F", "CL", "BR", "I"}
                           and not non_hyd_carbon)
        aromatic.append(nm in _AROMATIC_NAMES.get(rn, set()))

    acc_ideal = np.full(len(sel), np.nan)
    for i, (el, nm) in enumerate(zip(elements, names)):
        if acceptor[i]:
            acc_ideal[i] = (109.5 if nm in SP3_ACCEPTOR_NAMES else 120.0)

    # Heavy neighbours for the acceptor angle (global -> local remap).
    neighbors = []
    for i, a in enumerate(sel):
        nb = getattr(a, "neighbors", ()) or ()
        mapped = tuple(local[g] for g in nb if g in local)
        neighbors.append(mapped)

    donor_h = np.array(
        [bool(is_h[i] and parent[i] >= 0
              and elements[parent[i]] in {"N", "O", "S"})
         for i in range(len(sel))], dtype=bool)
    donor_rotatable = np.array(
        [bool(donor_h[i] and elements[parent[i]] in {"O", "S"})
         for i in range(len(sel))], dtype=bool)

    return AtomProps(
        xyz=xyz, radius=radius, charge=charge, formal=formal,
        donor=np.array(donor, dtype=bool),
        acceptor=np.array(acceptor, dtype=bool),
        hydrophobic=np.array(hydrophobic, dtype=bool),
        aromatic=np.array(aromatic, dtype=bool),
        is_hydrogen=is_h, parent=parent, donor_h=donor_h,
        donor_rotatable=donor_rotatable,
        acc_ideal=acc_ideal, neighbors=tuple(neighbors),
    )


def attachment_atoms(fragment, ligand, tol: float = 1e-3) -> set[int]:
    """Fragment atoms that sit on a cut bond to the rest of the ligand.

    The fragmenter replaces each dummy with H and folds it into implicit H;
    ``AddHs`` later puts an H back along the cut bond. That H points into the
    removed part of the ligand, so it must not be treated as a donor.
    """
    conf_f = fragment.GetConformer()
    conf_l = ligand.GetConformer()
    f_heavy = [a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum() > 1]
    if not f_heavy:
        return set()
    f_xyz = np.array([list(conf_f.GetAtomPosition(i)) for i in f_heavy])
    l_xyz = np.array([list(conf_l.GetAtomPosition(i))
                      for i in range(ligand.GetNumAtoms())])
    d = np.linalg.norm(f_xyz[:, None, :] - l_xyz[None, :, :], axis=-1)
    nearest = np.argmin(d, axis=1)
    if np.any(d[np.arange(len(f_heavy)), nearest] > tol):
        return set()
    mapped = set(int(j) for j in nearest)
    out = set()
    for fi, lj in zip(f_heavy, nearest):
        atom = ligand.GetAtomWithIdx(int(lj))
        if any(nb.GetIdx() not in mapped for nb in atom.GetNeighbors()):
            out.add(int(fi))
    return out


def fragment_atom_props(
    mol,
    charges: np.ndarray | None = None,
    *,
    exclude_donor_atoms: set[int] | None = None,
    legacy_hbond: bool = False,
) -> AtomProps:
    """Atom-level properties for a fragment, hydrogens included.

    ``charges`` must be aligned to the molecule's atom order (EEM, computed on
    the hydrogen-added molecule); it defaults to formal charges.
    ``exclude_donor_atoms`` are cut-point atoms whose attached H are capping
    artifacts, not real donors. ``legacy_hbond`` restores the v1 sets (no S
    donors, negative N excluded rather than any N).
    """
    n = mol.GetNumAtoms()
    excluded = exclude_donor_atoms or set()
    if n == 0:
        return empty_atom_props()

    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])
    zs = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    is_h = zs == 1
    radius = np.array([_VDW_BY_Z.get(int(z), 1.70) for z in zs])
    formal = np.array([float(a.GetFormalCharge()) for a in mol.GetAtoms()])
    charge = (np.array(charges, dtype=np.float64)
              if charges is not None else formal.copy())

    parent = np.full(n, -1, dtype=int)
    neighbors: list[tuple[int, ...]] = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        heavy_nb = tuple(nb.GetIdx() for nb in atom.GetNeighbors()
                         if nb.GetAtomicNum() > 1)
        if atom.GetAtomicNum() == 1:
            nb = atom.GetNeighbors()
            parent[i] = nb[0].GetIdx() if nb else -1
            neighbors.append(())
        else:
            neighbors.append(heavy_nb)

    donor, acceptor, hydrophobic, aromatic, donor_h = [], [], [], [], []
    donor_rotatable = []
    acc_ideal = np.full(n, np.nan)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        z = atom.GetAtomicNum()
        if z == 1:
            donor.append(False)
            acceptor.append(False)
            hydrophobic.append(False)
            aromatic.append(False)
            p = parent[i]
            real_parent = p >= 0 and p not in excluded
            donor_h.append(real_parent and zs[p] in {7, 8, 16})
            donor_rotatable.append(real_parent and zs[p] in {8, 16})
            continue
        # includeNeighbors matters after AddHs: the added H are graph
        # neighbours with no stored H count.
        has_h = atom.GetTotalNumHs(includeNeighbors=True) > 0
        donor.append(z in ({7, 8} if legacy_hbond else {7, 8, 16}) and has_h)
        acc = z in {7, 8, 16} and atom.GetFormalCharge() <= 0
        acceptor.append(acc)
        hydrophobic.append(z in {6, 9, 16, 17, 35, 53})
        aromatic.append(bool(atom.GetIsAromatic()))
        donor_h.append(False)
        donor_rotatable.append(False)
        if acc:
            hyb = str(atom.GetHybridization())
            acc_ideal[i] = 109.5 if hyb == "SP3" else 120.0

    return AtomProps(
        xyz=xyz, radius=radius, charge=charge, formal=formal,
        donor=np.array(donor, dtype=bool),
        acceptor=np.array(acceptor, dtype=bool),
        hydrophobic=np.array(hydrophobic, dtype=bool),
        aromatic=np.array(aromatic, dtype=bool),
        is_hydrogen=is_h, parent=parent,
        donor_h=np.array(donor_h, dtype=bool),
        donor_rotatable=np.array(donor_rotatable, dtype=bool),
        acc_ideal=acc_ideal, neighbors=tuple(neighbors),
    )


def _property_vector(props: AtomProps) -> np.ndarray:
    heavy = props.heavy
    if not np.any(heavy):
        return np.zeros(6)
    return np.array([
        float(props.hydrophobic[heavy].mean()),
        min(1.0, float((props.donor & heavy).sum()) / 2.0),
        min(1.0, float((props.acceptor & heavy).sum()) / 2.0),
        1.0 if np.any(props.formal[heavy] > 0.01) else 0.0,
        1.0 if np.any(props.formal[heavy] < -0.01) else 0.0,
        float(props.aromatic[heavy].mean()),
    ])


def residue_property_vector(res_name: str, atoms: list,
                            *, backbone: bool = True) -> np.ndarray:
    return _property_vector(residue_atom_props(
        res_name, atoms, backbone=backbone))


def fragment_property_vector(mol) -> np.ndarray:
    return _property_vector(fragment_atom_props(mol))


def complementarity(frag_vec: np.ndarray, res_vec: np.ndarray) -> float:
    """Cosine after crossing donor/acceptor and pos/neg axes; in [-1, 1]."""
    paired = np.asarray(res_vec, dtype=np.float64)[_PHARM_PAIR]
    frag = np.asarray(frag_vec, dtype=np.float64)
    denom = float(np.linalg.norm(frag) * np.linalg.norm(paired))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(frag, paired) / denom)


def _rbf(d: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((d - mu) / sigma) ** 2)


def _hbond_contributions(donor: AtomProps,
                         acceptor: AtomProps) -> list[float]:
    """D-H...A contributions with both angles; only passing contacts."""
    if not np.any(donor.donor_h) or not np.any(acceptor.acceptor):
        return []
    h_idx = np.flatnonzero(donor.donor_h)
    a_idx = np.flatnonzero(acceptor.acceptor)
    out: list[float] = []
    for i in h_idx:
        p = donor.parent[i]
        d_pos = donor.xyz[p]
        h_pos = donor.xyz[i]
        rotatable = bool(donor.donor_rotatable[i])
        for j in a_idx:
            a_pos = acceptor.xyz[j]
            d_da = float(np.linalg.norm(d_pos - a_pos))
            if d_da > _HBOND_DA_MAX or d_da < 1e-6:
                continue

            if rotatable:
                # The H orientation from AddHs is arbitrary on a hydroxyl or
                # thiol; let it rotate onto the D...A axis. No donor-angle
                # penalty, and H...A follows from the D-A distance.
                d_ha = max(0.8, d_da - 1.0)
                if d_ha > _HBOND_HA_MAX:
                    continue
                fd = 1.0
                h_dir = (d_pos - a_pos) / d_da   # A -> H
            else:
                ha = h_pos - a_pos
                d_ha = float(np.linalg.norm(ha))
                if d_ha > _HBOND_HA_MAX or d_ha < 1e-6:
                    continue
                v1 = d_pos - h_pos
                n1 = float(np.linalg.norm(v1))
                if n1 < 1e-6:
                    continue
                cos_d = float(np.dot(v1, -ha) / (n1 * d_ha))
                donor_angle = float(
                    np.degrees(np.arccos(np.clip(cos_d, -1, 1))))
                if donor_angle < _HBOND_DONOR_ANGLE:
                    continue
                fd = cos_d * cos_d
                h_dir = ha / d_ha

            ideal = acceptor.acc_ideal[j]
            if not np.isfinite(ideal):
                continue
            best = 0.0
            for y in acceptor.neighbors[j]:
                vy = acceptor.xyz[y] - a_pos
                ny = float(np.linalg.norm(vy))
                if ny < 1e-6:
                    continue
                cos_a = float(np.dot(vy, h_dir) / ny)
                acc_angle = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
                if acc_angle < _HBOND_ACCEPTOR_ANGLE:
                    continue
                c = float(np.cos(np.radians(acc_angle - ideal)))
                best = max(best, c * c)
            if best <= 0.0:
                continue
            out.append(float(_rbf(np.array(d_ha), 2.0, 0.6)) * fd * best)
    return out


def interaction_energy(
    frag: AtomProps,
    res: AtomProps,
    *,
    pooling: str = "mean",
    contact_cutoff: float = 6.0,
    use_angles: bool = True,
    return_contact: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """Smooth fragment-residue interaction score and minimum distance.

    Positive = favourable, negative = clash. With ``pooling="mean"`` every
    term is a per-contact average over matching pairs within ``contact_cutoff``
    (H-bonds and clashes use their own contact rules) so residue size cannot
    act as a weight; ``"sum"`` reproduces the old extensive behaviour.

    ``proximity`` is an ungated, always-positive distance term; it keeps the
    softmax from zeroing far residues but is *not* evidence of a contact.
    ``return_contact`` also returns the score without it, for gates that must
    mean "this residue actually touches the fragment".
    """
    if len(frag) == 0 or len(res) == 0:
        return (0.0, 99.0, 0.0) if return_contact else (0.0, 99.0)
    if pooling not in {"mean", "sum"}:
        raise ValueError(f"unknown pooling: {pooling}")

    d = np.linalg.norm(frag.xyz[:, None, :] - res.xyz[None, :, :], axis=-1)
    heavy_pair = frag.heavy[:, None] & res.heavy[None, :]
    dmin = float(d[heavy_pair].min()) if np.any(heavy_pair) else 99.0

    def pool(vals: np.ndarray, mask: np.ndarray) -> float:
        if pooling == "mean":
            contacts = mask & (d <= contact_cutoff)
            return float(vals[contacts].mean()) if np.any(contacts) else 0.0
        return float(vals[mask].sum())

    if use_angles:
        hb_vals = (_hbond_contributions(frag, res)
                   + _hbond_contributions(res, frag))
        hb = (float(np.mean(hb_vals)) if pooling == "mean"
              else float(np.sum(hb_vals))) if hb_vals else 0.0

        qq = frag.charge[:, None] * res.charge[None, :]
        qa = _rbf(d, 4.0, 1.5)
        signed = np.where(qq < 0, np.abs(qq) * qa, -0.5 * qq * qa)
        charge = pool(signed, np.abs(qq) > 1e-8)
    else:
        # Legacy heavy-atom, distance-only path (exact A/B reproduction).
        hb_mask = ((frag.donor[:, None] & res.acceptor[None, :])
                   | (frag.acceptor[:, None] & res.donor[None, :]))
        hb_mask &= heavy_pair
        hb = pool(_rbf(d, 2.9, 0.7), hb_mask)

        qq = frag.charge[:, None] * res.charge[None, :]
        qa = _rbf(d, 4.0, 1.5)
        signed = np.where(qq < 0, np.abs(qq) * qa, -0.5 * qq * qa)
        charge = pool(signed, heavy_pair & (np.abs(qq) > 1e-8))

    hyd_mask = (frag.hydrophobic[:, None] & res.hydrophobic[None, :]
                & heavy_pair)
    hydrophobic = pool(0.15 * _rbf(d, 3.9, 0.8), hyd_mask)

    ar_mask = frag.aromatic[:, None] & res.aromatic[None, :] & heavy_pair
    aromatic = pool(0.15 * _rbf(d, 4.2, 1.0), ar_mask)

    overlap = np.maximum(
        0.0, 0.85 * (frag.radius[:, None] + res.radius[None, :]) - d)
    overlap = overlap * heavy_pair
    clash_mask = overlap > 0
    if pooling == "mean":
        clash = (0.3 * float((overlap[clash_mask] ** 2).mean())
                 if np.any(clash_mask) else 0.0)
    else:
        clash = 0.3 * float((overlap ** 2).sum())

    contact = hb + charge + hydrophobic + aromatic - clash
    proximity = 0.1 * float(_rbf(np.array(dmin), 5.0, 2.0))
    if return_contact:
        return contact + proximity, dmin, contact
    return contact + proximity, dmin


def relevance_weights(
    frag_mol,
    env_residues: list,
    *,
    temperature: float = 1.0,
    use_energy: bool = True,
    use_content: bool = True,
    frag_charges: np.ndarray | None = None,
    exclude_donor_atoms: set[int] | None = None,
    pooling: str = "mean",
    contact_cutoff: float = 6.0,
    use_angles: bool = True,
    backbone: bool = True,
    energy_scale: float = 0.2,
    content_scale: float = 0.3,
    legacy_hbond: bool = False,
    return_energies: bool = False,
    return_contact: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """Softmax relevance of each environment residue to the fragment.

    ``env_residues`` items need ``res_name`` and ``atoms``; items carrying
    ``variants`` (histidine HID/HIE) are scored by their best variant, since
    only the tautomer that can form the contact should count. Softmax is
    deliberate: it is the attention normalization, and it keeps every residue
    in play (no hard cutoff). ``legacy_hbond`` selects the v1 donor/acceptor
    sets; it exists only for the exact A/B control, not for the angle ablation.
    """
    if not env_residues:
        return np.zeros(0)

    frag_props = fragment_atom_props(
        frag_mol, charges=frag_charges,
        exclude_donor_atoms=exclude_donor_atoms,
        legacy_hbond=legacy_hbond)
    frag_vec = _property_vector(frag_props)

    raw = []
    energies = []
    contacts = []
    for res in env_residues:
        res_vec = None
        energy = 0.0
        contact = 0.0
        variants = getattr(res, "variants", None) or [res.atoms]
        for atoms in variants:
            props = residue_atom_props(
                res.res_name, atoms, backbone=backbone,
                legacy_hbond=legacy_hbond)
            if res_vec is None:
                res_vec = _property_vector(props)
            if use_energy:
                e, _, c = interaction_energy(
                    frag_props, props, pooling=pooling,
                    contact_cutoff=contact_cutoff, use_angles=use_angles,
                    return_contact=True)
                energy = max(energy, e)
                contact = max(contact, c)
        score = 0.0
        if use_energy:
            score += 0.7 * np.tanh(energy / energy_scale)
        if use_content:
            score += content_scale * complementarity(frag_vec, res_vec)
        raw.append(score)
        energies.append(energy)
        contacts.append(contact)

    raw_arr = np.asarray(raw, dtype=np.float64)
    shifted = raw_arr - raw_arr.max()
    weights = np.exp(shifted / max(temperature, 1e-6))
    total = weights.sum()
    weights = (weights / total if total > 0
               else np.full(len(weights), 1.0 / len(weights)))
    if return_energies:
        if return_contact:
            return (weights, np.asarray(energies, dtype=np.float64),
                    np.asarray(contacts, dtype=np.float64))
        return weights, np.asarray(energies, dtype=np.float64)
    return weights
