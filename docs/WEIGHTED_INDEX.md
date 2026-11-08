# Weighted FRASE triplet index

## Pipeline

Two phases. The library is built once and stores `(33-bit key -> weight)` per
fragment; every query is reduced to the same structure before scoring.

```
INPUTS
  work/selected.tsv            pdb_id -> parent ligand SMILES
  rcsb_cache/<pid>.pdb         cached structures
  pdbbind_selected.tsv         PDBbind alternative (11,033 complexes)
  pdbbind_cache/<pid>.pdb      protein + HETATM LIG, made by
                               scripts/pdbbind_cache.py

+---------------------- BUILD (once) -- build.py ----------------------------+
|                                                                            |
|  block --> parse_pdb ----------------------+                               |
|    |                                       | fallback: _group_residues     |
|    +--> protein_residues                   | (heavy atoms, no H, no angles)|
|    |    explicit H, pH-7 charges,          |                               |
|    |    HID/HIE variants                   |                               |
|    |                                       |                               |
|    +--> _find_ligand(parent SMILES) --> parent ligand, deposited pose     |
|                   |                                                        |
|                   v                                                        |
|         fragment_ligand: all connected 50-300 Da pieces,                   |
|         H-capped, coordinates kept in the parent frame                     |
|                   |                                                        |
|                   v  per complex, one batch                                |
|         AddHs(addCoords=True) --> obabel EEM --> charges                   |
|         (mapped back by nearest coordinate, offset to formal net charge)   |
|                   |                                                        |
|                   v  per fragment                                          |
|  frag.centroid --> environment(residues, 12 A CA sphere)                   |
|                   |                                                        |
|                   v                                                        |
|  relevance_weights                                                         |
|    + fragment_atom_props  (heavy + H, donor/acceptor, partial charges)     |
|    + residue_atom_props   (heavy + H, backbone, best HID/HIE variant)      |
|    + interaction_energy   per residue:                                     |
|    |    H-bond RBF * cos^2(donor) * cos^2(acceptor - ideal)                |
|    |    + charge + hydrophobic + aromatic + proximity - clash              |
|    |    each term mean-pooled over contacting pairs (6 A)                  |
|    + raw_r = 0.7*tanh(E/0.2) + 0.3*complementarity(v_f, v_r)               |
|    + softmax(raw / 1.0)  --> residue weights, sum = 1                      |
|                   |                                                        |
|                   v                                                        |
|  fragment_fp_weights                                                       |
|    enumerate triplets: all 3 CA sides in [8, 12] A                         |
|    key    = sorted(11-bit residue code x 3)   <- no distance bits          |
|    w_trip = mean(w_i, w_j, w_k) * triplet_bias(seq sep, chain, distance)   |
|    aggregate per key by mean                                               |
|                   |                                                        |
|                   v                                                        |
|  add_fragment: rescale fp weights to mean 1                                |
|                postings: key -> [(frag_id, weight)]                        |
+----------------------------------------------------------------------------+
                   |
                   v
        rich_mean.pkl   (schema v2 + features dict)

+---------------- QUERY (per query) -- evaluate.py / index.query ------------+
|                                                                            |
|   stored mode                            refmol mode                       |
|   -----------                            -----------                       |
|   fragment's _env_coords + _env_codes    parent ligand @ deposited pose    |
|   (exactly what the builder saw)         + query complex residues          |
|         |                                        |                         |
|         |                                  AddHs + EEM                     |
|         |                                        |                         |
|         |                                  environment (12 A centroid)     |
|         |                                  relevance_weights   <- same     |
|         |                                  fragment_fp_weights  <- code    |
|         v                                        v                         |
|   enumerate triplets -> key set          per-key query weights (mean 1)    |
|         +----------------------+-----------------+                         |
|                                v                                           |
|   candidates: postings lookup on exact 33-bit keys                         |
|                                |                                           |
|   accumulate per candidate:    |                                           |
|     stored: n_match = #shared, W = sum of library weights                  |
|     refmol: W = sum min(w_q, w_l)   (or sum sqrt(w_q * w_l))               |
|                                |                                           |
|   score_jaccard  = n_match / (n_q + n_l - n_match)   <- nested baseline    |
|   score_weighted = W       / (n_q + n_l - W)                               |
|                                |                                           |
|   exclude own PDB -> sort (w, n_match, frag_id) -> first positive          |
+----------------------------------------------------------------------------+
```

Three invariants hold the flow together:

1. **All geometry is computed inside one PDB frame.** Distance appears only in
   the `[8, 12] A` triplet gate, `triplet_bias`, and `interaction_energy`, and
   each is evaluated on one side's own coordinates. What crosses between the
   sides is only `(key -> weight)` plus the key counts, which is why fragments
   from arbitrarily different crystal frames are comparable.
2. **The key carries no distance.** It is three 11-bit residue-property codes
   sorted into 33 bits; distance only decides whether a triplet exists and how
   much bias it gets.
3. **Both sides share `environment` / `relevance_weights` /
   `fragment_fp_weights` and normalize to mean 1.** `index.features` copies
   the build configuration into the query (the only override is
   `--allow-feature-mismatch`), so the all-uniform arm is exactly Jaccard.

### PDBbind input

`scripts/pdbbind_cache.py` turns the PDBbind P-L tree (per complex:
`{pid}_protein.pdb` + `{pid}_ligand.sdf`) into the single-PDB form the builder
and the ref-frag query provider expect: protein `ATOM` records plus one
`HETATM` `LIG` group from the SDF, and a `pdb_id -> ligand_smiles` TSV.

PDBbind is not small-molecule-curated, so the converter applies the upstream
BindingDB selection filter with a relaxed MW/logP cutoff: MW <= 700,
logP <= 5.5, HBD <= 5, HBA <= 10, has carbon, no phosphorus, allowed elements
only (H C N O F P S Cl Br I; drops metals, boron, selenium), and not
peptide-like (>= 2 peptide amide motifs or >= 2 acyclic amides). Of 19,037
PDBbind complexes, 11,033 pass (9,780 at the original 500/5.0 cutoffs).

The full pocket build (`--env-center pocket`) of that selection indexes 10,292
complexes, 184,186 fragments and 31.4M stored (fragment, key) postings over
680 unique fingerprints (skipped: 1,000 protein fallbacks, 16 ligand parses,
725 fragment sets; charges: EEM 184,130 / Gasteiger 56). The builder's
`_n_triplets` counter is the number of distinct keys per fragment, not the
number of enumerated triplets. Same-protein grouping
(`scripts/protein_groups.py`) finds 1,601 groups with >= 2 complexes covering
6,940 complexes. Results for the unified-pocket ref-frag query on this index
are in `REFMOL_QUERY.md`.

### Union with the RCSB/BindingDB cache

The RCSB side is built the same way in four disjoint shards over the cached
structures minus the PDBbind ids; `scripts/merge_indexes.py` then appends the
shards to the PDBbind index (features must match; complexes already in the
base are dropped fragment-for-fragment, ids are otherwise disjoint). The
merged index has 334,002 fragments over 16,663 complexes and 53.5M
(fragment, key) postings (PDBbind 184,186/10,292 plus 149,816/6,371 from the
shards; 680 unique fingerprints in both). Same-protein grouping over the
union: 19,219 grouped structures in 11,438 groups, 2,420 groups with >= 2
complexes covering 10,201 complexes.

## Motivation

The original FRASE index stores a discrete 33-bit key per residue triplet and
ranks fragments by plain Jaccard overlap. Everything the library side knows
beyond a residue's identity is discarded: the environment is a CA sphere, and
the scoring only asks whether the same three residue codes re-occur.

The query side cannot be enriched -- `orchestrate_v2.py` only has a sub-pocket
center and residue codes. But the library side is built once and can afford
expensive reasoning about the real fragment pose. This prototype exploits that
asymmetry: **rich library templates, coarse query probes.**

## Borrowed decomposition

The attention score

```
Score_ij = q_i^T k_j / sqrt(d) + Bias_topo(i,j) + Linear(RBF(||x_i - x_j||))
```

is used for its *decomposition*, not its learning. In the index:

| Term | Library-side realization | Query-side realization |
|---|---|---|
| content `q^T k` | fragment pharmacophore vector vs residue sidechain vector, donor/acceptor and +/- axes crossed | residue-code match (the discrete fingerprint) |
| pair bias | sequence separation + chain relation + in-band distance RBF between environment residues | not applied in the prototype, but *computable*: query residues carry chain, res_seq and CA coords |
| RBF potential | H-bond / charge / hydrophobic / aromatic / clash score between fragment atoms and sidechain atoms | not applied (rough query) |

No learned parameters. The first and third terms select and weight the indexed
environment; the query simply supplies keys.

## Index math

For fragment `f`, environment residues `r`:

1. `raw_r = 0.7 * tanh(E_r / energy_scale)
   + content_scale * complementarity(v_f, v_r)`
   where `E_r` is the mean-pooled interaction energy and `v` are 6-dim
   pharmacophore vectors. Defaults: `energy_scale = 0.2`,
   `content_scale = 0.3` (see *Energy model* below for their calibration).
2. `w_r = softmax(raw_r / T)`, summing to 1 over residues (no hard cutoff).
3. For each triplet `(i,j,k)`:
   `w_triplet = mean(w_i, w_j, w_k) * triplet_bias(i,j,k)`.
4. **Average** `w_triplet` over the residue triplets that produce each distinct
   33-bit fingerprint, then rescale the fragment's fingerprint weights to
   **mean 1**. Averaging (not summing) is required for the nested baseline:
   several triplets can hash to the same key, and summing would let that
   multiplicity act as a weight. With uniform residue weights every distinct
   fingerprint then gets exactly 1, and the weighted score reduces to Jaccard.

Query:

```
W_match = sum of library weights over shared fingerprints
score   = W_match / (n_query + n_library - W_match)
```

When all weights are 1 this is exactly the original Jaccard, so the A/B has a
strictly nested baseline.

## Energy model (v2)

The default build now uses the fragment's full chemistry, not a heavy-atom
approximation:

* **Charges.** Fragment partial charges come from Open Babel's classic
  electronegativity equalization (EEM), run once per complex on all fragments
  with explicit hydrogens. The batch output is mapped back by nearest 3D
  coordinate with a bijection check and offset to each fragment's formal net
  charge. A failure falls back to Gasteiger and records `charge_method` in the
  fragment metadata. Residue charges remain a residue/atom-name table (mixed
  scheme, documented approximation).
* **Hydrogens.** `AddHs(addCoords=True)` on fragments and residues. Hydrogens
  participate in electrostatics and H-bond geometry only; hydrophobic,
  aromatic and clash masks stay on heavy atoms.
* **Backbone.** N, CA, C, O are included by default. GLY has no sidechain, so
  without backbone it could never receive a weight.
* **H-bonds, both angles.** Donor D-H...A requires D-A <= 3.5 A and
  H...A <= 2.6 A, a donor angle D-H...A >= 120 deg, and an acceptor angle
  Y-A...H >= 90 deg for at least one heavy neighbour Y. The contribution is
  `RBF(d_HA; 2.0, 0.6) * cos^2(donor angle) * cos^2(acceptor angle - ideal)`,
  using 120 deg for sp2 and 109.5 deg for sp3 acceptors. Hydroxyl and thiol
  donors are treated as freely rotatable (the AddHs orientation is arbitrary):
  the H is placed along D...A and no donor-angle penalty is applied.
* **Histidine tautomers.** RDKit's PDB reader assigns every HIS as HIE. The
  builder does not accept that default: where the PDB has no hydrogen it
  enumerates HID and HIE and keeps the better H-bond geometry per residue.
  HIP is assigned only from the explicit protonation state.
* **Intensive pooling.** Each energy term is the **mean** of its contributions
  over atom pairs that match the term and are in contact (6 A cutoff; H-bonds
  and clashes use their own rules). This removes the size bias: a 36-pair
  aromatic stack no longer outscores a 2-pair salt bridge by pair count.
  `--energy-pooling sum` reproduces the old extensive behaviour.

### Calibration

Mean pooling divides every term by its contacting-pair count, and EEM shrinks
the per-atom charge magnitudes while redistributing the same total charge.
Neither is a simple constant factor, so the coefficients were refit on 50
complexes rather than rescaled by hand. With 4,154 contacting pairs, the
mean-pooled energy has median 0.073, p90 0.179, p95 0.257, p99 0.96
(max 1.168); complementarity has median 0.673. `energy_scale = 0.2` and
`content_scale = 0.3` keep the residue-weight distribution spread without
saturating `tanh`.

### Legacy A/B reproduction

`--no-explicit-h --no-backbone --no-angles --legacy-hbond
--energy-pooling sum --energy-scale 2.0 --content-scale 0.5
--temperature 0.1` reproduces the v1 index exactly
(max env-weight difference 6.7e-16, max fingerprint-weight difference
8.0e-15 over 17,934 posting entries). This is the control arm: any difference
in the new numbers comes from the chemistry and pooling, not from a changed
pipeline.

## Why mean-1 normalization

Attention softmax produces a distribution; the weighted index instead needs a
quantity comparable to a fingerprint count. Mean-1 normalization makes
`W_match` directly comparable to `n_match` and to the query's unweighted mass,
so the weighted Jaccard keeps the same union semantics as the original.

## Environment membership

The default builder uses residues with CA within 12 A of the **fragment
centroid**. This matches the original FRASE convention, and it was kept while
the weights were added so that the chemistry A/B isolates the weights, not the
membership.

`--env-center pocket` selects the alternative used by the ref-frag query: the
union of 12 A spheres around every parent-ligand heavy atom, computed once per
complex and shared by all of its fragments. The union (not a sphere at the
ligand centroid) keeps peripheral subpockets. This removes the per-fragment
membership confound, at the cost of one large triplet table per complex
(enumerated once and reused). `evaluate.py --query-mode reffrag` requires this
mode; see `REFMOL_QUERY.md` for the comparison.

## Evaluation protocol

`evaluate.py` runs leave-one-complex-out retrieval in two query modes:

* `--query-mode stored` (default): the query is the indexed fragment's stored
  CA coords + residue codes -- the coarse probe. This is the mode reported in
  the v1 findings below.
* `--query-mode refmol`: the query is the query complex's parent ligand in its
  deposited pose. Its environment is re-derived with the same rich feature
  code and matched by weighted exact-key overlap; the all-off arm is nested
  inside the same candidate list. See `REFMOL_QUERY.md`.

Common to both:

* queries = fragments with at least one eligible positive (see mode below);
* candidates from the query's own PDB id are excluded;
* positives are the same fragment SMILES from a different complex, optionally
  restricted by protein identity (`--positive-mode`):
  * `same-protein`: another PDB entry with the same longest-chain sequence
    (grouped by `frase_rich/proteins.py`, map from
    `scripts/protein_groups.py`). This is the meaningful transfer test.
  * `cross-protein`: same fragment in an unrelated protein. In a diverse
    sample this is what `any` becomes, and it is close to unretrievable
    (MRR ~0.06) because the environments need not agree.
  * `any`: no protein constraint.
* metrics: MRR, Recall@{10,50,100}, median rank, paired wins/losses.

Ranking fairness: hits are ordered with a deterministic `frag_id` tie-break
and 12-decimal keys, so the Jaccard column does not depend on the weighted
ordering or on float noise. The all-off arm must return exactly the Jaccard
column; it does (416/416 ties on the smoke build).

## Findings (2026-10-19, 500-complex same-protein sample)

These are the **v1** numbers: formal charges, no hydrogens, sidechain only,
distance-only H-bonds, sum pooling. They are kept as the control for the v2
runs; the v2 results are in `REFMOL_QUERY.md` and below.

Build: 447 complexes indexed from the 18 largest exact-sequence groups,
6,385 fragments, 268,568 triplets. Evaluation: same-protein positives,
968 scored queries, mean 5,506 candidates.

| arm | MRR | R@10 | R@50 | paired W/J |
|---|---|---|---|---|
| Jaccard | 0.4066 | 0.6818 | 0.8585 | — |
| rich T=1.0 | 0.4088 | 0.6880 | 0.8636 | 212 / 208 |
| rich T=2.0 | 0.4083 | 0.6829 | 0.8626 | 214 / 191 |
| rich T=0.5 | 0.3989 | 0.6725 | 0.8616 | 216 / 255 |
| no energy, T=1 | 0.4101 | 0.6798 | 0.8647 | 220 / 183 |
| no content, T=1 | 0.4048 | 0.6860 | 0.8647 | 208 / 201 |
| no pair bias, T=1 | 0.4041 | 0.6860 | 0.8626 | 204 / 212 |
| IDF (diagnostic) | 0.3901 | 0.6591 | 0.8585 | 174 / 253 |
| random weights σ=0.2 (seed 0 / 1) | 0.4107 / 0.3950 | 0.6818 / 0.6787 | 0.8636 / 0.8647 | — |
| random weights σ=0.5 (seed 2) | 0.3781 | 0.6653 | 0.8657 | 275 / 383 |

Random per-fingerprint noise at the same log-scale as the rich weights
(σ=0.2; the rich model's median log-weight std is 0.21) brackets the rich
model's MRR. The weighting effect is therefore inside the noise floor, and
sharpening (T=0.5) is consistently harmful. Cross-protein positives, for
contrast: Jaccard MRR 0.0629, weighted 0.0639, median rank ~340.

### v2 findings (2026-10-20, same 447 complexes, 6,385 fragments)

Build: EEM charges, explicit H, backbone, HID/HIE enumeration, either mean or
sum pooling, T=1.0. Stored coarse query, 968 scored queries, 5,506 candidates;
the candidate set and the Jaccard column are identical across these builds
because chemistry only affects weights. Every build has 268,568 triplets and
640 unique fingerprints.

| arm | MRR | R@10 | R@50 | paired W/J (ties) |
|---|---|---|---|---|
| Jaccard (nested baseline) | 0.4066 | 0.6818 | 0.8585 | — |
| EEM + H + backbone + angles, mean pool | **0.4101** | 0.6890 | 0.8667 | 208 / 198 (562) |
| same, sum pool | 0.4055 | 0.6890 | 0.8616 | 213 / 205 (550) |
| same, no angles | 0.4064 | 0.6901 | 0.8657 | 193 / 215 (560) |
| same, formal charges | 0.4084 | 0.6901 | 0.8678 | 211 / 200 (557) |
| same, sidechain only | 0.4108 | 0.6860 | 0.8657 | 224 / 202 (542) |

All v2 arms sit within ±0.005 MRR of Jaccard and the paired wins/losses are
balanced. Mean pooling is the best of the full-chemistry variants and is
marginally above its own nested baseline; the other ablations are
indistinguishable from noise. The chemistry changes therefore make the mark
more faithful without changing what the coarse query can exploit.

Why the flat result is structural **for this query protocol**: with a
fragment-centred environment and the whole parent ligand as the ref-mol query,
the per-triplet weights correlate with the true positive's library weights
(Spearman 0.594 over 137 stratified queries), but just as strongly with a
**random fragment in the positive's own complex** (0.617) and with a fragment
from another complex of the same protein (0.647). Paired per query, the true
positive is *lower* than its same-complex competitor (mean delta −0.024;
positive wins 45.6% of queries). The weights are dominated by which pocket
residues are near the ligand family, which is shared by all fragments of that
protein; the fragment-specific component is small. Plain key count carries the
discriminative signal, so weighted overlap cannot improve on it -- it only
adds variance. Per-atom `E/a` storage and top-k focus were tested and also fail
here; see `REFMOL_QUERY.md`.

The limitation is the query, not the mark. When the environment is unified per
complex (one pocket shared by every fragment, `--env-center pocket`) and the
query is the fragment itself in its deposited pose (`--query-mode reffrag`),
the rich scorings rise far above coverage, and the per-atom `E/a` vector adds
a further edge concentrated on positives with a different parent ligand. That
protocol and its controls are in `REFMOL_QUERY.md` under "Unified pocket +
ref-frag"; it is a distinct benchmark and its numbers are not comparable to
the stored-coarse rows above.

### Top-k separability

Full same-protein eligible set (3,535 scored queries, median 5,855 candidates):
the target's deterministic rank has MRR 0.4099 and median 4, far better than
chance (~1,600), so the index is a good **recall/coarse filter**. It is not a
fine ranker, though: 33.6% of queries have top score exactly 1.0, and among
those a median of 5 (mean 10.7, max 59) candidates tie at 1.0. The positive
ties with a median of 3 (mean 8.6) candidates, and 49.6% of positives are
strictly beaten by at least one candidate. Within a tie group the order is
decided by `frag_id`, not by evidence. Breaking those ties needs a
query-visible continuous term -- the symmetric pair/distance term.

### Why the headroom is small

The 33-bit fingerprint is not 33 bits of entropy. `residue_code` maps the 20
standard residues onto **15 distinct codes** (ALA/PRO, ASN/SER, ILE/LEU/VAL
and ARG/LYS collide), so a sorted triplet drawn with repetition has at most
C(17,3) = **680** possible values. The 500-complex index observes 640 of
them (94%). Weighting keys in a saturated 640-key universe can only reorder
candidates that already share most of their keys; it cannot create the
resolution the coarse query representation lacks.

## Ablations

* `--no-energy`: drop the interaction/potential term.
* `--no-content`: drop the pharmacophore-complementarity term.
* `--no-pair-bias`: drop the residue-residue pair bias.
* `--temperature`: sharpen/flatten the residue softmax.
* `--no-explicit-h`: heavy atoms only (forces formal charges).
* `--no-backbone`: sidechain atoms only (v1 behaviour).
* `--no-angles`: distance-only H-bonds, with v1 donor/acceptor sets.
* `--energy-pooling {mean,sum}`: intensive (default) vs extensive.
* `--charge-method {eem,formal}`: partial-charge source.
* `--strict-charges`: fail instead of falling back to Gasteiger.
* `--legacy-hbond`: v1 donor/acceptor sets (only for the exact A/B control).

Sanity checks: `--no-energy --no-content --no-pair-bias` must reproduce the
Jaccard column exactly; and the documented legacy flag set above must
reproduce the v1 index exactly.

## Known approximations

* Residue partial charges use a residue/atom-name table while fragment charges
  are EEM: a mixed scheme, not one Hamiltonian.
* Residue hydrogens carry zero charge; they contribute only to H-bond
  geometry. Fragment hydrogens carry EEM charge.
* Hydroxyl/thiol H orientations from `AddHs` are arbitrary and treated as
  freely rotatable; donor-angle terms are therefore not applied to them.
* Histidine tautomers are enumerated (HID/HIE) when the PDB has no hydrogen;
  HIP is not inferred from geometry.
* Fragment hydrophobic flags treat all C/S/halogen atoms as hydrophobic.
* Environment membership is still a CA sphere; this is the controlled variable.
* The coarse `stored` query still has no query-visible continuous term. The
  ref-mol mode is the answer to that: it replaces the coarse probe with a
  rich one rather than trying to enrich the codes.
