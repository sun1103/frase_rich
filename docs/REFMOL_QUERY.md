# Reference-molecule query

## Why

The weighted index is built from a fragment's real pose: interaction energies
with explicit hydrogens, two-sided H-bond angles, and pharmacophore
complementarity. The coarse query used by the original prototype -- CA
coordinates and 11-bit residue codes -- cannot see any of that. Library-only
weighting therefore lands inside the random-weight noise floor
(see `WEIGHTED_INDEX.md`): the rich information exists, but the query is too
blind to exploit it.

The reference-molecule (ref-mol) query fixes the asymmetry on the **query**
side. Instead of scoring the query with residue codes, it takes a molecule in
an experimental pose, recomputes the same rich environment features, and asks
which library fragments share the same weighted triplets.

Two inputs define the query:

* a **ref-mol** with 3D coordinates (in practice the query complex's parent
  ligand, `fragmentation.fragment_ligand` / the deposited HETATM residue);
* the **pocket sidechains** of the query complex.

The output is a set of 33-bit triplet keys with continuous weights, exactly
comparable to a library entry.

## Same code path as the library

`weighting.query_fp_weights` calls, in order:

1. `with_explicit_hydrogens(ref_mol)` and `charges_for(..., "eem")`;
2. `environment(residues, centroid, radius)` -- the same 12 A CA sphere;
3. `relevance_weights(...)` -- the same energy, angles, pooling and softmax;
4. `fragment_fp_weights(...)` -- the same per-triplet weight, pair bias and
   per-fingerprint mean aggregation.

Sharing the code is deliberate: if the two sides drift, weighted overlap stops
being symmetric, and the nested uniform baseline stops being Jaccard. The
index records its feature flags (schema v2) so the query provider can copy
them; `--allow-feature-mismatch` overrides the guard, it does not silently mix
settings.

## Scoring: weighted exact-key overlap

For a query with fingerprint weights `w_q` and a library fragment with weights
`w_l`, both normalized to mean 1 over their distinct keys:

```
W         = sum_{k in shared keys} min(w_q[k], w_l[k])      # default
W_geo     = sum_{k in shared keys} sqrt(w_q[k] * w_l[k])    # --combine geomean
score     = W / (n_q + n_l - W)
```

This is weighted Jaccard over identical 33-bit keys. There is no near-miss
matching and no assignment step: a key either matches exactly or it does not.
The `min` form never exceeds either weight, so `score <= 1`; `geomean` is
bounded by the arithmetic mean of the two weights and is slightly softer.

With uniform query weights (`w_q[k] = 1`) and mean-1 library weights,
`W = |shared|` and the score is exactly Jaccard. That is the nested baseline:
any gain of the weighted arm is attributable to the weights, not to a different
candidate set.

Mean-1 normalization matters here. If query weights summed to 1, a query with
many keys would be penalized relative to one with few; mean-1 keeps `W` on the
same scale as `n_match`.

## Evaluation protocol

`evaluate.py --query-mode refmol`:

1. Positives are fragments with the same SMILES in another complex (optionally
   restricted to `--positive-mode same-protein` / `cross-protein`).
2. The query is the query complex's parent ligand in its deposited pose,
   resolved with `build._find_ligand(block, parent_smiles)` and cached per PDB:
   all fragments of one complex share the query context.
3. Environments are rebuilt from the cached PDB, using
   `protein_residues` (explicit H) with the heavy-atom fallback.
4. Candidates from the query's own PDB id are excluded (leave-one-complex-out).
5. Ranking is deterministic: weights rounded to 12 decimals, then `n_match`,
   then `frag_id`. The Jaccard column is computed from the same hit list and
   never inherits the weighted order.

Both arms are scored on the same candidates:

| arm | query weights |
|---|---|
| `jaccard` | all 1.0 (nested baseline) |
| `weighted` | ref-mol weights, `--combine min` |
| `weighted-geomean` | ref-mol weights, `--combine geomean` |
| `query-random` | ref-mol weights multiplied by lognormal noise, `--query-noise sigma` |

The random arm is the noise floor: it keeps the query's key set and weight
*magnitude*, and randomizes the per-key relevance. If the weighted arm does not
beat it, the ranking signal is not in the weights.

## Commands

```bash
# build with the current defaults (EEM + explicit H + angles + mean pooling)
conda run -n rxn-env python -m frase_rich.build \
    --selected work/selected_sameprotein.tsv --out work/rich_mean.pkl

# ref-mol query, weighted vs nested Jaccard
conda run -n rxn-env python -m frase_rich.evaluate \
    --index work/rich_mean.pkl --query-mode refmol \
    --cache /Users/mai/Software/fragenv/data/rcsb_cache \
    --combine min --positive-mode same-protein \
    --protein-groups work/protein_groups.json \
    --output work/refmol_min.json

# noise floor (two seeds)
conda run -n rxn-env python -m frase_rich.evaluate \
    --index work/rich_mean.pkl --query-mode refmol --combine min \
    --query-noise 0.2 --seed 0 --output work/refmol_noise0.json
```

The ref-mol task is **harder** than the stored coarse query: the query
environment is derived from a pose, so a bad ligand placement or a mismatched
parent propagates. It is reported as its own arm with its own nested baseline,
and is never compared to the stored-query numbers as a drop-in improvement.
`refmol_context_failures` in the summary counts PDBs where the ligand or
environment could not be built.

## Results (2026-10-20, 447 complexes, 6,385 fragments)

Same-protein positives, 1,000 sampled queries (959 scored, 41 without a
candidate hit), mean 5,678 candidates. `jaccard` is the nested all-uniform
baseline computed on the same hits.

| arm | MRR | R@10 | R@50 | R@100 | median |
|---|---|---|---|---|---|
| Jaccard (uniform) | **0.0977** | 0.1856 | 0.3889 | 0.4943 | 105 |
| weighted, `min` | 0.0899 | 0.1856 | 0.3702 | 0.4901 | 105 |
| weighted, `geomean` | 0.0915 | 0.1825 | 0.3754 | 0.4849 | 104 |
| noise σ=0.2, seed 0 | 0.0888 | 0.1940 | 0.3858 | 0.4870 | 104 |
| noise σ=0.2, seed 1 | 0.0983 | 0.2004 | 0.4050 | 0.5067 | 98 |

The noise rows sample different query subsets (the seed drives both the noise
and the 1,000-query subsample), so each is compared to its **own** nested
baseline: seed 0 Jaccard 0.0977, seed 1 Jaccard 0.1027. Both weighted rows are
below their own baseline.

Ablations (same query set, `min`):

| build | MRR | R@10 | R@50 |
|---|---|---|---|
| mean pooling, angles | 0.0899 | 0.1856 | 0.3702 |
| sum pooling | 0.0906 | 0.1804 | 0.3712 |
| no angles | 0.0834 | 0.1721 | 0.3733 |
| formal charges | 0.0846 | 0.1814 | 0.3733 |
| sidechain only | 0.0892 | 0.1689 | 0.3618 |

The weighted arms span 0.083--0.091, all inside the same noise band; EEM and
the H-bond angles improve the mark slightly relative to formal charges and the
angle-free arm, but not enough to cross the uniform baseline.

Ranking is uniformly below the nested baseline. The reference-molecule query
is also much weaker in absolute terms than the stored coarse query
(MRR 0.098 vs 0.407): a 12 A sphere around the whole parent-ligand centroid
produces a large, pocket-level key set, whereas the stored query uses the
fragment's own local environment. The candidate list is barely filtered
(5,678 of 6,385 fragments share at least one key).

### Why the weights do not rank

The weights are informative about the *pocket*, not about the *fragment*.
This was measured with a stratified diagnostic that holds the protein fixed
(137 queries): on shared keys, the query weight vector is correlated with
the true positive at Spearman 0.594, with a **random fragment in the positive's
own complex** at 0.617, and with a fragment from another complex of the same
protein at 0.647. Paired per query, the true positive is *lower* than its
same-complex competitor in both comparisons (mean delta −0.024 and −0.053;
the positive wins in only 45.6% / 43.8% of queries). Cross-protein the
correlation collapses to 0.243. The profile identifies the fold; it does not
prefer the fragment.

The earlier same-protein figure (0.64 vs 0.61) was not controlled for the
positive's complex and overstated the gap; the stratified numbers above
replace it. A per-query AUC decomposition agrees: weighted AUC 0.745 vs plain
Jaccard 0.759 (61 queries). Shrinking the weights toward uniform
(`w' = 1 + s(w-1)`, s = 0.5 / 0.25) moves MRR from 0.0897 to 0.0898 / 0.0916
on a 478-query paired run where Jaccard is 0.1166 -- it does not recover the
gap.

### Tried and rejected: per-atom `E/a` and top-k focus

The working hypothesis was that the per-triplet interaction energy carries
fragment-specific signal even when the softmax weight does not. `build.py`
therefore stores, per fragment and per 33-bit key, the mean and max triplet
energy divided by heavy-atom count (`_fp_e_atom_mean`, `_fp_e_atom_max` in
`work/rich_eatom.pkl`); the per-atom division removes fragment size.

Stratified Spearman (same 137 queries) is the same story, slightly worse:

| quantity | positive | same complex | same protein, other complex | different protein |
|---|---|---|---|---|
| weight | 0.594 | 0.617 | 0.647 | 0.243 |
| `E/a` mean | 0.555 | 0.606 | 0.619 | 0.110 |
| `E/a` max | 0.576 | 0.609 | 0.598 | 0.096 |

`E/a` is *more* protein-specific than the weight (cross-protein 0.10 vs 0.24)
and still does not prefer the true fragment. Paired deltas remain negative.

Focusing retrieval on the most favorable keys also fails. Select the query's
top-k keys by `E/a`, score `sum min(q_e, l_e)` over those keys (or over keys
that are also in the candidate's top-k), 150 queries, 145 scored:

| arm | MRR | R@10 | R@50 | median |
|---|---|---|---|---|
| jaccard | 0.0840 | 0.179 | 0.366 | 140 |
| weighted (all keys) | 0.0868 | 0.179 | 0.372 | 146 |
| qfocus mean k=5 / 10 / 20 / 50 | 0.057 / 0.056 / 0.063 / 0.057 | 0.10 / 0.09 / 0.12 / 0.11 | 0.21 / 0.23 / 0.25 / 0.26 | 233 / 244 / 206 / 187 |
| qfocus max k=10 / 20 | 0.049 / 0.048 | 0.07 / 0.09 | 0.23 / 0.27 | 204 / 182 |
| both k=5 / 10 / 20 / 50 | 0.060 / 0.055 / 0.047 / 0.069 | 0.10 / 0.07 / 0.12 / 0.14 | 0.26 / 0.16 / 0.18 / 0.25 | 266 / 243 / 184 / 186 |

Every focus arm is 20--45% below both baselines and only approaches them as k
grows toward the full key set. Retrieval here is carried by **shared-key
coverage** -- how many distinct pocket triplets a candidate reproduces -- and
restricting to the most favorable keys discards exactly that. (The weighted
arm being 0.003 above Jaccard on this 150-query sample is sampling noise; on
the full run it is below.)

The negative result is therefore structural *for this query protocol*: with a
fragment-centred environment and a whole-ligand ref-mol query, the rich mark is
dominated by which pocket residues surround the ligand family, so it is nearly
constant among candidates from the same protein fold. See the next section for
the protocol that removes this confound.

## Unified pocket + ref-frag: the full E/a vector

Two changes isolate the fragment-specific component that the ref-mol protocol
could not see:

* **One pocket per complex.** The environment is the union of 12 A spheres
  around every parent-ligand heavy atom, shared by all fragments of that
  complex (`build.py --env-center pocket`; see `pocket_environment`). Library
  fragments and the query are scored against exactly the same residue set, so
  their per-key marks are directly comparable.
* **The query is the fragment itself in its deposited pose**
  (`evaluate.py --query-mode reffrag`). `fp_weights_in_env` scores it against
  the fixed pocket without re-centring. This is the replacement use case: the
  fragment sits in the pose it would occupy in the query complex.

The query-side extraction reproduces the builder exactly: same
`pocket_environment` (12 A union of spheres around the parent ligand), same
triplet table, and the feature flags read from the index itself. Checked on
the PDBbind index — 18 fragments across 8 complexes recompute to the stored
weights **and** per-key E/a marks with zero difference.

One subtlety: a symmetric parent produces the same fragment constitution at
two cut sites (`CCCc1ccccc1` twice in 184l), so 34,823 of 184,186 rows
(19%) share their `(pdb, frag_smiles)` key with another row. The provider
therefore keeps every copy and picks the one whose deposited heavy-atom
coordinates match the indexed fragment's `_frag_coords`; matching on SMILES
alone silently recomputed the other site's energies. `_trim_index` keeps
`_frag_coords` in `reffrag` mode for this.

Per key, the builder stores the three-leg mean interaction energy divided by
fragment heavy atoms (`_fp_e_atom_mean`), its max (`_fp_e_atom_max`), a
fully-contacted-triangle variant (`_fp_e_atom_min`), and a contact-gated mean
(`_fp_e_atom_cmean`). Scoring (`--score-mode emark`) compares mean-1 normalized
mark vectors with `W = sum min(q, l)` over shared keys and
`W / (|A| + |B| - W)`, where `|A|` and `|B|` are the vectors' **positive
support** (keys carrying a positive mark), not the fingerprint's distinct-key
count. For mean/max every key is positive, so the two coincide; for cmean
(median support 93 of 147 keys) and min_leg (median 3) they do not, and using
the distinct-key count as denominator penalizes the sparse marks.

150 queries, seed 0, same-SMILES positives in another complex, query PDB
excluded:

| arm | MRR | R@10 | R@50 | median rank |
|---|---|---|---|---|
| coverage Jaccard | 0.114 | 0.293 | 0.587 | 27.5 |
| softmax-weighted overlap (`--score-mode fp`) | **0.359** | 0.540 | 0.673 | 6.0 |
| E/a mean | **0.375** | **0.600** | **0.707** | **5.0** |
| E/a max | 0.371 | 0.560 | 0.700 | 5.5 |
| E/a min_leg (fully-contacted triangles) | 0.223 | 0.420 | 0.613 | 17.5 |
| E/a cmean (proximity removed) | 0.368 | 0.587 | 0.740 | 7.0 |
| best top-k key (max, m=10) | 0.242 | 0.500 | 0.673 | 10.5 |

The nested baseline is the softmax-weighted overlap, not Jaccard: unification
and the ref-frag query lift **both** rich scorings far above coverage
(0.11 -> 0.36), and the E/a vector adds a modest amount on top (+0.016 MRR,
+0.06 R@10). The E/a vector's advantage is concentrated where the pocket
context actually differs -- on the 97 queries whose positive fragment sits in
a *different* parent ligand:

| arm | MRR | R@10 | median rank |
|---|---|---|---|
| Jaccard | 0.044 | 0.113 | 108 |
| softmax-weighted | 0.167 | 0.289 | 70 |
| E/a mean | **0.234** | **0.423** | **24** |
| E/a max | 0.226 | 0.381 | 19 |
| E/a cmean | 0.206 | 0.371 | 37 |

Controls:

* **Amplitude matters.** Shrinking the query mark toward uniform
  (`1 + 0.5(v-1)`, support unchanged) drops MRR to 0.141, i.e. the coverage
  baseline. The gain is the contrast of the values, not extra key support.
* **Exact values do not.** 20% lognormal noise on every query mark gives
  0.318--0.330 MRR for the mean vector: most of the signal survives, but the
  arm falls below the noise-free weighted overlap (0.359). The cmean vector is
  essentially unaffected (0.368--0.369).
* **Within one pocket, the mark separates fragments.** Comparing the true
  positive with up to 8 random fragments of the positive's complex (same
  pocket, same candidate set): 0.362 vs 0.300, positive wins on 67.3% of
  queries, median delta +0.022. The earlier ref-mol protocol reported a
  negative median delta; removal of the membership confound flips it.
* **Top-k keys still lose.** Selecting the top-m most favorable keys and
  scoring set overlap caps at MRR 0.243, versus 0.375 for the full vector:
  restricting to the best keys discards the profile shape that carries the
  signal.
* **Contact gating is neutral overall, worse across parents.** Removing the
  ungated proximity term (`cmean`) gives 0.368 MRR, statistically level with
  mean (0.375; it is +0.013 on median rank and +0.033 on R@50) but clearly
  lower where the parent ligand changes (0.206 vs 0.234). The proximity term
  carries cross-context signal, not just a floor to subtract. Within one
  pocket the full cmean vector still separates TP from random same-complex
  fragments (0.289 vs 0.201, 60.7% wins, median +0.016), but less sharply than
  mean (0.362 vs 0.300, 67.3%). Top-k on cmean (0.243) remains far below the
  full cmean vector.

### Scale-up: PDBbind (10,292 complexes, 184k fragments)

The same protocol on the filtered PDBbind P-L set (`scripts/pdbbind_cache.py`,
MW <= 700, logP <= 5.5; see `WEIGHTED_INDEX.md`): 11,033 complexes converted,
10,292 built (skipped: protein fallback 1,000, ligand 16, fragments 725) ->
184,186 fragments, 31.4M triplets, 680 unique fingerprints. Charges are EEM
for 184,130 fragments and Gasteiger for 56. Same-protein groups: 1,601 groups
with >= 2 complexes, 6,940 complexes. 1,000 sampled queries (seed 0), positives
are the same fragment SMILES in another complex of the same protein, query PDB
excluded. Every query has nearly the entire library as candidates (mean
184,082.8 of 184,186), so R@10/R@50 carry the signal.

| arm | MRR | R@10 | R@50 | R@100 | median rank |
|---|---|---|---|---|---|
| coverage Jaccard | 0.070 | 0.179 | 0.452 | 0.591 | 62 |
| softmax-weighted overlap (`--score-mode fp`) | 0.428 | 0.592 | 0.733 | 0.798 | 5 |
| E/a max | **0.446** | **0.640** | 0.815 | 0.873 | 4 |
| E/a mean | 0.432 | 0.630 | **0.817** | **0.877** | 4 |
| E/a cmean | 0.375 | 0.603 | 0.793 | 0.842 | 5 |
| E/a min_leg (911 scored) | 0.302 | 0.542 | 0.729 | 0.793 | 8 |
| E/a mean, 20% lognormal query noise | 0.387 | 0.601 | 0.793 | 0.862 | 5 |

Paired against Jaccard, the mark wins 877 / 99 (mean), 859 / 114 (max),
838 / 136 (cmean) and 705 / 186 (min_leg) with 20--28 ties; the nested
softmax-weighted baseline itself wins 913 / 56 with 31 ties. Compared with the
447-complex sample, the ordering of the marks changes at scale: max now edges
mean (0.446 vs 0.432 MRR, 0.640 vs 0.630 R@10, equal median rank), while both
stay clearly above the weight-only baseline. cmean, statistically level with
mean on the small sample, is now clearly below it: the ungated proximity term
carries signal that matters more when more proteins are in play. min_leg loses
its fully-contacted triangles on 89 queries and ranks below cmean even on the
911 it scores.

The 20% noise control is unchanged in direction: mean falls to 0.387 MRR,
below the noise-free weighted overlap (0.428). The mark's edge comes from the
contrast of its values, so exact relative magnitudes matter more than support
alone.

The rough stored-query path (`--query-mode stored`: the library environment's
CA coordinates and 11-bit codes, no sidechain or energy information) scores
0.070 Jaccard / 0.082 weighted MRR on the same 1,000 queries, with 460
weighted-better / 430 Jaccard-better / 110 ties. The library's rich weights
are invisible to a query this coarse; the gain above requires the ref-frag
pose query, which is the replacement use case.

At this scale `evaluate.py` no longer materialises a dict row per candidate.
It stacks the library weights and marks into dense float64 matrices once,
scores every fragment per query in one numpy pass, and counts the rows that
beat the best positive instead of sorting the whole candidate list.
`_DenseLibrary` is checked against `WeightedTripletIndex.query` on both
indexes and both query paths (fp weights, stored): identical ranks and
candidate counts, and the sub-second-per-query loop makes the full 1,000-query
arm finish in minutes rather than hours.

Usage:

```bash
python -m frase_rich.evaluate --index work/rich_pocket_cmean.pkl \
    --query-mode reffrag --score-mode emark --mark mean \
    --cache /Users/mai/Software/fragenv/data/rcsb_cache
```

`--query-noise sigma` perturbs the query marks in `emark` mode. Both ends
must come from the same feature build: the query provider stores the mark
keys, and the builder records `features["marks"]`, which `evaluate.py` checks
before scoring.

### Scale-up: PDBbind + RCSB union (16,663 complexes, 334k fragments)

The PDBbind index merged with four RCSB/BindingDB shards (build and merge
details in `WEIGHTED_INDEX.md`): 334,002 fragments over 16,663 complexes,
680 unique fingerprints, 53.5M postings. Same-protein groups over the union:
2,420 groups with >= 2 complexes covering 10,201 of 19,219 grouped structures.

Protocol unchanged: 1,000 sampled queries (seed 0), positives are the same
fragment SMILES in another complex of the same protein, query PDB excluded.
997 scored (2 query contexts failed, 1 had no mark), mean 5.25
positives/query, 329,572.7 candidates on average. The RCSB cache stores full
deposited structures rather than trimmed P-L complexes, so the same-protein
sets mix both sources.

| arm | MRR | R@10 | R@50 | R@100 | median rank |
|---|---|---|---|---|---|
| coverage Jaccard | 0.085 | 0.201 | 0.505 | 0.644 | 48 |
| softmax-weighted overlap | **0.430** | 0.594 | 0.744 | 0.808 | **4** |
| E/a mean | 0.400 | **0.616** | **0.809** | **0.876** | 5 |
| E/a max | 0.403 | 0.600 | 0.800 | 0.862 | 5 |

Paired against Jaccard: the weighted overlap wins 899 / 61 (37 ties), mean
wins 840 / 133 (24 ties), max wins 822 / 159 (16 ties).

The mark arms keep their large edge over Jaccard at 1.8x the library size:
MRR 0.40 vs 0.085, median rank 5 vs 48. Both mark arms lose a few points to
the PDBbind-only numbers (0.400/0.403 vs 0.432/0.446 MRR), which is expected:
the union adds candidates from proteins and chemotypes the PDBbind filter
never saw, and same-protein positives are diluted by the larger library.

The PDBbind ordering of the marks does not survive the merge. There max
edged mean on MRR and R@10; here they are statistically level on MRR (max
+0.004) while mean is clearly better on R@10 (+0.016), R@50 (+0.009) and
R@100 (+0.014). The softmax-weighted overlap remains the best MRR arm, as on
PDBbind: the mark compresses the top of the list much harder than Jaccard but
does not beat the weight profile on mean reciprocal rank.

Running a 334k-fragment arm surfaced three evaluation-path costs, all fixed
in `evaluate.py` without changing scores:

* The ref-frag provider parsed the whole deposited structure with RDKit; a
  large assembly (8ana, 86k atoms) cost 105 s per context. Ligand lookup now
  runs on a HETATM-only block (identical ligand, 24 s -> 0.0 s) and the
  protein block is trimmed to residue records within 15 A of the ligand
  before `protein_residues` (8ana: 110 s -> 0.3 s; env residues, coordinates
  and weights are identical, apart from one added-H coordinate on a boundary
  nucleotide whose local bond graph changed).
* The per-PDB context cache is now a 32-entry LRU, and sampled query ids are
  sorted so all fragments of a complex are queried consecutively.
* Once the dense weight and mark matrices are built, the index drops its
  53.5M postings and per-fragment environment arrays; this cuts steady-state
  RSS from ~11 GB to ~4-5 GB.

With these, the full 1,000-query union arm takes about 45-60 min (it was
projected at >4 h and thrashed swap before the fixes).

### Confirmation slice (2026-11-08, seed 1, 120 queries)

An independent query subsample, disjoint from the seed-0 run above, scored
clean with no misses (`work/union_eval_confirm_seed1.json`):

| arm | MRR | R@10 | R@50 | R@100 | median rank |
|---|---|---|---|---|---|
| coverage Jaccard | 0.108 | 0.267 | 0.575 | 0.650 | 40.5 |
| softmax-weighted overlap | **0.489** | 0.625 | 0.825 | 0.858 | **2.0** |
| E/a mean | 0.468 | **0.708** | **0.858** | **0.892** | 3.0 |

Paired against Jaccard: weighted wins 108 / loses 8 (4 ties); mean wins
103 / loses 12 (5 ties). This reproduces the seed-0 structure -- a 4.5x MRR
gain over coverage, median rank 40 -> 2, and the mark leading on recall at
fixed depth while the softmax weight leads slightly on MRR. The absolute
numbers sit a little above the 997-query run, which is sampling noise at
n=120; the paired ratio is the stable part.

Reproduce (cache must match the index source -- the union index needs
`union_cache`; `rcsb_cache` lacks every PDBbind-only complex and turns them
into context failures):

```bash
python -m frase_rich.evaluate --index work/rich_pocket_union.pkl \
    --query-mode reffrag --score-mode emark --mark mean \
    --max-queries 120 --seed 1 --positive-mode same-protein \
    --protein-groups work/union_protein_groups.json \
    --cache /Users/mai/Software/fragenv/data/union_cache \
    --output work/union_eval_confirm_seed1.json
```
