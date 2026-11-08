# Residue-profile index — contact-local similarity across structures

Purpose: test the last untested route for carrying contact-local interaction
signal into retrieval. The placement prototype showed that the signal is real
(`shell_energy_r` separates analogs from random placements, and
persistence-weighted selection improves end-to-end retrieval) but that the
triplet index cannot express it: restricting triplet keys to the query shell
destroys retrieval, and the stored per-key energy marks are the wrong
projection. A residue-profile index stores the per-residue weights and contact
energies themselves and compares them residue by residue.

**Bottom line.** A standalone residue-profile index ties the same-universe
triplet index: the best profile arm is MRR 0.682 vs `triplet_ref` 0.668, paired
5W/2L/6T, mean ΔMRR +0.014, bootstrap CI −0.211 to +0.232. The profile signal
is real — the value-permutation control on the `combined` metric scores 0.285
and random order 0.176 — and the best profile arm is the *whole-pocket* weight
cosine, not any contact-local metric. Reciprocal-rank fusion of the triplet
index with that whole-pocket cosine reaches MRR 0.761 (5W/1L/7T, mean ΔMRR
+0.094), but its paired bootstrap interval also includes zero (−0.065 to
+0.256, P(Δ≤0) = 0.10), so the gain is suggestive, not established. The result
is a small complementary signal, not a replacement index.

All numbers below come from the store rebuilt after the duplicate-site fix
(see *Profile store*); the pre-fix outputs are kept as
`work/*_predup.json` and gave systematically lower profile scores.

## Why cross-structure residue keys

The evaluation needs residue *i* of a query structure and residue *i* of a
library structure to be the same residue. `(chain, res_seq)` cannot supply
that: PDBbind structures of one protein routinely relabel chains (`H` vs `A`)
or renumber, so raw keys are disjoint even for identical sequences. The
groups in `pdbbind_protein_groups.json` were built by exact longest-chain
sequence match, which is the identity available.

`sequence_residue_map` therefore keys a residue by

```
(CRC32 of the chain's one-letter sequence, position among that chain's CA records)
```

- Stable across chain labels and numbering frames.
- Chains with identical sequences (homodimer copies) share keys.
  `attach_sequence_keys` aggregates them per sequence position: weights and
  contact energies sum, the shell mask is the union, the first residue name is
  kept. A ligand binding the dimer interface is described by its aggregate
  contact with each sequence position, which is also robust to relabelling.
- Chains with different sequences (thrombin light chain, peptides, a different
  construct) get their own sequence id and never collide.

Mapping coverage on the store: median 1.00 of environment residues mapped
(p10 1.00), shell median 1.00. Homodimer merging is what made the earlier
`(chain, res_seq)` mapping failures (groups g0107, g0017, g0031, 1aqj) go
away.

## Profile store

Built by `scripts/build_profile_store.py`:

```bash
conda run --no-capture-output -n rxn-env python scripts/build_profile_store.py \
    --index work/rich_pocket_union.pkl \
    --cache /Users/mai/Software/fragenv/data/pdbbind_cache \
    --groups work/pdbbind_protein_groups.json \
    --queries work/patent_place_persist.json \
    --out work/profile_store.pkl
```

Only complexes in the groups of the 50 placement queries are profiled:
**103 complexes**, **2,268 fragments**, 21 groups, 0 missing, 0 failed, 244 s.
The whole library would be 16,663 complexes, and residues from different
proteins are not comparable anyway.

A symmetric parent can deposit the same fragment SMILES at two sites, and the
first version keyed the environment by SMILES alone, so both rows got the
profile of one arbitrary copy. `_group_copies` now resolves each row to the
copy it was built from using the index's stored canonical heavy-atom
coordinates (`_frag_coords`), profiles each distinct copy once, and maps fids
that share a copy onto the same profile; fids whose SMILES the re-cut parent
does not reproduce are reported as missing. The fixed store is 11.4 MB (was
8.6 MB) and lifts both sides — `p_all_weight_cos` MRR 0.620 → 0.682,
`p_weight_cos` 0.525 → 0.582, `triplet_ref` 0.648 → 0.668 (the triplet arm's
per-copy weights also reflect the corrected copy now).

Each fragment is profiled with exactly the index's feature settings (EEM +
explicit H, angle-aware H-bonds, mean-pooled contact energies, 12 Å pocket),
then projected onto sequence keys:

```
{
  schema: "frase_rich.profile_store.v1",
  features: {...},                 # index feature flags
  groups: {gid: [pdb, ...]},
  profiles: {fid: {
      frag_smiles, pdb_id, parent_smiles, n_heavy, n_attach,
      seq_id, seq_pos,             # cross-structure keys (int64)
      weights,                     # softmax relevance per residue
      contacts,                    # contact-gated energy per residue
      mask,                        # query-shell: heavy atom <= 5 A
      res_names,
      fp_norm,                     # mean-1 triplet weights (triplet baseline)
      n_env, n_mapped, n_shell, n_shell_mapped, n_keys,
  }},
}
```

`fp_norm` reproduces `index.add_fragment`'s mean-1 scaling, so weighted exact
key Jaccard over it *is* the triplet-index score. That gives every evaluation
an in-store triplet arm scored on exactly the same candidates — the only fair
comparison, because the profile index cannot score fragments it has no
structure for.

## Evaluation protocol

`scripts/eval_profile_index.py` runs the profile analogue of the triplet
leave-one-complex-out benchmark: for a query fragment, rank the fragments of
the *other* complexes of the same protein and find the same fragment SMILES.

- Candidates: all store profiles of the same group with a different `pdb_id`.
- Positives: candidate rows with the query's fragment SMILES.
- Ranks are deterministic: score descending, then
  `(frag_smiles, pdb_id, frag_id)`.
- 50 placement queries; 13 have at least one eligible positive. All numbers
  below are on those 13.
- `tie_at_positive` records how many candidates share the positive's best
  score, so a collapse to ties is visible rather than hidden.

```bash
conda run --no-capture-output -n rxn-env python scripts/eval_profile_index.py \
    --store work/profile_store.pkl \
    --queries work/patent_place_persist.json \
    --out work/profile_eval.json \
    --save-scores work/profile_scores.json
```

### Metrics

Contact-local metrics use the shared residues on the query's 5 Å shell:

| metric | definition |
|---|---|
| `weight_cos` | cosine between query and candidate query-shell weights |
| `energy_r` | Pearson r of per-residue contact energies |
| `weighted_energy_r` | same, weighted by the query's softmax relevance |
| `energy_r_fav` / `_engaged` | r restricted to query-favourable / either-favourable contacts |
| `weighted_energy_r_strength` | r weighted by the query's own contact strength |
| `shell_recall` / `shell_weight_recall` | fraction of the query shell (or query weight) the candidate actually touches |
| `contact_retention` | query-weight fraction of the query's favourable contacts the candidate also engages |
| `delta_energy` | query-weight mean of `candidate − query` contact energy |
| `contact_jaccard` | Jaccard of the two favourable-contact sets |
| `overlap_energy` | `contact_jaccard × (energy_r_engaged + 1)/2` |

Whole-pocket variants (`all_*`) use every shared residue, not just the query
shell. Missing correlations (too few shared residues, or a constant side) map
below every real value (`−2`) rather than to zero, because zero correlation is
a legitimate measurement.

## Results

Same-group universe, 13 eligible queries:

| arm | MRR | R@10 | R@50 | median | tie |
|---|---:|---:|---:|---:|---:|
| `p_all_weight_cos` | **0.682** | 0.92 | 0.92 | 1 | 8.2 |
| `triplet_ref` | 0.668 | 0.92 | 0.92 | 1 | 31.8 |
| `p_weight_cos` | 0.582 | 0.85 | 1.00 | 3 | 4.8 |
| `p_recall_energy` | 0.582 | 0.77 | 0.92 | 2 | 14.5 |
| `p_wenergy` | 0.574 | 0.77 | 0.77 | 2 | 16.8 |
| `p_all_energy` | 0.571 | 0.77 | 0.92 | 2 | 9.7 |
| `p_all_wenergy` | 0.571 | 0.77 | 0.92 | 2 | 10.3 |
| `p_overlap_energy` | 0.544 | 0.77 | 0.92 | 2 | 13.8 |
| `p_wenergy_strength` | 0.537 | 0.77 | 0.85 | 2 | 16.2 |
| `p_combined` | 0.536 | 0.77 | 0.77 | 2 | 17.3 |
| `p_energy` / `p_energy_fav` | 0.536 | 0.77 | 0.77 | 2 | 18.0 |
| `p_energy_engaged` | 0.535 | 0.77 | 0.77 | 2 | 19.0 |
| `p_contact_jaccard` | 0.372 | 0.77 | 0.85 | 3 | 55.6 |
| `p_retention` | 0.371 | 0.77 | 0.85 | 3 | 56.2 |
| `triplet_uniform` | 0.340 | 0.77 | 0.85 | 3 | 46.0 |
| `triplet_ref_shell` | 0.322 | 0.69 | 0.69 | 6 | 108.8 |
| `p_combined_permuted` (control) | 0.285 | 0.69 | 0.77 | 4 | 27.2 |
| `p_delta` | 0.274 | 0.85 | 0.92 | 6 | 22.7 |
| `random` (control) | 0.176 | 0.69 | 0.77 | 5 | — |

Notes:

- Ties are large: the triplet arm ties 31.8 candidates on average at the
  positive's score, the shell-restricted triplet 108.8, and the coarse
  contact-Jaccard/retention profiles 56. The two best arms have the fewest
  (`p_all_weight_cos` 8.2, `p_weight_cos` 4.8). 33-bit key saturation still
  caps resolution, so paired comparisons and tie counts matter more than MRR
  alone.
- No contact-local metric beats the whole-pocket weight cosine, though
  `p_weight_cos` (0.582) now closes the gap to it. The placement prototype's
  finding that the 12 Å profile is pocket-dominated does not mean the
  *weights* carry no fragment information; it means the contact-gated energy
  is the wrong thing to compare across structures, whose pocket contacts differ
  by pose and construct.
- `p_weight_cos` is the best contact-local arm and has the fewest ties (4.8).
  Its ranks are complementary to the triplet's: 1ajx triplet 388 vs profile 1,
  1a4m 3 vs 1; conversely 1a42 profile 18 vs triplet 1, 1a4w 18 vs 2, 1a69 8
  vs 1, 1af2 3 vs 1. In 10 of 13 queries one of the two ranks the positive
  1st, but they agree exactly (rank for rank) in only 3. The whole-pocket arm
  agrees exactly in 6 of 13 and is 1st alone in 2 (186l, 1a4w); at least one
  of the two is 1st in 10 of 13.
- The headline pairing: `p_all_weight_cos` vs `triplet_ref` is 5W/2L/6T,
  mean ΔMRR +0.014, 95% CI −0.211 to +0.232 — a tie. The profile arm wins
  mostly by moving second-place positives to rank 1 (186l 2 → 1, 1a4w 2 → 1)
  and loses on 1a42 (1 → 4) and 1a69 (1 → 8).
- Both arms fail the same query (1ajx, 920 candidates). Fusion moves it from
  388 to 169 but does not reach the top 50; it is the only R@50 miss.

## Fusion

Fusion is computed by `scripts/fuse_profile_scores.py` from
`work/profile_scores.json` (`--save-scores`), not stored in the index. RRF uses
`1/(60 + rank)` per arm; z-average sums the per-query z-scores; the tie-break
orders by the triplet and breaks ties with the profile. The permutation
control is fused the same way, as a test that fusion respects residue
correspondence.

| fusion | MRR | R@10 | vs triplet (W/L/T) | mean ΔMRR |
|---|---:|---:|---:|---:|
| RRF + `all_weight_cos` | **0.761** | 0.92 | 5/1/7 | +0.094 |
| top-10/25 rerank by `all_weight_cos` | 0.739 | 0.92 | 4/1/8 | +0.071 |
| z-average + `all_energy` | 0.713 | 0.92 | 5/1/7 | +0.045 |
| RRF + `all_energy` | 0.698 | 0.92 | 5/2/6 | +0.030 |
| z-average + `contact_jaccard` | 0.682 | 0.92 | 3/0/10 | +0.015 |
| z-average + `contact_retention` | 0.676 | 0.92 | 3/0/10 | +0.008 |
| z-average + `all_weighted_energy` | 0.674 | 0.92 | 4/1/8 | +0.007 |
| z-average + `all_weight_cos` | 0.644 | 0.92 | 3/1/9 | −0.023 |
| triplet, tie-break by any profile arm | 0.668 | 0.92 | 0/0/13 | 0.000 |
| RRF + permuted control | 0.484 | 0.92 | 2/7/4 | −0.184 |
| z-average + permuted control | 0.422 | 0.92 | 2/7/4 | −0.246 |
| oracle, triplet vs `all_weight_cos` | 0.807 | 1.00 | — | — |
| oracle, triplet vs `weight_cos` | 0.859 | 1.00 | — | — |

Two cautions on the z-average rows. First, `contact_jaccard` alone is one of
the worst arms (MRR 0.372), yet its z-average fusion (0.682) clears the
triplet: with 13 queries, a differently *ordered* weak arm can move a few
positives to the top even when its absolute signal is poor. Second, 10 of
those 13 queries are ties for that pairing, so the 0.682 is estimated from 3
informative queries. The fused permutation control (0.422–0.484) is the
useful check: fusion does not create gains from arbitrary reorderings, and
the profile fusions clear it.

RRF changes five queries and one against: 1a4m (3 → 1), 1a4w (2 → 1), 186l
(2 → 1), one of the two 184l queries (5 → 4) and 1ajx (388 → 169) improve;
1a69 loses (1 → 2); the other 7 are identical. On MRR the gain is dominated
by 1a4m (+0.67 reciprocal rank), 186l (+0.50) and 1a4w (+0.50), net of the
1a69 loss (−0.50); the 1ajx rank gain is large in rank terms but contributes
little MRR. Paired bootstrap over the 13 queries (5,000 resamples): mean ΔMRR
+0.094, 95% CI −0.065 to +0.256, P(Δ ≤ 0) ≈ 0.10. **Not significant.**

The tie-break changes nothing (0/0/13): although each arm has large tie
groups, reordering the triplet's ties by any profile score never moves the
first positive. Top-K rerank is better than the triplet but worse than RRF:
reranking the triplet's top 10/25 by `weight_cos` gives MRR 0.589/0.520, and
by `all_weight_cos` 0.739/0.739.

## Conclusions

1. The residue-profile index does not replace the triplet index. The best
   profile arm ties it (0.682 vs 0.668, CI −0.211 to +0.232); with the current
   energies and poses, whole-pocket profile agreement is not a stronger signal
   than exact triplet-key overlap.
2. The profile signal is genuine: destroying residue correspondence drops the
   `combined` metric from 0.536 to 0.285, near random (0.176), and fusing the
   permuted profile falls to 0.42–0.48 versus 0.67–0.76 for the real profile
   fusions.
3. The useful projection is the **whole-pocket weight cosine**, not the 5 Å
   contact-gated energy. Comparing contact energies across independently
   deposited poses adds pose/construct noise that cancels the contact signal.
4. Rank fusion is the only configuration above the triplet baseline (several
   pairings land at 0.69–0.76). The best RRF gain has a confidence interval
   that spans zero, and the z-average runner-up is estimated from 3 non-tied
   queries. Treat fusion as a hypothesis for a larger evaluation, not a
   default.
5. A larger eligible set (more same-group positives) is the prerequisite for
   any stronger claim; 13 queries cannot resolve a +0.09 MRR effect.

## Reproduce

```bash
# store (~4 min)
conda run --no-capture-output -n rxn-env python scripts/build_profile_store.py

# evaluation (<1 min) + per-candidate scores for fusion analysis
conda run --no-capture-output -n rxn-env python scripts/eval_profile_index.py \
    --save-scores work/profile_scores.json

# fusion table, paired wins and bootstrap (< 1 s)
conda run --no-capture-output -n rxn-env python scripts/fuse_profile_scores.py \
    --out work/profile_fusion.json
```

Outputs: `work/profile_store.pkl`, `work/build_profile_store.log`,
`work/profile_eval.json`, `work/profile_eval.log`, `work/profile_scores.json`,
`work/profile_fusion.json`.

## Caveats

- Profiles are only comparable within a protein group. There is no
  cross-protein profile similarity.
- The library profile is the fragment's own deposited complex, not the query
  pocket. A positive fragment that binds in a different pose (or a different
  conformation of the same protein) is still expected to match only partially;
  this is inherent to the task and is why the triplet arm is the reference.
- The 5 Å shell is defined on heavy atoms of the fragment pose; homodimer
  copies are merged by sequence position, so a fragment binding the interface
  is summarized per position rather than per copy.
- Charge assignment is EEM for all 2,268 profiles (no Gasteiger fallback
  fired), so charge-method mixture is not a confound here.
- `AddHs(addCoords=True)` is idempotent under RDKit (verified: re-adding to an
  explicit-H molecule leaves the atom count unchanged).
