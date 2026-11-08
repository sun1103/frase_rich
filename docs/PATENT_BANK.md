# Patent fragment bank — decomposition stage

Purpose: turn a 39M-compound patent corpus into candidates that are **the same
kind of object as a library fragment**, so they can be scored with the existing
`fp_weights_in_env` path and compared against 33-bit FRASE keys without a
translation layer.

Source corpus: `/Volumes/DATA_E/DATA/patent_compound` (SureChEMBL-derived),
parent pool `filtered_compounds.parquet` (10,744,313 id/smiles pairs), copied
to `patent_compound/source/`.

## Rule

Identical to the 3D library fragmenter (`fragenv/src/fragment.py`, the paper's
"Enumerate Fragments"):

1. Cut every bond that is single, non-cyclic, and not ring↔alpha. Ring–ring
   single bonds (biaryl) and amide C–N are cut (`cut_biaryl=True`,
   `cut_amide=True`, the library defaults).
2. Contract non-cuttable bonds into rigid blocks; cuttable bonds are the edges
   of the block graph.
3. Enumerate every connected block subset, gated to 50–300 Da, with
   `max_blocks=16`, `max_frags=200` — the same caps as `build.py`.
4. H-cap at the cut bonds; `n_attach` gives the role tag
   (`whole`/`substituent`/`linker`/`hub`) and `ring_class`
   (`acyclic`/`monocyclic`/`polycyclic`), matching the library's tags.

Verified equivalent to the conformer-carrying route: `fragment_ligand` and
`decompose_parent` produce the same fragment-SMILES multiset (3D coordinates
only add centroids). See `tests/test_patent_decomp.py`.

## Why not reuse `surechembl_data/decomposed_fragments_*`

That table is a **single-pass, ungated cut** (3.37 fragments/compound, median 8
heavy atoms, max 401, no 50–300 Da window) and the role-aware derivative is
H-capped with the attachment markers dropped. Different universe and no exit
vectors → candidates cannot be placed on a query pose or keyed against the
library.

The upstream `decomposed_fragments.parquet` (36.2M rows) *does* keep `*`
dummies, so its exit vectors are recoverable, but it is still the fine-grained
single-pass split.

## Exit vectors

The 3D library route recovers attachment atoms geometrically
(`features.attachment_atoms`). Here they are exact, read off each cut edge:

| column | meaning |
|---|---|
| `attach_atoms` | attachment atoms in **canonical fragment-SMILES** numbering |
| `attach_parent` | same atoms in **parent** numbering |
| `attach_partner` | parent atom on the other side of each cut bond |

`attach_atoms`/`attach_parent` are aligned pairwise with `attach_partner`, so
the exit vector of attachment *k* is the bond `(attach_parent[k], attach_partner[k])`.

## Output schema

One parquet part per input batch (`work/patent_frags/part_<offset>.parquet`),
zstd, one row per fragment:

```
compound_id int64, frag_smiles string, n_heavy int16, n_attach int8,
attach_atoms list<int32>, attach_parent list<int32>,
attach_partner list<int32>, role string, ring_class string
```

Parts are written atomically (`.tmp` + rename) and skipped on restart, so the
run is resumable.

## Run

```bash
conda run --no-capture-output -n rxn-env python scripts/decompose_patent.py \
    --parents /Users/mai/Software/patent_compound/source/filtered_compounds.parquet \
    --out-dir work/patent_frags --workers 8 --batch 10000
```

Scale measured on the corpus: **10,744,313 parents → 232,288,547 fragment rows
(21.6/parent), 2.11 GB across 1,075 parts**; 10.97% of parents yield no
fragment (unparseable, or nothing clears the 50–300 Da window). Wall time
~37 h on 8 workers, the rate degrading as compounds grow.

**Traversal memoization (required).** Upstream's enumeration only blocks
*duplicate emissions* (`seen`), not duplicate *traversal*: the frontier at a
state depends only on the selected blocks, so a branched block graph was
re-walked along every path. Two batches took over a day of CPU each. Tracking
visited selection masks (each state's subtree is identical) makes the walk
one pass per connected subset and preserves the output exactly — those two
batches then finished in 1.4 minutes.

## Candidate-pool gates

Applied at pool-build time (`scripts/patent_pool.py`), never at decomposition
time — the raw parts keep every fragment, so thresholds are reversible.

| gate | default | why |
|---|---|---|
| `--min-heavy` | 5 | 50 Da is ~4–5 heavy atoms; 5 is the floor for a stable rigid-body alignment. Costs 0.9% of unique keys but 9.5% of instances (generic caps like `CCCC`, `FCF`). |
| `--min-attach` | 1 | `0` = whole molecule, no exit vector, cannot be placed as a replacement. |
| `--max-attach` | 3 | 4+ is a scaffold swap and disproportionately hard to place; those rows are 19–40% of keys. Raise if a scaffold pool is wanted. |

On a 60-part sample these keep 2.16M of 3.00M unique keys and 9.05M of 13.0M
fragment instances. Not applied: prevalence (`n_instances`) — common fragments
such as benzene are legitimately ubiquitous, and ranking handles them.

## Pool

`work/patent_pool.parquet` (390 MB): 232,288,547 raw rows → 165,242,424 after
the gates → **31,370,292 unique (SMILES, attachment) candidates**. Attachments
1/2/3 = 8.67M / 12.45M / 10.25M; polycyclic 13.58M, monocyclic 11.81M,
acyclic 5.98M; heavy atoms median 16 (p10 10, p99 22); 48.8% of keys occur at
least twice, 5.1% at least ten times. Build time ~26 min (475 s pass 1,
1,110 s pass 2, peak RSS 4.6 GB).

Overlap with the 334,002-fragment library (union index): the library has only
**67,787 unique fragment SMILES**; **35,827 (52.9%)** appear exactly in the
pool. The pool adds 21.6M unique SMILES — ~320× the library's chemotype
diversity — so roughly half of the library's chemotypes can be reproduced from
patents and the rest need near-neighbour candidates.

## Downstream (not built yet)

1. Dedupe parts by `(frag_smiles, attach_atoms)` into a candidate pool; drop
   fragments whose attachment pattern already exists in the library.
2. Rank candidates against a query fragment by 2D similarity + attachment
   geometry, then place on the query pose using the exit vectors.
3. Score in time with `fp_weights_in_env` — keys and `E`/`a` marks are
   pose-dependent, so nothing is inherited from the library. The placement
   prototype (`docs/PLACEMENT.md`) shows the full 12 Å profile overlap is
   pocket-dominated and cannot rank replacements. On the query fragment's
   contact shell, per-residue contact-gated energy correlation
   (`shell_energy_r`, averaged over 4 poses) separates analogs from random
   placements: 0.343 vs 0.199 candidate-mean (38/50 wins), 41/44 for the gated
   best-of-6. Weighting those energies by per-residue contact persistence over
   the pose ensemble strengthens it (0.445 vs 0.229, 41/50; 43/44 gated), and
   selecting the analog by that score is the best placed index arm (MRR 0.527
   vs 0.553 for the query fragment, 0.435 for a random placement, on the 13
   queries with an eligible positive). Restricting the *triplet keys* to the
   same shell destroys
   retrieval (positives fall to rank 871–133,639), and selecting analogs by the
   contact score leaves index ranks identical to the random control in 37/44
   queries. The per-key energy marks already stored in the index were then
   tried as a direct query path: the true fragment's marks match its weights
   (MRR 0.138 vs 0.144), but a placed analog's marks are uniformly worse than
   its weights (0.079 vs 0.099 whole-pocket, 0.106 vs 0.144 contact-selected),
   because the analog is scored in the query pocket rather than its own. The
   contact-local signal therefore required its own **residue-profile
   index** — stored per-residue weights/energies scored directly — rather than
   a reweighting of the existing triplet index; the mark result made a large
   gain from an even coarser projection of the same energies unlikely.
   That index was built and evaluated (`docs/PROFILE_INDEX.md`): a standalone
   profile index does not beat the same-universe triplet index (best profile
   MRR 0.620 vs 0.648), the best profile projection is the whole-pocket weight
   cosine rather than contact energy, and rank fusion with the triplet index
   reaches 0.704 (4W/1L/8T) without statistical significance on 13 eligible
   queries. The contact-local idea is bounded, not disproved.
4. Patent prevalence (`compounds with fragment / total`) as a negative filter;
   `decoy/` family-enrichment tables as priors, not as chemotype matching.
