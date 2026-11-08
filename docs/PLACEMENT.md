# Placement prototype — patent analog → query pose → in-time score

Purpose: answer the risky question before any index work — **once a patent
analog is rigidly placed on a query fragment's exit vectors, can it reproduce
the query's environment fingerprint?** `fp_weights_in_env` is the same scorer
the library uses, so a positive answer means patent candidates can be compared
with library fragments without a translation layer.

The prototype is deliberately self-contained: retrieval from the 2D pool,
placement in the query's protein frame, scoring against the query's own pocket,
and a random no-retrieval control.

**Bottom line.** Placement works and is not the bottleneck. The whole-pocket
fingerprint overlap cannot rank replacements (it is pocket-dominated). The
contact-local per-residue energy correlation is a real analog-vs-random signal,
but it cannot be carried by the existing triplet index: shell-restricting the
keys destroys retrieval, and better candidate selection leaves pocket-dominated
ranks unchanged. A residue-profile index was built and evaluated to test this
directly (`docs/PROFILE_INDEX.md`): a standalone profile index does not beat
the triplet index, but whole-pocket weight cosine adds a small complementary
signal under rank fusion.

## Inputs

| input | default | meaning |
|---|---|---|
| queries | PDBbind cache + selected TSV | experimental poses; up to `--max-per-complex` fragments per complex, largest with `n_attach` 1 or 2, 6–25 heavy atoms |
| pool | `work/patent_pool.parquet` | 31,370,292 candidate rows from `docs/PATENT_BANK.md` |
| features | `DEFAULT_FEATURES` | mirrors the union index build (`env_center=pocket`, mean pooling, angles, backbone, EEM) |

## Retrieval index

Built once by `frase_rich.patent_place.build_fingerprints` into four memmaps
next to `--fp-prefix`:

- `_fp.npy` — Morgan radius 2, 1024 bits, packed to 128 bytes/row
  (31.4M rows ≈ 4.0 GB);
- `_pop.npy` — bit counts; `_n_attach.npy`, `_n_heavy.npy` — pool gates.

Tanimoto is computed blockwise with `np.bitwise_count(block & query)`:
`T = shared / (pop_q + pop_c − shared)`. Rows are gated to
`n_attach == query` and `|n_heavy − query| ≤ --heavy-band` (8) before scoring;
the gate is what makes the alignment meaningful, the fingerprint only ranks
within it. The build shards by parquet row group across worker processes and
caches on the `(n_rows, n_bits)` shape.

`--limit-fp-rows N` builds a truncated index for smoke runs.

Retrieval neutralizes the query before fingerprinting (`Uncharger`): the pool
stores fragments mostly in their neutral writing form and ECFP encodes formal
charge, so a protonated query under-matches the pool. On an amidine test case
the best Tanimoto over 31.4M rows rose from 0.387 (charged query) to 0.467
(neutral) and genuine amidine analogs appeared in the top hits. Only the
Morgan query is neutralized; pose generation, charges and scoring still use
the charged fragment.

### Attachment-aware channel

`--attach-fp-prefix work/patent_pool_attach` adds a second Morgan index built
on the *marked* fragment: every stored attachment atom (canonical positions in
`attach_atoms`; one mark per cut, so a geminally cut atom gets two or more) is
rewritten as an unlabeled dummy, i.e. the fragment as an R-group. The
fingerprint therefore encodes *where* the fragment attaches, not only what it
contains. Unlabeled marks keep the score invariant to how a symmetric linker
happened to be numbered. The build re-marks and then uncharges all 31.4M rows
(≈10–20 min on 6 workers, 4.0 GB) and needs its own prefix — the cache check
is shape-only. Rows whose cut cannot be marked (a saturated atom, ~0.06%) get
an all-zero row and are only reachable through the global channel.

The two channels rank independently and their *orders* are fused by
reciprocal rank (`fuse_rankings`, k0 = 10): Tanimotos from two different
fingerprints are not comparable numbers, so only the ranks are combined. Each
hit keeps `tanimoto_global` and `tanimoto_attach` (null when the row was not in
that channel's top-k); `tanimoto` is the larger of the two and is used for
display and the final tie-break.

`--attach-gate on` drops retrieved rows whose attachment identity — the
sorted (element, hybridization) multiset over the cut atoms — differs from the
query site: a carbon-attached candidate at a nitrogen site is a different edit
even when the geometry fits. `auto` (default) turns the gate on only for
`--query-smiles`, which pins a single site; the all-fragment benchmark keeps
it off so its numbers stay comparable. Dropped rows are counted per query in
`n_attach_gate_dropped`; the kept identity is recorded per hit as
`cand_attach_identity`.

Measured on the pinned amidine/azetidine test case (one attachment, cut at an
azetidine N): the global channel alone yields 26 placeable non-exact
candidates, the fused channels 25, and fused + gate 13 (14 of the 40 fused
hits attach through the wrong element). Without the gate three of the top six
by energy attach through a carbonyl carbon or an sp² amidine N and cannot
re-form the cut bond; with it the top six are all N-attached bicyclic amines.
Of the 25 fused candidates, 5 were found only by the attachment channel (best
E_r +0.04) and 10 only by the global channel (best E_r −0.09), while every
best-E_r candidate was in both. On this case the gate carries most of the
gain; the channel adds a handful of site-compatible candidates and costs an
index build, so it stays opt-in.

**Exact matches are excluded by default**, on constitution-only canonical
SMILES (`_nonisomeric`): a fragment cut from a 3D pose carries no stereo while
the pool row it came from may, so string equality both misses true matches
(one 13gs hit has Tanimoto 1.0 and no stereo in the query) and calls them
novel. `--keep-exact` disables the exclusion; the summary still reports how
many exact rows retrieval surfaced (`n_exact_available`).

The control arm samples a random pool row under the same gates and scores it
without any similarity ranking.

## Exit vectors

Query side (`query_exit_vectors`): the fragment mol is matched to the ligand
by 3D coordinates; attachment atoms are those on cut bonds, and the partner
atom of each cut gives the exit point. The vector's length is therefore the
true cut-bond length, not a generic 1.5 Å.

Candidate side: attachment atoms come from the pool in canonical-SMILES
numbering. The candidate is the H-capped fragment, so its free valences are
exactly the hydrogens added at the cut bonds; `exit_directions` uses those H
positions as unit directions in the candidate's own frame.

## Placement

Kabsch on the attachment atoms plus one synthetic point per exit direction
(`atom + unit_direction * cut_length`), target = attachment atom + partner
atom.

- **One attachment** leaves the roll about the exit axis free. The prototype
  enumerates `--rolls` steps (default 24 → 15° quantization). Same-conformer
  self-placement measured against the query pose: 0.84 Å at 8 rolls, 0.33 Å at
  16, flat at 24–48, ~0.30 Å asymptote. The floor is the cap H sitting ~4.3°
  off the true exit vector; more rolls buy nothing.
- **Two or more attachments** fix the roll but leave attachment-slot
  permutations and, at each attachment, a choice among the H directions. All
  combinations are tried and thinned evenly to `--max-poses`.

Conformers: `--n-conformers` ETKDGv3 conformers per candidate, MMFF-optimized
when it succeeds. EEM charges are computed once per conformer (one batched
obabel call), not once per pose — they are pose-invariant but
conformer-dependent. Cost per candidate molecule is ~240 ms EEM plus ~15 ms per
pose scored.

## Pose arms

| arm | choice rule | meaning |
|---|---|---|
| realistic | fewest protein heavy-atom clashes (< 2.2 Å), then lowest fit RMSD | what a deployable rule can do |
| oracle | best weighted key overlap | upper bound; separates retrieval failure from placement failure |

The realistic arm *ranks* by clash count but does not reject clashing poses;
a rigid 2D analog can still interpenetrate the subpocket.

## Scoring

`fp_weights_in_env` scores each placed pose against the query complex's pocket
(same environment membership the query fragment sees), then weights are mean-1
normalized over their positive support. Overlap is weighted exact-key:

```
W = Σ min(w_query[k], w_candidate[k]) over shared keys
weighted = W / (n_query + n_candidate − W)
```

Uniform weights reduce this to Jaccard exactly.

**Same-pocket caveat.** Query and candidate are scored against the same
residue set, so they share *every* environment triplet key: coverage and
Jaccard are 1.0 by construction and are not reported. The informative quantity
is the weight profile — how strongly each already-known key is expressed — not
key discovery. This is a property of the prototype, not of the index use case.

## Run

```bash
# smoke: truncated index, 3 queries
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_smoke_place \
    --out work/patent_place_smoke.json \
    --n-queries 3 --top-k 12 --n-candidates 3 --n-conformers 1 --rolls 8 \
    --limit-fp-rows 300000 --max-complexes 6

# full: 50 queries against the whole pool, with the contact-local index arms
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
    --out work/patent_place_full2.json \
    --n-queries 50 --top-k 120 --n-candidates 6 --n-conformers 3 --rolls 24 \
    --n-profile-poses 4 --shell-gate 0.6 \
    --index work/rich_pocket_union.pkl \
    --protein-groups work/pdbbind_protein_groups.json --positive-mode smiles \
    --workers 6

# same run plus per-key energy-mark arms and contact-persistence metrics
# (adds --mark mean; output includes the *_mark index arms and shell_persist_*)
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
    --out work/patent_place_persist.json \
    --n-queries 50 --top-k 120 --n-candidates 6 --n-conformers 3 --rolls 24 \
    --n-profile-poses 4 --shell-gate 0.6 \
    --index work/rich_pocket_union.pkl \
    --protein-groups work/pdbbind_protein_groups.json --positive-mode smiles \
    --mark mean --workers 6

# reference in, scored candidates out: select every eligible fragment and
# place every retrieved candidate, then rank by score
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
    --out work/patent_place_13gs_all.json --pdb-id 13gs \
    --all-fragments --place-all \
    --top-k 120 --n-conformers 3 --rolls 24

# one cached complex by PDB ID: default fragment choice (largest 1- and 2-attach)
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
    --out work/patent_place_13gs.json --pdb-id 13gs \
    --top-k 120 --n-candidates 6 --n-conformers 3 --rolls 24

# an explicit override: pin one fragment (matched on constitution; stereo ignored)
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
    --out work/patent_place_13gs_pin.json --pdb-id 13gs \
    --query-smiles 'Nc1ccc(S(=O)(=O)Nc2ccccn2)cc1' \
    --top-k 120 --n-candidates 6 --n-conformers 3 --rolls 24

# any target structure (protein + co-crystal ligand), not a cached complex
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
    --out work/patent_place_target.json \
    --structure /path/to/target.pdb \
    --ligand-smiles '<co-crystal ligand SMILES>' \
    --all-fragments --place-all \
    --top-k 120 --n-conformers 3 --rolls 24
```

`--pdb-id` reads `<cache>/<id>.pdb` (default cache:
`/Users/mai/Software/fragenv/data/pdbbind_cache`) and skips the `--selected`
batch; the ID is still looked up there for the parent ligand SMILES, or pass
`--ligand-smiles` for an ID absent from that TSV. The ligand must be a HETATM
record with experimental coordinates. The default selection takes the largest
1- and 2-attachment fragment per complex (`--max-per-complex`); with no flag
the script is already a complete select → retrieve → place → score run for
that reference.

`--all-fragments` removes the fragment-selection heuristic (every fragment
with 1–2 attachments inside the heavy band is a query), and `--place-all`
removes the stop-after-`--n-candidates` placement cutoff (every deduped,
non-exact retrieved row up to `--top-k` is placed and scored). Together they
leave the score ranking as the only selector. The output gains a flat
`candidates` list sorted by `--rank-by` (`energy` = `shell_energy_r_mean`,
the default; `persist` = `shell_persist_energy_r`; `weighted` = realistic
overlap, kept as the literal FRASE arm — on same-pocket scoring it is
saturated by generic keys and nearly ties) with
the query fragment, Tanimoto, retrieval rank, clash count and shell metrics
for every placed candidate; the same table is printed
(`--top-show`, default 10 in `--pdb-id` mode, 0 in batch). Each candidate
also carries `cand_dummy_smiles`, the H-capped fragment rewritten with one
`[*]` per attachment atom (the pool's canonical attachment positions), i.e.
the replacement as an R-group ready to join at the same site.

`--query-smiles` pins one exact fragment instead of the default pick and
bypasses the attachment/heavy filters. If the requested cut is not in the
generated fragment set (the paper cutter never breaks ring↔alpha bonds, e.g.
an amide at a ring nitrogen), the pinned fragment is built directly from the
parent ligand — substructure match, boundary bonds cut, H-capped in the parent
frame — and registered on the shared fragment table. A query that is not a
substructure of the ligand, covers the whole ligand, or matches more than one
site equally well is an error, not a silent skip. This is the single-site
replacement path: the pinned fragment is the part at the assigned cut that gets replaced,
and everything else in the parent ligand stays in its experimental pose. With
a pinned fragment, `--ligand-context` resolves to `clash`, so the retained
ligand atoms join the obstacle set used for pose selection and the clash
count (the atoms bonded across the cut are excluded, since the candidate's
attachment atom takes exactly that position); `--ligand-context ignore` keeps
the all-fragment benchmark protocol, where only protein atoms are obstacles.
Each query row records `ligand_context` and `n_obstacle_atoms`.

For a first look at which fragments a target offers (and therefore what to
pin), run once with `--all-fragments`; the printed `q1 …` lines and
`queries[].smiles` are the fragment SMILES. If the same fragment occurs twice
in one ligand, the pin resolves to the first match.

`--structure` replaces the cached PDBbind file with any target structure; the
run label is `--pdb-id` or the file stem. The reference ligand must be a
HETATM residue with experimental coordinates whose identity matches
`--ligand-smiles` (heavy-atom count plus bond-order template; `HOH` is
ignored), because the query fragments and their exit vectors are cut from
that pose. An apo structure cannot be used — with no ligand there is no
query fragment pose. Operationally the data requirement is: protein + one
posed ligand, i.e. exactly what a co-crystal structure provides.

Index arms (`--index`) still need the ID in `--protein-groups` to have
same-protein positives; a custom target without a group entry simply gets no
positives and those arms are skipped.

`--n-profile-poses` sets how many diverse poses per candidate are scored on the
contact shell (averaged into `shell_energy_r_mean` and used for the
`shell_persist_*` / displacement metrics); `--shell-gate` is the minimum 2D
Tanimoto for the contact-local ranking arms.

Output JSON: per-query retrieval hits, every pose-arm score, controls, a
summary of `best_realistic`/`best_oracle` distributions, the pose gap
(oracle − realistic), clash/min-distance means, the control arm, and the
flat score-ranked `candidates` list. The same flat table is also written as
CSV next to `--out` (`work/x.json` → `work/x.csv`); `--out-csv PATH` moves it
and `--out-csv ''` suppresses it. Columns are the union of record keys, so a
run with the attachment channel adds `tanimoto_global`, `tanimoto_attach` and
`cand_attach_identity`.

## Smoke findings (3 queries, 300k pool rows)

- realistic 0.970, oracle 0.972 → **placement is not the bottleneck**; the
  pose gap is ~0.
- random control 0.946 — retrieval separates from chance by only ~0.02 with
  the small index, because the softmax weight profile is fairly flat.
- realistic arm keeps 11.7 protein contacts < 2.2 Å at 1.36 Å minimum
  distance: 2D analogs are not guaranteed to fit the subpocket.

These numbers are from a 1% pool slice with one conformer and 8 rolls.

## Full run (50 queries, 31.4M-row pool)

- **46/50 scored.** The four unscored queries are the same fragment
  (`CC1OCC(O)C1O`) in four complexes: all 40 retrieved rows are exact
  (stereo variants of one constitution), so excluding exact matches consumed
  the whole `--top-k` depth and left nothing to place. Retrieval depth is the
  limiting factor when a fragment is both prevalent and exact; raise
  `--top-k` or dedupe the pool on constitution for those queries.
- best realistic **0.960**, best oracle **0.971**, random control **0.935**
  (mean over queries; 6 candidates placed per query).
- top-1 realistic **0.943**, top-1 oracle **0.959** → pose gap **+0.016**:
  placement still costs little relative to having the right analog.
- retrieval beats the control in **45/46** queries best-of-candidates but
  only **31/46** on the deployed top-1 rule.
- mean top-1 Tanimoto 0.748; the realistic rule selects **1.3** protein
  contacts < 2.2 Å with a 2.19 Å minimum distance (much better than the smoke
  run's 1 conformer / 8 rolls — more conformers and rolls find clash-free
  poses).
- retrieval 202 s, scoring 701 s for 50 queries (6 candidates, 3 conformers,
  24 rolls).

**Reading the numbers.** The ~0.94–0.96 band is the same-pocket baseline:
query and candidate already share every environment key, so the metric only
rewards reproducing the weight *profile*. The deployed signal above chance is
0.943 vs 0.935 (top-1) and the best-of-6 signal is 0.960 vs 0.935. The
end-to-end arm (below) shows why this does not transfer: the query pocket
dominates weighted min-overlap, so the fragment's contribution is too small to
rank analogs or to carry over into library retrieval.

**What this means for the pipeline.** Placement is solved; the full 12 Å
environment profile is the wrong replacement score. Ranking patent analogs
needs a contact-local signal — the interaction terms on the residues the query
fragment actually engages (the per-atom E/a marks, or an energy/clash score
restricted to its contact shell) — with 2D similarity and geometric fit as
independent filters, not a whole-pocket fingerprint overlap.

## End-to-end index arm (`--index`)

The same-pocket overlap saturates, so it cannot say whether a placed analog is a
usable replacement. With `--index`, the selected pose's mean-1 per-key weights
are fed to `WeightedTripletIndex.query(query_fp_weights=..., combine="min")`
and the rank of the first positive is recorded. Library fragments carry keys
from **their own** pockets, so the shared-key ceiling does not apply here.

Arms per query:

| arm | query weights |
|---|---|
| `ref` | the query fragment in its experimental pose (ceiling) |
| `placed_best_real` | realistic pose of the candidate with the best overlap |
| `placed_rank1` | realistic pose of the top retrieved candidate |
| `placed_oracle` | oracle pose of the best-overlap candidate (bound) |
| `control` | realistic pose of the random candidate |

Positives are controlled by `--positive-mode`:

- `smiles` (default): the query fragment's **chemotype** (same `frag_smiles`)
  in other complexes of the same protein. Fragment-specific, so it can test
  whether the analog stands in for the original.
- `protein`: any fragment from another complex of the same protein. This is
  pocket-dominated — a random placed fragment retrieves same-protein
  fragments about as well as the real query (smoke: control MRR 0.61 vs ref
  0.51), so it confirms context preservation but cannot rank analogs.

Reported per arm: MRR, R@10, R@50, median rank, and the number of eligible
queries on which no positive was retrieved (`no_positive`). An eligible query
has the same fragment in **another complex** of the same protein; queries whose
only exact fragment is in their own complex are excluded (`n` is lower than the
query count). The script writes `n_pos_fids` per query.

**Full result (50 queries, chemotype positives).** Only **13/50** queries have
an eligible positive: the same fragment in another complex of the same protein.
For the other 37 the fragment occurs only in the query's own complex, which the
ranker excludes; they are **excluded from the metrics**, not counted as misses
(the evaluator used to score them as rank 0, which is why older all-query MRR
values in this file are ~0.26× the numbers below).

| arm | n | MRR | R@10 | R@50 |
|---|---|---|---|---|
| `ref` (query fragment) | 13 | 0.553 | 0.77 | 0.92 |
| `placed_best_real` | 9 | 0.151 | 0.67 | 0.89 |
| `placed_rank1` | 9 | 0.260 | 0.56 | 0.78 |
| `placed_oracle` | 9 | 0.311 | 0.67 | 0.89 |
| `control` (random placed) | 13 | 0.435 | 0.69 | 0.77 |

(`n=9` for the placed arms: at top-k 40 the other four eligible queries have
only exact hits, so no analog is placed.) Paired, the original fragment ranks
the positive better than the placed analog in **7/9** queries. More tellingly,
the ranks are often *identical* across the ref, placed, and random-control arms
(e.g. rank 9 for every arm on 184l), and the random control ranks a positive in
all 13 eligible queries. Weighted min-overlap is dominated by the keys of the
query pocket, so any fragment placed there produces nearly the same ranking;
the fragment's own chemistry is a minor perturbation. A placed analog therefore
does **not** inherit the query fragment's chemotype-retrieval behaviour.

The four `CC1OCC(O)C1O` queries are the extreme case: `ref` ranks the positive
1st, but every retrieved analog is exact and excluded, so there is no placed
arm. Raising `--top-k` (or deduping the pool on constitution) would give those
queries analogs.

```bash
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
    --out work/patent_place_index.json \
    --n-queries 50 --top-k 40 --n-candidates 6 --n-conformers 3 --rolls 24 \
    --index work/rich_pocket_union.pkl \
    --protein-groups work/pdbbind_protein_groups.json --positive-mode smiles
```

## Fragment borrowing (`--borrow-index`)

The patent pool answers "which 2D analog replaces this fragment". The weighted
environment index answers a different question: **which fragment already
occupied a similar local protein environment elsewhere** — the borrowing
operation FRASE was built for. With `--borrow-index`, the query fragment is
scored in its target pocket exactly as the library was built, the index is
ranked, and the top hits are cut out of their source complexes and placed into
the query site through the same `score_candidate` path as pool candidates.

Feature parity is enforced by construction: `provider` is built from the
index's own `features` dict, and the query weights and marks come from
`fp_weights_in_env` — the same function the index build used, verified to
reproduce stored weights and marks with zero difference on a sample
(`docs/REFMOL_QUERY.md`). The query is scored once per fragment; the pocket
is the same for all candidates.

Retrieval (`--borrow-rank`):

- `mark` (default): per-key contact-energy marks, normalized mean-1, scored by
  weighted exact-key overlap `W/(|q|+|l|-W)`. Zero-contact keys carry no
  information and are excluded from the support. This is the discriminating
  signal: the 680-key fingerprints are saturated, so plain key overlap
  separates pockets poorly.
- `weight`: the same overlap over per-key interaction weights. Uniform
  weights reduce it to exact-key Jaccard.

Gates, applied before placement: equal attachment count, `|n_heavy - query| <=
--heavy-band`, source PDB different from the query's (`--borrow-include-self`
keeps it), exact constitution excluded unless `--keep-exact`, and the
attachment-identity gate when it is on. The indexed fragment is re-cut from its
deposited parent via `_RefFragProvider.ligand_context` — a ligand-only context
that skips the 12 A environment and triplet table the query site already
provides — and its cut atoms are mapped into canonical-SMILES order for
placement. Symmetric parents are disambiguated by the stored `_frag_coords`
(19% of rows share their `(pdb, smiles)` key).

Output: the per-query records gain `borrow_hits` plus `n_borrow_scanned`,
`n_borrow_gate_dropped`, `n_borrow_context_miss`, and `selected` (the joint
pool+borrow recommendation, below); the flat `candidates` table gains
`source` (`pool`/`borrow`), `selected`, `env_score`, `borrow_fid`,
`borrow_pdb`, and `pool_row` is null for borrowed rows. `summary["borrow"]`
reports the same realistic/oracle distributions and contact-shell means as
the pool arm, gated on `env_score` instead of 2D Tanimoto;
`summary["combined"]` reports the cross-channel selection. `--borrow-only`
skips pool retrieval entirely.

```bash
# borrow channel only, 50 queries from the PDBbind selection
conda run -n rxn-env python scripts/patent_place.py \
    --borrow-index work/rich_pocket_union.pkl --borrow-only \
    --cache /Users/mai/Software/fragenv/data/pdbbind_cache \
    --borrow-cache /Users/mai/Software/fragenv/data/union_cache \
    --n-queries 50 --n-candidates 6 --mark mean \
    --out work/patent_place_borrow.json

# both channels in one run (borrow hits join the flat table)
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
    --borrow-index work/rich_pocket_union.pkl \
    --index work/rich_pocket_union.pkl \
    --protein-groups work/pdbbind_protein_groups.json --positive-mode smiles \
    --n-queries 50 --top-k 120 --n-candidates 6 --mark mean \
    --out work/patent_place_borrow_pool.json

# borrow depth experiment: pool stays at 6, borrow retrieves 12
conda run -n rxn-env python scripts/patent_place.py \
    --pool work/patent_pool.parquet --fp-prefix work/patent_pool \
    --out work/patent_place_borrow_d12.json \
    --n-queries 50 --top-k 120 --n-candidates 6 --borrow-n-candidates 12 \
    --n-conformers 3 --rolls 24 --n-profile-poses 4 --shell-gate 0.6 \
    --index work/rich_pocket_union.pkl --borrow-index work/rich_pocket_union.pkl \
    --protein-groups work/pdbbind_protein_groups.json --positive-mode smiles \
    --mark mean --workers 6
```

### Full result (50 queries, both channels)

The run above on the 50-query pool set against the union index (334,002
fragments from 16,663 complexes, `--mark mean`, 6 candidates per channel,
3 conformers, 24 rolls, 4 profile poses, shell gate 0.6). Every query received
6 borrowed candidates (300 total); 0 attachment/heavy-band/self-PDB drops, 10
source-context misses (structure absent or parent-ligand parse failure), 6-33
rows scanned per query before the gates (mean 9).

| | pool | borrow |
|---|---|---|
| candidates / query | 6.0 | 6.0 |
| mean top-1 2D Tanimoto | 0.80 | n/a (not pool members) |
| top-1 realistic weighted | 0.941 | 0.945 |
| best realistic weighted | 0.958 | 0.958 |
| best oracle weighted | 0.972 | 0.970 |
| mean `env_score` of top-1 | — | 0.624 |
| `shell_energy_r_mean`, candidate mean | 0.323 | 0.270 |
| `shell_energy_r_mean`, top-1 | 0.334 | 0.320 |
| `shell_energy_r_mean`, gated best | 0.579 (n=46) | 0.505 (n=27) |
| `shell_persist_energy_r`, gated best | 0.603 (n=46) | 0.548 (n=27) |
| random control `shell_energy_r_mean` | 0.186 | — |

- **Borrowing recovers placement quality without 2D similarity.** Paired
  per-query best-oracle overlap is pool 0.972 vs borrow 0.970 (25 pool wins /
  24 borrow wins / 0 ties, mean delta −0.002). Borrowed fragments are not in
  the patent pool, so their 2D Tanimoto to the query is undefined; the channel
  adds genuinely new candidates that place into the query site as well as the
  best 2D analogs do.
- **`env_score` ranks global pose plausibility, not contact fit.** Across the
  300 borrowed candidates Spearman(env_score, oracle weighted) = +0.305
  (p=7e-8) but (env_score, `shell_energy_r_mean`) = +0.026 (p=0.65) and
  (env_score, persistence) = +0.140 (p=0.016). The env-ranked top-1 is the
  shell-best candidate in only 11/49 queries. Use environment overlap to gate
  the channel, and the placement contact profile to choose among hits.
- **The pool channel still produces better contact fits** (candidate-mean
  `shell_energy_r_mean` 0.323 vs 0.270; gated best 0.579 vs 0.505), which is
  expected — the 50 queries are patent-pool queries, and their analogs were
  selected to be close in 2D. The borrow result is the right baseline for a
  target whose chemotype is absent from the pool, where only the index channel
  has candidates.

The earlier smoke run (13gs, 2 queries, 3 candidates each) placed borrowed
fragments from 2gss/3gus/3ie3 with realistic weighted overlap 0.92-0.97 and
stored-mark environment overlap ~0.53-0.61; that overlap range matches the
full-run top-1 mean (0.624), confirming the fingerprint overlap is near
saturation and the contact profile is the informative placement score, exactly
as in the pool arm.

### Joint selection across channels

With both channels enabled, each query carries one recommended replacement in
`row["selected"]`: the gate-passing candidate with the highest
`shell_persist_energy_r` (ties broken by `shell_energy_r_mean`). Each channel
keeps its own gate — 2D Tanimoto ≥ 0.6 for patent-pool analogs, environment
overlap ≥ 0.6 for borrowed fragments — and the two retrieval scores are never
compared directly. `summary["combined"]` reports the per-channel gated means,
the union, the paired channel delta, and the source mix.

From the full run (50 queries; pool gated in 46, borrow in 27, union 46):

| selection | mean `shell_persist_energy_r` |
|---|---|
| pool only (gated) | 0.659 |
| borrow only (gated) | 0.621 |
| union (better channel per query) | 0.698 |
| selected source | pool 35, borrow 11, none 4 |

- Paired on the 27 queries where both channels pass a gate: pool wins 16,
  borrow 11, 0 ties; mean Δ (borrow − pool) −0.090, median −0.019. Borrow
  wins average +0.160, losses −0.262 — the channel is not uniformly better.
- The union is optimistic by construction (best of two channels). A Monte
  Carlo null that adds six more draws from the *pool* candidate distribution
  raises the per-query max by +0.036 on the same subset; the observed union
  gain there is +0.065 (+0.039 over the full 46). Roughly +0.03 is
  complementarity rather than selection noise: the borrow channel's tail
  occasionally exceeds the pool's best. Treat it as a complementary
  generator, not a replacement.

### Borrow depth: 6 vs 12 candidates

`--borrow-n-candidates` sets the borrow channel's depth independently of the
pool's `--n-candidates`. Re-running the same 50 queries with 12 borrow
candidates (3,995 candidates total; all 50 queries filled 12 slots; median 15
rows scanned per query, max 92, so the 400-row scan budget was never the
limiter) gives a strictly nested comparison: the pool hits are bit-identical
to the depth-6 run and the first six borrow hits are a prefix of the twelve.

| | depth 6 | depth 12 |
|---|---:|---:|
| borrow candidates / query | 6.0 | 12.0 |
| borrow gated best `shell_persist_energy_r` | 0.621 (n=27) | 0.693 (n=27) |
| union persist (better channel per query) | 0.698 (n=46) | 0.711 (n=46) |
| selected source (pool/borrow/none) | 35/11/4 | 32/14/4 |
| paired channel delta, both gated (n=27) | pool 16 / borrow 11 | pool 13 / borrow 14 |
| mean Δ (borrow − pool) | −0.090 | −0.019 |

- **Borrow gain.** Paired over the 27 gate-passing queries: 6 wins, 0 losses,
  21 ties, mean +0.071, bootstrap CI [+0.008, +0.155]. All of the gain is a
  tail effect: the four informative queries are rescues of bad depth-6 fits
  (1ai4 0.043 → 0.889, 1ai5 0.180 → 0.839, 1ajn 0.440 → 0.662, 1a42 0.022 →
  0.156). In 21 queries the depth-6 best remains the best.
- **End-to-end gain.** The union moves 0.698 → 0.711: 4 wins, 0 losses, 42
  ties, mean +0.014, CI [+0.000, +0.034]. The selection changes on 1aht
  (pool 0.836 → borrow 0.844), 1ai4 (pool 0.527 → borrow 0.889), 1ajn (pool
  0.441 → borrow 0.662) and 1ajx (borrow 0.826 → 0.861). The channel remains
  nearly balanced on the gated subset (−0.019 vs −0.090 at depth 6), so depth
  is what lets the borrow channel compete with the pool on contact fit.
- **How much is selection noise?** Treating the 12 placed hits as fixed and
  asking what the max of six random hits from the same twelve would be, the
  extra-draw null raises the per-query max by +0.029 — the observed gain is
  2.4× that. The excess (+0.042) has CI [−0.003, +0.102], so most of the gain
  is genuine depth, but not established at 27-query resolution.
- **Depth 20 is not warranted yet.** The benefit comes from rescuing queries
  whose shallow hits have poor contact fit: only 6 of the 27 gated queries
  improve, and the end-to-end gain (+0.014) already sits at the noise
  boundary. A depth-20 run (~2 h) is only worth it if the borrow channel
  becomes the primary generator (targets with no pool analogs), where it is
  not second-guessed by the pool. Depth 12 is the sensible default: zero
  regressions, 4/46 selections improved, and it covers the tail that depth 6
  misses.

### Does the 3D profile add anything over 2D?

Computed from the 50-query placement-only run (`work/patent_place.json`), over
queries with at least three placed candidates:

- Spearman(2D Tanimoto, realistic profile overlap) = **−0.11** (median −0.09):
  the two rankings are essentially unrelated, so the profile is not a
  restatement of 2D similarity.
- The best-overlap candidate is the top-Tanimoto candidate in only **8/45**
  queries.
- Within-query spread is small: median 0.028 for profile overlap vs 0.078 for
  Tanimoto.
- Profile overlap correlates weakly with a clash-free pose: Spearman(overlap,
  −clashes) = +0.14 mean / +0.31 median.

So the profile carries information the 2D fingerprint does not, but the
signal is weak and only partially tracks physical fit; the index arm is what
decides whether it is strong enough to rank replacements.

## Contact-local score

The end-to-end result says the whole-pocket profile cannot rank replacements,
so the prototype now also computes a **fragment-specific** comparison on the
query's own contact shell: residues with a heavy atom within 5 Å of the query
fragment (`contact_shell`). Nothing about the pocket membership changes; only
the comparison set is restricted.

`residue_profile` keeps the per-residue values behind `relevance_weights`
(softmax weight, contact-gated energy, and shell mask) instead of collapsing
them onto triplet keys. `profile_overlap` then scores a placed candidate:

| score | meaning |
|---|---|
| `shell_recall` | fraction of the query's contact residues the candidate also touches |
| `shell_jaccard` | union-normalized version of the same |
| `shell_weight_cos` | cosine of the two softmax weight vectors on the query shell |
| `shell_energy_r` | Pearson correlation of per-residue contact-gated energies on shared shell residues (`None` if <3 shared or one side is constant) |

### Full result (50 queries, 6 candidates, 3 conformers, 24 rolls, 4 profiles)

All 50 queries are now scorable. `--top-k 120` (up from 40) surfaces non-exact
analogs for the prevalent `CC1OCC(O)C1O` fragment, which previously consumed the
whole depth with stereo variants. Mean top-1 Tanimoto 0.759; mean contact shell
14.8 residues.

| score | top-1 | best whole-pocket | best shell-ranked (gated) | oracle | random control |
|---|---|---|---|---|---|
| `shell_recall` | 0.661 | 0.756 | 0.669 | 0.841 | 0.649 |
| `shell_weight_cos` | 0.989 | 0.993 | 0.989 | 0.995 | 0.985 |
| `shell_energy_r` | 0.347 | 0.573 | 0.572 | 0.649 | 0.137 |
| `shell_energy_r_mean` | 0.355 | 0.398 | **0.620** | 0.376 | 0.199 |

`best shell-ranked` selects the candidate with the highest multi-pose
`shell_energy_r_mean` among analogs with Tanimoto ≥ 0.6 (44/50 queries have
one). It beats the random control in **41/44** queries (+0.415 mean). The
unrestricted **candidate-mean** comparison avoids best-of-k selection bias:

| score | candidate mean | control | paired wins |
|---|---|---|---|
| `shell_energy_r_mean` | 0.343 | 0.199 | **38/50** |
| `shell_energy_r` | 0.345 | 0.137 | 36/50 |
| `shell_jaccard` | 0.558 | 0.524 | 34/50 |
| `shell_recall` | 0.652 | 0.649 | 22/50 |
| `shell_weight_cos` | 0.988 | 0.985 | 39/50 |

Only the per-residue energy correlation has a material gap. `shell_recall`
(which residues are touched) is dead: a random fragment placed in the same
pocket touches the same shell. `shell_weight_cos` is saturated. Persistence
weighting of the same energies improves this further (next subsection).

**What the signal is.** Within a query, `shell_energy_r` is independent of 2D
similarity (Spearman +0.04 median) and of clash count (+0.05 median), so it is
not a restatement of Tanimoto or geometric fit. The multi-pose-ranked analogs
are still chemically related (mean Tanimoto 0.74).

**Multi-pose averaging.** Within-query scatter of `shell_energy_r_mean` is 0.192
(single pose 0.303). The multi-pose rule and the single-pose rule pick the same
candidate in 20/44 queries and the multi criterion is never worse by
construction (mean +0.126). But an independent check — the selected candidate
scored in its *other* (oracle) pose — gives mean +0.108, **median 0.000**,
15/44 positive, 10th percentile −0.21. The gain is concentrated in a few
queries; typical candidate ranking is not more reliable. Averaging is a variance
reduction, not a fix for pose noise.

### Contact persistence and interaction displacement

The same profiled poses give per-residue contact frequencies and pose-averaged
energies. `shell_persist_energy_r` correlates the query's per-residue energies
with candidate energies averaged over the ensemble, so a contact seen in one
pose contributes in proportion to how often it survives;
`shell_contact_retention` asks whether the query's favourable contacts are made
again across the ensemble; `shell_delta_energy` is the signed energy change
(candidate − query, favourable positive). Unrestricted candidate mean over all
hits, same 50 queries and control as above:

| score | candidate mean | control | paired wins |
|---|---|---|---|
| `shell_energy_r_mean` | 0.343 | 0.199 | 38/50 |
| `shell_persist_energy_r` | 0.445 | 0.229 | **41/50** |
| `shell_delta_energy` | −0.087 | −0.110 | 37/50 |
| `shell_contact_retention` | 0.729 | 0.702 | 29/50 |
| `shell_persist_recall` | 0.679 | 0.660 | 28/50 |
| `shell_persist_all` | 0.533 | 0.536 | 23/50 |

Gated best-of-candidates: `shell_persist_energy_r` 0.725 vs 0.222 control,
**43/44** wins (`shell_energy_r_mean`: 41/44). The single loss is 1ahy — a
4-heavy-atom fragment with 6 hits, where the control's 0.668 is noise.

`shell_persist_energy_r` and `shell_energy_r_mean` are correlated within a
query (median r = 0.85 across candidates), but persistence weighting selects a
different analog in 21/44 queries and raises the selected candidate's P_E from
0.495 to 0.695. Binary contact persistence, retention and unsigned coverage add
little: a random fragment placed in the same pocket touches the same residues
and keeps a similar fraction. The signed displacement is a moderate signal on
its own and is dominated by the persistence-weighted correlation.

**End-to-end.** Selecting the analog by `shell_persist_energy_r` gives the best
placed index arm: MRR 0.527 vs 0.487 (energy-selected), 0.435 (random control),
0.553 (`ref`). Paired on the 13 eligible queries: 3W/0L/10T against the
energy-selected arm, 7W/1L/5T against control, and 2W/4L/7T against `ref`
(mean ΔMRR −0.025). The mark path still degrades through the placed analog
(0.356).

### Contact-local index arm

The score is residue-level, while the library index stores whole-pocket triplet
keys. Two ways to bridge them were tested:

1. **Shell-restricted triplet query** (`residue_mask` in
   `fragment_fp_weights`): keep only triplets whose three residues are inside
   the fragment's own contact shell, then query the index. Identical keys to
   the full query, just a subset.
2. **Contact-local candidate selection**: use the whole-pocket weights of the
   analog chosen by the gated `shell_energy_r_mean` rule.

Restricted to the 13 queries with an eligible positive (same convention as
above; the other 37 are excluded, not scored as misses):

| arm | n | MRR | R@10 | R@50 |
|---|---|---|---|---|
| `ref` (query fragment, whole pocket) | 13 | 0.553 | 0.77 | 0.92 |
| `placed_shell_best` (contact-ranked analog) | 13 | 0.487 | 0.69 | 0.85 |
| `placed_persist_best` (persistence-ranked analog) | 13 | 0.527 | 0.69 | 0.85 |
| `placed_best_real` (whole-pocket-ranked analog) | 13 | 0.380 | 0.69 | 0.92 |
| `placed_rank1` (top Tanimoto) | 13 | 0.416 | 0.69 | 0.77 |
| `placed_oracle` | 13 | 0.483 | 0.77 | 0.92 |
| `control` (random placed) | 13 | 0.435 | 0.69 | 0.77 |
| **any `*_shell` triplet query** | 12–13 | **0.000** | 0.00 | 0.00 |

- **Shell restriction destroys retrieval.** The true positive's rank moves from
  1–512 (`ref`) to **871–133,639** (`ref_shell`; median 9,514); placed analogs'
  shell queries rank it 1,240–74,368. No shell query puts a positive in the top
  50. A contact shell of ~15 residues yields too few triplets to identify
  anything, and the surviving keys are the least specific ones.
- **Selection barely moves retrieval.** `placed_shell_best` (0.487) is below
  `ref` (0.553) and only modestly above the random control (0.435); in **6/13**
  eligible queries its rank is *identical* to the control's, 7/13 for
  `placed_best_real` and `placed_rank1`, 8/13 for `placed_oracle`. The
  persistence-selected analog is the best of the placed arms (0.527, 5/13
  identical to control) but still does not reach `ref`. Whole-pocket triplet
  weights are dominated by the pocket, so any fragment placed there produces
  nearly the same ranking (see the end-to-end section above).
- The four previously unscored `CC1OCC(O)C1O` queries now rank their positive
  1st with the placed analog, confirming that the earlier gap was retrieval
  depth, not placement.

**Conclusion.** The contact-local residue score is a real analog-vs-random
filter, and its best form is the persistence-weighted energy correlation
(41/50 unbiased wins, 43/44 gated best-of-6; binary persistence and displacement
add little). It still cannot be expressed through the current triplet index:
reweighting triplets to the shell removes the signal. Selecting the analog by
persistence is the first selection that improves end-to-end retrieval (0.527 vs
0.487), but it remains below the query fragment itself (0.553) and only
modestly above the random control (0.435). A contact-local retrieval would need
the library side to store per-residue contact profiles (the same
`residue_profile` values) and to score query/candidate profile similarity
directly, rather than triplet keys. That index was built and tested in
`docs/PROFILE_INDEX.md`: contact-local profile metrics underperform the
same-universe triplet index (best profile arm MRR 0.620 vs 0.648); the
whole-pocket weight cosine is the useful projection and adds a small,
not-yet-significant gain under rank fusion (RRF 0.704, 4W/1L/8T).

### Per-key energy marks (existing index data)

The index already stores four per-key energy marks (`_fp_e_atom_mean`,
`_fp_e_atom_max`, `_fp_e_atom_min_leg`, `_fp_e_atom_cmean`), so a placed analog
can be returned as a mean-1 normalized mark vector (`return_marks=True`) and
ranked with `_dense_mark_scores` without any new index. This was a coarser
projection of the same per-key energies than a residue-profile index, but it
was what the then-existing index supported. Same 50 queries/candidates,
`--mark mean`,
restricted to the 13 eligible queries:

| arm | n | MRR | R@10 | R@50 |
|---|---|---|---|---|
| `ref` (stored weights) | 13 | 0.553 | 0.77 | 0.92 |
| `ref_mark` | 13 | 0.532 | 0.92 | 1.00 |
| `placed_best_real` / `_mark` | 13 | 0.380 / 0.302 | 0.69 / 0.54 | 0.92 / 0.69 |
| `placed_rank1` / `_mark` | 13 | 0.416 / 0.195 | 0.69 / 0.62 | 0.77 / 0.62 |
| `placed_shell_best` / `_mark` | 13 | 0.487 / 0.359 | 0.69 / 0.62 | 0.85 / 0.69 |
| `placed_persist_best` / `_mark` | 13 | 0.527 / 0.356 | 0.69 / 0.62 | 0.85 / 0.62 |
| `control` / `control_mark` | 13 | 0.435 / 0.195 | 0.69 / 0.62 | 0.77 / 0.62 |

- **The direct query transfers.** `ref_mark` matches `ref` (MRR 0.532 vs
  0.553) and improves R@10/R@50 (0.92/1.00 vs 0.77/0.92). Paired ranks are
  6 wins / 2 losses / 5 ties, but the two losses are large top-1 displacements
  (1a42, 1a69), so the mean ΔMRR is −0.03. Both sides are scored in their own
  deposited pose here, so the mark vector is directly comparable.
- **The placed analog does not.** Every mark arm is worse than its weight
  counterpart (paired 2–3 wins vs 6–8 losses): the analog is scored in the
  query pocket, and per-key energy marks amplify the pose/conformer and
  molecule differences that the mean-1 weights absorb.
- **Selection still differentiates, but not enough.** The contact-selected
  analog beats the random control more clearly under marks (8W/3L/2T, mean
  ΔMRR +0.164) than under weights (5W/2L/6T, +0.052), because the mark control
  is especially noisy — absolute ranks remain pocket-dominated and worse than
  the stored-weight query.
- This bounds the *coarse projection*, not the contact signal itself: the
  persistence-weighted score is real and improves end-to-end retrieval when
  used for selection (0.487 → 0.527). The per-residue index it pointed to was
  subsequently built and tested (`docs/PROFILE_INDEX.md`); per-key marks are
  simply the wrong projection of the signal, and the direct profile comparison
  recovers only a small complementary gain.

The script reports the arms above plus paired gaps. Full output:
`work/patent_place_full2.json` (weights), `work/patent_place_marks.json`
(weights + marks); the earlier single-pose run is
`work/patent_place_shell.json`.

## Caveats

- Global heavy-atom RMSD after placement is not meaningful: ETKDG conformers
  differ from the crystallographic conformation (biaryl/carbonyl flips) even
  for the same molecule. Only the local attachment frame is reproduced exactly
  (attachment atom < 0.05 Å, fit < 0.05 Å in tests).
- Scores are per-complex and not comparable across queries.
- The oracle arm is never deployable and is reported only to bound placement.
