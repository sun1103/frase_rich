# frase_rich

Weighted FRASE triplet fingerprints for fragment retrieval inside protein
pockets.

FRASE (FRAgments in Structural Environments) encodes a fragment's molecular
graph as discrete triplet keys (atom types, distances, sequence). Every key is
weighted equally, but a triplet's contribution to binding depends on the
environment it sits in.
This project keeps the discrete fingerprint for candidate recall and adds a
continuous weight per triplet, computed from the fragment's real protein
environment — explicit hydrogens, EEM partial charges, two-sided H-bond
geometry, interaction energy, and pharmacophore complementarity. It then asks
the only question that matters: **do those weights improve retrieval?**

## Result

The weights carry signal, but only when the query is fragment-local. With a
coarse query (CA coordinates and residue codes) or a whole-ligand reference
molecule, the weighting lands inside the noise floor. With a unified pocket and
the fragment in its deposited pose (the `reffrag` protocol), the weighted
overlap beats the nested all-uniform baseline by a wide margin.

Union benchmark — 16,663 complexes, 334,002 fragments, 680 keys, 997 scored
queries, positives are the same fragment SMILES in another complex of the same
protein:

| arm | MRR | R@10 | R@50 | R@100 | median rank |
|---|---|---|---|---|---|
| coverage Jaccard (uniform) | 0.085 | 0.201 | 0.505 | 0.644 | 48 |
| softmax-weighted overlap | **0.430** | 0.594 | 0.744 | 0.808 | 4 |
| E/a mark, mean | 0.400 | **0.616** | **0.809** | **0.876** | 5 |

Paired against Jaccard: weighted wins 899 / loses 61 (37 ties); the E/a mark
wins 840 / 133 (24 ties). An independent 120-query confirmation slice (seed 1)
reproduces the structure: Jaccard 0.108, weighted 0.489, mark 0.468 MRR, with
paired wins 108 / 8 and 103 / 12.

The same weights on the PDBbind-only index (184,186 fragments) give 0.070
Jaccard → 0.428 weighted → 0.432/0.446 for the mean/max marks, median rank
62 → 4–5.

What did *not* work is documented, not hidden: whole-ligand reference-molecule
query (weighted 0.090 vs 0.098 Jaccard), coarse stored query (0.082 vs 0.070),
top-k key focusing, and per-atom `E/a` as a replacement for the weights. See
[`docs/REFMOL_QUERY.md`](docs/REFMOL_QUERY.md).

## How it works

1. **Environment** — residues within 12 Å of the fragment (or the union of
   spheres around the parent ligand, one pocket per complex).
2. **Per-residue relevance** — a softmax over interaction energy: EEM partial
   charges with explicit hydrogens, backbone and sidechain atoms, electrostatics
   / van der Waals / H-bonds / aromatic and hydrophobic contacts, angle-aware
   H-bond geometry, and intensive (mean) pooling that removes the atom-count
   bias. A pharmacophore complementarity term is added.
3. **Per-triplet weights** — each triplet inherits the relevance of its
   residues, modulated by a pair-geometry prior, then aggregated per
   fingerprint key and normalized to mean 1.
4. **Scoring** — weighted exact-key overlap,
   `W = Σ min(w_q, w_l)`, `score = W / (n_q + n_l − W)`. Uniform weights give
   exactly Jaccard, which is the nested baseline.
5. **E/a marks** — an optional second vector: per key, mean interaction energy
   divided by fragment heavy-atom count. It trades some MRR for recall at fixed
   depth.

## Quick start

Requires Python ≥ 3.10, RDKit, numpy, scipy, and the `obabel` binary for EEM
charges (`OBABEL` to point elsewhere). The PDB parser and the paper
fragmentation rule are vendored into `frase_rich/pdb_io.py` and
`frase_rich/fragmentation.py`, so the package is standalone:

```bash
pip install -e .        # numpy + rdkit + scipy
```

Building needs two inputs: a TSV of `pdb_id` / `ligand_smiles` (the complexes
to index) and a directory of `<pdb_id>.pdb` structure files. Both are yours to
supply; the scripts have no bundled data.

```bash
python scripts/build_index.py \
    --selected data/selected.tsv --cache data/pdb_cache \
    --max-complexes 50 --out work/smoke.pkl

python scripts/evaluate.py \
    --index work/smoke.pkl --max-queries 500 \
    --output work/smoke_eval.json
```

Pass `--protein-groups work/protein_groups.json --positive-mode same-protein`
for the transfer benchmark instead of the default leave-one-complex-out mode.

The full benchmark chain — PDBbind cache, protein groups, sharded builds, the
union merge, and every evaluation arm — is in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Layout

```
frase_rich/
  geometry.py       residue codes, triplet enumeration, RBF / pair bias
  chem.py           explicit hydrogens, EEM charges (obabel batching)
  protein.py        PDB → residues with explicit H, protonation, HID/HIE
  features.py       atom props, two-sided H-bonds, mean-pooled energy
  weighting.py      environment, residue weights, per-fingerprint weights
  index.py          WeightedTripletIndex, weighted Jaccard query, save/load
  build.py          structure cache → weighted index
  evaluate.py       leave-one-complex-out retrieval (stored / refmol / reffrag)
  proteins.py       exact-sequence protein grouping
  profiles.py       cross-structure residue keys and profile metrics
  patent_decomp.py  parent SMILES → library-rule fragments with exit vectors
  patent_place.py   pool fingerprints, exit-vector placement, borrow
  pdb_io.py         PDB parsing and atom typing (vendored)
  fragmentation.py  paper FRASE fragmentation rule (vendored)
scripts/            build, merge, evaluate, pipeline, patent/profile drivers
tests/              unit tests (unittest)
docs/               design notes and reproduction commands
```

## Docs

- [`docs/WEIGHTED_INDEX.md`](docs/WEIGHTED_INDEX.md) — design, math, energy
  model, findings, caveats.
- [`docs/REFMOL_QUERY.md`](docs/REFMOL_QUERY.md) — reference-molecule and
  ref-fragment queries, scoring, full benchmark tables, negative results.
- [`docs/PLACEMENT.md`](docs/PLACEMENT.md) — patent placement prototype.
- [`docs/PROFILE_INDEX.md`](docs/PROFILE_INDEX.md) — residue-profile retrieval.
- [`docs/PATENT_BANK.md`](docs/PATENT_BANK.md) — patent fragment bank schema.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md) — every command used for the
  numbers above.

## Limitations

- The benchmark measures **environment transfer**, not binding affinity:
  positives are the same fragment SMILES observed in another complex of the
  same protein. Cross-protein transfer is harder and is reported separately.
- The `reffrag` protocol needs the query fragment **already posed**. This is a
  retrieval / fragment-borrowing tool, not a de novo pose scorer.
- The fingerprint space saturates (680 distinct keys over 334k fragments), so
  resolution is limited; the mean vs max ordering of the E/a mark is not stable
  across libraries.
- No structures, caches, or indexes are shipped. PDBbind is available to
  academic users on registration; RCSB entries are open.

## References

The FRASE fingerprint and the fragmentation rule this repository builds on:

> An, Y., Lim, J., Glavatskikh, M. et al. In silico fragment-based discovery
> of CIB1-directed anti-tumor agents by FRASE-bot. *Nat. Commun.* **15**, 5564
> (2024). https://doi.org/10.1038/s41467-024-49892-9

The environment-aware weighting, the reference-fragment query protocol, and the
retrieval benchmark are this project's work; the vendored parser and fragmenter
implement the paper's rules.

## License

MIT — see [LICENSE](LICENSE).
