# Reproducing the benchmark

Operational command log. The overview lives in `README.md`; this file keeps the
exact invocations used to build the indexes and produce the evaluation tables
in `REFMOL_QUERY.md` and `WEIGHTED_INDEX.md`.

All commands assume the `rxn-env` interpreter (RDKit, numpy, scipy, pyarrow,
Open Babel on `PATH` or `OBABEL`). Set the data root once:

```bash
export DATA=/path/to/fragenv/data            # structure caches and selections
```

## 1. Tests

```bash
conda run -n rxn-env python -m unittest discover -s tests -v
```

## 2. Small smoke build and evaluation

```bash
conda run -n rxn-env python scripts/build_index.py \
    --max-complexes 50 --out work/smoke.pkl

conda run -n rxn-env python scripts/evaluate.py \
    --index work/smoke.pkl --max-queries 500 \
    --output work/smoke_eval.json
```

## 3. Same-protein transfer benchmark

```bash
conda run -n rxn-env python scripts/protein_groups.py \
    --groups-out work/protein_groups.json \
    --selected-out work/selected_sameprotein.tsv --max-pids 500

conda run -n rxn-env python scripts/build_index.py \
    --selected work/selected_sameprotein.tsv --out work/rich_mean.pkl

conda run -n rxn-env python scripts/evaluate.py --index work/rich_mean.pkl \
    --protein-groups work/protein_groups.json --positive-mode same-protein \
    --max-queries 1000 --output work/sp_eval.json
```

## 4. Query modes

Reference-molecule query (whole-ligand environment; the arm that failed):

```bash
conda run -n rxn-env python scripts/evaluate.py --index work/rich_mean.pkl \
    --query-mode refmol --combine min \
    --cache $DATA/rcsb_cache \
    --protein-groups work/protein_groups.json --positive-mode same-protein \
    --output work/refmol_min.json
```

Unified pocket + ref-fragment query (the working protocol):

```bash
conda run -n rxn-env python scripts/build_index.py --env-center pocket \
    --selected work/selected_sameprotein.tsv --out work/rich_pocket_cmean.pkl

conda run -n rxn-env python scripts/evaluate.py \
    --index work/rich_pocket_cmean.pkl --query-mode reffrag \
    --score-mode emark --mark mean \
    --cache $DATA/rcsb_cache \
    --protein-groups work/protein_groups.json --positive-mode same-protein \
    --output work/eval_mark_mean.json
```

## 5. PDBbind build (10,292 complexes)

```bash
conda run -n rxn-env python scripts/pdbbind_cache.py \
    --root $DATA/pdbbind_pl/P-L \
    --out-cache $DATA/pdbbind_cache \
    --out-tsv $DATA/pdbbind_selected.tsv

conda run -n rxn-env python scripts/build_index.py \
    --selected $DATA/pdbbind_selected.tsv \
    --cache $DATA/pdbbind_cache \
    --env-center pocket --out work/rich_pocket_pdbbind.pkl

conda run -n rxn-env python scripts/protein_groups.py \
    --selected $DATA/pdbbind_selected.tsv \
    --cache $DATA/pdbbind_cache \
    --groups-out work/pdbbind_protein_groups.json \
    --selected-out work/pdbbind_sameprotein.tsv

conda run -n rxn-env python scripts/evaluate.py \
    --index work/rich_pocket_pdbbind.pkl \
    --query-mode reffrag --score-mode emark --mark mean \
    --cache $DATA/pdbbind_cache \
    --protein-groups work/pdbbind_protein_groups.json \
    --positive-mode same-protein --max-queries 1000 \
    --output work/pdbbind_eval_mark_mean.json
```

## 6. PDBbind + RCSB/BindingDB union (16,663 complexes, 334k fragments)

Build the RCSB side in disjoint shards, then merge:

```bash
conda run -n rxn-env python scripts/build_index.py --cache $DATA/union_cache \
    --selected work/rcsb_q0.tsv --env-center pocket --out work/rich_pocket_rcsbq0.pkl
# ... repeat for rcsb_q1..q3

conda run -n rxn-env python scripts/merge_indexes.py \
    --base work/rich_pocket_pdbbind.pkl \
    --shards work/rich_pocket_rcsbq{0,1,2,3}.pkl \
    --out work/rich_pocket_union.pkl
```

Evaluate (**the cache must match the index source** — `union_cache`, not
`rcsb_cache`; the latter lacks every PDBbind-only complex and reports them as
context failures):

```bash
conda run -n rxn-env python scripts/evaluate.py \
    --index work/rich_pocket_union.pkl --query-mode reffrag \
    --score-mode emark --mark mean \
    --cache $DATA/union_cache \
    --protein-groups work/union_protein_groups.json \
    --positive-mode same-protein --max-queries 1000 \
    --output work/union_eval_mark_mean.json
```

The 120-query confirmation slice:

```bash
conda run -n rxn-env python scripts/evaluate.py \
    --index work/rich_pocket_union.pkl \
    --query-mode reffrag --score-mode emark --mark mean \
    --max-queries 120 --seed 1 --positive-mode same-protein \
    --protein-groups work/union_protein_groups.json \
    --cache $DATA/union_cache \
    --output work/union_eval_confirm_seed1.json
```

## 7. Full cached build (8,401 RCSB structures)

```bash
conda run -n rxn-env python scripts/build_index.py \
    --out work/frase_rich_index.pkl
```

## 8. Orchestrated pipeline

`scripts/run_pipeline.py` chains every stage in dependency order (tests →
PDBbind cache/groups/index → RCSB shards + union → union evaluation → patent
decomposition → pool → placement → profile store/evaluation → fusion). A
command is skipped when its outputs exist; the long patent decomposition
(~37 h on the full corpus) is gated behind `--allow-long`.

```bash
# inspect stages and their exact commands
conda run --no-capture-output -n rxn-env python scripts/run_pipeline.py --list

# plan only: prints skip decisions and command lines
conda run --no-capture-output -n rxn-env python scripts/run_pipeline.py --stages all --dry-run

# run a range (0 tests, 1-5 indexes/eval, 6 decompose, 7-11 patent/profile)
conda run --no-capture-output -n rxn-env python scripts/run_pipeline.py --from 7 --workers 6

# rerun one stage even though its outputs exist
conda run --no-capture-output -n rxn-env python scripts/run_pipeline.py --stages fusion --force
```

Per-stage output is teed to `work/logs/<nn>_<stage>.log`. The runner uses the
interpreter it was launched with, so invoke it from `rxn-env`; it does not wrap
commands in a nested `conda run`. `--decompose-batch` must match the batch the
part directory was built with (10000 for the shipped parts) — the runner
refuses to mix batches because the offsets would cover different parent
ranges.
