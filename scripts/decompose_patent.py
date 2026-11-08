"""Decompose 2D parent SMILES with the library fragmentation rule.

Emits one parquet part per input batch (resumable: existing parts are skipped)
with the H-capped canonical fragment SMILES, attachment count, attachment atoms
in canonical and parent numbering, and the parent partner atom of each cut
bond. That is the candidate pool stage of the patent-bank expansion; the 33-bit
key and E/a marks are computed later, in the query pose.

Usage:
    python scripts/decompose_patent.py \
        --parents /Users/mai/Software/patent_compound/source/filtered_compounds.parquet \
        --out-dir work/patent_frags --workers 8 --batch 10000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCHEMA = pa.schema([
    ("compound_id", pa.int64()),
    ("frag_smiles", pa.string()),
    ("n_heavy", pa.int16()),
    ("n_attach", pa.int8()),
    ("attach_atoms", pa.list_(pa.int32())),
    ("attach_parent", pa.list_(pa.int32())),
    ("attach_partner", pa.list_(pa.int32())),
    ("role", pa.string()),
    ("ring_class", pa.string()),
])


def _init_worker() -> None:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _run_batch(job: tuple[str, int, list[int], list[str]]) -> tuple[str, int, int, int]:
    from frase_rich.patent_decomp import decompose_parent

    path, n_in, ids, smiles = job
    cols = {name: [] for name in SCHEMA.names}
    n_fail = 0
    for cid, smi in zip(ids, smiles):
        frags = decompose_parent(smi)
        if not frags:
            n_fail += 1
            continue
        for f in frags:
            cols["compound_id"].append(cid)
            for key in ("frag_smiles", "n_heavy", "n_attach", "attach_atoms",
                        "attach_parent", "attach_partner", "role", "ring_class"):
                cols[key].append(f[key])
    table = pa.table(cols, schema=SCHEMA)
    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path, n_in, table.num_rows, n_fail


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", required=True, help="parquet with id+smiles")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=25_000)
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--smiles-col", default="smiles")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    pf = pq.ParquetFile(args.parents)
    total = pf.metadata.num_rows

    jobs = []
    cursor = 0
    for b in pf.iter_batches(batch_size=args.batch,
                             columns=[args.id_col, args.smiles_col]):
        ids = b.column(0).to_pylist()
        smiles = b.column(1).to_pylist()
        path = os.path.join(args.out_dir, f"part_{cursor:09d}.parquet")
        n = len(ids)
        if not os.path.exists(path):
            jobs.append((path, n, ids, smiles))
        cursor += n
    print(f"{total:,} parents, {cursor // args.batch + 1} batches, "
          f"{len(jobs)} to run ({cursor - sum(j[1] for j in jobs):,} already done)")

    t0 = time.time()
    n_parents = n_frags = n_fail = done = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as pool:
        futures = [pool.submit(_run_batch, j) for j in jobs]
        for fut in as_completed(futures):
            _path, n_in, nf, nf_fail = fut.result()
            n_parents += n_in
            n_frags += nf
            n_fail += nf_fail
            done += 1
            if done % 10 == 0 or done == len(jobs):
                dt = time.time() - t0
                rate = n_parents / dt
                eta = (sum(j[1] for j in jobs) - n_parents) / max(rate, 1e-9)
                print(f"  [{done}/{len(jobs)}] {n_parents:,} parents, "
                      f"{n_frags:,} fragments, {n_fail:,} failures, "
                      f"{rate:,.0f} parents/s, eta {eta/60:.1f} min", flush=True)
    print(f"done: {n_frags:,} fragment rows in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
