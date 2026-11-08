"""Dedupe decomposed patent fragments into one candidate pool.

Key = (fragment SMILES, attachment atoms): the same H-capped fragment cut at a
different position is a different candidate (different exit vectors).

Scale: ~244M fragment rows collapse to ~70M keys, so the keys are never held
as strings for every row. Two passes over int64 hashes:

  pass 1  hash every row key, ``np.unique`` the concatenated hashes into a
          sorted key table with instance counts
  pass 2  re-hash each row, look its hash up with ``searchsorted`` and emit the
          first occurrence of each key

``n_instances`` counts rows, not distinct compounds: a symmetric parent can
yield the same (fragment, attachment) twice. Prevalence over distinct parents
is a later filter, computed only for the subset that matters.

Usage:
    python scripts/patent_pool.py --parts 'work/patent_frags/part_*.parquet' \
        --out work/patent_pool.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCHEMA = pa.schema([
    ("frag_smiles", pa.string()),
    ("n_heavy", pa.int16()),
    ("n_attach", pa.int8()),
    ("attach_atoms", pa.list_(pa.int32())),
    ("role", pa.string()),
    ("ring_class", pa.string()),
    ("n_instances", pa.int32()),
])

COLS = ["frag_smiles", "n_heavy", "n_attach", "attach_atoms", "role",
        "ring_class"]


def _hashes(smiles: list[str], attach: list[list[int]]) -> np.ndarray:
    """64-bit key hashes; stable within one process (pass 1 and 2 agree)."""
    return np.fromiter((hash(s + "|" + ",".join(map(str, a)))
                        for s, a in zip(smiles, attach)),
                       dtype=np.int64, count=len(smiles))


def _batches(paths: list[str], batch_rows: int):
    for path in paths:
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=batch_rows, columns=COLS):
            yield b


def _filtered(b, args):
    """Apply the candidate-pool gates to one batch (same mask in both passes)."""
    nh = b.column("n_heavy").to_numpy(zero_copy_only=False)
    na = b.column("n_attach").to_numpy(zero_copy_only=False)
    keep = ((nh >= args.min_heavy) & (na >= args.min_attach)
            & (na <= args.max_attach))
    return b.filter(pa.array(keep)) if not keep.all() else b


def _rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", required=True, help="glob of part parquets")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=250_000)
    parser.add_argument("--limit-parts", type=int, default=0,
                        help="debug: use only the first N parts")
    parser.add_argument("--min-heavy", type=int, default=5,
                        help="drop fragments below this heavy-atom count "
                             "(50 Da is ~4-5 heavy; 5 is the rigid-alignment floor)")
    parser.add_argument("--min-attach", type=int, default=1,
                        help="0 would keep whole molecules, which have no exit vector")
    parser.add_argument("--max-attach", type=int, default=3,
                        help="4+ attachments are scaffold/hub pieces; raise to keep them")
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.parts))
    if args.limit_parts:
        paths = paths[:args.limit_parts]
    if not paths:
        raise SystemExit(f"no parts match {args.parts}")
    print(f"{len(paths)} parts")

    t0 = time.time()
    chunks: list[np.ndarray] = []
    n_rows = 0
    for b in _batches(paths, args.batch):
        b = _filtered(b, args)
        if b.num_rows == 0:
            continue
        h = _hashes(b.column("frag_smiles").to_pylist(),
                    b.column("attach_atoms").to_pylist())
        chunks.append(h)
        n_rows += b.num_rows
        if n_rows % 20_000_000 < b.num_rows:
            print(f"  {n_rows:,} rows, RSS {_rss_gb():.1f} GB, "
                  f"{n_rows/(time.time()-t0):,.0f} rows/s", flush=True)
    hashes = np.concatenate(chunks)
    del chunks
    keys, counts = np.unique(hashes, return_counts=True)
    del hashes
    print(f"pass 1: {n_rows:,} rows -> {len(keys):,} unique keys "
          f"({len(keys)/n_rows:.3f}/row) in {time.time()-t0:.0f}s, "
          f"RSS {_rss_gb():.1f} GB")

    seen = np.zeros(len(keys), dtype=bool)
    writer = pq.ParquetWriter(args.out + ".tmp", SCHEMA, compression="zstd")
    buf = {name: [] for name in SCHEMA.names}
    out_rows = 0

    def flush() -> None:
        nonlocal buf, out_rows
        if not buf["frag_smiles"]:
            return
        table = pa.table(buf, schema=SCHEMA)
        writer.write_table(table)
        out_rows += table.num_rows
        buf = {name: [] for name in SCHEMA.names}

    t1 = time.time()
    for b in _batches(paths, args.batch):
        b = _filtered(b, args)
        if b.num_rows == 0:
            continue
        cols = {c: b.column(c).to_pylist() for c in COLS}
        h = _hashes(cols["frag_smiles"], cols["attach_atoms"])
        pos = np.searchsorted(keys, h)
        pos = np.minimum(pos, len(keys) - 1)
        hit = keys[pos] == h
        if not hit.all():
            raise SystemExit("hash table lookup missed; key hashing is not "
                             "stable between passes")
        # one row per key per batch: np.unique keeps the first occurrence
        uniq_pos, first = np.unique(pos, return_index=True)
        fresh = ~seen[uniq_pos]
        if not fresh.any():
            continue
        seen[uniq_pos[fresh]] = True
        idx = first[fresh]
        for name in COLS:
            vals = cols[name]
            src = [vals[i] for i in idx] if not isinstance(vals, np.ndarray) \
                else vals[idx].tolist()
            buf[name].extend(src)
        buf["n_instances"].extend(counts[pos[idx]].tolist())
        if len(buf["frag_smiles"]) >= 500_000:
            flush()
    flush()
    writer.close()
    os.replace(args.out + ".tmp", args.out)
    print(f"pass 2: {out_rows:,} pool rows in {time.time()-t1:.0f}s")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
