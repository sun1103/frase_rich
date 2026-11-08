#!/usr/bin/env python
"""Run the whole frase_rich pipeline in dependency order, resumably.

Usage
-----
    # must run inside the environment that has the dependencies
    conda run --no-capture-output -n rxn-env python scripts/run_pipeline.py --list
    conda run --no-capture-output -n rxn-env python scripts/run_pipeline.py --stages all --dry-run
    conda run --no-capture-output -n rxn-env python scripts/run_pipeline.py --stages 6-11
    conda run --no-capture-output -n rxn-env python scripts/run_pipeline.py --from union_index --allow-long

Each stage is a list of commands. A command is skipped when all of its
recorded outputs exist and are non-empty; a stage is skipped when all of its
commands are complete. ``--force`` reruns the selected stages regardless.

The decompose stage is treated as long (~37 h on the full patent corpus) and
is skipped unless ``--allow-long`` is passed. Everything else is minutes.

The orchestrator does NOT wrap commands in ``conda run``; it launches them
with the interpreter running this script, so invoke it from the target
environment (``rxn-env`` for this project).

Examples
--------
After a fresh checkout, only ``--stages 0`` (tests) is cheap; every other
stage needs the PDBbind corpus and the patent corpus on disk. Use
``--dry-run`` first: it prints every command line and the skip decision
without executing anything.

A failure stops the run (exit 1) unless ``--keep-going`` is given. Per-stage
console output is teed to ``work/logs/<nn>_<stage>.log``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/Users/mai/Software/fragenv/data"
DEFAULT_PARENTS = "/Users/mai/Software/patent_compound/source/filtered_compounds.parquet"


@dataclass
class Command:
    argv: Sequence[str]
    outputs: Sequence[str] = ()
    check: Callable[[], tuple[bool, str]] | None = None
    note: str = ""


@dataclass
class Stage:
    name: str
    description: str
    commands: list[Command] = field(default_factory=list)
    long: bool = False
    # Runs before the skip logic and may raise SystemExit for configurations
    # that are wrong regardless of what is already on disk.
    precheck: Callable[[], None] | None = None


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _nonempty(path: str) -> bool:
    if os.path.isdir(path):
        try:
            with os.scandir(path) as entries:
                return any(True for _ in entries)
        except OSError:
            return False
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _complete(command: Command, force: bool) -> tuple[bool, str]:
    if force:
        return False, "forced"
    # A command with no outputs (the test suite) has no completion marker and
    # therefore always runs when its stage is selected.
    if not command.outputs and command.check is None:
        return False, "no outputs to check"
    if command.check is not None:
        ok, detail = command.check()
        if not ok:
            return False, detail
    for path in command.outputs:
        if not _nonempty(path):
            return False, f"missing or empty: {path}"
    return True, "up to date"


def _parent_rows(path: str) -> int:
    import pyarrow.parquet as pq  # local import: only the decompose check needs it

    return int(pq.ParquetFile(path).metadata.num_rows)


_PART_RE = re.compile(r"^part_(\d{9})\.parquet$")


def _part_starts(out_dir: str) -> list[int]:
    try:
        names = os.listdir(out_dir)
    except OSError:
        return []
    return [int(m.group(1)) for m in map(_PART_RE.match, names) if m]


def _decompose_precheck(out_dir: str, batch: int) -> Callable[[], None]:
    """Refuse to mix parts written under different ``--batch`` values.

    Resumability keys parts by their parent offset. With a different batch,
    the offsets still exist but cover different ranges, so the pool would
    duplicate parents instead of resuming.
    """

    def check() -> None:
        bad = sorted({start for start in _part_starts(out_dir) if start % batch})
        if bad:
            raise SystemExit(
                f"decompose: {out_dir} already holds part_{bad[0]:09d}.parquet, "
                f"which is not a multiple of --decompose-batch {batch}. "
                f"Use the batch the directory was built with (10000 for the "
                f"shipped parts) or point --work elsewhere.")

    return check


def _decompose_check(out_dir: str, parents: str, batch: int) -> Callable[[], tuple[bool, str]]:
    """Completeness for a resumable, one-part-per-batch decomposition.

    A plain "output dir is non-empty" check would accept a run killed halfway,
    so this recomputes the exact part set from the parent count.
    """

    def check() -> tuple[bool, str]:
        if not os.path.isfile(parents):
            return False, f"missing parent parquet: {parents}"
        try:
            total = _parent_rows(parents)
        except Exception as exc:  # unreadable parquet -> let the stage try
            return False, f"cannot read parent rows: {exc}"
        expected = list(range(0, total, batch))
        have = set(_part_starts(out_dir))
        missing = [s for s in expected if s not in have]
        if missing:
            return False, f"{len(missing)} of {len(expected)} parts missing"
        return True, f"{len(expected)} parts"

    return check


def _resolve_one(token: str, names: list[str]) -> int:
    token = token.strip()
    if token.isdigit():
        idx = int(token)
        if not 0 <= idx < len(names):
            raise SystemExit(f"stage index out of range: {token} (0-{len(names) - 1})")
        return idx
    if token in names:
        return names.index(token)
    raise SystemExit(f"unknown stage: {token}\nstages: {', '.join(names)}")


def _select(stages: list[Stage], spec: str, from_: str | None,
            to: str | None) -> list[int]:
    names = [stage.name for stage in stages]
    chosen: set[int] = set()
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if token == "all":
            chosen.update(range(len(stages)))
        elif "-" in token:
            left, _, right = token.partition("-")
            lo = _resolve_one(left, names)
            hi = _resolve_one(right, names)
            chosen.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            chosen.add(_resolve_one(token, names))
    if from_ is not None:
        lo = _resolve_one(from_, names)
        chosen = {idx for idx in chosen if idx >= lo}
    if to is not None:
        hi = _resolve_one(to, names)
        chosen = {idx for idx in chosen if idx <= hi}
    return sorted(chosen)


def _build_stages(args: argparse.Namespace) -> list[Stage]:
    py = args.python
    work = _abs(args.work)

    cache = _abs(args.cache)
    union_cache = _abs(args.union_cache)
    pdbbind_root = _abs(args.pdbbind_root)
    selected = _abs(args.pdbbind_selected)
    parents = _abs(args.patent_parents)
    index = _abs(args.index)
    queries = _abs(args.queries)
    eval_out = _abs(args.eval_out)

    sameprotein = os.path.join(work, "pdbbind_sameprotein.tsv")
    pdbbind_groups = os.path.join(work, "pdbbind_protein_groups.json")
    base_index = os.path.join(work, "rich_pocket_pdbbind.pkl")
    union_groups = os.path.join(work, "union_protein_groups.json")
    union_selected = os.path.join(work, "union_selected.tsv")
    frag_dir = os.path.join(work, "patent_frags")
    pool = os.path.join(work, "patent_pool.parquet")
    store = os.path.join(work, "profile_store.pkl")
    profile_eval = os.path.join(work, "profile_eval.json")
    profile_scores = os.path.join(work, "profile_scores.json")
    fusion = os.path.join(work, "profile_fusion.json")

    shard_pattern = _abs(args.union_shards)
    shard_tsvs = sorted(glob.glob(shard_pattern))

    stages = [
        Stage(
            "tests",
            "unit tests (always runs; no completion marker)",
            [Command([py, "-m", "unittest", "discover", "-s", "tests", "-v"])],
        ),
        Stage(
            "pdbbind_cache",
            "cache PDBbind protein/ligand complexes and the selected TSV",
            [Command(
                [py, "scripts/pdbbind_cache.py", "--root", pdbbind_root,
                 "--out-cache", cache, "--out-tsv", selected],
                outputs=[selected, cache],
                note="reuses existing per-PDB caches",
            )],
        ),
        Stage(
            "groups",
            "protein groups + same-protein selection for PDBbind",
            [Command(
                [py, "scripts/protein_groups.py", "--selected", selected,
                 "--cache", cache, "--groups-out", pdbbind_groups,
                 "--selected-out", sameprotein],
                outputs=[pdbbind_groups, sameprotein],
            )],
        ),
        Stage(
            "pdbbind_index",
            "rich weighted index over the PDBbind same-protein set",
            [Command(
                [py, "scripts/build_index.py", "--selected", sameprotein,
                 "--cache", cache, "--env-center", "pocket", "--out", base_index],
                outputs=[base_index],
            )],
        ),
        Stage(
            "union_index",
            "RCSB shard indexes, merge with the PDBbind base, union groups",
            _union_commands(args, py, work, union_cache, base_index, index,
                            union_groups, union_selected, shard_tsvs),
        ),
        Stage(
            "eval_union",
            "leave-one-complex-out evaluation on the union index (energy marks)",
            [Command(
                [py, "scripts/evaluate.py", "--index", index,
                 "--query-mode", "reffrag", "--score-mode", "emark",
                 "--mark", "mean", "--cache", union_cache,
                 "--protein-groups", union_groups,
                 "--positive-mode", "same-protein", "--max-queries", "1000",
                 "--output", eval_out],
                outputs=[eval_out],
            )],
        ),
        Stage(
            "decompose",
            f"decompose the patent parent corpus into fragments (~{args.decompose_batch}-compound parts)",
            [Command(
                [py, "scripts/decompose_patent.py", "--parents", parents,
                 "--out-dir", frag_dir, "--workers", str(args.workers),
                 "--batch", str(args.decompose_batch)],
                check=_decompose_check(frag_dir, parents, args.decompose_batch),
                note="long-running; resumable",
            )],
            long=True,
            precheck=_decompose_precheck(frag_dir, args.decompose_batch),
        ),
        Stage(
            "pool",
            "gate and deduplicate patent fragments into the candidate pool",
            [Command(
                [py, "scripts/patent_pool.py",
                 "--parts", os.path.join(frag_dir, "part_*.parquet"),
                 "--out", pool],
                outputs=[pool],
            )],
        ),
        Stage(
            "place",
            "select patent candidates, place them on query poses, persist queries",
            [Command(
                [py, "scripts/patent_place.py", "--pool", pool,
                 "--fp-prefix", os.path.join(work, "patent_pool"),
                 "--selected", selected, "--cache", cache, "--out", queries,
                 "--n-queries", "50", "--max-per-complex", "2",
                 "--top-k", str(args.place_top_k), "--n-candidates", "6",
                 "--index", index, "--protein-groups", pdbbind_groups,
                 "--mark", "mean", "--positive-mode", "smiles",
                 "--workers", str(args.workers), "--seed", "0"],
                outputs=[queries],
                note="builds Morgan fingerprints next to the pool on first run",
            )],
        ),
        Stage(
            "profile_store",
            "per-residue profiles for the complexes of the placement queries",
            [Command(
                [py, "scripts/build_profile_store.py", "--index", index,
                 "--cache", cache, "--groups", pdbbind_groups,
                 "--queries", queries, "--out", store],
                outputs=[store],
            )],
        ),
        Stage(
            "profile_eval",
            "same-fragment retrieval with the profile index vs the triplet index",
            [Command(
                [py, "scripts/eval_profile_index.py", "--store", store,
                 "--queries", queries, "--groups", pdbbind_groups,
                 "--cache", cache,
                 "--eligibility", os.path.join(work, "placement_eligible.json"),
                 "--out", profile_eval, "--save-scores", profile_scores],
                outputs=[profile_eval, profile_scores],
            )],
        ),
        Stage(
            "fusion",
            "triplet/profile rank fusion table with paired bootstrap",
            [Command(
                [py, "scripts/fuse_profile_scores.py", "--scores", profile_scores,
                 "--out", fusion],
                outputs=[fusion],
            )],
        ),
    ]
    return stages


def _union_commands(args: argparse.Namespace, py: str, work: str,
                    union_cache: str, base_index: str, index: str,
                    union_groups: str, union_selected: str,
                    shard_tsvs: list[str]) -> list[Command]:
    if not shard_tsvs:
        # Failing here is better than silently building a union from the base
        # alone, which would look like success and change every later number.
        raise SystemExit(f"union_index: no shard TSVs match {args.union_shards!r}")
    commands: list[Command] = []
    shard_indexes: list[str] = []
    for tsv in shard_tsvs:
        tag = os.path.splitext(os.path.basename(tsv))[0].replace("_", "")
        out = os.path.join(work, f"rich_pocket_{tag}.pkl")
        shard_indexes.append(out)
        commands.append(Command(
            [py, "scripts/build_index.py", "--selected", tsv,
             "--cache", union_cache, "--env-center", "pocket", "--out", out],
            outputs=[out],
        ))
    commands.append(Command(
        [py, "scripts/merge_indexes.py", "--base", base_index,
         "--shards", *shard_indexes, "--out", index],
        outputs=[index],
    ))
    commands.append(Command(
        [py, "scripts/protein_groups.py", "--selected", union_selected,
         "--cache", union_cache, "--groups-out", union_groups],
        outputs=[union_groups],
    ))
    return commands


def _print_list(stages: list[Stage]) -> None:
    print(f"{'#':>2}  {'stage':<16} {'cmds':>4}  description")
    for idx, stage in enumerate(stages):
        marker = " (long, --allow-long)" if stage.long else ""
        print(f"{idx:>2}  {stage.name:<16} {len(stage.commands):>4}  "
              f"{stage.description}{marker}")
        for command in stage.commands:
            print(f"      $ {shlex.join(command.argv)}")
            if command.note:
                print(f"        note: {command.note}")


def _run(argv: Sequence[str], log) -> tuple[int, float]:
    header = f"$ {shlex.join(argv)}\n"
    sys.stdout.write(header)
    sys.stdout.flush()
    log.write(header)
    log.flush()
    started = time.monotonic()
    proc = subprocess.Popen(list(argv), cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        log.flush()
    return proc.wait(), time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all",
                    help="comma-separated names/indices/ranges, or 'all'")
    ap.add_argument("--from", dest="from_", default=None,
                    help="keep only stages at or after this name/index")
    ap.add_argument("--to", dest="to", default=None,
                    help="keep only stages at or before this name/index")
    ap.add_argument("--list", action="store_true",
                    help="list stages and their commands, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="show skip decisions and commands, run nothing")
    ap.add_argument("--force", action="store_true",
                    help="rerun selected stages even when outputs exist")
    ap.add_argument("--allow-long", action="store_true",
                    help="permit the decompose stage (~37 h on the full corpus)")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing stage")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for subprocesses (default: this one)")
    ap.add_argument("--work", default="work")
    ap.add_argument("--cache", default=os.path.join(DATA, "pdbbind_cache"))
    ap.add_argument("--union-cache", default=os.path.join(DATA, "union_cache"))
    ap.add_argument("--pdbbind-root", default=os.path.join(DATA, "pdbbind_pl", "P-L"))
    ap.add_argument("--pdbbind-selected", default=os.path.join(DATA, "pdbbind_selected.tsv"))
    ap.add_argument("--union-shards", default=os.path.join("work", "rcsb_q*.tsv"),
                    help="glob of TSVs, one per RCSB index shard")
    ap.add_argument("--patent-parents", default=DEFAULT_PARENTS)
    ap.add_argument("--index", default=os.path.join("work", "rich_pocket_union.pkl"),
                    help="merged union index (written by union_index, read later)")
    ap.add_argument("--queries", default=os.path.join("work", "patent_place_persist.json"),
                    help="placement query file (written by place, read later)")
    ap.add_argument("--eval-out", default=os.path.join("work", "union_eval_mark_mean.json"))
    ap.add_argument("--workers", type=int, default=6,
                    help="worker processes for decompose/place (default 6: leave cores free)")
    ap.add_argument("--decompose-batch", type=int, default=10_000,
                    help="must match any existing parts in the fragment directory")
    ap.add_argument("--place-top-k", type=int, default=120)
    args = ap.parse_args(argv)

    stages = _build_stages(args)
    if args.list:
        _print_list(stages)
        return 0

    chosen = _select(stages, args.stages, args.from_, args.to)
    if not chosen:
        print("no stages selected")
        return 0

    work = _abs(args.work)
    log_dir = os.path.join(work, "logs")
    if not args.dry_run:
        os.makedirs(log_dir, exist_ok=True)

    print(f"python: {args.python}")
    print(f"repo:   {ROOT}")
    print(f"stages: {', '.join(stages[i].name for i in chosen)}")
    if args.dry_run:
        print("dry run: no commands will execute")

    failed: list[str] = []
    ran = skipped_stages = skipped_cmds = 0
    for idx in chosen:
        stage = stages[idx]
        incomplete = [cmd for cmd in stage.commands
                      if not _complete(cmd, args.force)[0]]
        print(f"\n[{idx}] {stage.name} — {stage.description}")
        if stage.precheck is not None:
            stage.precheck()
        if stage.long and not args.allow_long and incomplete:
            print("  skipped: long stage; pass --allow-long to run it")
            skipped_stages += 1
            continue
        if not incomplete:
            print("  up to date")
            skipped_stages += 1
            continue

        log_path = os.path.join(log_dir, f"{idx:02d}_{stage.name}.log")
        stage_failed = False
        log = None
        if not args.dry_run:
            log = open(log_path, "w")
        try:
            for n, command in enumerate(stage.commands):
                ok, detail = _complete(command, args.force)
                if ok:
                    print(f"  [{n + 1}/{len(stage.commands)}] skip: {detail}")
                    skipped_cmds += 1
                    continue
                if args.dry_run:
                    print(f"  [{n + 1}/{len(stage.commands)}] would run: {detail}")
                    print(f"      $ {shlex.join(command.argv)}")
                    continue
                print(f"  [{n + 1}/{len(stage.commands)}] run: {detail}")
                code, elapsed = _run(command.argv, log)
                ran += 1
                if code != 0:
                    print(f"  FAILED (exit {code}) after {elapsed:.1f}s; "
                          f"log: {log_path}")
                    stage_failed = True
                    failed.append(f"{stage.name}#{n + 1}")
                    break
                if elapsed >= 60:
                    print(f"  done in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")
                else:
                    print(f"  done in {elapsed:.1f}s")
        finally:
            if log is not None:
                log.close()

        if stage_failed and not args.keep_going:
            break

    print("\nsummary: "
          f"{ran} command(s) run, {skipped_cmds} skipped, "
          f"{skipped_stages} stage(s) up to date/skipped")
    if failed:
        print(f"failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
