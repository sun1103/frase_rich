"""Weighted FRASE triplet index.

The discrete key and candidate generation are identical to the original
``TripletIndex`` (src/triplet.py in ``fragenv``). The difference is that each
fragment's distinct fingerprints carry a weight normalized to mean 1, so the
query can score a *weighted* Jaccard:

    W_match = sum of library weights over shared fingerprints
    score   = W_match / (n_query + n_library - W_match)

When every weight is 1 this is exactly the original Jaccard, which keeps the
baseline comparison honest.

The stored-query path stays deliberately rough (residue coordinates + 11-bit
codes). The ref-mol path supplies per-fingerprint query weights computed by the
same library-side code and scores them by weighted exact-key overlap:

    W = sum over shared keys of min(w_query, w_library)   (or geometric mean)
    score = W / (n_query + n_library - W)

With uniform query and library weights this is again exactly Jaccard.

Schema v2 adds a top-level ``features`` dict so a query can refuse to compare
against an index built with a different feature configuration. v1 indexes still
load (with an empty feature dict).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

from .geometry import canonical_triplet_fp, enumerate_triplet_idxs

SCHEMA = "frase_rich.weighted_index.v2"
SCHEMA_V1 = "frase_rich.weighted_index.v1"


@dataclass
class EnvResidue:
    """One environment residue. ``atoms`` is build-time only and not stored."""

    res_name: str
    chain: str
    res_seq: str
    coords: np.ndarray
    code: int
    atoms: list = field(default_factory=list)
    weight: float = 1.0


class WeightedTripletIndex:
    def __init__(self, *, weight_mode: str = "rich",
                 features: dict | None = None) -> None:
        self.weight_mode = weight_mode
        self.features = dict(features or {})
        self._postings: dict[int, list[tuple[str, float]]] = {}
        self._frags: dict[str, dict] = {}

    # ── build ──────────────────────────────────────────────────────────────

    def add_fragment(
        self,
        frag_id: str,
        env: list[EnvResidue],
        fp_weights: dict[int, float],
        *,
        meta: dict | None = None,
        frag_coords: np.ndarray | None = None,
    ) -> int:
        """Add one fragment; ``fp_weights`` may be raw sums (normalized here).

        Weights are rescaled to mean 1 over the fragment's distinct
        fingerprints. That makes ``W_match`` directly comparable to
        ``n_match`` and lets the weighted score collapse to Jaccard when the
        rich signal is uniform.
        """
        raw = {int(k): float(v) for k, v in fp_weights.items() if v > 0}
        if raw:
            scale = len(raw) / float(sum(raw.values()))
            weights = {k: v * scale for k, v in raw.items()}
        else:
            weights = {}

        info = dict(meta or {})
        if env:
            info["_env_coords"] = np.stack([r.coords for r in env]).astype(np.float32)
            info["_env_codes"] = np.array([r.code for r in env], dtype=np.int64)
            info["_env_weights"] = np.array([r.weight for r in env], dtype=np.float64)
        else:
            info["_env_coords"] = np.zeros((0, 3), dtype=np.float32)
            info["_env_codes"] = np.zeros((0,), dtype=np.int64)
            info["_env_weights"] = np.zeros((0,), dtype=np.float64)
        if frag_coords is not None:
            info["_frag_coords"] = np.asarray(frag_coords, dtype=np.float32)

        info["_n_distinct"] = len(weights)
        info["_n_triplets"] = len(fp_weights)
        self._frags[frag_id] = info
        for fp, weight in weights.items():
            self._postings.setdefault(fp, []).append((frag_id, weight))
        return len(fp_weights)

    # ── query ──────────────────────────────────────────────────────────────

    def query(
        self,
        residues: list,
        *,
        top_k: int | None = 50,
        min_matches: int = 1,
        d_lo: float = 8.0,
        d_hi: float = 12.0,
        exclude_pdb: str | None = None,
        method: str = "weighted",
        query_fp_weights: dict[int, float] | None = None,
        combine: str = "min",
    ) -> list[dict]:
        """Rank fragments by weighted (or plain) triplet overlap.

        ``query_fp_weights`` switches to the ref-mol path: no enumeration, and
        matching is a weighted exact-key overlap. ``combine`` is ``min``
        (default) or ``geomean``.
        """
        if query_fp_weights is not None:
            raw_q = {int(k): float(v) for k, v in query_fp_weights.items()
                     if v > 0}
            if not raw_q:
                return []
            scale = len(raw_q) / float(sum(raw_q.values()))
            q_weights = {k: v * scale for k, v in raw_q.items()}
            query_fps = set(q_weights)
            n_query = len(query_fps)
            acc: dict[str, list[float]] = {}
            for fp, qw in q_weights.items():
                for fid, weight in self._postings.get(fp, ()):
                    entry = acc.get(fid)
                    match = (min(qw, weight) if combine == "min"
                             else float(np.sqrt(qw * weight)))
                    if entry is None:
                        acc[fid] = [1, match]
                    else:
                        entry[0] += 1
                        entry[1] += match
        else:
            if len(residues) < 3:
                return []
            pts = np.stack([r.coords for r in residues])
            codes = np.array([r.code for r in residues], dtype=np.int64)
            triplets = enumerate_triplet_idxs(pts, d_lo=d_lo, d_hi=d_hi)
            query_fps = {
                canonical_triplet_fp((codes[i], codes[j], codes[k]))
                for i, j, k in triplets
            }
            n_query = len(query_fps)
            if n_query == 0:
                return []

            acc: dict[str, list[float]] = {}
            for fp in query_fps:
                for fid, weight in self._postings.get(fp, ()):
                    entry = acc.get(fid)
                    if entry is None:
                        acc[fid] = [1, weight]
                    else:
                        entry[0] += 1
                        entry[1] += weight

        out = []
        for fid, (n_match, w_match) in acc.items():
            if n_match < min_matches:
                continue
            info = self._frags[fid]
            if exclude_pdb and str(info.get("pdb_id", "")).lower() == exclude_pdb.lower():
                continue
            n_lib = int(info.get("_n_distinct", 0))
            union = max(n_query + n_lib - n_match, 1e-9)
            w_union = max(n_query + n_lib - w_match, 1e-9)
            row = {
                "frag_id": fid,
                "n_match": int(n_match),
                "score_jaccard": float(n_match / union),
                "score_weighted": float(w_match / w_union),
                "n_query": n_query,
                "n_lib": n_lib,
            }
            row.update({k: v for k, v in info.items() if not k.startswith("_")})
            row["_env_coords"] = info.get("_env_coords")
            row["_env_codes"] = info.get("_env_codes")
            row["_env_weights"] = info.get("_env_weights")
            row["_frag_coords"] = info.get("_frag_coords")
            out.append(row)

        key = "score_weighted" if method == "weighted" else "score_jaccard"
        out.sort(key=lambda r: (-r[key], -r["n_match"]))
        return out if top_k is None else out[:top_k]

    # ── persistence ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._frags)

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump({
                "schema": SCHEMA,
                "weight_mode": self.weight_mode,
                "features": self.features,
                "postings": self._postings,
                "frags": self._frags,
            }, handle)

    @classmethod
    def load(cls, path: str) -> "WeightedTripletIndex":
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        schema = data.get("schema")
        if schema not in {SCHEMA, SCHEMA_V1}:
            raise ValueError(f"unsupported index schema: {schema}")
        idx = cls(weight_mode=data.get("weight_mode", "rich"),
                  features=data.get("features") if schema == SCHEMA else {})
        idx._postings = data["postings"]
        idx._frags = data["frags"]
        return idx
