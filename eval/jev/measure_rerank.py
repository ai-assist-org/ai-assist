#!/usr/bin/env python3
"""Measure whether a jev Score rerank would improve KG semantic retrieval.

This is the "eval harness first" step for the remaining Phase 3 item (rerank KG
retrieval with jev Score). It does NOT touch the production retrieval path. It
ingests a labeled corpus (``rerank_cases.yaml``) into a throwaway knowledge
graph and, for every query, compares two orderings of the SAME cosine-retrieved
candidate pool against graded relevance labels:

* **baseline** — the order ``KnowledgeGraph.semantic_search`` returns today
  (pure vector cosine similarity).
* **jev**      — that pool reordered by a jev Score relevance judgment (the
  prototype rerank we are deciding whether to ship).

Both arms score with nDCG@k over the *pool* (identical IDCG), so the number
isolates rerank quality: can reordering what cosine already retrieved beat cosine
order? If baseline nDCG is already ~1.0 there is no headroom and a jev rerank
cannot help; if it is lower, the jev column shows whether jev closes the gap.

Requirements: the embedding model (fastembed, a core dep; first run downloads
all-MiniLM-L6-v2) for every run, and ``AI_ASSIST_JEV_API_KEY`` for the jev arm.

Usage::

    uv run --extra eval python eval/jev/measure_rerank.py                 # both arms
    uv run --extra eval python eval/jev/measure_rerank.py --arm baseline  # headroom only
    uv run --extra eval python eval/jev/measure_rerank.py --pool 8 --k 5
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Query:
    name: str
    query: str
    labels: dict[str, int]  # entity key -> graded relevance (0..3)


@dataclass
class QueryResult:
    name: str
    base_ndcg: float | None  # None => no relevant entity in the retrieved pool
    jev_ndcg: float | None
    pool_size: int


@dataclass
class Summary:
    results: list[QueryResult] = field(default_factory=list)

    def _mean(self, attr: str) -> float | None:
        vals = [getattr(r, attr) for r in self.results if getattr(r, attr) is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def base_mean(self) -> float | None:
        return self._mean("base_ndcg")

    @property
    def jev_mean(self) -> float | None:
        return self._mean("jev_ndcg")

    def win_loss(self) -> tuple[int, int, int]:
        """Queries where jev improved / hurt / tied nDCG vs baseline (both scored)."""
        win = loss = tie = 0
        for r in self.results:
            if r.base_ndcg is None or r.jev_ndcg is None:
                continue
            if r.jev_ndcg > r.base_ndcg + 1e-9:
                win += 1
            elif r.jev_ndcg < r.base_ndcg - 1e-9:
                loss += 1
            else:
                tie += 1
        return win, loss, tie


def load_queries(data: dict) -> list[Query]:
    return [
        Query(
            name=q["name"],
            query=" ".join(q["query"].split()),
            labels={str(k): int(v) for k, v in q["relevant"].items()},
        )
        for q in data["queries"]
    ]


def _dcg(rels: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(ranked_keys: list[str], labels: dict[str, int], k: int) -> float | None:
    """nDCG@k with the ideal taken over the retrieved pool (so both arms share IDCG)."""
    pool_rels = [labels.get(key, 0) for key in ranked_keys]
    idcg = _dcg(sorted(pool_rels, reverse=True)[:k])
    if idcg == 0:  # no relevant entity was retrieved — nothing a rerank could fix
        return None
    return _dcg([labels.get(key, 0) for key in ranked_keys[:k]]) / idcg


def build_graph(entities: list[dict]) -> tuple[Any, str]:
    """Ingest the corpus into a fresh on-disk KG; return (graph, db_path)."""
    from ai_assist.knowledge_graph import KnowledgeGraph

    fd, path = tempfile.mkstemp(suffix=".db", prefix="jev_rerank_")
    os.close(fd)
    kg = KnowledgeGraph(db_path=path)
    now = datetime.now()
    for ent in entities:
        kg.insert_knowledge(
            entity_type="project_context",
            key=ent["key"],
            content=" ".join(ent["content"].split()),
            valid_from=now,
        )
    return kg, path


async def run(cases_path: Path, pool: int, k: int, arm: str) -> Summary:
    from ai_assist.config import AiAssistConfig
    from ai_assist.jev_client import jev_configured, rerank_candidates

    data = yaml.safe_load(cases_path.read_text())
    queries = load_queries(data)
    config = AiAssistConfig.from_env()
    config.jev_enabled = True
    want_jev = arm in ("jev", "both")
    if want_jev and not jev_configured(config):
        raise SystemExit("jev arm needs AI_ASSIST_JEV_API_KEY (and jev enabled).")

    kg, db_path = build_graph(data["entities"])
    summary = Summary()
    try:
        for q in queries:
            candidates = kg.semantic_search(q.query, limit=pool)
            base_keys = [c["key"] for c in candidates]
            base_ndcg = ndcg_at_k(base_keys, q.labels, k)
            jev_ndcg = None
            if want_jev:
                reranked = await rerank_candidates(config, q.query, candidates)
                jev_ndcg = ndcg_at_k([c["key"] for c in reranked], q.labels, k)
            summary.results.append(QueryResult(q.name, base_ndcg, jev_ndcg, len(candidates)))
            b = f"{base_ndcg:.3f}" if base_ndcg is not None else "n/a"
            j = f"{jev_ndcg:.3f}" if jev_ndcg is not None else ("n/a" if want_jev else "-")
            print(f"  {q.name:20} pool={len(candidates):2}  base={b:6}  jev={j}")
    finally:
        kg.close()
        Path(db_path).unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parent / "rerank_cases.yaml")
    parser.add_argument("--arm", choices=["baseline", "jev", "both"], default="both")
    parser.add_argument("--pool", type=int, default=8, help="candidates retrieved by cosine before rerank")
    parser.add_argument("--k", type=int, default=5, help="nDCG cutoff")
    args = parser.parse_args()

    summary = asyncio.run(run(args.cases, args.pool, args.k, args.arm))

    base = summary.base_mean
    jev = summary.jev_mean
    print(f"\nKG rerank headroom over {len(summary.results)} queries (pool={args.pool}, nDCG@{args.k})\n")
    print(f"  baseline (cosine)  mean nDCG@{args.k}: {base:.3f}" if base is not None else "  baseline: n/a")
    if base is not None:
        print(f"  headroom (1 - baseline)         : {1 - base:.3f}")
    if args.arm in ("jev", "both"):
        print(f"  jev rerank         mean nDCG@{args.k}: {jev:.3f}" if jev is not None else "  jev: n/a")
        if base is not None and jev is not None:
            wins, losses, ties = summary.win_loss()
            print(f"  delta (jev - baseline)          : {jev - base:+.3f}")
            print(f"  per-query  win/loss/tie         : {wins}/{losses}/{ties}")


if __name__ == "__main__":
    main()
