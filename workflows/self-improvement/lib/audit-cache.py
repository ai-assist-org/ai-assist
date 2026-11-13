"""Report prompt-cache misses inside queries, from ai-assist's query traces.

Usage: audit-cache.py <query_traces.jsonl> [--since YYYY-MM-DD[THH:MM]] [--model SUBSTR] [--queries]

Within one query, each turn resends the previous turn's prompt plus new
content, so its cache read should cover nearly all of the previous prompt. A
turn whose cache read is below 90% of the previous turn's prompt tokens is a
miss; the difference was rewritten (billed as a cache write or uncached input
instead of a cache read). Each miss is classified from the turns' prefix_sha
(fingerprint of tools + system prompt):

- prefix-changed: the prefix bytes differ from the previous turn's;
- cache-absent: same prefix, so the cache was not where the request landed;
- unclassified: traces written before prefix_sha was recorded.

Turn 0 of each query is not judged (its predecessor is another query).
Lines that are not JSON objects are skipped. Does not import ai-assist.
"""

import argparse
import ast
import json
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

MISS_RATIO = 0.9
# Below the smallest cacheable prompt, a zero read is expected, not a miss
MIN_CACHEABLE = 4096


def prompt_tokens(entry: dict) -> int:
    return sum(
        int(entry.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )


def load_queries(path: Path, since: str = "", model: str = "claude") -> list[dict]:
    queries = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or str(row.get("timestamp", "")) < since:
                continue
            if model not in str(row.get("model", "")):
                continue
            usage = row.get("token_usage")
            if isinstance(usage, str):
                try:
                    usage = ast.literal_eval(usage)
                except ValueError, SyntaxError:
                    continue
            if isinstance(usage, list) and usage:
                queries.append({**row, "token_usage": [e for e in usage if isinstance(e, dict)]})
    return queries


def find_misses(turns: list[dict]) -> list[dict]:
    misses = []
    for prev, cur in pairwise(turns):
        expected = prompt_tokens(prev)
        read = int(cur.get("cache_read_input_tokens") or 0)
        if expected < MIN_CACHEABLE or read >= MISS_RATIO * expected:
            continue
        if prev.get("prefix_sha") and cur.get("prefix_sha"):
            kind = "cache-absent" if prev["prefix_sha"] == cur["prefix_sha"] else "prefix-changed"
        else:
            kind = "unclassified"
        misses.append({"turn": cur.get("turn"), "kind": kind, "rewritten": expected - read})
    return misses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", type=Path)
    parser.add_argument("--since", default="", help="ISO timestamp prefix, e.g. 2026-11-11T14:00")
    parser.add_argument("--model", default="claude", help="model name substring (default: claude)")
    parser.add_argument("--queries", action="store_true", help="one line per query as well")
    args = parser.parse_args()

    totals: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for query in load_queries(args.traces.expanduser(), args.since, args.model):
        turns = query["token_usage"]
        misses = find_misses(turns)
        t = totals[query.get("model", "?")]
        t["queries"] += 1
        t["pairs"] += max(len(turns) - 1, 0)
        t["rewritten"] += sum(m["rewritten"] for m in misses)
        for m in misses:
            t[m["kind"]] += 1
        if args.queries:
            pattern = " ".join(
                f"{int(e.get('cache_read_input_tokens') or 0) // 1000}r/"
                f"{int(e.get('cache_creation_input_tokens') or 0) // 1000}w"
                for e in turns
            )
            kinds = ",".join(m["kind"] for m in misses) or "-"
            print(f"{query.get('timestamp', '?')[:19]}\t{query.get('model', '?')}\t{pattern}\t{kinds}")

    print("model\tqueries\tturn_pairs\tcache_absent\tprefix_changed\tunclassified\ttokens_rewritten")
    for name, t in sorted(totals.items()):
        print(
            f"{name}\t{t['queries']}\t{t['pairs']}\t{t['cache-absent']}\t{t['prefix-changed']}"
            f"\t{t['unclassified']}\t{t['rewritten']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
