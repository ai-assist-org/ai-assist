#!/usr/bin/env python3
"""Run the LoCoMo benchmark against ai-assist's KnowledgeGraph.

Usage:
    python run_benchmark.py [--dataset locomo10.json] [--db-dir /tmp/locomo-kg] \
                            [--model claude-sonnet-4-6] [--synthesis-model claude-sonnet-4-6] \
                            [--limit N] [--keep-db] [--skip-synthesis]
"""

import argparse
import logging
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

# Add project root to path so we can import ai_assist
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate import evaluate_dataset  # noqa: E402
from ingest import ingest_dataset  # noqa: E402

from ai_assist.knowledge_graph import KnowledgeGraph  # noqa: E402

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_DATASET = Path(__file__).parent / "locomo10.json"

logger = logging.getLogger(__name__)


def download_dataset(dest: Path) -> None:
    """Download locomo10.json if not present."""
    if dest.exists():
        logger.info("Dataset already present: %s", dest)
        return
    logger.info("Downloading LoCoMo dataset to %s ...", dest)
    urllib.request.urlretrieve(DATASET_URL, dest)  # nosec B310 — hardcoded HTTPS URL
    logger.info("Download complete (%.1f KB)", dest.stat().st_size / 1024)


def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark for ai-assist KnowledgeGraph")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to locomo10.json")
    parser.add_argument("--db-dir", type=Path, default=None, help="Directory for temporary KG database")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="LLM judge model")
    parser.add_argument(
        "--synthesis-model", default=None, help="Model for synthesis/connection discovery (default: same as --model)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to N conversations")
    parser.add_argument("--keep-db", action="store_true", help="Keep the KG database after run")
    parser.add_argument(
        "--skip-synthesis", action="store_true", help="Skip synthesis — store raw conversations only (vector baseline)"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip ingestion — evaluate on existing KG (requires --db-dir)"
    )
    parser.add_argument("--output", type=Path, default=None, help="Write results to markdown file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    synthesis_model = args.synthesis_model or args.model

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    download_dataset(args.dataset)

    db_dir = args.db_dir or Path(tempfile.mkdtemp(prefix="locomo-kg-"))
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "knowledge_graph.db")
    logger.info("KG database: %s", db_path)

    try:
        kg = KnowledgeGraph(db_path=db_path)

        if not args.eval_only:
            logger.info("=== Phase 1: Ingestion ===")
            totals = ingest_dataset(
                kg,
                args.dataset,
                model=synthesis_model,
                limit=args.limit,
                skip_synthesis=args.skip_synthesis,
            )
            logger.info(
                "Ingested: %d conversations, %d insights, %d relationships",
                totals["conversations"],
                totals["insights"],
                totals["relationships"],
            )
        else:
            logger.info("=== Skipping ingestion (--eval-only) ===")

        logger.info("=== Phase 2: Evaluation ===")
        results = evaluate_dataset(kg, args.dataset, model=args.model, limit=args.limit)

        kg.close()

        report = results.to_markdown()
        print()
        print(report)

        if args.output:
            args.output.write_text(report)
            logger.info("Results written to %s", args.output)

    finally:
        if not args.keep_db and not args.db_dir:
            shutil.rmtree(db_dir, ignore_errors=True)
            logger.info("Cleaned up temp KG: %s", db_dir)
        else:
            logger.info("KG database kept at: %s", db_dir)


if __name__ == "__main__":
    main()
