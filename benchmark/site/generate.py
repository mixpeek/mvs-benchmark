#!/usr/bin/env python3
"""
Generate the benchmark results website from raw JSON results.

Usage:
    python benchmark/site/generate.py --results results/ --output docs/
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark results site")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--output", default="docs/", help="Output directory")
    return parser.parse_args()


def load_results(results_dir: Path) -> list[dict]:
    """Load all result JSON files."""
    results = []
    for f in results_dir.rglob("*.json"):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def main():
    args = parse_args()
    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files")

    # TODO: Generate HTML/charts from results
    # - Recall vs QPS scatter plots
    # - Latency time-series for streaming ingest
    # - Memory usage bar charts
    # - TCO comparison tables
    # - Hybrid search NDCG heatmaps
    print("Site generation not yet implemented — results available as JSON")


if __name__ == "__main__":
    main()
