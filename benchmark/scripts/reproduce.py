#!/usr/bin/env python3
"""
Reproduce a specific published benchmark result.

Usage:
    python benchmark/scripts/reproduce.py --result results/2026-04/steady-state-mvs.json
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce a benchmark result")
    parser.add_argument(
        "--result",
        required=True,
        help="Path to a published result JSON file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_path = Path(args.result)

    if not result_path.exists():
        print(f"Result file not found: {result_path}")
        sys.exit(1)

    with open(result_path) as f:
        result = json.load(f)

    print(f"Reproducing: {result['scenario']} x {result['engine']}")
    print(f"Original timestamp: {result['timestamp']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Hardware: {json.dumps(result.get('hardware', {}), indent=2)}")

    # Extract the config and pin the exact Docker image
    config = result.get("config", {})
    engine_config = config.get("engine", {})
    docker_image = engine_config.get("docker_image", "unknown")

    print(f"\nDocker image: {docker_image}")
    print("\nTo reproduce, run:")
    print(f"  python benchmark/scripts/run.py \\")
    print(f"    --scenario {result['scenario']} \\")
    print(f"    --engine {result['engine']} \\")
    print(f"    --dataset {result['dataset']} \\")
    print(f"    --output results/reproduce/")


if __name__ == "__main__":
    main()
