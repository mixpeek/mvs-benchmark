"""Validation helpers for benchmark result JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when a benchmark result file is internally inconsistent."""


class UnsupportedScenario(ValueError):
    """Raised when a result file uses a schema this validator does not cover."""


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    return value


def _require_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{path} must be numeric")
    return float(value)


def _require_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{path} must be an integer")
    return value


def _validate_latency(latency: dict[str, Any], path: str) -> None:
    required = ["min_ms", "p50_ms", "p95_ms", "p99_ms", "p999_ms", "max_ms", "mean_ms"]
    values = {key: _require_number(latency.get(key), f"{path}.{key}") for key in required}

    ordered = ["min_ms", "p50_ms", "p95_ms", "p99_ms", "p999_ms", "max_ms"]
    for lower, upper in zip(ordered, ordered[1:]):
        if values[lower] > values[upper]:
            raise ValidationError(
                f"{path}.{lower} must be <= {path}.{upper} "
                f"({values[lower]} > {values[upper]})"
            )

    if values["mean_ms"] < values["min_ms"] or values["mean_ms"] > values["max_ms"]:
        raise ValidationError(f"{path}.mean_ms must be between min_ms and max_ms")


def validate_steady_state_result(data: dict[str, Any]) -> None:
    """Validate a steady-state result payload emitted by ``benchmark.runner``."""
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValidationError("results must be a non-empty list")

    config = data.get("config")
    expected_num_queries = None
    if config is not None:
        config = _require_mapping(config, "config")
        expected_num_queries = _require_int(config.get("num_queries"), "config.num_queries")
        _require_int(config.get("num_vectors"), "config.num_vectors")
        _require_int(config.get("dimensions"), "config.dimensions")
        _require_int(config.get("top_k"), "config.top_k")

    seen_runs: set[tuple[str, int, int, int]] = set()
    for idx, raw_result in enumerate(results):
        result = _require_mapping(raw_result, f"results[{idx}]")
        engine = result.get("engine")
        if not isinstance(engine, str) or not engine:
            raise ValidationError(f"results[{idx}].engine must be a non-empty string")
        scenario = result.get("scenario")
        if scenario != "steady-state":
            raise ValidationError(f"results[{idx}].scenario must be 'steady-state'")
        concurrency = _require_int(result.get("concurrency"), f"results[{idx}].concurrency")

        num_queries = _require_int(result.get("num_queries"), f"results[{idx}].num_queries")
        if expected_num_queries is not None and num_queries != expected_num_queries:
            raise ValidationError(
                f"results[{idx}].num_queries ({num_queries}) does not match "
                f"config.num_queries ({expected_num_queries})"
            )
        num_vectors = _require_int(result.get("num_vectors"), f"results[{idx}].num_vectors")
        dimensions = _require_int(result.get("dimensions"), f"results[{idx}].dimensions")

        errors = _require_int(result.get("errors"), f"results[{idx}].errors")
        if errors < 0 or errors > num_queries:
            raise ValidationError(f"results[{idx}].errors must be between 0 and num_queries")

        run_key = (engine, num_vectors, dimensions, concurrency)
        if run_key in seen_runs:
            raise ValidationError(
                f"duplicate result for engine={engine} num_vectors={num_vectors} "
                f"dimensions={dimensions} concurrency={concurrency}"
            )
        seen_runs.add(run_key)

        successful_queries = num_queries - errors
        total_time_s = _require_number(result.get("total_time_s"), f"results[{idx}].total_time_s")
        qps = _require_number(result.get("qps"), f"results[{idx}].qps")
        if successful_queries and total_time_s <= 0:
            raise ValidationError(f"results[{idx}].total_time_s must be positive")
        if successful_queries == 0 and qps != 0:
            raise ValidationError(f"results[{idx}].qps must be 0 when all queries fail")
        if successful_queries and abs(qps - (successful_queries / total_time_s)) > max(1.0, qps * 0.02):
            raise ValidationError(f"results[{idx}].qps is inconsistent with num_queries/errors/total_time_s")

        recall = _require_number(result.get("recall_at_k"), f"results[{idx}].recall_at_k")
        if recall < 0 or recall > 1:
            raise ValidationError(f"results[{idx}].recall_at_k must be in [0, 1]")

        latency = _require_mapping(result.get("latency"), f"results[{idx}].latency")
        _validate_latency(latency, f"results[{idx}].latency")


def validate_result_file(path: Path) -> None:
    """Load and validate a benchmark result JSON file."""
    import json

    with path.open() as f:
        data = json.load(f)

    scenario = data.get("scenario")
    result_scenarios = {
        result.get("scenario")
        for result in data.get("results", [])
        if isinstance(result, dict)
    }
    if scenario == "steady-state" or result_scenarios == {"steady-state"}:
        validate_steady_state_result(data)
        return

    raise UnsupportedScenario(f"unsupported scenario: {scenario!r}")
