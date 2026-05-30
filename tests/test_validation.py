import copy
import unittest

from benchmark.validation import ValidationError, validate_steady_state_result


def valid_payload():
    return {
        "engine": "mvs",
        "scenario": "steady-state",
        "config": {
            "num_vectors": 1000,
            "dimensions": 768,
            "num_queries": 100,
            "top_k": 10,
        },
        "results": [
            {
                "engine": "mvs",
                "scenario": "steady-state",
                "num_vectors": 1000,
                "dimensions": 768,
                "concurrency": 10,
                "num_queries": 100,
                "qps": 50.0,
                "recall_at_k": 0.85,
                "latency": {
                    "min_ms": 10.0,
                    "p50_ms": 15.0,
                    "p95_ms": 35.0,
                    "p99_ms": 45.0,
                    "p999_ms": 50.0,
                    "mean_ms": 20.0,
                    "max_ms": 60.0,
                },
                "total_time_s": 2.0,
                "errors": 0,
            }
        ],
    }


class ValidateSteadyStateResultTest(unittest.TestCase):
    def test_accepts_valid_result(self):
        validate_steady_state_result(valid_payload())

    def test_rejects_latency_percentiles_out_of_order(self):
        payload = valid_payload()
        payload["results"][0]["latency"]["p95_ms"] = 8.0

        with self.assertRaisesRegex(ValidationError, "p50_ms must be <= .*p95_ms"):
            validate_steady_state_result(payload)

    def test_rejects_recall_outside_unit_interval(self):
        payload = valid_payload()
        payload["results"][0]["recall_at_k"] = 1.2

        with self.assertRaisesRegex(ValidationError, r"recall_at_k must be in \[0, 1\]"):
            validate_steady_state_result(payload)

    def test_rejects_duplicate_concurrency(self):
        payload = valid_payload()
        payload["results"].append(copy.deepcopy(payload["results"][0]))

        with self.assertRaisesRegex(ValidationError, "duplicate result for engine=mvs"):
            validate_steady_state_result(payload)

    def test_allows_same_concurrency_for_different_engine_or_dataset(self):
        payload = valid_payload()
        other_engine = copy.deepcopy(payload["results"][0])
        other_engine["engine"] = "qdrant"
        larger_dataset = copy.deepcopy(payload["results"][0])
        larger_dataset["num_vectors"] = 5000
        payload["results"].extend([other_engine, larger_dataset])

        validate_steady_state_result(payload)

    def test_rejects_qps_inconsistent_with_successful_queries(self):
        payload = valid_payload()
        payload["results"][0]["qps"] = 90.0

        with self.assertRaisesRegex(ValidationError, "qps is inconsistent"):
            validate_steady_state_result(payload)

    def test_rejects_error_count_larger_than_query_count(self):
        payload = valid_payload()
        payload["results"][0]["errors"] = 101

        with self.assertRaisesRegex(ValidationError, "errors must be between"):
            validate_steady_state_result(payload)


if __name__ == "__main__":
    unittest.main()
