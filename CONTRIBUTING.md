# Contributing to MVS Benchmark

We actively encourage contributions, especially from teams behind the engines being benchmarked.

## How to contribute

### Improve an engine configuration

The most impactful contribution is an optimized configuration for your engine:

1. Fork this repo
2. Edit or create `benchmark/engines/{your-engine}/config.yaml`
3. Add comments explaining each parameter choice
4. Run the benchmark locally to verify improvement
5. Submit a PR with before/after results

### Add a new engine

1. Create `benchmark/engines/{engine}/config.yaml`
2. Implement the engine adapter in `benchmark/engines/{engine}/adapter.py`
3. Add Docker setup instructions
4. Run at least the steady-state scenario
5. Submit a PR

### Report an issue with methodology

Open an issue describing:
- Which scenario is affected
- What the methodological concern is
- How you'd suggest fixing it

### Submit improved results

If you can reproduce our results and find different numbers:
1. Include the full result JSON
2. Describe your hardware
3. Note any configuration differences

## Code style

- Python 3.11+
- Type hints on all public functions
- `ruff` for formatting and linting
- `pytest` for tests

## Ground rules

1. **No gaming.** Configs must be usable in production, not just on benchmarks.
2. **Reproducibility.** Every result must be independently reproducible.
3. **Transparency.** If a scenario favors your engine, say so. If it doesn't, don't hide it.
4. **Docker images only.** Pin to specific versions, no custom builds with benchmark-specific patches.

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
