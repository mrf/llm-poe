# Performance Benchmarking

This document describes the performance benchmarking system for the llm-poe project.

## Overview

The benchmarking system measures performance of key operations in the llm-poe plugin:

- **Model list fetching time**: Cold vs cached API calls
- **Cache performance**: Hit vs miss scenarios
- **API response handling**: Simulated request/response cycles
- **Model registration time**: Registration with type detection
- **Model type detection**: Classification performance
- **Model instantiation**: Creation overhead for different model types

## Running Benchmarks Locally

### Install Dependencies

```bash
pip install -e ".[test]"
```

### Run All Benchmarks

```bash
pytest tests/test_benchmarks.py --benchmark-only
```

### Run Specific Benchmark Classes

```bash
# Only cache performance tests
pytest tests/test_benchmarks.py::TestCachePerformance --benchmark-only

# Only model type detection tests
pytest tests/test_benchmarks.py::TestModelTypeDetectionPerformance --benchmark-only
```

### Generate JSON Output

```bash
pytest tests/test_benchmarks.py --benchmark-only --benchmark-json=output.json
```

### Compare Results

```bash
# Save baseline
pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=baseline

# Compare against baseline
pytest tests/test_benchmarks.py --benchmark-only --benchmark-compare=baseline
```

## Benchmark Metrics

The benchmarks measure the following metrics:

- **min**: Minimum execution time across all rounds
- **max**: Maximum execution time across all rounds
- **mean**: Average execution time
- **stddev**: Standard deviation of execution times
- **median**: Median execution time
- **ops**: Operations per second (1/mean)
- **rounds**: Number of iterations performed

## CI/CD Integration

### Automated Benchmarking on PRs

The GitHub Actions workflow (`.github/workflows/benchmark.yml`) automatically:

1. Runs benchmarks on the PR branch
2. Checks out and runs benchmarks on the main branch
3. Compares results and generates a report
4. Posts the report as a PR comment
5. Fails if performance regressions exceed 20%

### Performance Thresholds

- **Regression threshold**: 20% slower than main branch
- **Improvement threshold**: 20% faster than main branch (highlighted positively)

### Workflow Artifacts

The following artifacts are uploaded for each benchmark run:

- `benchmark-current.json`: Full benchmark results for the PR branch
- `benchmark-main.json`: Full benchmark results for the main branch
- `benchmark-report.md`: Human-readable comparison report

## Benchmark Categories

### 1. Model Fetching Performance

**Tests:**
- `test_benchmark_fetch_models_cold`: Cold API fetch (no cache)
- `test_benchmark_fetch_models_cached`: Cached model list retrieval
- `test_benchmark_fetch_models_fallback`: Fallback model retrieval

**What it measures:** Time to fetch and parse the model list from the Poe API

**Expected range:**
- Cold: 1-5ms (mocked API)
- Cached: < 1μs
- Fallback: < 1μs

### 2. Cache Performance

**Tests:**
- `test_benchmark_cache_miss`: Cache miss with API call
- `test_benchmark_cache_hit_performance`: Pure cache hit
- `test_benchmark_cache_expiration_check`: Cache validation logic

**What it measures:** Cache efficiency and overhead

**Expected range:**
- Cache hit: < 1μs
- Cache miss: Similar to cold fetch

### 3. Model Type Detection

**Tests:**
- `test_benchmark_get_model_type_text`: Text model detection
- `test_benchmark_get_model_type_image`: Image model detection
- `test_benchmark_get_model_type_video`: Video model detection
- `test_benchmark_get_model_type_audio`: Audio model detection
- `test_benchmark_multiple_model_types`: Batch type detection

**What it measures:** Pattern matching performance for model classification

**Expected range:** 1-5μs per model name

### 4. Model Registration

**Tests:**
- `test_benchmark_register_models_success`: Full registration with API
- `test_benchmark_register_models_cached`: Registration with cache
- `test_benchmark_register_models_fallback`: Registration with fallback models

**What it measures:** Time to register all models including type detection and instantiation

**Expected range:** 20-200ms depending on number of models

### 5. Model Instantiation

**Tests:**
- `test_benchmark_text_model_creation`: PoeModel creation
- `test_benchmark_image_model_creation`: PoeImageModel creation
- `test_benchmark_video_model_creation`: PoeVideoModel creation
- `test_benchmark_audio_model_creation`: PoeAudioModel creation
- `test_benchmark_batch_model_creation`: Multiple model creation

**What it measures:** Object instantiation overhead

**Expected range:** < 1μs per model

### 6. API Response Simulation

**Tests:**
- `test_benchmark_model_execute_nonstreaming`: Non-streaming execution
- `test_benchmark_image_model_execute`: Image generation execution

**What it measures:** Request/response cycle including payload construction

**Expected range:** 5-20ms (mocked network)

## Interpreting Results

### Example Benchmark Output

```
| Benchmark | Current (mean) | Main (mean) | Change | Status |
|-----------|----------------|-------------|--------|--------|
| fetch_models_cold | 0.750ms | 0.800ms | -6.3% | ✅ IMPROVED |
| fetch_models_cached | 0.001ms | 0.001ms | +0.5% | ✓ OK |
| cache_miss | 0.780ms | 0.750ms | +4.0% | ✓ OK |
| register_models | 25.5ms | 20.1ms | +26.9% | ⚠️ REGRESSION |
```

### Status Indicators

- **✓ OK**: Performance change within acceptable range (±20%)
- **✅ IMPROVED**: Significant performance improvement (>20% faster)
- **⚠️ REGRESSION**: Significant performance degradation (>20% slower)
- **NEW**: Benchmark not present in baseline

### When Regressions Are Acceptable

Document in your PR description if a regression is expected due to:

- New functionality that inherently requires more computation
- Trade-off for better correctness or robustness
- Technical debt paydown (e.g., removing unsafe optimizations)
- Mocked vs real API differences

## Best Practices

### Writing New Benchmarks

1. **Focus on user-facing operations**: Benchmark what impacts actual usage
2. **Use realistic test data**: Match production scenarios
3. **Avoid micro-optimizations**: Focus on meaningful performance
4. **Document expectations**: Add comments about expected ranges
5. **Test both fast and slow paths**: Cover edge cases

### Example Benchmark

```python
def test_benchmark_my_operation(self, benchmark, mock_api_key):
    """Benchmark my new operation.

    Expected range: < 10ms
    """
    def operation():
        return my_function()

    result = benchmark(operation)
    assert result is not None
```

### Interpreting Failures

1. **Check the baseline**: Is main branch slower than expected?
2. **Review recent changes**: What changed in your PR?
3. **Run locally**: Can you reproduce the regression?
4. **Check variance**: High stddev indicates unstable benchmarks
5. **Compare absolute values**: Is the regression meaningful in absolute terms?

## Troubleshooting

### Benchmarks too slow

- Check if test fixtures are inefficient
- Ensure mocks are properly configured
- Verify cache clearing between runs

### Inconsistent results

- Increase rounds: `--benchmark-min-rounds=100`
- Disable GC: Already configured via pytest-benchmark
- Run on dedicated hardware (CI is consistent)

### Comparison fails

- Ensure main branch has benchmarks
- Check for breaking changes in test structure
- Verify artifact upload/download in workflow

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [GitHub Actions workflow](.github/workflows/benchmark.yml)
- [Benchmark test suite](tests/test_benchmarks.py)
