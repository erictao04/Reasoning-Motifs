# Benchmark Notes

The benchmark loader expects either CSV or JSON files with:

- `question`
- `answer`

Recommended next real benchmarks to plug in locally:

- `math_500`
  - Expected path: `data/benchmarks/math_500.json`
  - Good default for real mathematical reasoning with exact-answer checking.
- `gsm8k`
  - Expected path: `data/benchmarks/gsm8k_test.json`
  - Good first real benchmark for word-problem reasoning.
- `aime_2024`
  - Expected path: `data/benchmarks/aime_2024.json`
  - Good harder benchmark when you want longer and more failure-prone traces.

You can use either:

- `--benchmark-name <preset>`
- `--benchmark-path /absolute/or/relative/path`
