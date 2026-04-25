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
- `aimo_math_level5`
  - Expected path: `data/benchmarks/aimo_math_level5.json`
  - 721 integer-answer MATH level 5 questions from AI-MO.
- `aimo_aime_validation`
  - Expected path: `data/benchmarks/aimo_aime_validation.json`
  - 90 AIME validation questions from AI-MO.
- `expanded_motif_pool`
  - Expected path: `data/benchmarks/expanded_motif_pool.json`
  - Deduped combined pool for broad adaptive motif collection.

You can use either:

- `--benchmark-name <preset>`
- `--benchmark-path /absolute/or/relative/path`
