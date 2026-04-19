---
name: sample-math-online
provider: together
execution_mode: batch
model: Qwen/Qwen3.5-9B
benchmark_path: ../data/benchmarks/sample_math.csv
samples_per_question: 4
greedy_first: true
temperature: 1.0
top_p: 0.95
max_completion_tokens: 256
seed: 17
enable_thinking: true
api_base: https://api.together.xyz/v1
api_key_env: TOGETHER_API_KEY
request_timeout_seconds: 60
batch_completion_window: 24h
batch_poll_interval_seconds: 10
---

# Sample Math Online Experiment

This experiment collects reasoning traces from a hosted Qwen thinking model through
Together AI's Batch API.
