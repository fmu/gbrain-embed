# Reranker benchmark scripts

These are the scripts used to produce
[`docs/reranker-benchmark.md`](../../docs/reranker-benchmark.md): Qwen3-Reranker-0.6B
Q8_0 (ggml-org) vs locally re-quantized Q4_K_M/Q5_K_M/Q6_K from that verified
Q8_0, on BEIR SciFact.

They were written for a scratch directory (`/tmp/gbrain-rerank-bench`) and
still reference those absolute paths. To reproduce, create that directory with
`models/`, `logs/`, `results/`, the SciFact split (`scifact/` with
`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`), and put the GGUF files in
`models/`; then run `prep_eval.py`, `run_bench.py` per model and
`make_report.py` / `make_report_md.py`.

The important checks encoded here:

- `inspect_gguf.py` parses the GGUF tensor table and asserts
  `cls.output.weight` exists (community reranker conversions silently drop it;
  without it llama-server returns ~1e-9 scores).
- `verify_models.py` checks exact size + SHA-256 of every artifact.
- `prep_eval.py` freezes the seed-42 candidate set (and the harder
  confusable-negative set) so every model sees identical inputs.
- `run_bench.py` measures cold start, warm latency at 1/10/30 documents and
  RSS (via libproc on macOS, because `ps` was unavailable in the sandbox).

`bench_lib.py` uses `/opt/homebrew/bin/llama-server`; adjust paths for another
setup.
