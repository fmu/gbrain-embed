#!/bin/bash
# Produce the quantized candidates locally from the verified ggml-org Q8_0 GGUF.
#
# Why: every published community conversion we checked (mradermacher Q4_K_M/Q6_K,
# QuantFactory Q4_K_M/Q6_K) was converted as a plain causal LM and therefore lacks
# the reranker classifier head (cls.output.weight / score.weight); llama-server
# then produces degenerate ~1e-9 relevance scores. Re-quantizing the correct Q8_0
# conversion preserves the classifier head, so it is the only way to measure what a
# quantized Qwen3-Reranker-0.6B would actually do in this runtime.
#
# NOTE: quantizing from Q8_0 (not F16) adds a small extra quantization error on top
# of the target type; this is documented in the report as a caveat.
set -euo pipefail
M=/tmp/qwen3-rerank-bench/models
SRC="$M/qwen3-reranker-0.6b-q8_0.gguf"
Q=/opt/homebrew/bin/llama-quantize

quantize() { # type outname
  local type=$1 out=$2
  echo "=== llama-quantize $type -> $out ==="
  "$Q" --allow-requantize "$SRC" "$M/$out" "$type" 2>&1 | tail -25
  echo "size: $(stat -f %z "$M/$out")"
  echo "sha256: $(shasum -a 256 "$M/$out" | awk '{print $1}')"
  echo "cls.output.weight: $(strings -a "$M/$out" | grep -c 'cls.output.weight')"
}

quantize Q4_K_M "qwen3-reranker-0.6b-q8_0-requant-Q4_K_M.gguf"
quantize Q5_K_M "qwen3-reranker-0.6b-q8_0-requant-Q5_K_M.gguf"
quantize Q6_K   "qwen3-reranker-0.6b-q8_0-requant-Q6_K.gguf"
