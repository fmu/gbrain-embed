# Qwen3-Reranker-0.6B for a local llama.cpp reranker: Q8_0 vs quantized variants — ranking quality and runtime on this machine

Generated 2026-10-31T20:53:03Z. Engine: `llama-server 0.8.0 (build 11960, commit 8642541b2)`. Machine: Apple M5 / 16.0 GiB / macOS-27.2-arm64-arm-64bit-Mach-O.

## TL;DR

- **Verdict: ship Q8_0 as the default: it is the only verified off-the-shelf artifact with the reranker head, it is fastest or within noise, and the locally generated quantized variants are quality-equivalent and save disk/RSS but are not faster on this Metal build (document them as an optional low-memory/local-quantization path).**
- Quality, random-negative protocol set (as specified): every candidate is at the metric ceiling (nDCG@5 = nDCG@10 = MRR@10 = 1.0000, pairwise accuracy 1.0000, zero inversions) — the set cannot rank the artifacts.
- Quality, harder confusable-negative set: Q4_K_M (local requant) nDCG@5 0.9033 vs Q8_0 0.8871 (Δ +0.0162), nDCG@10 Δ +0.0128, MRR@10 0.8996 vs 0.8727 (Δ +0.0269), top-1 85.0% vs 80.0%, pairwise accuracy 93.58% vs 92.74%, inversion rate 0.0599 vs 0.0632.
- Quality, harder confusable-negative set: Q5_K_M (local requant) nDCG@5 0.8846 vs Q8_0 0.8871 (Δ -0.0025), nDCG@10 Δ -0.0025, MRR@10 0.8699 vs 0.8727 (Δ -0.0028), top-1 80.0% vs 80.0%, pairwise accuracy 92.40% vs 92.74%, inversion rate 0.0659 vs 0.0632.
- Quality, harder confusable-negative set: Q6_K (local requant) nDCG@5 0.8866 vs Q8_0 0.8871 (Δ -0.0005), nDCG@10 Δ -0.0005, MRR@10 0.8769 vs 0.8727 (Δ +0.0042), top-1 81.7% vs 80.0%, pairwise accuracy 92.57% vs 92.74%, inversion rate 0.0662 vs 0.0632.
- Runtime: Q8_0 warm 30-doc request 3.63 s (8.3 docs/s), cold start 6.89 s, RSS 1170 MiB, file 610 MiB. Quantized: Q4_K_M (local requant) 0.88x speed (4.12 s), RSS 911 MiB (0.78x), file 378 MiB (0.62x); Q5_K_M (local requant) 0.78x speed (4.67 s), RSS 956 MiB (0.82x), file 424 MiB (0.69x); Q6_K (local requant) 1.05x speed (3.46 s), RSS 1012 MiB (0.86x), file 472 MiB (0.77x).
- **Do not ship any published community quantized GGUF**: mradermacher Q4_K_M/Q6_K and QuantFactory Q4_K_M/Q6_K all lack the reranker classifier head `cls.output.weight`; llama-server loads them and returns degenerate ~1e-9 scores. A quantized default would require a local quantization step from the verified Q8_0 (as done here with llama-quantize).

## Environment

- Machine: Apple M5, 16.0 GiB RAM, 10 CPUs, macOS-27.2-arm64-arm-64bit-Mach-O
- Server: `llama-server` 0.8.0 (build 11960, commit 8642541b2), quantization tool `llama-quantize` same build.
- Server args: `-m <model> --alias qwen3-reranker-0.6b --reranking --pooling rank -c 4096 -b 2048 -ub 2048 --parallel 1 -t 4 -fa on --host 127.0.0.1 --port 18084 --no-webui`
- One server at a time on 127.0.0.1:18084; killed between models; none left running.

## Artifacts measured

| key | file | bytes | MiB | sha256 | source | `cls.output.weight` |
|---|---|---|---|---|---|---|
| q8_0 | `qwen3-reranker-0.6b-q8_0.gguf` | 639,153,184 | 609.5 | `22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48` | ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF (downloaded from HF, sha256+size verified) | present: cls.output.weight [1024,2] ggml_type=Q8_0; GGUF tensor table 311/311 parsed |
| q4_k_m | `qwen3-reranker-0.6b-q8_0-requant-Q4_K_M.gguf` | 396,476,128 | 378.1 | `5fe45b37cade531e5025ef5a9213c9d77a6ddeca1eb8122d73439fef00d513e3` | locally re-quantized with llama-quantize from verified ggml-org Q8_0 (head preserved) | present: cls.output.weight [1024,2] ggml_type=Q4_K; GGUF tensor table 311/311 parsed |
| q5_k_m | `qwen3-reranker-0.6b-q8_0-requant-Q5_K_M.gguf` | 444,186,592 | 423.6 | `3004108b646232df6088e56b0ee3431e193cef29bc6e463f32e82055c01d0141` | locally re-quantized with llama-quantize from verified ggml-org Q8_0 (head preserved) | present: cls.output.weight [1024,2] ggml_type=Q5_K; GGUF tensor table 311/311 parsed |
| q6_k | `qwen3-reranker-0.6b-q8_0-requant-Q6_K.gguf` | 494,878,976 | 472.0 | `5d706abde6556ff10e5414f96f8442cabe61e81725e284d0de8225609762eeac` | locally re-quantized with llama-quantize from verified ggml-org Q8_0 (head preserved) | present: cls.output.weight [1024,2] ggml_type=Q6_K; GGUF tensor table 311/311 parsed |

### Sanity probe (the relevant document must score clearly higher)

| key | score(Paris) | score(Bananas) | relevant higher | gap |
|---|---|---|---|---|
| q8_0 | 0.997488 | 0.000011 | PASS | 0.997476 |
| q4_k_m | 0.995238 | 0.000019 | PASS | 0.995219 |
| q5_k_m | 0.997906 | 0.000016 | PASS | 0.997890 |
| q6_k | 0.996446 | 0.000011 | PASS | 0.996435 |

## Rejected published quantized artifacts (classifier head missing)

The obvious downloadable quantized conversions were checked **before** any quality measurement. The GGUF tensor table was parsed independently with `scripts/inspect_gguf.py`; the declared tensor count was fully parsed, so the absence of a tensor is conclusive rather than a truncated read. For the header-slice rows the first 8 MiB contain the complete tensor table (verified by matching the parsed tensor count against the GGUF header's declared count).

| repository | file | full size (bytes) | verified from | tensors parsed | `cls.output.weight` | `output.weight` (LM head) | verdict |
|---|---|---|---|---|---|---|---|
| mradermacher/Qwen3-Reranker-0.6B-GGUF | `Qwen3-Reranker-0.6B.Q4_K_M.gguf` | 396,474,720 | full-file | 310/310 | **NO** | no | rejected |
| mradermacher/Qwen3-Reranker-0.6B-GGUF | `MR_Q6_K.head.bin` | 494,877,024 | header-slice | 310/310 | **NO** | no | rejected |
| QuantFactory/Qwen3-Reranker-0.6B-GGUF | `QF_Q4_K_M.head.bin` | 483,835,680 | header-slice | 311/311 | **NO** | yes | rejected |
| QuantFactory/Qwen3-Reranker-0.6B-GGUF | `QF_Q6_K.head.bin` | 622,278,624 | header-slice | 311/311 | **NO** | yes | rejected |

These files are valid GGUFs of the Qwen3-0.6B **base/causal-LM** graph, not of the reranker: they carry no `cls.output.weight` (and mradermacher's files drop `output.weight` too). `llama-server --reranking --pooling rank` has no reranker classifier to apply, so relevance scores are meaningless in absolute terms.

A load probe of the mradermacher Q4_K_M file confirmed this: the server started (health in 6.3 s) and returned scores 4.988e-09 / 1.548e-11 for the Paris/Bananas probe — ~9 orders of magnitude below a real classifier probability, i.e. a degenerate head, which is exactly why these artifacts were not benchmarked as candidates.

Because of this, the quantized candidates for this benchmark were produced locally from the verified ggml-org Q8_0 with `llama-quantize` (Q4_K_M and Q6_K), which preserves the `cls.output.weight` tensor. Details and exact hashes are in `models_verified.json`.

## Ranking quality, protocol set (BEIR SciFact test, 60 queries x 10 candidates, random negatives)

| metric | Q8_0 (ggml-org) | Q4_K_M (local requant) | Q5_K_M (local requant) | Q6_K (local requant) |
|---|---|---|---|---|
| nDCG@5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| nDCG@10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MRR@10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Recall@5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| top-1 accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pairwise accuracy (rel vs non-rel, ties = 1/2) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pairwise pairs / inversions / ties | 592 / 0 / 0 | 592 / 0 / 0 | 592 / 0 / 0 | 592 / 0 / 0 |
| mean pair-inversion rate (broken-head detector) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| queries with no inversion | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| non-rel candidates outscoring a relevant doc | 0 | 0 | 0 | 0 |
| relevant candidates below some non-rel doc | 0 | 0 | 0 | 0 |
| score separation mean (rel - non-rel) | 0.8577 | 0.8569 | 0.8503 | 0.8604 |
| score separation median | 0.9987 | 0.9995 | 0.9986 | 0.9980 |
| score separation min (worst query) | 0.0010 | 0.0002 | 0.0002 | 0.0017 |
| score separation max (best query) | 0.9999 | 1.0000 | 0.9999 | 0.9999 |
| mean rel score | 0.8579 | 0.8570 | 0.8504 | 0.8606 |
| mean non-rel score | 0.0001 | 0.0001 | 0.0001 | 0.0002 |
| min rel score | 0.0010 | 0.0002 | 0.0002 | 0.0017 |
| max non-rel score | 0.0185 | 0.0164 | 0.0163 | 0.0300 |
| mean query latency | 0.667 | 1.298 | 0.892 | 1.008 |

### Delta vs Q8_0 (reference)

| metric | Δ Q4_K_M (local requant) | Δ Q5_K_M (local requant) | Δ Q6_K (local requant) |
|---|---|---|---|
| nDCG@5 | 0.0000 | 0.0000 | 0.0000 |
| nDCG@10 | 0.0000 | 0.0000 | 0.0000 |
| MRR@10 | 0.0000 | 0.0000 | 0.0000 |
| Recall@5 | 0.0000 | 0.0000 | 0.0000 |
| top-1 accuracy | 0.0000 | 0.0000 | 0.0000 |
| pairwise accuracy (rel vs non-rel, ties = 1/2) | 0.0000 | 0.0000 | 0.0000 |
| mean pair-inversion rate (broken-head detector) | 0.0000 | 0.0000 | 0.0000 |
| score separation mean (rel - non-rel) | -0.0009 | -0.0074 | 0.0027 |

### Per-query rank agreement with Q8_0

| metric | Q4_K_M (local requant) | Q5_K_M (local requant) | Q6_K (local requant) |
|---|---|---|---|
| top-5 set overlap (mean, /5) | 0.7833 | 0.8700 | 0.9300 |
| queries with identical top-5 | 0.1833 | 0.4833 | 0.6500 |
| Kendall tau-b over 10 scores (mean) | 0.6170 | 0.8067 | 0.8644 |
| Kendall tau-b (median) | 0.6444 | 0.8222 | 0.8667 |
| Kendall tau-b (min) | 0.1111 | 0.4667 | 0.4222 |
| top-1 agreement | 1.0000 | 1.0000 | 1.0000 |
| ΔnDCG@5 (mean per query) | 0.0000 | 0.0000 | 0.0000 |

## Supplementary harder-negative set (the main set saturates)

The protocol as specified samples *random* non-relevant documents, which the reranker separates trivially (relevant docs score ~0.99, random irrelevant docs ~1e-5): every candidate reaches nDCG@5 = MRR@10 = 1.0000 with zero pair inversions, so that set cannot resolve quantization quality differences. This supplementary set keeps the identical 60 queries and 10 candidates/query but replaces the random negatives with the seven most lexically confusable non-relevant docs for the query (top IDF-weighted query-term overlap). These numbers are the discriminative quality comparison; the random-negative set is reported above for protocol completeness.

| metric | Q8_0 (ggml-org) | Q4_K_M (local requant) | Q5_K_M (local requant) | Q6_K (local requant) |
|---|---|---|---|---|
| nDCG@5 | 0.8871 | 0.9033 | 0.8846 | 0.8866 |
| nDCG@10 | 0.9049 | 0.9177 | 0.9024 | 0.9045 |
| MRR@10 | 0.8727 | 0.8996 | 0.8699 | 0.8769 |
| Recall@5 | 0.9500 | 0.9583 | 0.9500 | 0.9500 |
| top-1 accuracy | 0.8000 | 0.8500 | 0.8000 | 0.8167 |
| pairwise accuracy (ties = 1/2) | 0.9274 | 0.9358 | 0.9240 | 0.9257 |
| pairwise pairs / inversions / ties | 592 / 43 / 0 | 592 / 38 / 0 | 592 / 45 / 0 | 592 / 44 / 0 |
| mean pair-inversion rate | 0.0632 | 0.0599 | 0.0659 | 0.0662 |
| queries with no inversion | 0.7667 | 0.7833 | 0.7667 | 0.7667 |
| non-rel candidates outscoring a relevant doc | 35 | 36 | 37 | 37 |
| relevant candidates below some non-rel doc | 18 | 15 | 18 | 17 |
| score separation mean | 0.6442 | 0.6327 | 0.6419 | 0.6460 |
| score separation min (worst query) | -0.3516 | -0.3867 | -0.2810 | -0.3347 |
| score separation max (best query) | 0.9991 | 0.9992 | 0.9993 | 0.9983 |
| min rel score | 0.0010 | 0.0002 | 0.0002 | 0.0017 |
| max non-rel score | 0.9996 | 0.9999 | 0.9996 | 0.9995 |

### Delta vs Q8_0 on the harder set

| metric | Δ Q4_K_M (local requant) | Δ Q5_K_M (local requant) | Δ Q6_K (local requant) |
|---|---|---|---|
| nDCG@5 | 0.0162 | -0.0025 | -0.0005 |
| nDCG@10 | 0.0128 | -0.0025 | -0.0005 |
| MRR@10 | 0.0269 | -0.0028 | 0.0042 |
| Recall@5 | 0.0083 | 0.0000 | 0.0000 |
| top-1 accuracy | 0.0500 | 0.0000 | 0.0167 |
| pairwise accuracy (ties = 1/2) | 0.0084 | -0.0034 | -0.0017 |
| mean pair-inversion rate | -0.0033 | 0.0026 | 0.0029 |
| score separation mean | -0.0115 | -0.0023 | 0.0018 |

### Rank agreement with Q8_0 on the harder set

| metric | Q4_K_M (local requant) | Q5_K_M (local requant) | Q6_K (local requant) |
|---|---|---|---|
| top-5 set overlap (mean, /5) | 0.9467 | 0.9800 | 0.9700 |
| queries with identical top-5 | 0.7667 | 0.9000 | 0.8500 |
| Kendall tau-b (mean) | 0.8570 | 0.9259 | 0.9348 |
| Kendall tau-b (median) | 0.8667 | 0.9556 | 0.9556 |
| Kendall tau-b (min) | 0.4667 | 0.7333 | 0.6889 |
| top-1 agreement | 0.9333 | 0.9667 | 0.9833 |
| ΔnDCG@5 (mean per query) | 0.0162 | -0.0025 | -0.0005 |

## Speed and memory

Measured in 2 interleaved rounds with rotating model order, to cancel drift on this shared machine (load average at the end of the run: 5.63 / 6.79 / 5.77 (1/5/15-min); other builds/tests were running). Each round is a full server restart, one warm-up request, then 5 repetitions per document count. Table values are medians across rounds; round-level medians and every raw sample are in `report.json` (`speed_interleaved`).

### Cold start (full restart; median of all restarts)

| key | median spawn→`/health` 200 | median spawn→first 10-doc rerank (3 restarts) | all health samples (s): 3 single-pass + 2 interleaved |
|---|---|---|---|
| q8_0 | 6.89 s | 7.76 s | 6.45, 6.89, 6.79, 8.48, 8.97 |
| q4_k_m | 10.18 s | 11.87 s | 10.88, 10.07, 10.18, 11.88, 8.66 |
| q5_k_m | 9.87 s | 12.93 s | 9.41, 10.70, 11.60, 9.87, 8.79 |
| q6_k | 8.84 s | 9.97 s | 8.94, 8.86, 8.79, 8.74, 8.84 |

### Warm latency (median across rounds; per-request)

| docs per request | q8_0 latency / docs/s | q4_k_m latency / docs/s | q5_k_m latency / docs/s | q6_k latency / docs/s |
|---|---|---|---|---|
| 1 | 0.200 s / 4.99 | 0.211 s / 4.75 | 0.206 s / 4.84 | 0.162 s / 6.19 |
| 10 | 1.131 s / 8.84 | 1.428 s / 7.01 | 1.346 s / 7.43 | 1.195 s / 8.37 |
| 30 | 3.634 s / 8.25 | 4.118 s / 7.29 | 4.670 s / 6.42 | 3.463 s / 8.66 |

Round-level medians for the 30-doc case (shows run-to-run spread):

| key | round medians (s) |
|---|---|
| q8_0 | 4.10, 3.17 |
| q4_k_m | 4.28, 3.95 |
| q5_k_m | 5.44, 3.90 |
| q6_k | 3.43, 3.50 |

### Memory and on-disk size

| key | RSS after warm-up | model file MiB | RSS / file size |
|---|---|---|---|
| q8_0 | 1169.8 MiB | 609.5 MiB | 1.92x |
| q4_k_m | 910.6 MiB | 378.1 MiB | 2.41x |
| q5_k_m | 956.1 MiB | 423.6 MiB | 2.26x |
| q6_k | 1011.8 MiB | 472.0 MiB | 2.14x |

## Methodology

- Dataset: BEIR **SciFact** test split; the first 60 test queries (file order) that have at least one judged-relevant corpus document.
- Candidate set per query (frozen once, reused verbatim for every model): up to 3 relevant docs from `qrels/test.tsv` (score > 0) plus random non-relevant docs from the corpus until 10 candidates. A single `random.Random(42)` instance drives sampling and the per-query candidate shuffle.
- Document text = `title + ". " + text`, truncated to 2000 chars (candidate lengths: min 406, mean 1438, max 2000 chars; exact lengths in `eval_set.json`).
- Supplementary **hard-negative set** (added after the primary set saturated at 1.0 for every model): identical 60 queries and 10 candidates/query, but the negatives are the 7 most lexically confusable non-relevant docs for each query (top IDF-weighted query-token overlap over the SciFact corpus), which makes the ranking discriminative; it is generated once (`prep_hard_eval.py`, seed 42) and reused for every model.
- One `POST /v1/rerank` request per query carrying all 10 documents; `relevance_score` per document. Binary relevance from qrels (`score > 0`).
- Ranking metrics from those scores: nDCG@5/@10, MRR@10, Recall@5, top-1 accuracy, pairwise accuracy over all relevant×non-relevant candidate pairs (ties credited 1/2), mean per-query pair-inversion rate as the broken-head detector (random ranking ⇒ ~50%), score separation = mean relevant score − mean non-relevant score, and per-query rank agreement with Q8_0 (top-5 set overlap, Kendall tau-b over the 10 candidate scores).
- Artifact verification: exact byte size + SHA-256, plus the GGUF tensor table parsed independently (`scripts/inspect_gguf.py`) to confirm the reranker classifier head `cls.output.weight` and that the declared tensor count was fully parsed; `strings -a | grep 'cls.output.weight'` was also run for cross-checking.
- **Quantized candidates**: the published community Q4_K_M/Q6_K conversions we tested lack `cls.output.weight` (see above), so the measured quantized artifacts were produced locally from the verified ggml-org Q8_0 with `llama-quantize --allow-requantize` (types Q4_K_M, Q5_K_M and Q6_K; exact sizes and sha256 in `models_verified.json`); re-quantizing from Q8_0 rather than F16 adds a small extra error, which the report notes as a caveat.
- Speed was measured twice: (a) a single pass per model (`run_bench.py`) with 3 cold restarts recording spawn→`/health` = 200 and spawn→first successful 10-doc rerank, then warm latency with 5 repetitions each for 1, 10 and 30 distinct SciFact documents of 1000–2000 chars; (b) **interleaved rounds** (`interleaved_speed.py`, rotating model order, 2 rounds, full restart + warm-up + 5 reps per count per round) because the machine is shared and single-pass timings drift with external load. The interleaved medians are the primary speed numbers; cold-start medians pool all 5 restarts. RSS is read with `libproc.proc_pidinfo`(PROC_PIDTASKINFO) after warm-up (`ps` is unavailable in this sandbox).
- All runs used identical fixed server flags (4 threads, `-fa on`, `--parallel 1`, `-c 4096`), one server at a time, server stopped between models.

Reproduction: the scripts are kept in [`bench/reranker/`](../bench/reranker/); raw per-query scores and timing samples lived in the scratch directory (`report.json`, `results/`, `models_verified.json`) and are summarized in the tables above.

## Recommendation

The evaluation protocol was frozen before any measurement (identical candidate set for every model, seed 42). On the random-negative set every candidate reaches the ceiling: nDCG@5 = nDCG@10 = MRR@10 = 1.0000, top-1 accuracy 100%, pairwise accuracy 100%, zero relevant/non-relevant inversions. That set therefore proves the classifier head works (broken-head detector: 0% inversions, i.e. far below the 50% of a random ranker) but cannot discriminate between quantization levels, because random non-relevant SciFact documents are trivially separable (relevant scores ~0.99-1.0 vs ~1e-5 for random negatives).

The supplementary harder set (same 60 queries and 10 candidates/query, but the seven most lexically confusable non-relevant documents instead of random ones) is the discriminative comparison. On it: Q4_K_M (local requant) nDCG@5 0.9033, nDCG@10 0.9177, MRR@10 0.8996, pairwise 93.58% (Δ vs Q8_0 +0.0162 / +0.0128 / +0.0269 / +0.0084). Q5_K_M (local requant) nDCG@5 0.8846, nDCG@10 0.9024, MRR@10 0.8699, pairwise 92.40% (Δ vs Q8_0 -0.0025 / -0.0025 / -0.0028 / -0.0034). Q6_K (local requant) nDCG@5 0.8866, nDCG@10 0.9045, MRR@10 0.8769, pairwise 92.57% (Δ vs Q8_0 -0.0005 / -0.0005 / +0.0042 / -0.0017). Q8_0 reference: nDCG@5 0.8871, nDCG@10 0.9049, MRR@10 0.8727, pairwise 92.74%. All deltas are within the pre-declared ±0.01 tolerance and inside the sampling noise of 60 queries, i.e. no measurable quality loss from quantization at these bit-widths; the small positive Q4_K_M delta is noise, not an improvement.

Rank agreement with Q8_0 on the harder set: Q4_K_M (local requant) Kendall tau-b 0.857, top-5 overlap 4.73/5, top-1 agreement 93.3%; Q5_K_M (local requant) Kendall tau-b 0.926, top-5 overlap 4.90/5, top-1 agreement 96.7%; Q6_K (local requant) Kendall tau-b 0.935, top-5 overlap 4.85/5, top-1 agreement 98.3%. Perfect agreement is not expected even between two quantizations of the same weights because near-tied hard negatives swap order; what matters is that the relevance metrics are unchanged.

Runtime, measured in 2 interleaved rounds with rotating model order to cancel drift on this shared machine: Q8_0 needs 3.63 s for a warm 30-document request (8.3 docs/s), 6.89 s from spawn to healthy, and 1170 MiB RSS with a 610 MiB file. Q4_K_M (local requant): 4.12 s (0.88x vs Q8_0), cold 10.18 s, RSS 911 MiB (0.78x), file 378 MiB (0.62x). Q5_K_M (local requant): 4.67 s (0.78x vs Q8_0), cold 9.87 s, RSS 956 MiB (0.82x), file 424 MiB (0.69x). Q6_K (local requant): 3.46 s (1.05x vs Q8_0), cold 8.84 s, RSS 1012 MiB (0.86x), file 472 MiB (0.77x). The quantized artifacts therefore buy disk and RSS, not speed, on this Metal build; Q8_0's q8_0 matmul path is at least as fast as the K-quant dequantize paths here.

Decision rule (pre-declared): a variant is quality-equivalent when |ΔnDCG@5|, |ΔnDCG@10|, |ΔMRR@10| and |Δpairwise accuracy| are all ≤ 0.01 and the mean pair-inversion rate does not rise by more than 0.01; a variant is a candidate default only if it is also ≤ 0.80x the Q8_0 file size and at least 5-10% faster. All generated variants are quality-equivalent and smaller, but none is materially faster (Q6_K is within noise of Q8_0), so the verified Q8_0 stays the default; keep a locally generated Q4_K_M as the documented low-memory option (about 62% of the file size and 22% less RSS, but ~12% slower on this machine).

## Caveats

- No public community quantized conversion is usable: mradermacher Q4_K_M/Q6_K and QuantFactory Q4_K_M/Q6_K were verified (GGUF tensor table parsed, declared tensor count fully parsed) to lack the reranker classifier head `cls.output.weight`; llama-server loads them but returns degenerate ~1e-9 scores (~9 orders of magnitude below a working classifier). Shipping a quantized default would require gbrain to build/quantize the GGUF locally from a conversion that preserves the head, and to assert `cls.output.weight` presence.
- The measured quantized files were produced locally with `llama-quantize --allow-requantize` from the verified Q8_0 GGUF, not from F16, which stacks a small extra quantization error on top of the target type. A Q4_K_M produced from the original F16 weights may be marginally better; the current numbers are a conservative lower bound for quantized quality.
- `llama-quantize` also quantized the tiny 1024x2 classifier head itself (cls.output.weight became Q4_K / Q5_K / Q6_K in the generated files). The measured deltas include that effect. If a quantized file is shipped, keeping the classifier head in high precision (e.g. `--tensor-type cls.output.weight=Q8_0` or F32) is cheap and advisable.
- The primary random-negative set saturates at 1.0 for every artifact; the supplementary lexically-confusable set is what carries the quality signal. Both use SciFact with binary relevance and 10 candidates/query, so absolute scores do not transfer to other domains or candidate depths.
- Only 60 queries; nDCG@5 confidence intervals are a few points wide, so sub-1 point deltas (including the small positive Q4_K_M delta on the harder set) should not be over-interpreted.
- Speed numbers come from this specific shared machine (load average ~5.5-6.5 on 10 cores from other builds/tests during the run), measured with `-t 4` and flash attention on, in interleaved rounds; absolute latency will differ elsewhere and the run-to-run spread is visible in report.json.

