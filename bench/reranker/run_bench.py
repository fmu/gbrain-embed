#!/usr/bin/env python3
"""Measure sanity probe, ranking quality on the frozen SciFact eval set, cold
start, warm latency and RSS for one candidate GGUF.

Usage: python3 run_bench.py --key q8_0 --path /path/model.gguf \
         --size-bytes N --sha256 HEX --head-info "..."
Writes /tmp/gbrain-rerank-bench/results/<key>.json
"""
import argparse
import json
import os
import statistics
import time

import bench_lib as B

ROOT = "/tmp/gbrain-rerank-bench"
RES = os.path.join(ROOT, "results")
SANITY_QUERY = "What is the capital of France?"
SANITY_DOCS = [
    "Paris is the capital of France.",
    "Bananas are yellow and grow in the tropics.",
]
WARM_REPS = 5
COLD_RESTARTS = 3


def p50(xs):
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--size-bytes", type=int, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--head-info", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--file-size-actual", type=int, default=None)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "eval_set.json"), encoding="utf-8") as f:
        ev = json.load(f)
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, args.key + ".json")

    result = {
        "key": args.key,
        "model_path": args.path,
        "declared_size_bytes": args.size_bytes,
        "actual_size_bytes": args.file_size_actual,
        "sha256": args.sha256,
        "head_info": args.head_info,
        "source": args.source,
        "engine": "llama-server 0.8.0 (build 11960, commit 8642541b2)",
        "server_args": " ".join(B.SERVER_ARGS),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    p = log = None
    try:
        # ---------- phase 1: sanity + quality on a freshly started server ------
        p, log, t0, log_path = B.start_server(args.path, f"server-{args.key}-quality.log")
        result["quality_phase_log"] = log_path
        result["quality_phase_health_s"] = B.wait_health(p, t0)

        sanity_lat, sanity_results, sanity_raw = B.rerank(SANITY_QUERY, SANITY_DOCS)
        s = [float(r["relevance_score"]) for r in sanity_results]
        result["sanity"] = {
            "query": SANITY_QUERY,
            "documents": SANITY_DOCS,
            "scores": s,
            "relevant_higher": s[0] > s[1],
            "gap": s[0] - s[1],
            "latency_s": sanity_lat,
            "raw_response": sanity_raw,
        }
        print(f"[{args.key}] sanity scores={s} relevant_higher={s[0] > s[1]}", flush=True)

        # ---------- quality -----------------------------------------------------
        qrows = []
        t_phase = time.perf_counter()
        for i, q in enumerate(ev["queries"]):
            docs = [c["text"] for c in q["candidates"]]
            lat, results, _ = B.rerank(q["query"], docs)
            scores = B.score_list(results, len(docs))
            qrows.append(
                {
                    "qid": q["qid"],
                    "query": q["query"],
                    "latency_s": lat,
                    "scores": scores,
                    "labels": [c["label"] for c in q["candidates"]],
                    "doc_ids": [c["doc_id"] for c in q["candidates"]],
                    "doc_chars": [len(d) for d in docs],
                }
            )
            if (i + 1) % 10 == 0:
                print(f"[{args.key}] quality {i + 1}/{len(ev['queries'])}", flush=True)
        result["quality"] = {
            "elapsed_s": time.perf_counter() - t_phase,
            "mean_query_latency_s": statistics.mean(r["latency_s"] for r in qrows),
            "mean_doc_chars": statistics.mean(c for r in qrows for c in r["doc_chars"]),
            "queries": qrows,
        }
        B.stop_server(p, log)
        p = log = None
        B.wait_port_free()
        print(f"[{args.key}] quality done", flush=True)

        # ---------- phase 2: cold starts (restart from scratch) -----------------
        cold = []
        for i in range(COLD_RESTARTS):
            B.wait_port_free()
            p, log, t0, log_path = B.start_server(
                args.path, f"server-{args.key}-cold{i + 1}.log")
            t_health = B.wait_health(p, t0)
            t_rr0 = time.perf_counter()
            lat0, res0, _ = B.rerank(ev["latency"]["query"], ev["latency"]["docs_10_coldstart"])
            t_first_rerank = time.perf_counter() - t0
            cold.append(
                {
                    "run": i + 1,
                    "cold_start_health_s": t_health,
                    "cold_start_first_rerank_s": t_first_rerank,
                    "first_rerank_request_s": time.perf_counter() - t_rr0,
                    "scores": [float(r["relevance_score"]) for r in res0],
                    "log": log_path,
                }
            )
            print(f"[{args.key}] cold run {i + 1}: health={t_health:.3f}s "
                  f"first_rerank={t_first_rerank:.3f}s", flush=True)
            if i < COLD_RESTARTS - 1:
                B.stop_server(p, log)
                p = log = None
        result["cold_start"] = {
            "runs": cold,
            "median_health_s": p50([c["cold_start_health_s"] for c in cold]),
            "median_first_rerank_s": p50([c["cold_start_first_rerank_s"] for c in cold]),
        }

        # ---------- phase 3: warm latency + RSS (server from last cold start) ---
        warm = {}
        counts = [("docs_1", 1), ("docs_10", 10), ("docs_30", 30)]
        for slot, n in counts:
            docs = ev["latency"][slot]
            lats = []
            for _ in range(WARM_REPS):
                lat, results, _ = B.rerank(ev["latency"]["query"], docs)
                lats.append(lat)
            med = p50(lats)
            warm[f"n_{n}"] = {
                "n_docs": n,
                "doc_chars": [len(d) for d in docs],
                "reps": WARM_REPS,
                "latencies_s": lats,
                "median_latency_s": med,
                "min_latency_s": min(lats),
                "max_latency_s": max(lats),
                "docs_per_s": n / med,
            }
            print(f"[{args.key}] warm n={n}: median={med:.3f}s "
                  f"({n / med:.2f} docs/s)", flush=True)

        # settle then RSS
        time.sleep(1.0)
        rss = [B.rss_mib(p.pid) for _ in range(3)]
        time.sleep(0.5)
        rss.append(B.rss_mib(p.pid))
        result["warm"] = warm
        result["memory"] = {
            "rss_mib_samples": rss,
            "rss_mib": p50(rss),
            "rss_mib_max": max(rss),
            "model_file_bytes": args.file_size_actual,
            "model_file_mib": (args.file_size_actual or 0) / (1024.0 * 1024.0),
        }
        print(f"[{args.key}] RSS={result['memory']['rss_mib']:.1f} MiB", flush=True)

        B.stop_server(p, log)
        p = log = None
        B.wait_port_free()
        result["ok"] = True
    finally:
        if p is not None:
            B.stop_server(p, log)
            B.wait_port_free()

    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, out_path)
    print(f"[{args.key}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
