#!/usr/bin/env python3
"""Interleaved speed benchmark.

The machine is shared (load average ~5 on this 10-core box while the benchmark
ran), so single-pass measurements taken one model after another can drift with
transient load. This script restarts each model in rotating order for several
rounds and measures the same protocol every time:

  round 1: q8_0  -> q5_k_m -> q4_k_m -> q6_k
  round 2: q6_k  -> q4_k_m -> q5_k_m -> q8_0
  round 3: q4_k_m -> q8_0 -> q6_k -> q5_k_m

Per start: spawn→/health 200, one warm-up 10-doc request (discarded), then 5
repetitions each for 1/10/30 documents, RSS sample, SIGTERM. Medians across
rounds are reported as the primary speed comparison.

Writes /tmp/gbrain-rerank-bench/results/speed_interleaved.json
"""
import json
import os
import statistics
import time

import bench_lib as B

ROOT = "/tmp/gbrain-rerank-bench"
RES = os.path.join(ROOT, "results")
REPS = 5
ROUNDS = 2

MODELS = [
    ("q8_0", "/tmp/gbrain-rerank-bench/models/qwen3-reranker-0.6b-q8_0.gguf"),
    ("q5_k_m", "/tmp/gbrain-rerank-bench/models/qwen3-reranker-0.6b-q8_0-requant-Q5_K_M.gguf"),
    ("q4_k_m", "/tmp/gbrain-rerank-bench/models/qwen3-reranker-0.6b-q8_0-requant-Q4_K_M.gguf"),
    ("q6_k", "/tmp/gbrain-rerank-bench/models/qwen3-reranker-0.6b-q8_0-requant-Q6_K.gguf"),
]
ORDERS = [
    ["q8_0", "q5_k_m", "q4_k_m", "q6_k"],
    ["q6_k", "q4_k_m", "q5_k_m", "q8_0"],
    ["q4_k_m", "q8_0", "q6_k", "q5_k_m"],
]


def p50(xs):
    return statistics.median(xs)


def main():
    with open(os.path.join(ROOT, "eval_set.json"), encoding="utf-8") as f:
        ev = json.load(f)
    lat = ev["latency"]
    docs = {"n_1": lat["docs_1"], "n_10": lat["docs_10"], "n_30": lat["docs_30"]}
    paths = dict(MODELS)

    out = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rounds": ROUNDS,
        "reps_per_round": REPS,
        "orders": ORDERS,
        "samples": [],
    }
    for rnd in range(ROUNDS):
        for key in ORDERS[rnd]:
            B.wait_port_free()
            p, log, t0, log_path = B.start_server(paths[key], f"server-{key}-il{rnd + 1}.log")
            sample = {"round": rnd + 1, "key": key, "log": log_path}
            try:
                sample["cold_start_health_s"] = B.wait_health(p, t0)
                # warm-up request (not measured)
                B.rerank(lat["query"], lat["docs_10_coldstart"])
                warm = {}
                for slot, n in (("n_1", 1), ("n_10", 10), ("n_30", 30)):
                    ts = []
                    for _ in range(REPS):
                        t, _, _ = B.rerank(lat["query"], docs[slot])
                        ts.append(t)
                    warm[slot] = {"n_docs": n, "latencies_s": ts,
                                  "median_latency_s": p50(ts),
                                  "min_latency_s": min(ts),
                                  "max_latency_s": max(ts),
                                  "docs_per_s": n / p50(ts)}
                sample["warm"] = warm
                sample["rss_mib"] = B.rss_mib(p.pid)
                sample["ok"] = True
            finally:
                B.stop_server(p, log)
                B.wait_port_free()
            out["samples"].append(sample)
            print(f"round {rnd + 1} {key}: health={sample.get('cold_start_health_s', -1):.2f}s "
                  f"n1={sample.get('warm', {}).get('n_1', {}).get('median_latency_s', -1):.3f}s "
                  f"n10={sample.get('warm', {}).get('n_10', {}).get('median_latency_s', -1):.3f}s "
                  f"n30={sample.get('warm', {}).get('n_30', {}).get('median_latency_s', -1):.3f}s "
                  f"rss={sample.get('rss_mib', -1):.0f}MiB load={open('/dev/null').read() if False else ''}",
                  flush=True)

    # aggregate
    agg = {}
    for key, _ in MODELS:
        ss = [s for s in out["samples"] if s["key"] == key and s.get("ok")]
        if not ss:
            continue
        per_n = {}
        for slot in ("n_1", "n_10", "n_30"):
            vals = [s["warm"][slot]["median_latency_s"] for s in ss]
            per_n[slot] = {
                "n_docs": ss[0]["warm"][slot]["n_docs"],
                "round_medians_s": vals,
                "median_latency_s": p50(vals),
                "min_latency_s": min(vals),
                "docs_per_s": ss[0]["warm"][slot]["n_docs"] / p50(vals),
            }
        agg[key] = {
            "rounds": len(ss),
            "cold_start_health_s_samples": [s["cold_start_health_s"] for s in ss],
            "cold_start_health_s": p50([s["cold_start_health_s"] for s in ss]),
            "rss_mib_samples": [s["rss_mib"] for s in ss],
            "rss_mib": p50([s["rss_mib"] for s in ss]),
            "warm": per_n,
        }
    out["aggregate"] = agg
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["loadavg_at_finish"] = os.popen("sysctl -n vm.loadavg").read().strip()
    path = os.path.join(RES, "speed_interleaved.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}\nloadavg at finish: {out['loadavg_at_finish']}")
    for k, a in agg.items():
        print(f"{k:8s} health={a['cold_start_health_s']:.2f}s "
              f"n1={a['warm']['n_1']['median_latency_s']:.3f}s "
              f"n10={a['warm']['n_10']['median_latency_s']:.3f}s "
              f"n30={a['warm']['n_30']['median_latency_s']:.3f}s "
              f"({a['warm']['n_30']['docs_per_s']:.2f} docs/s) "
              f"rss={a['rss_mib']:.0f}MiB")


if __name__ == "__main__":
    main()
