#!/usr/bin/env python3
"""Run one candidate model over a supplementary evaluation set and save raw scores.

Usage: python3 run_quality_extra.py --key q8_0 --path <gguf> --eval-set hard
Writes /tmp/gbrain-rerank-bench/results/<key>.<eval-set>.json
"""
import argparse
import json
import os
import statistics
import time

import bench_lib as B

ROOT = "/tmp/gbrain-rerank-bench"
RES = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--eval-set", required=True, choices=["hard"])
    args = ap.parse_args()

    ev_path = os.path.join(ROOT, f"eval_set_{args.eval_set}.json")
    with open(ev_path, encoding="utf-8") as f:
        ev = json.load(f)
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, f"{args.key}.{args.eval_set}.json")
    out = {"key": args.key, "eval_set": args.eval_set, "model_path": args.path,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = log = None
    try:
        B.wait_port_free()
        p, log, t0, log_path = B.start_server(args.path, f"server-{args.key}-{args.eval_set}.log")
        out["log"] = log_path
        out["health_s"] = B.wait_health(p, t0)
        rows = []
        t_phase = time.perf_counter()
        for i, q in enumerate(ev["queries"]):
            docs = [c["text"] for c in q["candidates"]]
            lat, results, _ = B.rerank(q["query"], docs)
            scores = B.score_list(results, len(docs))
            rows.append({"qid": q["qid"], "query": q["query"], "latency_s": lat,
                         "scores": scores,
                         "labels": [c["label"] for c in q["candidates"]],
                         "doc_ids": [c["doc_id"] for c in q["candidates"]]})
            if (i + 1) % 20 == 0:
                print(f"[{args.key}/{args.eval_set}] {i + 1}/{len(ev['queries'])}", flush=True)
        out["elapsed_s"] = time.perf_counter() - t_phase
        out["mean_query_latency_s"] = statistics.mean(r["latency_s"] for r in rows)
        out["queries"] = rows
        out["ok"] = True
    finally:
        if p is not None:
            B.stop_server(p, log)
            B.wait_port_free()
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
