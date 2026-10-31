#!/usr/bin/env python3
"""Probe whether a GGUF that lacks the reranker classifier head can be loaded and
used with `--reranking --pooling rank`, and record what the server does.

This is a diagnostic for rejected artifacts, NOT a quality benchmark: if the
classifier head (`cls.output.weight`) is absent the relevance scores are
meaningless by construction.

Usage: python3 probe_broken_head.py --path <model.gguf> --key <name>
Writes /tmp/gbrain-rerank-bench/results/<key>.broken_probe.json
"""
import argparse
import json
import os
import time
import urllib.error

import bench_lib as B

ROOT = "/tmp/gbrain-rerank-bench"
RES = os.path.join(ROOT, "results")
SANITY_QUERY = "What is the capital of France?"
SANITY_DOCS = [
    "Paris is the capital of France.",
    "Bananas are yellow and grow in the tropics.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--head-count", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    out = {
        "key": args.key,
        "model_path": args.path,
        "classifier_head_occurrences": args.head_count,
        "server_args": " ".join(B.SERVER_ARGS),
        "attempt_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p = log = None
    try:
        B.wait_port_free()
        p, log, t0, log_path = B.start_server(args.path, f"server-{args.key}-brokenprobe.log")
        out["log"] = log_path
        try:
            out["health_ok_after_s"] = B.wait_health(p, t0, deadline_s=120)
            out["loaded"] = True
        except Exception as e:
            out["loaded"] = False
            out["load_error"] = f"{type(e).__name__}: {e}"
        if out.get("loaded"):
            try:
                lat, results, raw = B.rerank(SANITY_QUERY, SANITY_DOCS)
                scores = [float(r["relevance_score"]) for r in results]
                out["rerank"] = {
                    "ok": True,
                    "scores": scores,
                    "relevant_higher": scores[0] > scores[1],
                    "gap": scores[0] - scores[1],
                    "latency_s": lat,
                    "raw_response": raw,
                }
            except urllib.error.HTTPError as e:
                out["rerank"] = {
                    "ok": False,
                    "http_status": e.code,
                    "error_body": e.read().decode("utf-8", "replace")[:2000],
                }
            except Exception as e:
                out["rerank"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        with open(log_path, "rb") as f:
            tail = f.read()[-4000:].decode("utf-8", "replace")
        out["log_tail"] = tail
    finally:
        if p is not None:
            B.stop_server(p, log)
            B.wait_port_free()

    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = os.path.join(RES, args.key + ".broken_probe.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "log_tail"}, indent=1))
    print(f"\nwrote {path}")
    print("\n--- server log tail ---\n" + out.get("log_tail", "")[-2500:])


if __name__ == "__main__":
    main()
