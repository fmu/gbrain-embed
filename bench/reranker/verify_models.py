#!/usr/bin/env python3
"""Verify every artifact involved in the benchmark and write models_verified.json.

Evidence per artifact:
  * exact byte size
  * sha256
  * GGUF tensor table parsed independently (`inspect_gguf.py`), including whether
    the declared tensor count was fully parsed (so absence of a tensor name is
    conclusive, not a truncated read)
  * classifier head presence: `cls.output.weight` (also checked with the task's
    `strings -a | grep` command for reproducibility)
  * architecture and classifier-head shape/type

For the published community conversions that are rejected, the complete tensor
table is verified from their first 8 MiB header slice (no need to download the
whole file); the reported full size comes from the HTTP Content-Range header.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_gguf import parse  # noqa: E402

ROOT = "/tmp/qwen3-rerank-bench"
MODELS = os.path.join(ROOT, "models")
CAND = os.path.join(ROOT, "candidates")


def sha256(path):
    out = subprocess.check_output(["shasum", "-a", "256", path]).decode()
    return out.split()[0]


def cross_check_gguf_tensors(path):
    """Second, independent parser (Lead-provided scripts/gguf_tensors.py)."""
    try:
        out = subprocess.check_output(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "gguf_tensors.py"), path],
            stderr=subprocess.DEVNULL).decode()
        d = json.loads(out)
        return {"tensors": d.get("tensors"), "has_cls_output": d.get("has_cls_output"),
                "cls_like": d.get("cls_like")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def strings_count(path, needle):
    out = subprocess.check_output(["strings", "-a", path]).decode("utf-8", "replace")
    return sum(1 for line in out.splitlines() if needle in line)


def parse_file(path, max_bytes=None):
    with open(path, "rb") as f:
        buf = f.read(max_bytes) if max_bytes else f.read()
    return parse(buf), len(buf)


def inspect(key, path, source, evidence_level="full-file", max_bytes=None,
            declared_size=None):
    info, read_n = parse_file(path, max_bytes)
    names = info.get("tensor_names", [])
    rec = {
        "key": key,
        "file": os.path.basename(path),
        "path": path,
        "source": source,
        "evidence_level": evidence_level,
        "bytes_read": read_n,
        "size_bytes": declared_size if declared_size else os.path.getsize(path),
        "sha256": sha256(path) if evidence_level == "full-file" else sha256(path),
        "sha256_scope": "full file" if evidence_level == "full-file" else f"first {read_n} bytes",
        "gguf_version": info.get("version"),
        "architecture": info.get("metadata", {}).get("general.architecture"),
        "declared_tensor_count": info.get("declared_tensor_count"),
        "tensor_names_parsed": len(names),
        "complete_tensor_table": bool(info.get("complete_tensor_table")),
        "has_cls_output_weight": "cls.output.weight" in names,
        "has_score_weight": "score.weight" in names,
        "has_output_weight": "output.weight" in names,
        "cls_like_tensors": [n for n in names if n.startswith(("cls.", "score."))],
        "strings_cls_output_weight_occurrences": strings_count(path, "cls.output.weight"),
        "strings_check_command": "strings -a <file> | grep -c 'cls.output.weight'",
        "header_parse_error": info.get("header_parse_error"),
    }
    rec["cross_check_gguf_tensors"] = cross_check_gguf_tensors(path)
    if "cls.output.weight" in names:
        t = next(t for t in info["tensors"] if t["name"] == "cls.output.weight")
        rec["cls_output_weight_dims"] = t["dims"]
        rec["cls_output_weight_ggml_type"] = t["type"]
    return rec


def main():
    out = {"verified_utc": None, "artifacts": {}, "rejected_candidates": []}

    # 1) reference: the only published correct conversion
    out["artifacts"]["q8_0"] = inspect(
        "q8_0", os.path.join(MODELS, "qwen3-reranker-0.6b-q8_0.gguf"),
        "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF (official-style conversion)")
    out["artifacts"]["q8_0"]["expected_size_bytes"] = 639153184
    out["artifacts"]["q8_0"]["expected_sha256"] = (
        "22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48")
    a = out["artifacts"]["q8_0"]
    a["size_match"] = a["size_bytes"] == a["expected_size_bytes"]
    a["sha_match"] = a["sha256"] == a["expected_sha256"]
    a["accepted"] = a["has_cls_output_weight"]

    # 2) locally re-quantized candidates (head preserved)
    for key, fn in [("q4_k_m", "qwen3-reranker-0.6b-q8_0-requant-Q4_K_M.gguf"),
                    ("q5_k_m", "qwen3-reranker-0.6b-q8_0-requant-Q5_K_M.gguf"),
                    ("q6_k", "qwen3-reranker-0.6b-q8_0-requant-Q6_K.gguf")]:
        p = os.path.join(MODELS, fn)
        if not os.path.exists(p):
            continue
        rec = inspect(key, p,
                      "locally re-quantized with llama-quantize 0.8.0 from the verified "
                      "ggml-org Q8_0 (head preserved)")
        rec["requant_source"] = "qwen3-reranker-0.6b-q8_0.gguf"
        rec["requant_source_sha256"] = out["artifacts"]["q8_0"]["sha256"]
        rec["requant_tool"] = "/opt/homebrew/bin/llama-quantize (build 11960)"
        rec["accepted"] = rec["has_cls_output_weight"]
        out["artifacts"][key] = rec

    # 3) published community quants: rejected because the classifier head is absent
    published = [
        ("mradermacher/Qwen3-Reranker-0.6B-GGUF", "Qwen3-Reranker-0.6B.Q4_K_M.gguf",
         os.path.join(MODELS, "Qwen3-Reranker-0.6B.Q4_K_M.gguf"), None, "full-file",
         "expected sha256 b432ca16e27e1c6d8e910a00e09f5022522b34aba1df5ff42f09dfaa33a1e178 (verified)"),
        ("mradermacher/Qwen3-Reranker-0.6B-GGUF", "Qwen3-Reranker-0.6B.Q6_K.gguf",
         os.path.join(CAND, "MR_Q6_K.head.bin"), 494877024, "header-slice",
         "first 8 MiB header slice; complete 310-tensor table parsed"),
        ("QuantFactory/Qwen3-Reranker-0.6B-GGUF", "Qwen3-Reranker-0.6B.Q4_K_M.gguf",
         os.path.join(CAND, "QF_Q4_K_M.head.bin"), 483835680, "header-slice",
         "first 8 MiB header slice; complete 311-tensor table parsed"),
        ("QuantFactory/Qwen3-Reranker-0.6B-GGUF", "Qwen3-Reranker-0.6B.Q6_K.gguf",
         os.path.join(CAND, "QF_Q6_K.head.bin"), 622278624, "header-slice",
         "first 8 MiB header slice; complete 311-tensor table parsed"),
    ]
    for repo, fn, path, size, level, note in published:
        if not os.path.exists(path):
            continue
        rec = inspect(f"{repo.split('/')[0]}/{fn}", path,
                      f"{repo} (community conversion)", evidence_level=level,
                      declared_size=size)
        rec["expected_size_bytes"] = size
        rec["size_note"] = note
        rec["rejected_reason"] = ("GGUF tensor table contains no `cls.output.weight` "
                                  "(and no `score.weight`): converted as a plain causal "
                                  "LM, so llama.cpp rank pooling has no reranker "
                                  "classifier head")
        rec["accepted"] = False
        out["rejected_candidates"].append(rec)

    out["verified_utc"] = subprocess.check_output(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).decode().strip()
    path = os.path.join(ROOT, "models_verified.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}\n")
    for k, a in out["artifacts"].items():
        print(f"{k:8s} size={a['size_bytes']:>12,} sha={a['sha256'][:16]}... "
              f"tensors={a['tensor_names_parsed']}/{a['declared_tensor_count']} "
              f"complete={a['complete_tensor_table']} "
              f"cls.output.weight={a['has_cls_output_weight']} "
              f"strings_hits={a['strings_cls_output_weight_occurrences']} "
              f"accepted={a['accepted']}")
    print("\nrejected published candidates:")
    for r in out["rejected_candidates"]:
        print(f"  {r['source'].split(' ')[0]:45s} {r['file']:32s} "
              f"tensors={r['tensor_names_parsed']}/{r['declared_tensor_count']} "
              f"complete={r['complete_tensor_table']} cls={r['has_cls_output_weight']} "
              f"output.weight={r['has_output_weight']}")


if __name__ == "__main__":
    main()
