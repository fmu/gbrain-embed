#!/usr/bin/env python3
"""Compute ranking-quality metrics for every candidate from results/<key>.json,
Q8_0 agreement metrics, and write /tmp/gbrain-rerank-bench/report.json.

Also builds the comparison tables consumed by make_report_md.py.
"""
import json
import math
import os
import statistics

ROOT = "/tmp/gbrain-rerank-bench"
RES = os.path.join(ROOT, "results")
REF_KEY = "q8_0"


# --------------------------------- metrics ------------------------------------
def dcg(rels, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(labels, scores, k):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    rels = [labels[i] for i in order]
    ideal = [1] * min(k, sum(labels))
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def first_relevant_rank(labels, scores):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            return rank
    return None


def kendall_tau_b(x, y):
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = x[i] - x[j]
            b = y[i] - y[j]
            if a == 0 and b == 0:
                tx += 1
                ty += 1
            elif a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif (a > 0) == (b > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    if denom == 0:
        return 1.0
    return (conc - disc) / denom


def query_metrics(row):
    labels = row["labels"]
    scores = row["scores"]
    n_rel = sum(labels)
    n_non = len(labels) - n_rel
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranked_labels = [labels[i] for i in order]
    rank_first = first_relevant_rank(labels, scores)
    top5 = set(order[:5])

    rel_scores = [s for s, l in zip(scores, labels) if l == 1]
    non_scores = [s for s, l in zip(scores, labels) if l == 0]
    rel_min = min(rel_scores) if rel_scores else None
    nonrel_max = max(non_scores) if non_scores else None
    nonrel_above_any_rel = (sum(1 for b in non_scores if b > rel_min)
                            if rel_scores and non_scores else 0)
    rel_below_any_nonrel = (sum(1 for a in rel_scores if a < nonrel_max)
                            if rel_scores and non_scores else 0)
    inv = 0
    total_pairs = 0
    for a in rel_scores:
        for b in non_scores:
            total_pairs += 1
            if b > a:
                inv += 1
    ties = 0
    for a in rel_scores:
        for b in non_scores:
            if a == b:
                ties += 1

    return {
        "qid": row["qid"],
        "scores": scores,
        "labels": labels,
        "n_relevant": n_rel,
        "n_candidates": len(labels),
        "ndcg5": ndcg_at_k(labels, scores, 5),
        "ndcg10": ndcg_at_k(labels, scores, 10),
        "mrr10": (1.0 / rank_first) if rank_first and rank_first <= 10 else 0.0,
        "recall5": sum(1 for l in ranked_labels[:5] if l == 1) / n_rel if n_rel else 0.0,
        "top1": 1.0 if ranked_labels and ranked_labels[0] == 1 else 0.0,
        "pairs_inverted": inv,
        "pairs_tied": ties,
        "pairs_total": total_pairs,
        "pair_inversion_rate": inv / total_pairs if total_pairs else 0.0,
        "nonrel_above_any_rel": nonrel_above_any_rel,
        "rel_below_any_nonrel": rel_below_any_nonrel,
        "has_inversion": inv > 0,
        "rel_score_mean": statistics.mean(rel_scores) if rel_scores else None,
        "nonrel_score_mean": statistics.mean(non_scores) if non_scores else None,
        "rel_score_min": min(rel_scores) if rel_scores else None,
        "rel_score_max": max(rel_scores) if rel_scores else None,
        "nonrel_score_max": max(non_scores) if non_scores else None,
        "nonrel_score_min": min(non_scores) if non_scores else None,
        "score_min": min(scores),
        "score_max": max(scores),
        "top5_set": sorted(order[:5]),
        "order": order,
        "tied_scores": len(scores) - len(set(scores)),
        "latency_s": row.get("latency_s"),
    }


def aggregate(per_q):
    def mean(key):
        vals = [q[key] for q in per_q if q[key] is not None]
        return statistics.mean(vals) if vals else None

    pair_inv_total = sum(q["pairs_inverted"] for q in per_q)
    nonrel_above_total = sum(q.get("nonrel_above_any_rel", 0) for q in per_q)
    rel_below_total = sum(q.get("rel_below_any_nonrel", 0) for q in per_q)
    nonrel_candidates_total = sum(sum(1 for l in q.get("labels", [])
                                      if l == 0) for q in per_q) if per_q else 0
    rel_candidates_total = sum(sum(1 for l in q.get("labels", [])
                                   if l == 1) for q in per_q) if per_q else 0
    pair_tie_total = sum(q.get("pairs_tied", 0) for q in per_q)
    pair_total = sum(q["pairs_total"] for q in per_q)
    rel_means = [q["rel_score_mean"] for q in per_q]
    nonrel_means = [q["nonrel_score_mean"] for q in per_q]
    seps = [a - b for a, b in zip(rel_means, nonrel_means)]
    return {
        "n_queries": len(per_q),
        "ndcg5": mean("ndcg5"),
        "ndcg10": mean("ndcg10"),
        "mrr10": mean("mrr10"),
        "recall5": mean("recall5"),
        "top1_accuracy": mean("top1"),
        "pairwise_accuracy": ((pair_total - pair_inv_total - 0.5 * pair_tie_total) / pair_total
                              if pair_total else None),
        "pairwise_pairs": pair_total,
        "pairwise_inversions": pair_inv_total,
        "pairwise_ties": pair_tie_total,
        "nonrel_above_any_rel_candidates": nonrel_above_total,
        "rel_below_any_nonrel_candidates": rel_below_total,
        "nonrel_candidates": nonrel_candidates_total,
        "rel_candidates": rel_candidates_total,
        "mean_pair_inversion_rate": mean("pair_inversion_rate"),
        "queries_with_any_inversion_rate": statistics.mean(1.0 if q["has_inversion"] else 0.0 for q in per_q),
        "queries_fully_correct_rate": statistics.mean(1.0 if q["pairs_inverted"] == 0 else 0.0 for q in per_q),
        "score_separation_mean": statistics.mean(seps) if seps else None,
        "score_separation_median": statistics.median(seps) if seps else None,
        "score_separation_min": min(seps) if seps else None,
        "score_separation_max": max(seps) if seps else None,
        "rel_score_mean": statistics.mean(rel_means) if rel_means else None,
        "nonrel_score_mean": statistics.mean(nonrel_means) if nonrel_means else None,
        "rel_score_min": min(q["rel_score_min"] for q in per_q if q["rel_score_min"] is not None),
        "nonrel_score_max": max(q["nonrel_score_max"] for q in per_q if q["nonrel_score_max"] is not None),
        "mean_query_latency_s": mean("latency_s"),
        "tied_score_queries": sum(1 for q in per_q if q["tied_scores"] > 0),
    }


def agreement_vs_ref(ref_per_q, cur_per_q):
    ref_by = {q["qid"]: q for q in ref_per_q}
    overlaps, taus, top1s, ndcg_deltas = [], [], [], []
    for q in cur_per_q:
        r = ref_by[q["qid"]]
        overlap = len(set(q["top5_set"]) & set(r["top5_set"])) / 5.0
        overlaps.append(overlap)
        taus.append(kendall_tau_b(q["scores"], r["scores"]))
        top1s.append(1.0 if q["order"][0] == r["order"][0] else 0.0)
        ndcg_deltas.append(q["ndcg5"] - r["ndcg5"])
    return {
        "reference": REF_KEY,
        "top5_set_overlap_mean": statistics.mean(overlaps),
        "top5_identical_rate": statistics.mean(1.0 if o == 1.0 else 0.0 for o in overlaps),
        "kendall_tau_b_mean": statistics.mean(taus),
        "kendall_tau_b_median": statistics.median(taus),
        "kendall_tau_b_min": min(taus),
        "top1_agreement_rate": statistics.mean(top1s),
        "ndcg5_delta_mean": statistics.mean(ndcg_deltas),
    }


def main():
    keys = [k for k in ("q8_0", "q4_k_m", "q5_k_m", "q6_k")
            if os.path.exists(os.path.join(RES, k + ".json"))]
    if REF_KEY not in keys:
        raise SystemExit("reference q8_0 results missing")
    raw = {k: json.load(open(os.path.join(RES, k + ".json"), encoding="utf-8")) for k in keys}
    per_q = {k: [query_metrics(r) for r in raw[k]["quality"]["queries"]] for k in keys}
    agg = {k: aggregate(per_q[k]) for k in keys}
    agree = {k: agreement_vs_ref(per_q[REF_KEY], per_q[k]) for k in keys if k != REF_KEY}

    # ---------- supplementary hard-negative set ----------
    hard_keys = [k for k in keys
                 if os.path.exists(os.path.join(RES, k + ".hard.json"))]
    quality_hard = {}
    agree_hard = {}
    if hard_keys:
        hard_raw = {k: json.load(open(os.path.join(RES, k + ".hard.json"),
                                      encoding="utf-8")) for k in hard_keys}
        hard_per_q = {k: [query_metrics(r) for r in hard_raw[k]["queries"]]
                      for k in hard_keys}
        for k in hard_keys:
            hq = aggregate(hard_per_q[k])
            quality_hard[k] = {
                "eval_set": "hard",
                "n_queries": hq["n_queries"],
                "ndcg5": hq["ndcg5"], "ndcg10": hq["ndcg10"],
                "mrr10": hq["mrr10"], "recall5": hq["recall5"],
                "top1_accuracy": hq["top1_accuracy"],
                "pairwise_accuracy": hq["pairwise_accuracy"],
                "pairwise_pairs": hq["pairwise_pairs"],
                "pairwise_inversions": hq["pairwise_inversions"],
                "pairwise_ties": hq["pairwise_ties"],
                "mean_pair_inversion_rate": hq["mean_pair_inversion_rate"],
                "queries_fully_correct_rate": hq["queries_fully_correct_rate"],
                "nonrel_above_any_rel_candidates": hq["nonrel_above_any_rel_candidates"],
                "rel_below_any_nonrel_candidates": hq["rel_below_any_nonrel_candidates"],
                "nonrel_candidates": hq["nonrel_candidates"],
                "rel_candidates": hq["rel_candidates"],
                "score_separation_mean": hq["score_separation_mean"],
                "score_separation_median": hq["score_separation_median"],
                "score_separation_min": hq["score_separation_min"],
                "score_separation_max": hq["score_separation_max"],
                "rel_score_mean": hq["rel_score_mean"],
                "nonrel_score_mean": hq["nonrel_score_mean"],
                "rel_score_min": hq["rel_score_min"],
                "nonrel_score_max": hq["nonrel_score_max"],
                "per_query": [
                    {kk: qq[kk] for kk in ("qid", "n_relevant", "ndcg5", "ndcg10",
                                           "mrr10", "recall5", "top1",
                                           "pair_inversion_rate", "rel_score_mean",
                                           "nonrel_score_mean", "rel_score_min",
                                           "nonrel_score_max", "scores", "labels")}
                    for qq in hard_per_q[k]
                ],
            }
        if REF_KEY in hard_keys:
            agree_hard = {k: agreement_vs_ref(hard_per_q[REF_KEY], hard_per_q[k])
                          for k in hard_keys if k != REF_KEY}
        hard_meta = json.load(open(os.path.join(ROOT, "eval_set_hard.json"),
                                   encoding="utf-8")).get("meta", {})
    else:
        hard_meta = None

    # ---------- interleaved speed (preferred over single-pass under load) ------
    speed_il = None
    il_path = os.path.join(RES, "speed_interleaved.json")
    if os.path.exists(il_path):
        speed_il = json.load(open(il_path, encoding="utf-8"))

    machine = {}
    mpath = os.path.join(ROOT, "machine.json")
    if os.path.exists(mpath):
        machine = json.load(open(mpath, encoding="utf-8"))

    verification = None
    vpath = os.path.join(ROOT, "models_verified.json")
    if os.path.exists(vpath):
        verification = json.load(open(vpath, encoding="utf-8"))
    broken_probe = None
    for fn in sorted(os.listdir(RES)) if os.path.isdir(RES) else []:
        if fn.endswith(".broken_probe.json"):
            broken_probe = json.load(open(os.path.join(RES, fn), encoding="utf-8"))
            break

    artifacts = []
    for k in keys:
        r = raw[k]
        artifacts.append({
            "key": k,
            "path": r["model_path"],
            "size_bytes": r["actual_size_bytes"],
            "size_mib": (r["actual_size_bytes"] or 0) / 1048576.0,
            "sha256": r["sha256"],
            "head_info": r["head_info"],
            "source": r.get("source", ""),
            "sanity": r["sanity"],
            "rss_mib": r["memory"]["rss_mib"],
            "cold_start_health_s": r["cold_start"]["median_health_s"],
            "cold_start_health_s_single_pass_samples": [
                c["cold_start_health_s"] for c in r["cold_start"]["runs"]],
            "cold_start_health_s_all_samples": (
                [c["cold_start_health_s"] for c in r["cold_start"]["runs"]]
                + [s["cold_start_health_s"] for s in
                   ((speed_il or {}).get("samples") or []) if s.get("key") == k
                   and s.get("ok")]),
            "cold_start_first_rerank_s": r["cold_start"]["median_first_rerank_s"],
            "warm": {n: {"median_latency_s": v["median_latency_s"],
                         "docs_per_s": v["docs_per_s"],
                         "latencies_s": v["latencies_s"]}
                     for n, v in r["warm"].items()},
        })

    for a in artifacts:
        smpl = a.get("cold_start_health_s_all_samples") or []
        a["cold_start_health_s_all_median"] = (
            statistics.median(smpl) if smpl else a["cold_start_health_s"])
        a["cold_start_health_s_all_samples_count"] = len(smpl)

    quality = {}
    for k in keys:
        q = dict(agg[k])
        q["per_query"] = [
            {kk: qq[kk] for kk in ("qid", "n_relevant", "ndcg5", "ndcg10", "mrr10",
                                   "recall5", "top1", "pair_inversion_rate",
                                   "rel_score_mean", "nonrel_score_mean",
                                   "rel_score_min", "nonrel_score_max", "latency_s",
                                   "scores", "labels")}
            for qq in per_q[k]
        ]
        quality[k] = q

    report = {
        "meta": {
            "generated_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
            "engine": "llama-server 0.8.0 (build 11960, commit 8642541b2)",
            "server_args": "-m <model> --alias qwen3-reranker-0.6b --reranking --pooling rank "
                           "-c 4096 -b 2048 -ub 2048 --parallel 1 -t 4 -fa on "
                           "--host 127.0.0.1 --port 18084 --no-webui",
            "machine": machine,
            "protocol": {
                "dataset": "BEIR SciFact test split",
                "eval_seed": 42,
                "n_queries": 60,
                "candidates_per_query": 10,
                "relevant_cap": 3,
                "doc_text": "title + '. ' + text, truncated to 2000 chars",
                "metrics": ["nDCG@5", "nDCG@10", "MRR@10", "Recall@5",
                            "top-1 accuracy", "pairwise accuracy",
                            "score separation", "pair inversion rate",
                            "top-5 overlap and Kendall tau-b vs Q8_0"],
                "notes": "binary relevance from qrels score > 0; single /v1/rerank "
                         "request per query with all 10 documents; identical frozen "
                         "candidate set for every model",
            },
        },
        "artifacts": artifacts,
        "quality": quality,
        "agreement_vs_q8_0": agree,
        "rejected_published_candidates": (
            verification.get("rejected_candidates", []) if verification else []),
        "rejected_probe": broken_probe,
        "quality_hard": quality_hard,
        "agreement_hard_vs_q8_0": agree_hard,
        "eval_set_hard_meta": hard_meta,
        "speed_interleaved": speed_il,
        "verification": {k: v for k, v in (verification or {}).get("artifacts", {}).items()
                         if k in keys},
    }

    out = os.path.join(ROOT, "report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {out}")
    print("\n== quality ==")
    for k in keys:
        a = quality[k]
        print(f"{k:7s} nDCG@5={a['ndcg5']:.4f} nDCG@10={a['ndcg10']:.4f} "
              f"MRR@10={a['mrr10']:.4f} R@5={a['recall5']:.4f} top1={a['top1_accuracy']:.4f} "
              f"pair={a['pairwise_accuracy']:.4f} inv={a['mean_pair_inversion_rate']:.4f} "
              f"sep={a['score_separation_mean']:.4f}")
    for k, a in agree.items():
        print(f"{k:7s} vs q8: top5overlap={a['top5_set_overlap_mean']:.4f} "
              f"tau_b={a['kendall_tau_b_mean']:.4f} top1agree={a['top1_agreement_rate']:.4f}")
    if quality_hard:
        print("\n== quality on HARD-negative set ==")
        for k in [kk for kk in keys if kk in quality_hard]:
            a = quality_hard[k]
            print(f"{k:7s} nDCG@5={a['ndcg5']:.4f} nDCG@10={a['ndcg10']:.4f} "
                  f"MRR@10={a['mrr10']:.4f} R@5={a['recall5']:.4f} "
                  f"top1={a['top1_accuracy']:.4f} pair={a['pairwise_accuracy']:.4f} "
                  f"inv={a['mean_pair_inversion_rate']:.4f} "
                  f"sep={a['score_separation_mean']:.4f}")
        for k, a in agree_hard.items():
            print(f"{k:7s} vs q8 (hard): top5overlap={a['top5_set_overlap_mean']:.4f} "
                  f"tau_b={a['kendall_tau_b_mean']:.4f} "
                  f"top1agree={a['top1_agreement_rate']:.4f}")


if __name__ == "__main__":
    main()
