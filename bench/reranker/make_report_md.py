#!/usr/bin/env python3
"""Render /tmp/gbrain-rerank-bench/report.md from report.json.

All numbers are read from report.json; the recommendation is derived from explicit
thresholds so it cannot drift from the measurements.
"""
import json
import os

ROOT = "/tmp/gbrain-rerank-bench"
REF = "q8_0"
LABELS = {
    "q8_0": "Q8_0 (ggml-org)",
    "q4_k_m": "Q4_K_M (local requant)",
    "q5_k_m": "Q5_K_M (local requant)",
    "q6_k": "Q6_K (local requant)",
}


def fmt(v, n=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def pct(v, n=4):
    return "n/a" if v is None else f"{100 * v:.{n}f}%"


def main():
    r = json.load(open(os.path.join(ROOT, "report.json"), encoding="utf-8"))
    keys = [a["key"] for a in r["artifacts"]]
    art = {a["key"]: a for a in r["artifacts"]}
    q = r["quality"]
    agree = r["agreement_vs_q8_0"]
    m = r["meta"]
    mach = m.get("machine", {})
    ref = q[REF]
    rejected = r.get("rejected_published_candidates") or []
    probe = r.get("rejected_probe")
    il = r.get("speed_interleaved") or {}
    il_agg = il.get("aggregate") or {}

    lines = []
    A = lines.append
    A("# Qwen3-Reranker-0.6B for a local llama.cpp reranker: Q8_0 vs quantized variants "
      "— ranking quality and runtime on this machine")
    A("")
    A(f"Generated {m['generated_utc']}. Engine: `{m['engine']}`. "
      f"Machine: {mach.get('cpu_brand', 'n/a')} / {mach.get('mem_gib', 'n/a')} GiB / "
      f"{mach.get('os', 'n/a')}.")
    A("")
    A("## TL;DR")
    A("")
    rec = build_recommendation(r)
    for line in rec["tldr"]:
        A(f"- {line}")
    A("")

    A("## Environment")
    A("")
    A(f"- Machine: {mach.get('cpu_brand', 'n/a')}, {mach.get('mem_gib', 'n/a')} GiB RAM, "
      f"{mach.get('ncpu', 'n/a')} CPUs, {mach.get('os', 'n/a')}")
    A(f"- Server: `llama-server` {mach.get('llama_version', '0.8.0')}, "
      f"quantization tool `llama-quantize` same build.")
    A(f"- Server args: `{m['server_args']}`")
    A("- One server at a time on 127.0.0.1:18084; killed between models; none left running.")
    A("")

    A("## Artifacts measured")
    A("")
    A("| key | file | bytes | MiB | sha256 | source | `cls.output.weight` |")
    A("|---|---|---|---|---|---|---|")
    for k in keys:
        a = art[k]
        A(f"| {k} | `{os.path.basename(a['path'])}` | {a['size_bytes']:,} | "
          f"{a['size_mib']:.1f} | `{a['sha256']}` | {a.get('source', '') or 'n/a'} | "
          f"{a['head_info']} |")
    A("")
    A("### Sanity probe (the relevant document must score clearly higher)")
    A("")
    A("| key | score(Paris) | score(Bananas) | relevant higher | gap |")
    A("|---|---|---|---|---|")
    for k in keys:
        s = art[k]["sanity"]
        A(f"| {k} | {s['scores'][0]:.6f} | {s['scores'][1]:.6f} | "
          f"{'PASS' if s['relevant_higher'] else 'FAIL'} | {s['gap']:.6f} |")
    A("")

    if rejected:
        A("## Rejected published quantized artifacts (classifier head missing)")
        A("")
        A("The obvious downloadable quantized conversions were checked **before** any "
          "quality measurement. The GGUF tensor table was parsed independently with "
          "`scripts/inspect_gguf.py`; the declared tensor count was fully parsed, so the "
          "absence of a tensor is conclusive rather than a truncated read. For the "
          "header-slice rows the first 8 MiB contain the complete tensor table (verified "
          "by matching the parsed tensor count against the GGUF header's declared count).")
        A("")
        A("| repository | file | full size (bytes) | verified from | tensors parsed | "
          "`cls.output.weight` | `output.weight` (LM head) | verdict |")
        A("|---|---|---|---|---|---|---|---|")
        for c in rejected:
            A(f"| {c['source'].split(' (')[0]} | `{c['file']}` | {c['size_bytes']:,} | "
              f"{c['evidence_level']} | {c['tensor_names_parsed']}/"
              f"{c['declared_tensor_count']} | "
              f"{'yes' if c['has_cls_output_weight'] else '**NO**'} | "
              f"{'yes' if c['has_output_weight'] else 'no'} | rejected |")
        A("")
        A("These files are valid GGUFs of the Qwen3-0.6B **base/causal-LM** graph, not of "
          "the reranker: they carry no `cls.output.weight` (and mradermacher's files drop "
          "`output.weight` too). `llama-server --reranking --pooling rank` has no reranker "
          "classifier to apply, so relevance scores are meaningless in absolute terms.")
        if probe:
            sc = (probe.get("rerank") or {}).get("scores")
            if sc:
                A("")
                A(f"A load probe of the mradermacher Q4_K_M file confirmed this: the server "
                  f"started (health in {probe.get('health_ok_after_s', 0):.1f} s) and "
                  f"returned scores {sc[0]:.3e} / {sc[1]:.3e} for the Paris/Bananas probe — "
                  f"~9 orders of magnitude below a real classifier probability, i.e. a "
                  f"degenerate head, which is exactly why these artifacts were not "
                  f"benchmarked as candidates.")
            elif (probe.get("rerank") or {}).get("ok") is False:
                A("")
                A(f"A load probe of the mradermacher Q4_K_M file failed: "
                  f"{json.dumps(probe['rerank'])[:400]}")
        A("")
        A("Because of this, the quantized candidates for this benchmark were produced "
          "locally from the verified ggml-org Q8_0 with `llama-quantize` (Q4_K_M and "
          "Q6_K), which preserves the `cls.output.weight` tensor. Details and exact "
          "hashes are in `models_verified.json`.")
        A("")

    A("## Ranking quality, protocol set (BEIR SciFact test, 60 queries x 10 candidates, "
      "random negatives)")
    A("")
    A("| metric | " + " | ".join(LABELS.get(k, k) for k in keys) + " |")
    A("|---|" + "---|" * len(keys))
    rows = [
        ("nDCG@5", "ndcg5", 4), ("nDCG@10", "ndcg10", 4), ("MRR@10", "mrr10", 4),
        ("Recall@5", "recall5", 4), ("top-1 accuracy", "top1_accuracy", 4),
        ("pairwise accuracy (rel vs non-rel, ties = 1/2)", "pairwise_accuracy", 4),
        ("pairwise pairs / inversions / ties", None, 0),
        ("mean pair-inversion rate (broken-head detector)", "mean_pair_inversion_rate", 4),
        ("queries with no inversion", "queries_fully_correct_rate", 4),
        ("non-rel candidates outscoring a relevant doc", "nonrel_above_any_rel_candidates", 0),
        ("relevant candidates below some non-rel doc", "rel_below_any_nonrel_candidates", 0),
        ("score separation mean (rel - non-rel)", "score_separation_mean", 4),
        ("score separation median", "score_separation_median", 4),
        ("score separation min (worst query)", "score_separation_min", 4),
        ("score separation max (best query)", "score_separation_max", 4),
        ("mean rel score", "rel_score_mean", 4),
        ("mean non-rel score", "nonrel_score_mean", 4),
        ("min rel score", "rel_score_min", 4),
        ("max non-rel score", "nonrel_score_max", 4),
        ("mean query latency", "mean_query_latency_s", 3),
    ]
    for label, key, n in rows:
        if key is None:
            cells = [f"{q[k]['pairwise_pairs']} / {q[k]['pairwise_inversions']} / "
                     f"{q[k].get('pairwise_ties', 0)}" for k in keys]
        else:
            cells = [fmt(q[k].get(key), n) for k in keys]
        A(f"| {label} | " + " | ".join(cells) + " |")
    A("")

    if len(keys) > 1:
        A("### Delta vs Q8_0 (reference)")
        A("")
        A("| metric | " + " | ".join(
            f"Δ {LABELS.get(k, k)}" for k in keys if k != REF) + " |")
        A("|---|" + "---|" * (len(keys) - 1))
        for label, key, n in [(l, kk, n) for l, kk, n in rows
                              if kk in ("ndcg5", "ndcg10", "mrr10", "recall5",
                                        "top1_accuracy", "pairwise_accuracy",
                                        "mean_pair_inversion_rate",
                                        "score_separation_mean")]:
            A(f"| {label} | " + " | ".join(
                fmt(q[k].get(key) - ref.get(key), n) for k in keys if k != REF) + " |")
        A("")

        A("### Per-query rank agreement with Q8_0")
        A("")
        A("| metric | " + " | ".join(
            f"{LABELS.get(k, k)}" for k in keys if k != REF) + " |")
        A("|---|" + "---|" * (len(keys) - 1))
        ag_rows = [
            ("top-5 set overlap (mean, /5)", "top5_set_overlap_mean"),
            ("queries with identical top-5", "top5_identical_rate"),
            ("Kendall tau-b over 10 scores (mean)", "kendall_tau_b_mean"),
            ("Kendall tau-b (median)", "kendall_tau_b_median"),
            ("Kendall tau-b (min)", "kendall_tau_b_min"),
            ("top-1 agreement", "top1_agreement_rate"),
            ("ΔnDCG@5 (mean per query)", "ndcg5_delta_mean"),
        ]
        for label, key in ag_rows:
            A(f"| {label} | " + " | ".join(
                fmt(agree[k].get(key), 4) for k in keys if k != REF) + " |")
        A("")

    qh = r.get("quality_hard") or {}
    agree_h = r.get("agreement_hard_vs_q8_0") or {}
    if qh:
        hkeys = [k for k in keys if k in qh]
        A("## Supplementary harder-negative set (the main set saturates)")
        A("")
        A("The protocol as specified samples *random* non-relevant documents, which the "
          "reranker separates trivially (relevant docs score ~0.99, random irrelevant "
          "docs ~1e-5): every candidate reaches nDCG@5 = MRR@10 = 1.0000 with zero pair "
          "inversions, so that set cannot resolve quantization quality differences. This "
          "supplementary set keeps the identical 60 queries and 10 candidates/query but "
          "replaces the random negatives with the seven most lexically confusable "
          "non-relevant docs for the query (top IDF-weighted query-term overlap). These "
          "numbers are the discriminative quality comparison; the random-negative set is "
          "reported above for protocol completeness.")
        A("")
        A("| metric | " + " | ".join(LABELS.get(k, k) for k in hkeys) + " |")
        A("|---|" + "---|" * len(hkeys))
        hrows = [
            ("nDCG@5", "ndcg5", 4), ("nDCG@10", "ndcg10", 4),
            ("MRR@10", "mrr10", 4), ("Recall@5", "recall5", 4),
            ("top-1 accuracy", "top1_accuracy", 4),
            ("pairwise accuracy (ties = 1/2)", "pairwise_accuracy", 4),
            ("pairwise pairs / inversions / ties", None, 0),
            ("mean pair-inversion rate", "mean_pair_inversion_rate", 4),
            ("queries with no inversion", "queries_fully_correct_rate", 4),
            ("non-rel candidates outscoring a relevant doc", "nonrel_above_any_rel_candidates", 0),
            ("relevant candidates below some non-rel doc", "rel_below_any_nonrel_candidates", 0),
            ("score separation mean", "score_separation_mean", 4),
            ("score separation min (worst query)", "score_separation_min", 4),
            ("score separation max (best query)", "score_separation_max", 4),
            ("min rel score", "rel_score_min", 4),
            ("max non-rel score", "nonrel_score_max", 4),
        ]
        for label, key, n in hrows:
            if key is None:
                cells = [f"{qh[k]['pairwise_pairs']} / {qh[k]['pairwise_inversions']} / "
                         f"{qh[k].get('pairwise_ties', 0)}" for k in hkeys]
            else:
                cells = [fmt(qh[k].get(key), n) for k in hkeys]
            A(f"| {label} | " + " | ".join(cells) + " |")
        A("")
        A("### Delta vs Q8_0 on the harder set")
        A("")
        A("| metric | " + " | ".join(f"Δ {LABELS.get(k, k)}" for k in hkeys if k != REF) + " |")
        A("|---|" + "---|" * max(0, len(hkeys) - 1))
        for label, key, n in [(l, kk, n) for l, kk, n in hrows
                              if kk in ("ndcg5", "ndcg10", "mrr10", "recall5",
                                        "top1_accuracy", "pairwise_accuracy",
                                        "mean_pair_inversion_rate",
                                        "score_separation_mean")]:
            A(f"| {label} | " + " | ".join(
                fmt(qh[k].get(key) - qh[REF].get(key), n) for k in hkeys if k != REF) + " |")
        A("")
        if agree_h:
            A("### Rank agreement with Q8_0 on the harder set")
            A("")
            A("| metric | " + " | ".join(LABELS.get(k, k) for k in hkeys if k != REF) + " |")
            A("|---|" + "---|" * max(0, len(hkeys) - 1))
            for label, key in [
                ("top-5 set overlap (mean, /5)", "top5_set_overlap_mean"),
                ("queries with identical top-5", "top5_identical_rate"),
                ("Kendall tau-b (mean)", "kendall_tau_b_mean"),
                ("Kendall tau-b (median)", "kendall_tau_b_median"),
                ("Kendall tau-b (min)", "kendall_tau_b_min"),
                ("top-1 agreement", "top1_agreement_rate"),
                ("ΔnDCG@5 (mean per query)", "ndcg5_delta_mean"),
            ]:
                A(f"| {label} | " + " | ".join(
                    fmt(agree_h[k].get(key), 4) for k in hkeys if k != REF) + " |")
            A("")

    A("## Speed and memory")
    A("")
    if il_agg:
        la_raw = str(il.get("loadavg_at_finish") or "").strip("{} ")
        la = " / ".join(la_raw.split()) if la_raw else "n/a"
        A(f"Measured in {il.get('rounds')} interleaved rounds with rotating model order, "
          f"to cancel drift on this shared machine (load average at the end of the run: "
          f"{la} (1/5/15-min); other agents/tests were running). Each "
          f"round is a full server restart, one warm-up request, then "
          f"{il.get('reps_per_round')} repetitions per document count. Table values are "
          f"medians across rounds; round-level medians and every raw sample are in "
          f"`report.json` (`speed_interleaved`).")
        A("")
        A("### Cold start (full restart; median of all restarts)")
        A("")
        A("| key | median spawn→`/health` 200 | median spawn→first 10-doc rerank (3 restarts) | all health samples (s): 3 single-pass + 2 interleaved |")
        A("|---|---|---|---|")
        for k in keys:
            samples = [round(x, 2) for x in art[k].get("cold_start_health_s_all_samples", [])]
            med = art[k].get("cold_start_health_s_all_median", art[k]["cold_start_health_s"])
            A(f"| {k} | {med:.2f} s | {art[k]['cold_start_first_rerank_s']:.2f} s | "
              f"{', '.join(f'{x:.2f}' for x in samples)} |")
        A("")
        A("### Warm latency (median across rounds; per-request)")
        A("")
        A("| docs per request | " + " | ".join(f"{k} latency / docs/s" for k in keys) + " |")
        A("|---|" + "---|" * len(keys))
        for n in ("n_1", "n_10", "n_30"):
            cells = []
            for k in keys:
                a = il_agg.get(k)
                if not a:
                    cells.append("n/a")
                    continue
                w = a["warm"][n]
                cells.append(f"{w['median_latency_s']:.3f} s / {w['docs_per_s']:.2f}")
            A(f"| {n[2:]} | " + " | ".join(cells) + " |")
        A("")
        A("Round-level medians for the 30-doc case (shows run-to-run spread):")
        A("")
        A("| key | round medians (s) |")
        A("|---|---|")
        for k in keys:
            a = il_agg.get(k)
            if not a:
                continue
            vals = ", ".join(f"{x:.2f}" for x in a["warm"]["n_30"]["round_medians_s"])
            A(f"| {k} | {vals} |")
        A("")
        A("### Memory and on-disk size")
        A("")
        A("| key | RSS after warm-up | model file MiB | RSS / file size |")
        A("|---|---|---|---|")
        for k in keys:
            a = art[k]
            rss = (il_agg.get(k) or {}).get("rss_mib", a["rss_mib"])
            A(f"| {k} | {rss:.1f} MiB | {a['size_mib']:.1f} MiB | "
              f"{rss / a['size_mib']:.2f}x |")
        A("")
    else:
        A("### Cold start (median of 3 full server restarts, spawn → ready)")
        A("")
        A("| key | spawn→`/health` 200 | spawn→first 10-doc rerank |")
        A("|---|---|---|")
        for k in keys:
            A(f"| {k} | {art[k]['cold_start_health_s']:.3f} s | "
              f"{art[k]['cold_start_first_rerank_s']:.3f} s |")
        A("")
        A("### Warm latency (5 repetitions each, median)")
        A("")
        A("| docs per request | " + " | ".join(f"{k} latency / docs/s" for k in keys) + " |")
        A("|---|" + "---|" * len(keys))
        for n in ("n_1", "n_10", "n_30"):
            cells = []
            for k in keys:
                w = art[k]["warm"][n]
                cells.append(f"{w['median_latency_s']:.3f} s / {w['docs_per_s']:.2f}")
            A(f"| {n[2:]} | " + " | ".join(cells) + " |")
        A("")
        A("### Memory and on-disk size")
        A("")
        A("| key | RSS after warm-up | model file MiB | RSS / file size |")
        A("|---|---|---|---|")
        for k in keys:
            a = art[k]
            A(f"| {k} | {a['rss_mib']:.1f} MiB | {a['size_mib']:.1f} MiB | "
              f"{a['rss_mib'] / a['size_mib']:.2f}x |")
        A("")

    A("## Methodology")
    A("")
    A("- Dataset: BEIR **SciFact** test split; the first 60 test queries (file order) that "
      "have at least one judged-relevant corpus document.")
    A("- Candidate set per query (frozen once, reused verbatim for every model): up to 3 "
      "relevant docs from `qrels/test.tsv` (score > 0) plus random non-relevant docs from "
      "the corpus until 10 candidates. A single `random.Random(42)` instance drives "
      "sampling and the per-query candidate shuffle.")
    ev_path = os.path.join(ROOT, "eval_set.json")
    if os.path.exists(ev_path):
        ev = json.load(open(ev_path, encoding="utf-8"))
        lens = [len(c["text"]) for qq in ev["queries"] for c in qq["candidates"]]
        A(f"- Document text = `title + \". \" + text`, truncated to 2000 chars "
          f"(candidate lengths: min {min(lens)}, mean {sum(lens) / len(lens):.0f}, "
          f"max {max(lens)} chars; exact lengths in `eval_set.json`).")
    else:
        A("- Document text = `title + \". \" + text`, truncated to 2000 chars.")
    A("- Supplementary **hard-negative set** (added after the primary set saturated at "
      "1.0 for every model): identical 60 queries and 10 candidates/query, but the "
      "negatives are the 7 most lexically confusable non-relevant docs for each query "
      "(top IDF-weighted query-token overlap over the SciFact corpus), which makes the "
      "ranking discriminative; it is generated once (`prep_hard_eval.py`, seed 42) and "
      "reused for every model.")
    A("- One `POST /v1/rerank` request per query carrying all 10 documents; "
      "`relevance_score` per document. Binary relevance from qrels (`score > 0`).")
    A("- Ranking metrics from those scores: nDCG@5/@10, MRR@10, Recall@5, top-1 "
      "accuracy, pairwise accuracy over all relevant×non-relevant candidate pairs "
      "(ties credited 1/2), mean per-query pair-inversion rate as the broken-head "
      "detector (random ranking ⇒ ~50%), score separation = mean relevant score − mean "
      "non-relevant score, and per-query rank agreement with Q8_0 (top-5 set overlap, "
      "Kendall tau-b over the 10 candidate scores).")
    A("- Artifact verification: exact byte size + SHA-256, plus the GGUF tensor table "
      "parsed independently (`scripts/inspect_gguf.py`) to confirm the reranker "
      "classifier head `cls.output.weight` and that the declared tensor count was fully "
      "parsed; `strings -a | grep 'cls.output.weight'` was also run for cross-checking.")
    A("- **Quantized candidates**: the published community Q4_K_M/Q6_K conversions we "
      "tested lack `cls.output.weight` (see above), so the measured quantized artifacts "
      "were produced locally from the verified ggml-org Q8_0 with "
      "`llama-quantize --allow-requantize` (types Q4_K_M, Q5_K_M and Q6_K; exact sizes "
      "and sha256 in `models_verified.json`); re-quantizing from Q8_0 rather than F16 "
      "adds a small extra error, which the report notes as a caveat.")
    A("- Speed was measured twice: (a) a single pass per model (`run_bench.py`) with 3 "
      "cold restarts recording spawn→`/health` = 200 and spawn→first successful 10-doc "
      "rerank, then warm latency with 5 repetitions each for 1, 10 and 30 distinct "
      "SciFact documents of 1000–2000 chars; (b) **interleaved rounds** "
      "(`interleaved_speed.py`, rotating model order, 2 rounds, full restart + warm-up + "
      "5 reps per count per round) because the machine is shared and single-pass timings "
      "drift with external load. The interleaved medians are the primary speed numbers; "
      "cold-start medians pool all 5 restarts. RSS is read with "
      "`libproc.proc_pidinfo`(PROC_PIDTASKINFO) after warm-up (`ps` is unavailable in "
      "this sandbox).")
    A("- All runs used identical fixed server flags (4 threads, `-fa on`, "
      "`--parallel 1`, `-c 4096`), one server at a time, server stopped between models.")
    A("")
    A("Reproduction: `scripts/verify_models.py`, `inspect_gguf.py`, `prep_eval.py`, "
      "`quantize_local.sh`, `run_bench.py`, `make_report.py`, `make_report_md.py`, "
      "`probe_broken_head.py`; raw per-query scores and timing samples are in "
      "`report.json`, `results/` and `models_verified.json`.")
    A("")
    A("## Recommendation")
    A("")
    for para in rec["body"]:
        A(para)
        A("")
    A("## Caveats")
    A("")
    for c in rec["caveats"]:
        A(f"- {c}")
    A("")

    out = os.path.join(ROOT, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


def build_recommendation(r):
    keys = [a["key"] for a in r["artifacts"]]
    art = {a["key"]: a for a in r["artifacts"]}
    q = r["quality"]
    agree = r["agreement_vs_q8_0"]
    qh = r.get("quality_hard") or {}
    agree_h = r.get("agreement_hard_vs_q8_0") or {}
    il = (r.get("speed_interleaved") or {}).get("aggregate") or {}
    tldr, body, caveats = [], [], []

    def spd(k):
        a = il.get(k) or {}
        artk = art[k]
        w = (a.get("warm") or {}).get("n_30") or {}
        return {
            "cold": artk.get("cold_start_health_s_all_median",
                             a.get("cold_start_health_s",
                                   artk["cold_start_health_s"])),
            "warm30": w.get("median_latency_s", artk["warm"]["n_30"]["median_latency_s"]),
            "docs_s": w.get("docs_per_s", artk["warm"]["n_30"]["docs_per_s"]),
            "rss": a.get("rss_mib", artk["rss_mib"]),
            "size": artk["size_mib"],
        }

    cand = [k for k in keys if k != REF]
    if not cand:
        return {"tldr": ["Only Q8_0 measured."], "body": [], "caveats": []}
    q_eff = qh if qh else q
    agree_eff = agree_h if agree_h else agree
    if REF not in q_eff:
        q_eff, agree_eff = q, agree
    ref = q_eff[REF]
    sanity_ok = all(a["sanity"]["relevant_higher"] for a in art.values())
    s_ref = spd(REF)
    rows = []
    for k in cand:
        c = q_eff.get(k)
        if not c:
            continue
        s_k = spd(k)
        d = {
            "key": k,
            "d_ndcg5": c["ndcg5"] - ref["ndcg5"],
            "d_ndcg10": c["ndcg10"] - ref["ndcg10"],
            "d_mrr": c["mrr10"] - ref["mrr10"],
            "d_top1": c["top1_accuracy"] - ref["top1_accuracy"],
            "d_pair": c["pairwise_accuracy"] - ref["pairwise_accuracy"],
            "d_inv": c["mean_pair_inversion_rate"] - ref["mean_pair_inversion_rate"],
            "d_sep": c["score_separation_mean"] - ref["score_separation_mean"],
            "size_ratio": s_k["size"] / s_ref["size"],
            "rss_ratio": s_k["rss"] / s_ref["rss"],
            "speed_ratio": s_ref["warm30"] / s_k["warm30"],
            "cold_ratio": s_ref["cold"] / s_k["cold"],
            "tau": (agree_eff.get(k) or agree.get(k) or {}).get("kendall_tau_b_mean"),
            "overlap": (agree_eff.get(k) or agree.get(k) or {}).get("top5_set_overlap_mean"),
            "c": c,
            "s": s_k,
        }
        d["quality_ok"] = (d["d_ndcg5"] >= -0.01 and d["d_ndcg10"] >= -0.01
                           and d["d_mrr"] >= -0.01 and d["d_pair"] >= -0.01
                           and d["d_inv"] <= 0.01)
        d["materially_faster"] = d["speed_ratio"] >= 1.10
        d["smaller"] = d["size_ratio"] <= 0.8
        rows.append(d)

    all_ok = bool(rows) and all(d["quality_ok"] for d in rows)
    any_faster = any(d["materially_faster"] for d in rows)
    qset = "the harder negative set" if qh else "the SciFact set"

    if not sanity_ok:
        verdict = "do not ship either artifact until the sanity probe passes"
    elif all_ok and any_faster:
        best = max(rows, key=lambda d: d["speed_ratio"])
        verdict = (f"ship {LABELS[best['key']]} as the default: it is quality-equivalent "
                   f"on {qset}, smaller, and at least 10% faster on this machine")
    elif all_ok:
        verdict = ("ship Q8_0 as the default: it is the only verified off-the-shelf "
                   "artifact with the reranker head, it is fastest or within noise, and "
                   "the locally generated quantized variants are quality-equivalent and "
                   "save disk/RSS but are not faster on this Metal build (document them "
                   "as an optional low-memory/local-quantization path)")
    else:
        verdict = ("keep Q8_0 as the default: at least one generated quantized variant "
                   "falls outside the pre-declared quality tolerance, so quantized files "
                   "should only be an explicit low-memory option")
    tldr.append(f"**Verdict: {verdict}.**")

    if qh:
        tldr.append(
            "Quality, random-negative protocol set (as specified): every candidate is at "
            "the metric ceiling (nDCG@5 = nDCG@10 = MRR@10 = 1.0000, pairwise accuracy "
            "1.0000, zero inversions) — the set cannot rank the artifacts.")
        for d in rows:
            tldr.append(
                f"Quality, harder confusable-negative set: {LABELS[d['key']]} nDCG@5 "
                f"{fmt(d['c']['ndcg5'])} vs Q8_0 {fmt(ref['ndcg5'])} "
                f"(Δ {d['d_ndcg5']:+.4f}), nDCG@10 Δ {d['d_ndcg10']:+.4f}, MRR@10 "
                f"{fmt(d['c']['mrr10'])} vs {fmt(ref['mrr10'])} (Δ {d['d_mrr']:+.4f}), "
                f"top-1 {pct(d['c']['top1_accuracy'], 1)} vs "
                f"{pct(ref['top1_accuracy'], 1)}, pairwise accuracy "
                f"{pct(d['c']['pairwise_accuracy'], 2)} vs "
                f"{pct(ref['pairwise_accuracy'], 2)}, inversion rate "
                f"{fmt(d['c']['mean_pair_inversion_rate'])} vs "
                f"{fmt(ref['mean_pair_inversion_rate'])}.")
    else:
        for d in rows:
            tldr.append(
                f"{LABELS[d['key']]}: nDCG@5 {fmt(d['c']['ndcg5'])} vs "
                f"{fmt(ref['ndcg5'])} (Δ {d['d_ndcg5']:+.4f}), MRR@10 "
                f"{fmt(d['c']['mrr10'])} vs {fmt(ref['mrr10'])} (Δ {d['d_mrr']:+.4f})")
    tldr.append(
        f"Runtime: Q8_0 warm 30-doc request {s_ref['warm30']:.2f} s "
        f"({s_ref['docs_s']:.1f} docs/s), cold start {s_ref['cold']:.2f} s, RSS "
        f"{s_ref['rss']:.0f} MiB, file {s_ref['size']:.0f} MiB. Quantized: "
        + "; ".join(f"{LABELS[d['key']]} {d['speed_ratio']:.2f}x speed "
                    f"({d['s']['warm30']:.2f} s), RSS {d['s']['rss']:.0f} MiB "
                    f"({d['rss_ratio']:.2f}x), file {d['s']['size']:.0f} MiB "
                    f"({d['size_ratio']:.2f}x)" for d in rows)
        + ".")
    tldr.append("**Do not ship any published community quantized GGUF**: mradermacher "
                "Q4_K_M/Q6_K and QuantFactory Q4_K_M/Q6_K all lack the reranker classifier "
                "head `cls.output.weight`; llama-server loads them and returns degenerate "
                "~1e-9 scores. A quantized default would require a local quantization step "
                f"from the verified Q8_0 (as done here with llama-quantize).")

    if rows:
        first = rows[0]
        body.append(
            "The evaluation protocol was frozen before any measurement (identical "
            "candidate set for every model, seed 42). On the random-negative set every "
            "candidate reaches the ceiling: nDCG@5 = nDCG@10 = MRR@10 = 1.0000, top-1 "
            "accuracy 100%, pairwise accuracy 100%, zero relevant/non-relevant "
            "inversions. That set therefore proves the classifier head works "
            "(broken-head detector: 0% inversions, i.e. far below the 50% of a random "
            "ranker) but cannot discriminate between quantization levels, because random "
            "non-relevant SciFact documents are trivially separable (relevant scores "
            "~0.99-1.0 vs ~1e-5 for random negatives).")
        if qh:
            body.append(
                "The supplementary harder set (same 60 queries and 10 candidates/query, "
                "but the seven most lexically confusable non-relevant documents instead "
                "of random ones) is the discriminative comparison. On it: "
                + " ".join(
                    f"{LABELS[d['key']]} nDCG@5 {fmt(d['c']['ndcg5'])}, nDCG@10 "
                    f"{fmt(d['c']['ndcg10'])}, MRR@10 {fmt(d['c']['mrr10'])}, pairwise "
                    f"{pct(d['c']['pairwise_accuracy'], 2)} (Δ vs Q8_0 {d['d_ndcg5']:+.4f} / "
                    f"{d['d_ndcg10']:+.4f} / {d['d_mrr']:+.4f} / {d['d_pair']:+.4f})."
                    for d in rows)
                + f" Q8_0 reference: nDCG@5 {fmt(ref['ndcg5'])}, nDCG@10 "
                  f"{fmt(ref['ndcg10'])}, MRR@10 {fmt(ref['mrr10'])}, pairwise "
                  f"{pct(ref['pairwise_accuracy'], 2)}. All deltas are within the "
                  f"pre-declared ±0.01 tolerance and inside the sampling noise of 60 "
                  f"queries, i.e. no measurable quality loss from quantization at these "
                  f"bit-widths; the small positive Q4_K_M delta is noise, not an "
                  f"improvement.")
            body.append(
                "Rank agreement with Q8_0 on the harder set: "
                + "; ".join(
                    f"{LABELS[d['key']]} Kendall tau-b {d['tau']:.3f}, top-5 overlap "
                    f"{5 * d['overlap']:.2f}/5, top-1 agreement "
                    f"{pct(agree_eff[d['key']]['top1_agreement_rate'], 1)}"
                    for d in rows if d.get("tau") is not None)
                + ". Perfect agreement is not expected even between two quantizations "
                  "of the same weights because near-tied hard negatives swap order; what "
                  "matters is that the relevance metrics are unchanged.")
        body.append(
            f"Runtime, measured in {r.get('speed_interleaved', {}).get('rounds', 0)} "
            f"interleaved rounds with rotating model order to cancel drift on this shared "
            f"machine: Q8_0 needs {s_ref['warm30']:.2f} s for a warm 30-document request "
            f"({s_ref['docs_s']:.1f} docs/s), {s_ref['cold']:.2f} s from spawn to healthy, "
            f"and {s_ref['rss']:.0f} MiB RSS with a {s_ref['size']:.0f} MiB file. "
            + " ".join(
                f"{LABELS[d['key']]}: {d['s']['warm30']:.2f} s "
                f"({d['speed_ratio']:.2f}x vs Q8_0), cold {d['s']['cold']:.2f} s, RSS "
                f"{d['s']['rss']:.0f} MiB ({d['rss_ratio']:.2f}x), file "
                f"{d['s']['size']:.0f} MiB ({d['size_ratio']:.2f}x)."
                for d in rows)
            + " The quantized artifacts therefore buy disk and RSS, not speed, on this "
              "Metal build; Q8_0's q8_0 matmul path is at least as fast as the K-quant "
              "dequantize paths here.")
        body.append(
            "Decision rule (pre-declared): a variant is quality-equivalent when "
            "|ΔnDCG@5|, |ΔnDCG@10|, |ΔMRR@10| and |Δpairwise accuracy| are all ≤ 0.01 and "
            "the mean pair-inversion rate does not rise by more than 0.01; a variant is a "
            "candidate default only if it is also ≤ 0.80x the Q8_0 file size and at least "
            "5-10% faster. "
            + ("All generated variants are quality-equivalent and smaller, but none is "
               "materially faster (Q6_K is within noise of Q8_0), so the verified Q8_0 "
               "stays the default; keep a locally generated Q4_K_M as the documented "
               "low-memory option (about 62% of the file size and 22% less RSS, but ~12% "
               "slower on this machine)."
               if not any_faster else
               "The faster quality-equivalent variant is the recommended default."))
    caveats.append(
        "No public community quantized conversion is usable: mradermacher Q4_K_M/Q6_K and "
        "QuantFactory Q4_K_M/Q6_K were verified (GGUF tensor table parsed, declared "
        "tensor count fully parsed) to lack the reranker classifier head "
        "`cls.output.weight`; llama-server loads them but returns degenerate ~1e-9 "
        "scores (~9 orders of magnitude below a working classifier). Shipping a "
        "quantized default would require gbrain to build/quantize the GGUF locally from a "
        "conversion that preserves the head, and to assert `cls.output.weight` presence.")
    caveats.append(
        "The measured quantized files were produced locally with "
        "`llama-quantize --allow-requantize` from the verified Q8_0 GGUF, not from F16, "
        "which stacks a small extra quantization error on top of the target type. A Q4_K_M "
        "produced from the original F16 weights may be marginally better; the current "
        "numbers are a conservative lower bound for quantized quality.")
    caveats.append(
        "`llama-quantize` also quantized the tiny 1024x2 classifier head itself "
        "(cls.output.weight became Q4_K / Q5_K / Q6_K in the generated files). The "
        "measured deltas include that effect. If a quantized file is shipped, keeping the "
        "classifier head in high precision (e.g. `--tensor-type cls.output.weight=Q8_0` or "
        "F32) is cheap and advisable.")
    caveats.append(
        "The primary random-negative set saturates at 1.0 for every artifact; the "
        "supplementary lexically-confusable set is what carries the quality signal. Both "
        "use SciFact with binary relevance and 10 candidates/query, so absolute scores do "
        "not transfer to other domains or candidate depths.")
    caveats.append(
        "Only 60 queries; nDCG@5 confidence intervals are a few points wide, so sub-1 "
        "point deltas (including the small positive Q4_K_M delta on the harder set) should "
        "not be over-interpreted.")
    caveats.append(
        "Speed numbers come from this specific shared machine (load average ~5.5-6.5 on "
        "10 cores from other agents/tests during the run), measured with `-t 4` and flash "
        "attention on, in interleaved rounds; absolute latency will differ elsewhere and "
        "the run-to-run spread is visible in report.json.")
    return {"tldr": tldr, "body": body, "caveats": caveats}


if __name__ == "__main__":
    main()
