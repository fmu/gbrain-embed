#!/usr/bin/env python3
"""Build the supplementary HARD-negative evaluation set.

Motivation: the primary protocol (random non-relevant documents) saturates the
metrics at 1.0 for every candidate, because random SciFact docs are trivially
separable from the cited relevant docs. This set keeps the identical 60 queries
and 10-candidates-per-query layout, but replaces the random negatives with
lexically confusable non-relevant docs: the top-7 non-relevant docs by
IDF-weighted query-term overlap. That makes the task discriminative enough to
reveal quantization quality differences.

Writes /tmp/qwen3-rerank-bench/eval_set_hard.json
"""
import json
import math
import os
import random
import re
import sys
from collections import Counter

ROOT = "/tmp/qwen3-rerank-bench"
DATA = os.path.join(ROOT, "scifact")
OUT = os.path.join(ROOT, "eval_set_hard.json")
SEED = 42
DOC_CHARS = 2000
N_CANDIDATES = 10
MAX_REL = 3

STOP = set("""a an and are as at be been by for from has have how in is it its of on or
that the their there these this to was were what when where which who why will with
does do did not no yes can could should would may might if then than into about
over under between among during before after above below up down out off again
further once here all any both each few more most other some such only own same so
too very s t just""".split())

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1 and t not in STOP]


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def doc_text(d):
    t = (d.get("title") or "").strip()
    b = (d.get("text") or "").strip()
    return f"{t}. {b}" if t else b


def main():
    corpus = {str(d["_id"]): d for d in load_jsonl(os.path.join(DATA, "corpus.jsonl"))}
    queries = {str(q["_id"]): q["text"] for q in load_jsonl(os.path.join(DATA, "queries.jsonl"))}
    qrels = {}
    with open(os.path.join(DATA, "qrels", "test.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            qid, cid, score = line.strip().split("\t")[:3]
            qrels.setdefault(str(qid), []).append((str(cid), float(score)))

    with open(os.path.join(ROOT, "eval_set.json"), encoding="utf-8") as f:
        primary = json.load(f)
    qids = [q["qid"] for q in primary["queries"]]

    # doc token sets + document frequencies
    doc_tokens = {}
    df = Counter()
    for cid, d in corpus.items():
        toks = tokens(doc_text(d))
        doc_tokens[cid] = set(toks)
        df.update(set(toks))
    n_docs = len(corpus)
    idf = {t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    rng = random.Random(SEED)
    hard_queries = []
    for qid in qids:
        qtext = queries[qid]
        rels = [(cid, s) for cid, s in qrels[qid] if s > 0 and cid in corpus]
        rel_ids = [cid for cid, _ in rels][:MAX_REL]
        excluded = {cid for cid, _ in qrels[qid]}
        qtok = list(dict.fromkeys(tokens(qtext)))
        scored = []
        for cid in corpus:
            if cid in excluded:
                continue
            dt = doc_tokens[cid]
            s = sum(idf.get(t, 1.0) for t in qtok if t in dt)
            if s > 0:
                scored.append((s, cid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        n_neg = N_CANDIDATES - len(rel_ids)
        hard = [cid for _, cid in scored[:n_neg]]
        if len(hard) < n_neg:  # extremely defensive; fall back to random fill
            pool = [cid for cid in corpus if cid not in excluded and cid not in hard]
            hard += rng.sample(pool, n_neg - len(hard))
        cand = [{"doc_id": cid, "label": 1} for cid in rel_ids] + [
            {"doc_id": cid, "label": 0} for cid in hard
        ]
        rng.shuffle(cand)
        for c in cand:
            c["text"] = doc_text(corpus[c["doc_id"]])[:DOC_CHARS]
        hard_queries.append(
            {"qid": qid, "query": qtext, "n_relevant": len(rel_ids), "candidates": cand}
        )

    out = {
        "meta": {
            "dataset": "BEIR SciFact",
            "variant": "hard negatives: top non-relevant docs by IDF-weighted query-token "
                       "overlap (lexically confusable)",
            "seed": SEED,
            "n_queries": len(hard_queries),
            "n_candidates": N_CANDIDATES,
            "max_relevant": MAX_REL,
            "doc_char_limit": DOC_CHARS,
        },
        "queries": hard_queries,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    lens = [len(c["text"]) for q in hard_queries for c in q["candidates"]]
    print(f"wrote {OUT}: {len(hard_queries)} queries, doc chars "
          f"min/mean/max = {min(lens)}/{sum(lens)/len(lens):.0f}/{max(lens)}")
    # quick difficulty signal: overlap of query terms between relevant and hard docs
    print("sample qid", hard_queries[0]["qid"], "query:", hard_queries[0]["query"])
    print("labels:", [c["label"] for c in hard_queries[0]["candidates"]])


if __name__ == "__main__":
    main()
