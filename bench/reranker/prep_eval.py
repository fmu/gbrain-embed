#!/usr/bin/env python3
"""Build the fixed SciFact evaluation set used for every candidate model.

Protocol (frozen before any model is measured):
  - first 60 test queries (queries.jsonl order) that have >= 1 relevant corpus doc
  - candidates per query = up to 3 relevant docs (qrels score>0) + random
    non-relevant docs until 10 candidates total
  - one random.Random(42) instance drives sampling and candidate shuffling, so
    the exact same candidate lists are reused for every model.
  - document text = title + ". " + text, truncated to 2000 chars.
Also stores a disjoint latency document pool (~1000-2000 char docs).
"""
import json, os, random, sys

ROOT = "/tmp/gbrain-rerank-bench"
DATA = os.path.join(ROOT, "scifact")
OUT = os.path.join(ROOT, "eval_set.json")
N_QUERIES = 60
N_CANDIDATES = 10
MAX_REL = 3
DOC_CHARS = 2000
SEED = 42


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def doc_text(d):
    t = (d.get("title") or "").strip()
    b = (d.get("text") or "").strip()
    return f"{t}. {b}" if t else b


def main():
    corpus = {str(d["_id"]): d for d in load_jsonl(os.path.join(DATA, "corpus.jsonl"))}
    queries = load_jsonl(os.path.join(DATA, "queries.jsonl"))

    qrels = {}
    with open(os.path.join(DATA, "qrels", "test.tsv"), encoding="utf-8") as f:
        header = f.readline()
        assert header.strip().split("\t")[:3] == ["query-id", "corpus-id", "score"], header
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, cid, score = line.split("\t")[:3]
            qrels.setdefault(str(qid), []).append((str(cid), float(score)))

    # first 60 test queries with >=1 relevant corpus doc
    selected = []
    for q in queries:
        qid = str(q["_id"])
        if qid not in qrels:
            continue
        rels = [(cid, s) for cid, s in qrels[qid] if s > 0 and cid in corpus]
        if not rels:
            continue
        selected.append((qid, q["text"], rels))
        if len(selected) == N_QUERIES:
            break
    if len(selected) != N_QUERIES:
        sys.exit(f"only {len(selected)} usable test queries, need {N_QUERIES}")

    rng = random.Random(SEED)
    all_ids = list(corpus.keys())
    eval_queries = []
    for qid, qtext, rels in selected:
        rel_ids = [cid for cid, _ in rels][:MAX_REL]
        excluded = {cid for cid, _ in qrels[qid]}
        negatives = [cid for cid in all_ids if cid not in excluded]
        n_neg = N_CANDIDATES - len(rel_ids)
        neg_ids = rng.sample(negatives, n_neg)
        cand = [{"doc_id": cid, "label": 1} for cid in rel_ids] + [
            {"doc_id": cid, "label": 0} for cid in neg_ids
        ]
        rng.shuffle(cand)
        for c in cand:
            c["text"] = doc_text(corpus[c["doc_id"]])[:DOC_CHARS]
        eval_queries.append(
            {
                "qid": qid,
                "query": qtext,
                "n_relevant": len(rel_ids),
                "candidates": cand,
            }
        )

    # latency pool: disjoint real docs of typical length 1000-2000 chars
    latency_pool = []
    used = set()
    for cid, d in corpus.items():
        if cid in used:
            continue
        txt = doc_text(d)
        if 1000 <= len(txt) <= 2000:
            latency_pool.append({"doc_id": cid, "text": txt})
            used.add(cid)
        if len(latency_pool) >= 80:
            break
    if len(latency_pool) < 51:
        sys.exit(f"latency pool too small: {len(latency_pool)}")

    latency = {
        "query": eval_queries[0]["query"],
        "docs_1": [d["text"] for d in latency_pool[0:1]],
        "docs_10": [d["text"] for d in latency_pool[1:11]],
        "docs_30": [d["text"] for d in latency_pool[11:41]],
        "docs_10_coldstart": [d["text"] for d in latency_pool[41:51]],
        "doc_ids_10": [d["doc_id"] for d in latency_pool[1:11]],
    }

    out = {
        "meta": {
            "dataset": "BEIR SciFact",
            "seed": SEED,
            "n_queries": len(eval_queries),
            "n_candidates": N_CANDIDATES,
            "max_relevant": MAX_REL,
            "doc_char_limit": DOC_CHARS,
            "doc_text_format": "title + '. ' + text",
            "latency_doc_char_range": [1000, 2000],
        },
        "queries": eval_queries,
        "latency": latency,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    lens = [len(c["text"]) for q in eval_queries for c in q["candidates"]]
    nrel = [q["n_relevant"] for q in eval_queries]
    print(f"wrote {OUT}: {len(eval_queries)} queries, "
          f"{sum(len(q['candidates']) for q in eval_queries)} candidates, "
          f"relevant/query min/mean/max = {min(nrel)}/{sum(nrel)/len(nrel):.2f}/{max(nrel)}, "
          f"doc chars min/mean/max = {min(lens)}/{sum(lens)/len(lens):.0f}/{max(lens)}, "
          f"latency pool docs = {len(latency_pool)}")


if __name__ == "__main__":
    main()
