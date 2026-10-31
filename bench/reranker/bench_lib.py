#!/usr/bin/env python3
"""Shared helpers: llama-server lifecycle, rerank HTTP, RSS measurement."""
import ctypes
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

LLAMA = "/opt/homebrew/bin/llama-server"
HOST = "127.0.0.1"
PORT = 18084
BASE = f"http://{HOST}:{PORT}"
ALIAS = "qwen3-reranker-0.6b"
ROOT = "/tmp/gbrain-rerank-bench"
LOGS = os.path.join(ROOT, "logs")

SERVER_ARGS = [
    "--alias", ALIAS,
    "--reranking",
    "--pooling", "rank",
    "-c", "4096",
    "-b", "2048",
    "-ub", "2048",
    "--parallel", "1",
    "-t", "4",
    "-fa", "on",
    "--host", HOST,
    "--port", str(PORT),
    "--no-webui",
]


# ---------------- RSS via libproc (ps is unavailable in this sandbox) ----------
class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libproc.proc_pidinfo.restype = ctypes.c_int
_libproc.proc_pidinfo.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
    ctypes.c_void_p, ctypes.c_int,
]
PROC_PIDTASKINFO = 4


def rss_mib(pid):
    """Resident set size of pid in MiB (macOS libproc.proc_pidinfo)."""
    info = _ProcTaskInfo()
    rv = _libproc.proc_pidinfo(pid, PROC_PIDTASKINFO, 0, ctypes.byref(info),
                               ctypes.sizeof(info))
    if rv != ctypes.sizeof(info):
        raise OSError(ctypes.get_errno(), f"proc_pidinfo failed rv={rv}")
    return info.pti_resident_size / (1024.0 * 1024.0)


# ---------------- HTTP ---------------------------------------------------------
def http_json(url, payload=None, timeout=300.0):
    """Returns (latency_seconds, parsed_json)."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return time.perf_counter() - t0, json.loads(body)


def health(timeout=2.0):
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=timeout) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return False
    except Exception:
        return False


# ---------------- server lifecycle --------------------------------------------
def start_server(model_path, log_name):
    os.makedirs(LOGS, exist_ok=True)
    log_path = os.path.join(LOGS, log_name)
    log = open(log_path, "wb")
    cmd = [LLAMA, "-m", model_path] + SERVER_ARGS
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    return p, log, t0, log_path


def wait_health(p, t0, deadline_s=240.0, poll=0.05):
    """Wait until /health returns 200; returns seconds since spawn t0."""
    while time.perf_counter() - t0 < deadline_s:
        rc = p.poll()
        if rc is not None:
            raise RuntimeError(f"llama-server exited early rc={rc}")
        if health():
            return time.perf_counter() - t0
        time.sleep(poll)
    raise TimeoutError(f"llama-server not healthy after {deadline_s}s")


def stop_server(p, log):
    if p is None:
        return
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGKILL)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    try:
        log.close()
    except Exception:
        pass


def wait_port_free(timeout=20.0):
    """Ensure nothing still answers on the port before the next start."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if not health(timeout=0.5):
            return True
        time.sleep(0.2)
    return not health(timeout=0.5)


def rerank(query, documents, timeout=300.0):
    """POST /v1/rerank; returns (latency_s, results sorted by index, raw)."""
    lat, resp = http_json(
        BASE + "/v1/rerank",
        {"model": ALIAS, "query": query, "documents": documents},
        timeout=timeout,
    )
    if "results" not in resp:
        raise RuntimeError(f"unexpected rerank response: {str(resp)[:500]}")
    return lat, sorted(resp["results"], key=lambda r: r["index"]), resp


def score_list(results, n):
    """Return per-document relevance scores aligned with document index."""
    scores = [None] * n
    for r in results:
        scores[int(r["index"])] = float(r["relevance_score"])
    if any(s is None for s in scores):
        raise RuntimeError(f"missing indices in rerank results: {results}")
    return scores
