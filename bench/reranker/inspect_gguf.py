#!/usr/bin/env python3
"""Minimal GGUF header/tensor-table parser used to verify that a candidate GGUF
carries the Qwen3 reranker classifier head (`cls.output.weight`).

This is deliberately independent of llama.cpp: it parses the documented GGUF
layout (magic, version, tensor_count, metadata k/v pairs, tensor infos) and
reports the *complete* tensor-name list when the parsed name count equals the
declared tensor_count. Works on a full file or on a header slice (first N MiB),
so a broken published conversion can be rejected from its header alone without
downloading the whole file.

Usage:
  python3 inspect_gguf.py <file> [--max-bytes N] [--json]
"""
import json
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# gguf metadata value types
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY,
 UINT64, INT64, FLOAT64) = range(13)


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def need(self, n):
        if self.o + n > len(self.b):
            raise EOFError(f"need {n} bytes at {self.o}, have {len(self.b) - self.o}")

    def u8(self):
        self.need(1)
        v = self.b[self.o]
        self.o += 1
        return v

    def u16(self):
        self.need(2)
        v = struct.unpack_from("<H", self.b, self.o)[0]
        self.o += 2
        return v

    def u32(self):
        self.need(4)
        v = struct.unpack_from("<I", self.b, self.o)[0]
        self.o += 4
        return v

    def u64(self):
        self.need(8)
        v = struct.unpack_from("<Q", self.b, self.o)[0]
        self.o += 8
        return v

    def i32(self):
        self.need(4)
        v = struct.unpack_from("<i", self.b, self.o)[0]
        self.o += 4
        return v

    def f32(self):
        self.need(4)
        v = struct.unpack_from("<f", self.b, self.o)[0]
        self.o += 4
        return v

    def f64(self):
        self.need(8)
        v = struct.unpack_from("<d", self.b, self.o)[0]
        self.o += 8
        return v

    def string(self):
        n = self.u64()
        if n > 1 << 28:
            raise ValueError(f"implausible string length {n} at {self.o}")
        self.need(n)
        v = self.b[self.o:self.o + n]
        self.o += n
        return v.decode("utf-8", "replace")


FIXED = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<B", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}


def read_scalar(r, t):
    fmt, n = FIXED[t]
    r.need(n)
    v = struct.unpack_from(fmt, r.b, r.o)[0]
    r.o += n
    return v


def skip_value(r, t):
    if t == ARRAY:
        et = r.u32()
        n = r.u64()
        if et == STRING:
            for _ in range(n):
                r.string()
        elif et in FIXED:
            r.need(FIXED[et][1] * n)
            r.o += FIXED[et][1] * n
        elif et == ARRAY:
            for _ in range(n):
                skip_value(r, ARRAY)
        else:
            raise ValueError(f"unknown array elem type {et}")
        return {"type": "array", "elem_type": et, "count": n}
    if t == STRING:
        return r.string()
    if t in FIXED:
        return read_scalar(r, t)
    raise ValueError(f"unknown metadata value type {t}")


def parse(buf, max_meta_items=1_000_000):
    r = Reader(buf)
    magic = r.u32()
    if magic != GGUF_MAGIC:
        raise ValueError(f"not a GGUF file (magic=0x{magic:08x})")
    version = r.u32()
    tensor_count = r.u64()
    kv_count = r.u64()
    out = {"magic": "GGUF", "version": version, "declared_tensor_count": tensor_count,
           "metadata_kv_count": kv_count, "metadata": {}}
    for _ in range(min(kv_count, max_meta_items)):
        key = r.string()
        t = r.u32()
        try:
            out["metadata"][key] = skip_value(r, t)
        except (EOFError, ValueError) as e:
            out["header_parse_error"] = f"{type(e).__name__}: {e}"
            out["tensor_names"] = []
            out["complete_tensor_table"] = False
            return out
    names = []
    try:
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ggml_type = r.u32()
            offset = r.u64()
            names.append({"name": name, "dims": dims, "type": ggml_type,
                          "offset": offset})
    except (EOFError, ValueError) as e:
        out["header_parse_error"] = f"{type(e).__name__}: {e}"
    out["tensors"] = names
    out["tensor_names"] = [t["name"] for t in names]
    out["complete_tensor_table"] = len(names) == tensor_count
    out["header_bytes_consumed"] = r.o
    return out


def main():
    path = sys.argv[1]
    max_bytes = None
    as_json = "--json" in sys.argv
    if "--max-bytes" in sys.argv:
        max_bytes = int(sys.argv[sys.argv.index("--max-bytes") + 1])
    with open(path, "rb") as f:
        buf = f.read(max_bytes) if max_bytes else f.read()
    info = parse(buf)
    names = info["tensor_names"]
    result = {
        "file": path,
        "file_bytes_read": len(buf),
        "parsed_bytes_consumed": info.get("header_bytes_consumed"),
        "gguf_version": info.get("version"),
        "architecture": info.get("metadata", {}).get("general.architecture"),
        "declared_tensor_count": info.get("declared_tensor_count"),
        "tensor_names_parsed": len(names),
        "complete_tensor_table": info.get("complete_tensor_table"),
        "header_parse_error": info.get("header_parse_error"),
        "has_cls_output_weight": "cls.output.weight" in names,
        "has_score_weight": "score.weight" in names,
        "has_output_weight": "output.weight" in names,
        "cls_like_tensors": [n for n in names if n.startswith(("cls.", "score."))],
        "output_like_tensors": [n for n in names
                                if n in ("output.weight", "output_norm.weight")],
    }
    if "cls.output.weight" in names:
        result["cls_output_weight_shape"] = next(
            t["dims"] for t in info["tensors"] if t["name"] == "cls.output.weight")
        result["cls_output_weight_ggml_type"] = next(
            t["type"] for t in info["tensors"] if t["name"] == "cls.output.weight")
    if as_json:
        print(json.dumps(result, indent=1))
    else:
        for k, v in result.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
