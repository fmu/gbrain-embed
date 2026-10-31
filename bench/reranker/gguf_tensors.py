#!/usr/bin/env python3
"""List GGUF tensor names, fetching only the header bytes via HTTP Range."""
import json, struct, sys, urllib.request

CHUNK = 16 * 1024 * 1024

class Reader:
    def __init__(self, url=None, path=None):
        self.url = url; self.path = path
        self.data = b""; self.pos = 0
        self.fh = open(path, "rb") if path else None
    def _fetch(self, end):
        if end <= len(self.data): return
        target = max(end, len(self.data) + CHUNK)
        if self.fh is not None:
            self.fh.seek(len(self.data))
            self.data += self.fh.read(target - len(self.data))
            return
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={len(self.data)}-{target-1}", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            chunk = r.read()
        if not chunk: raise EOFError("empty range response")
        self.data += chunk
    def read(self, n):
        if self.pos + n > len(self.data):
            self._fetch(self.pos + n)
        out = self.data[self.pos:self.pos + n]
        if len(out) != n: raise EOFError(f"short read {len(out)} != {n}")
        self.pos += n
        return out
    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.read(8))[0]
    def string(self):
        n = self.u64(); return self.read(n).decode("utf-8", "replace")

SCALARS = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def skip_value(r, t):
    if t == 8: r.read(r.u64()); return
    if t == 9:
        et = r.u32(); n = r.u64()
        if et == 8:
            for _ in range(n): r.read(r.u64())
        else:
            r.read(SCALARS[et]*n)
        return
    r.read(SCALARS[t])

def main():
    src = sys.argv[1]
    r = Reader(path=src) if not src.startswith("http") else Reader(url=src)
    magic = r.read(4)
    if magic != b"GGUF": print("not GGUF", magic); sys.exit(1)
    version = r.u32(); ntensors = r.u64(); nkv = r.u64()
    for _ in range(nkv):
        r.string(); t = r.u32(); skip_value(r, t)
    names = []
    for _ in range(ntensors):
        names.append(r.string())
        nd = r.u32()
        for _ in range(nd): r.u64()
        r.u32(); r.u64()
    found = [n for n in names if "cls" in n.lower() or "classifier" in n.lower() or "score" in n.lower()]
    print(json.dumps({"version": version, "tensors": ntensors, "kv": nkv,
                      "cls_like": found, "has_cls_output": "cls.output.weight" in names,
                      "header_bytes": r.pos, "first_tensors": names[:3], "last_tensors": names[-3:]}))

main()
