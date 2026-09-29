"""A/B: the same suite rows to two servers (alternating who goes first); time and answer diff."""
import json, statistics, sys, time, zlib
import httpx
from decision_index.suite.io import Suite
from pathlib import Path

A, B, MOD = sys.argv[1], sys.argv[2], int(sys.argv[3])
suite = Suite(Path("suite-0.2"), "0.2.1")
rows = [r for r in suite.rows() if zlib.crc32(r["_evaluation"]["run_id"].encode()) % MOD == 7]
c = httpx.Client(timeout=600)
ta, tb, same, n, maxdiff = [], [], 0, 0, 0.0
per = {}
for i, r in enumerate(rows):
    body = {"model": "x", "state": r["state"], "questions": r["questions"]}
    res = {}
    for url in ((A, B) if i % 2 == 0 else (B, A)):
        t = time.perf_counter(); j = c.post(url + "/v1/systemone", json=body).json(); res[url] = (time.perf_counter() - t, j)
    ta.append(res[A][0]); tb.append(res[B][0])
    ja, jb = res[A][1], res[B][1]
    for k, qa in (ja.get("answers") or {}).items():
        qb = (jb.get("answers") or {}).get(k, {})
        n += 1; same += qa.get("answer") == qb.get("answer")
        pa, pb = qa.get("probabilities") or {}, qb.get("probabilities") or {}
        for o in pa:
            d = abs(pa[o] - pb.get(o, 0))
            if d > maxdiff:
                maxdiff = d; worst = (r["_evaluation"]["run_id"], k, o, pa[o], pb.get(o), qa.get("answer"), type(pa[o]).__name__)
    b = r["_evaluation"].get("dataset") or r["_evaluation"]["catalog_id"]
    per.setdefault(b, [0, 0.0, 0.0]); per[b][0] += 1; per[b][1] += res[A][0]; per[b][2] += res[B][0]
print("worst", worst)
print(f"rows {len(rows)}  questions {n}  same answer {same}/{n}  max prob diff {maxdiff:.4f}")
print(f"A {A}: total {sum(ta):.1f}s median {statistics.median(ta)*1000:.0f} ms | B {B}: total {sum(tb):.1f}s median {statistics.median(tb)*1000:.0f} ms | B/A total {sum(tb)/sum(ta):.2f}x")
for b, (k, sa, sb) in sorted(per.items(), key=lambda x: -x[1][2])[:8]:
    print(f"  {str(b)[:28]:28s} n={k:3d}  A {sa:6.1f}s  B {sb:6.1f}s")
