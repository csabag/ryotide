"""Compare two runs item by item: accuracy, agreement, probability drift, marker mass.

    uv run python bench/compare_runs.py <reference-tag> <new-tag>

Tags are directories under results/jevbench/. Only items present in both runs are
compared, so a hard-tier run can be checked against a full 231-item reference.
"""
from __future__ import annotations

import json
import os
import sys


def load(tag: str) -> dict:
    path = os.path.join("results", "jevbench", tag, "results.jsonl")
    if not os.path.exists(path):
        sys.exit(f"no results at {path}")
    return {r["task_id"]: r for r in map(json.loads, open(path))}


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ref, new = load(sys.argv[1]), load(sys.argv[2])
    ids = sorted(ref.keys() & new.keys())
    if not ids:
        sys.exit("the two runs share no task ids")
    agree = sum(ref[i]["predicted"] == new[i]["predicted"] for i in ids)
    diffs = sorted(abs(ref[i]["probs"][k] - new[i]["probs"][k])
                   for i in ids for k in new[i]["probs"] if k in ref[i]["probs"])
    mass = [new[i].get("runtime", {}).get("marker_mass") for i in ids]
    mass = [m for m in mass if m is not None]
    rt = new[ids[0]].get("runtime", {})
    print(f"compared items         {len(ids)}")
    print(f"accuracy  reference    {sum(ref[i]['correct'] for i in ids)}/{len(ids)}   ({sys.argv[1]})")
    print(f"accuracy  new          {sum(new[i]['correct'] for i in ids)}/{len(ids)}   ({sys.argv[2]})")
    print(f"identical predictions  {agree}/{len(ids)}  ({100 * agree / len(ids):.1f}%)")
    print(f"probability drift      median {diffs[len(diffs) // 2]:.4f}   p95 "
          f"{diffs[int(len(diffs) * .95)]:.4f}   max {diffs[-1]:.4f}")
    if mass:
        low = sum(m < 0.9 for m in mass)
        print(f"marker mass (new)      min {min(mass):.3f}   items below 0.9: {low}")
    print(f"new run                device={rt.get('device')} quant={rt.get('quant')} "
          f"offloaded={rt.get('low_vram_offloaded')} prefix={rt.get('answer_prefix')!r}")
    flips = [i for i in ids if ref[i]["predicted"] != new[i]["predicted"]]
    for i in flips[:10]:
        print(f"  differs: {i}: {ref[i]['predicted']} -> {new[i]['predicted']}"
              f"{'  (now correct)' if new[i]['correct'] else '  (now wrong)' if ref[i]['correct'] else ''}")
    verdict = ("MATCH" if agree == len(ids) else
               "CLOSE" if agree / len(ids) >= 0.9 and (not mass or min(mass) >= 0.9) else
               "INVESTIGATE")
    print(f"verdict                {verdict}")


if __name__ == "__main__":
    main()
